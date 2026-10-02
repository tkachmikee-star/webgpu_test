import { mat4 } from 'https://unpkg.com/gl-matrix@3.4.3/esm/index.js';

const GRID_N     = 48;
const SURFACE_SZ = 4.0;
const DT         = 1 / 60;
const MAX_VERTS  = (GRID_N + 1) * (GRID_N + 1);

const cam = { yaw: 0.6, pitch: 0.55, dist: 7.5, target: [0, 0, 0] };
let dragging = false, lastX = 0, lastY = 0;

const canvas = document.getElementById('gpuCanvas');

canvas.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup',   () => dragging = false);
window.addEventListener('mousemove', e => {
    if (!dragging) return;
    cam.yaw   -= (e.clientX - lastX) * 0.008;
    cam.pitch -= (e.clientY - lastY) * 0.008;
    cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
    lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener('wheel', e => {
    e.preventDefault();
    cam.dist = Math.max(2.5, Math.min(20, cam.dist * (1 + Math.sign(e.deltaY) * 0.1)));
}, { passive: false });

function resize() {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = Math.floor(canvas.clientWidth  * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
}

function buildMVP(aspect) {
    const proj = mat4.create();
    mat4.perspective(proj, Math.PI / 4, aspect, 0.1, 100);
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw),   sy = Math.sin(cam.yaw);
    const eye = [
        cam.target[0] + cam.dist * cp * sy,
        cam.target[1] + cam.dist * sp,
        cam.target[2] + cam.dist * cp * cy,
    ];
    const view = mat4.create();
    mat4.lookAt(view, eye, cam.target, [0, 1, 0]);
    const mvp = mat4.create();
    mat4.multiply(mvp, proj, view);
    return mvp;
}

async function main() {
    if (!navigator.gpu) { alert('WebGPU not supported.'); return; }

    const adapter = await navigator.gpu.requestAdapter();
    const device  = await adapter.requestDevice();

    const context = canvas.getContext('webgpu');
    const format  = navigator.gpu.getPreferredCanvasFormat();

    resize();
    context.configure({ device, format, alphaMode: 'opaque' });

    const simCode    = await (await fetch('./sim.wgsl')).text();
    const renderCode = await (await fetch('./render.wgsl')).text();
    const simModule    = device.createShaderModule({ code: simCode });
    const renderModule = device.createShaderModule({ code: renderCode });

    const PARTICLE_FLOATS = 8;
    const storageBuffer = device.createBuffer({
        size: MAX_VERTS * PARTICLE_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const initial = new Float32Array(MAX_VERTS * PARTICLE_FLOATS);
    for (let r = 0; r <= GRID_N; r++) {
        for (let c = 0; c <= GRID_N; c++) {
            const i = r * (GRID_N + 1) + c;
            const fu = c / GRID_N;
            const fv = r / GRID_N;
            initial[i * 8 + 0] = (fu - 0.5) * SURFACE_SZ;
            initial[i * 8 + 1] = 0;
            initial[i * 8 + 2] = (fv - 0.5) * SURFACE_SZ;
        }
    }
    device.queue.writeBuffer(storageBuffer, 0, initial);

    const PARAMS_SIZE = 128;
    const paramsBuffer = device.createBuffer({
        size: PARAMS_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const paramsData = new Float32Array(PARAMS_SIZE / 4);

    const cameraBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const cameraData = new Float32Array(16);

    const simBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE,
              buffer: { type: 'uniform' } },
        ],
    });

    const renderBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX,
              buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX,
              buffer: { type: 'uniform' } },
        ],
    });

    const simBindGroup = device.createBindGroup({
        layout: simBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: storageBuffer } },
            { binding: 1, resource: { buffer: paramsBuffer } },
        ],
    });

    const renderBindGroup = device.createBindGroup({
        layout: renderBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: storageBuffer } },
            { binding: 1, resource: { buffer: cameraBuffer } },
        ],
    });

    const simPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [simBindGroupLayout],
        }),
        compute: { module: simModule, entryPoint: 'cs_main' },
    });

    const renderPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [renderBindGroupLayout],
        }),
        vertex:   { module: renderModule, entryPoint: 'vs_main' },
        fragment: { module: renderModule, entryPoint: 'fs_main',
                    targets: [{ format }] },
        primitive: { topology: 'line-list' },
        depthStencil: {
            format: 'depth24plus',
            depthWriteEnabled: true,
            depthCompare: 'less',
        },
    });

    let depthTex = null;
    function ensureDepth() {
        if (depthTex &&
            depthTex.width  === canvas.width &&
            depthTex.height === canvas.height) return;
        if (depthTex) depthTex.destroy();
        depthTex = device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    const vertsPerRow = GRID_N + 1;
    const segCount    = GRID_N * GRID_N * 3;
    const indexData   = new Uint32Array(segCount * 2);
    let p = 0;
    for (let r = 0; r < GRID_N; r++) {
        for (let c = 0; c < GRID_N; c++) {
            const a  =  r       * vertsPerRow +  c;
            const b  =  r       * vertsPerRow + (c + 1);
            const c_ = (r + 1)  * vertsPerRow +  c;
            const d  = (r + 1)  * vertsPerRow + (c + 1);
            indexData[p++] = a; indexData[p++] = b;
            indexData[p++] = a; indexData[p++] = c_;
            indexData[p++] = a; indexData[p++] = d;
        }
    }
    const indexBuffer = device.createBuffer({
        size: indexData.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, indexData);

    const WORKGROUP_SIZE = 64;
    const numWorkgroups  = Math.ceil(MAX_VERTS / WORKGROUP_SIZE);

    const gravityEl = document.getElementById('gravityToggle');
    const anchorEl = document.getElementById('anchorToggle');
    const ampEl     = document.getElementById('amp');
    const freqEl    = document.getElementById('freq');
    const springEl  = document.getElementById('spring');
    const dampEl    = document.getElementById('damping');

    const start = performance.now();

    function frame(now) {
        resize();
        ensureDepth();

        const t = (now - start) / 1000;
        const gravityOn = gravityEl.checked ? 1.0 : 0.0;
        const anchorOn = anchorEl.checked ? 1.0 : 0.0;

        const mvp = buildMVP(canvas.width / canvas.height);
        paramsData.set(mvp, 0);                       
        paramsData[16] = t;                           // time
        paramsData[17] = DT;                          // dt
        paramsData[18] = SURFACE_SZ;                  // size
        paramsData[19] = GRID_N;                      // gridN
        paramsData[20] = parseFloat(ampEl.value);     // centerAmp
        paramsData[21] = parseFloat(freqEl.value);    // centerFreq
        paramsData[22] = parseFloat(springEl.value);  // springK
        paramsData[23] = parseFloat(dampEl.value);    // damping
        paramsData[24] = gravityOn;                   // gravity flag
        paramsData[25] = 0.9;                         // gravityG - real 9.8 seems too much
        paramsData[26] = anchorOn;                    // anchor flag
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        cameraData.set(mvp, 0);
        device.queue.writeBuffer(cameraBuffer, 0, cameraData);

        const encoder = device.createCommandEncoder();

        const computePass = encoder.beginComputePass();
        computePass.setPipeline(simPipeline);
        computePass.setBindGroup(0, simBindGroup);
        computePass.dispatchWorkgroups(numWorkgroups);
        computePass.end();

        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0.043, g: 0.055, b: 0.075, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: depthTex.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });
        renderPass.setPipeline(renderPipeline);
        renderPass.setBindGroup(0, renderBindGroup);
        renderPass.setIndexBuffer(indexBuffer, 'uint32');
        renderPass.drawIndexed(indexData.length);
        renderPass.end();

        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main().catch(err => { console.error(err); alert(err.message); });