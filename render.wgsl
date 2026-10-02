struct Particle {
    pos : vec3<f32>,
    _p  : f32,
    vel : vec3<f32>,
    _v  : f32,
};

struct Camera {
    mvp : mat4x4<f32>,
};

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform>        camera    : Camera;

struct VertexOut {
    @builtin(position) pos : vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid : u32) -> VertexOut {
    let p = particles[vid].pos;
    var out : VertexOut;
    out.pos = camera.mvp * vec4<f32>(p, 1.0);
    return out;
}

@fragment
fn fs_main(in : VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.55, 0.75, 0.95, 1.0);
}