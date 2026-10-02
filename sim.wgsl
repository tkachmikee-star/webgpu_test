struct Particle {
    pos : vec3<f32>,
    _p  : f32,
    vel : vec3<f32>,
    _v  : f32,
};

struct Params {
    mvp        : mat4x4<f32>,
    time       : f32,
    dt         : f32,
    size       : f32,
    gridN      : f32,
    centerAmp  : f32,
    centerFreq : f32,
    springK    : f32,
    damping    : f32,
    gravity    : f32,   // 1.0 = on, 0.0 = off
    gravityG   : f32,
    anchor     : f32,  // 1.0 = on, 0.0 = off
    _pad0      : f32,
    _pad1      : f32,
};

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform>             params    : Params;

fn isCorner(row : u32, col : u32, N : u32) -> bool {
    let atLeft   = col == 0u;
    let atRight  = col == N;
    let atBottom = row == 0u;
    let atTop    = row == N;
    return (atLeft || atRight) && (atBottom || atTop);
}

fn isCenter(row : u32, col : u32, N : u32) -> bool {
    return row == N / 2u && col == N / 2u;
}

fn springForce(
    pi   : vec3<f32>,
    pj   : vec3<f32>,
    L0   : f32,
    k    : f32,
) -> vec3<f32> {
    let d = pj - pi;
    let r = length(d);
    if (r < 1e-6) { return vec3<f32>(0.0); }
    let stretch = r - L0;
    return k * stretch * (d / r);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    let total = arrayLength(&particles);
    if (i >= total) { return; }

    let N = u32(params.gridN);
    let vertsPerRow = N + 1u;
    let row = i / vertsPerRow;
    let col = i % vertsPerRow;

    let fu = f32(col) / f32(N);
    let fv = f32(row) / f32(N);

    let rest = vec3<f32>(
        (fu - 0.5) * params.size,
        0.0,
        (fv - 0.5) * params.size
    );

    // Kinematic: corners pinned
    if (isCorner(row, col, N)) {
        particles[i].pos = rest;
        particles[i].vel = vec3<f32>(0.0);
        return;
    }

    // Kinematic: center forced oscillation
    if (isCenter(row, col, N)) {
        let y = params.centerAmp * sin(params.centerFreq * params.time);
        particles[i].pos = vec3<f32>(rest.x, y, rest.z);
        particles[i].vel = vec3<f32>(0.0);
        return;
    }

    // Dynamic: integrate
    let p = particles[i].pos;
    let v = particles[i].vel;

    // Rest lengths
    let cell    = params.size / params.gridN;
    let L_orth  = cell;
    let L_diag  = cell * sqrt(2.0);
    let diagK = params.springK/2; // make diag springs less rigid

    var force = vec3<f32>(0.0);

    if (col > 0u) {
        let j = row * vertsPerRow + (col - 1u);
        force += springForce(p, particles[j].pos, L_orth, params.springK);
    }
    if (col < N) {
        let j = row * vertsPerRow + (col + 1u);
        force += springForce(p, particles[j].pos, L_orth, params.springK);
    }
    if (row > 0u) {
        let j = (row - 1u) * vertsPerRow + col;
        force += springForce(p, particles[j].pos, L_orth, params.springK);
    }
    if (row < N) {
        let j = (row + 1u) * vertsPerRow + col;
        force += springForce(p, particles[j].pos, L_orth, params.springK);
    }

    if (col > 0u && row > 0u) {
        let j = (row - 1u) * vertsPerRow + (col - 1u);
        force += springForce(p, particles[j].pos, L_diag, diagK);
    }
    if (col < N && row > 0u) {
        let j = (row - 1u) * vertsPerRow + (col + 1u);
        force += springForce(p, particles[j].pos, L_diag, diagK);
    }
    if (col > 0u && row < N) {
        let j = (row + 1u) * vertsPerRow + (col - 1u);
        force += springForce(p, particles[j].pos, L_diag, diagK);
    }
    if (col < N && row < N) {
        let j = (row + 1u) * vertsPerRow + (col + 1u);
        force += springForce(p, particles[j].pos, L_diag, diagK);
    }

    // Gravity
    if (params.gravity > 0.0) {
        force += vec3<f32>(0.0, -params.gravityG, 0.0);
    }

    // Anchor force: pull toward rest position
    let anchorK = 3.0;   // small value, doesn't dominate springs
    if (params.anchor > 0.0) {
        force += (rest - p) * anchorK;
    }

    // Integrate
    let newV = (v + force * params.dt) * params.damping;
    let newP = p + newV * params.dt;

    particles[i].vel = newV;
    particles[i].pos = newP;
}