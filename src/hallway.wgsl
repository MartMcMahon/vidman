struct Cam {
    view_proj: mat4x4<f32>,
    write_row: u32,
    history:   u32,
    db_min:    f32,
    db_max:    f32,
    band_half_width_m: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var spec_tex: texture_2d<f32>;
@group(0) @binding(2) var spec_samp: sampler;
// LO world position (in meters) for each historical texture row. Indexed by
// the same row index that spec_tex is sampled at.
@group(0) @binding(3) var<storage, read> lo_per_row: array<f32>;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv:  vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_x: f32,
    @location(1) uv_y:    f32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.world_x = in.pos.x;
    out.uv_y = in.uv.y;
    return out;
}

// synthwave: black → indigo → hot pink → electric cyan → white-hot pop
fn colormap(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.03, 0.00, 0.08); // near-black
    let c1 = vec3<f32>(0.20, 0.05, 0.50); // indigo
    let c2 = vec3<f32>(1.00, 0.20, 0.55); // hot pink
    let c3 = vec3<f32>(0.20, 0.95, 1.00); // electric cyan
    let c4 = vec3<f32>(1.10, 1.10, 1.10); // white pop (overbright)

    let s = clamp(t, 0.0, 1.0);
    if s < 0.3 {
        return mix(c0, c1, s / 0.3);
    } else if s < 0.6 {
        return mix(c1, c2, (s - 0.3) / 0.3);
    } else if s < 0.85 {
        return mix(c2, c3, (s - 0.6) / 0.25);
    } else {
        return mix(c3, c4, (s - 0.85) / 0.15);
    }
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let dim = textureDimensions(spec_tex);
    let width = i32(dim.x);
    let history = i32(dim.y);

    // uv_y = 1 (top of wall) → most recent row; uv_y = 0 (floor) → oldest
    let age = i32((1.0 - in.uv_y) * f32(history));
    var row = i32(cam.write_row) - 1 - age;
    row = ((row % history) + history) % history;

    // sample this row's LO position; gate by per-row band membership
    let lo_at_row = lo_per_row[row];
    let dx = in.world_x - lo_at_row;
    if abs(dx) > cam.band_half_width_m {
        // dead zone: a touch darker than the colormap's low end
        return vec4<f32>(0.02, 0.00, 0.05, 1.0);
    }

    let norm = dx / cam.band_half_width_m * 0.5 + 0.5;
    let col = clamp(i32(norm * f32(width)), 0, width - 1);
    let v = textureLoad(spec_tex, vec2<i32>(col, row), 0).r;
    let t = clamp((v - cam.db_min) / (cam.db_max - cam.db_min), 0.0, 1.0);
    return vec4<f32>(colormap(t), 1.0);
}
