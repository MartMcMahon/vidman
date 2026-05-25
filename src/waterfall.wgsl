struct Params {
  write_row: u32,
  history  : u32,
  db_min   : f32,
  db_max   : f32,
  tuned_x  : f32,
}

@group(0) @binding(0) var waterfall: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct VsOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  var corners = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  let p = corners[vi];
  var out: VsOut;
  out.clip = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>((p.x + 1.0) *0.5, (1.0 - p.y) * 0.5);
  return out;
}

fn colormap(t: f32) -> vec3<f32> {
  let c0 = vec3<f32>(0.0, 0.0, 0.1);  // noise floor
  let c1 = vec3<f32>(0.1, 0.3, 0.9);  // blue
  let c2 = vec3<f32>(0.9, 0.2, 0.5);  // magenta
  let c3 = vec3<f32>(1.0, 0.95, 0.4); // peaks
  if (t < 0.33) { return mix(c0, c1, t / 0.33); }
  if (t < 0.66) { return mix(c1, c2, (t - 0.33) / 0.33); }
  return mix(c2, c3, (t - 0.66) / 0.34);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let dims = vec2<i32>(textureDimensions(waterfall));
  let hist = i32(params.history);

  let col = clamp(i32(in.uv.x * f32(dims.x)), 0, dims.x - 1);
  let age = clamp(i32(in.uv.y * f32(hist)), 0, hist - 1);
  let row = (i32(params.write_row) - 1 - age + 2 * hist) % hist;
  let db = textureLoad(waterfall, vec2<i32>(col, row), 0).r;
  let t = clamp((db - params.db_min) / (params.db_max - params.db_min), 0.0, 1.0);
  let base = colormap(t);

  // thin vertical line at the tuned-to frequency
  let line_half_width = 0.5 / f32(dims.x); // ~1 texel wide
  let marker = step(abs(in.uv.x - params.tuned_x), line_half_width);
  let color = mix(base, vec3<f32>(1.0, 1.0, 1.0), marker);
  return vec4<f32>(color, 1.0);
}
