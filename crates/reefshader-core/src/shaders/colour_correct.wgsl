struct Params {
    width: u32,
    height: u32,
    downsample_win: u32,
    ds_width: u32,
    ds_height: u32,
    colour_norm_enabled: u32,
    gamma_enabled: u32,
    side_by_side: u32,
    gamma: f32,
    blackpoint: f32,
    _pad0: vec2<f32>,
    mins: vec4<f32>,
    gains: vec4<f32>,
}

const HIST_BINS: u32 = 4096u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> in_pixels: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_pixels: array<u32>;
@group(0) @binding(3) var<storage, read_write> histogram: array<atomic<u32>>;

fn load_pixel(x: u32, y: u32) -> vec4<f32> {
    let cx = min(x, params.width - 1u);
    let cy = min(y, params.height - 1u);
    let idx = (cy * params.width + cx) * 2u;
    let rg = unpack2x16unorm(in_pixels[idx]);
    let ba = unpack2x16unorm(in_pixels[idx + 1u]);
    return vec4<f32>(rg.x, rg.y, ba.x, ba.y);
}

fn store_pixel(x: u32, y: u32, rgba: vec4<f32>) {
    let idx = (y * params.width + x) * 2u;
    out_pixels[idx] = pack2x16unorm(rgba.xy);
    out_pixels[idx + 1u] = pack2x16unorm(rgba.zw);
}

@compute @workgroup_size(16, 16, 1)
fn histogram_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ds_width || gid.y >= params.ds_height) {
        return;
    }

    let win = max(params.downsample_win, 1u);
    let base_x = gid.x * win;
    let base_y = gid.y * win;

    var sum = vec3<f32>(0.0);
    for (var dy = 0u; dy < win; dy = dy + 1u) {
        for (var dx = 0u; dx < win; dx = dx + 1u) {
            sum = sum + load_pixel(base_x + dx, base_y + dy).xyz;
        }
    }

    let count = f32(win * win);
    let avg = clamp(sum / count, vec3<f32>(0.0), vec3<f32>(1.0));
    let scale = f32(HIST_BINS - 1u);

    let bin_r = u32(round(avg.x * scale));
    let bin_g = u32(round(avg.y * scale));
    let bin_b = u32(round(avg.z * scale));

    atomicAdd(&histogram[bin_r], 1u);
    atomicAdd(&histogram[HIST_BINS + bin_g], 1u);
    atomicAdd(&histogram[2u * HIST_BINS + bin_b], 1u);
}

@compute @workgroup_size(16, 16, 1)
fn apply_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let orig = load_pixel(gid.x, gid.y);
    if (params.side_by_side != 0u && gid.x >= params.width / 2u) {
        store_pixel(gid.x, gid.y, orig);
        return;
    }

    var rgb = orig.xyz;
    if (params.colour_norm_enabled != 0u) {
        rgb = clamp((rgb - params.mins.xyz) * params.gains.xyz + params.blackpoint, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    if (params.gamma_enabled != 0u) {
        rgb = pow(max(rgb, vec3<f32>(0.0)), vec3<f32>(params.gamma));
    }

    store_pixel(gid.x, gid.y, vec4<f32>(rgb, orig.w));
}
