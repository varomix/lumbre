// The reflection ray march, shared by the trace pass (glossy rays, half
// resolution) and the resolve (a mirror's exact ray, full resolution). The
// includer declares `scene_color` (the lit image with its blur mips, read
// trilinearly), `normal_target` and `hiz` before including this.

static const int MAX_STEPS = 96;
// tan of a reflection cone's half-width per unit of GGX alpha (= roughness):
// D falls to half at tan(theta_h) = 0.64 alpha, and the reflected direction
// turns twice as far as the half vector. The resolve pass shares each ray
// among its taps, so one ray covers about 1/sqrt(taps) of that.
static const float CONE_TAN_PER_ALPHA = 1.3;
// A 2x2 box pyramid, read trilinearly, blurs about as much as a Gaussian whose
// half-maximum radius is this many times smaller than a texel of the level.
static const float CONE_TEXELS_PER_RADIUS = 2.5;
// The widest blur a hit is read at.
static const float BLUR_MAX = 3.0;

// GGX visible-normal sample (Heitz 2017), in the frame where n is +z.
float3 sample_vndf(float3 wo, float alpha, float2 u) {
    float3 v = normalize(float3(wo.x * alpha, wo.y * alpha, wo.z));
    float3 t1 = v.z < 0.9999 ? normalize(cross(v, float3(0.0, 0.0, 1.0))) : float3(1.0, 0.0, 0.0);
    float3 t2 = cross(t1, v);
    float a = 1.0 / (1.0 + v.z);
    float r = sqrt(u.x);
    float phi = u.y < a ? u.y / a * LUMBRE_PI : LUMBRE_PI + (u.y - a) / (1.0 - a) * LUMBRE_PI;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi) * (u.y < a ? 1.0 : v.z);
    float3 h = p1 * t1 + p2 * t2 + sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0)) * v;
    return normalize(float3(h.x * alpha, h.y * alpha, max(h.z, 1e-6)));
}

// Traces one reflection ray from `origin` along `dir`. On a hit, returns the
// lit colour it found, read at the blur level for a cone of half-width
// `cone_tan`, and how far to trust it (0 for a miss).
float trace_ray(float3 world_pos, float3 origin, float origin_z, float3 dir, float cone_tan, out float3 colour) {
    colour = float3(0.0);
    // A reflection aimed back at the viewer shows what is behind the camera
    // or the hidden side of what is in front of it. Neither is on screen.
    float along = dot(dir, ssr.forward.xyz);
    float trust = smoothstep(-0.85, -0.5, along);
    if (trust <= 0.0) return 0.0;

    // End the segment before it passes behind the near plane, where the
    // projection turns inside out.
    float reach = ssr.eye.w;
    float near = ssr.depth.x * 2.0;
    if (along < 0.0) reach = min(reach, (origin_z - near) / -along);

    float2 hit_px;
    float confidence;
    if (!trace(to_screen(origin), to_screen(origin + dir * reach), hit_px, confidence)) return 0.0;
    trust *= confidence;

    // What was hit must face the ray. The depth buffer holds only the side of
    // a surface the camera sees, and a ray arriving from the other side would
    // be handed that side's colour.
    float2 hit_uv = hit_px * ssr.size.zw;
    float3 hit_pos = world_at(hit_uv, hiz.SampleLevel(hit_uv, 0.0).r);
    float3 hit_n = decode_normal(normal_target.SampleLevel(hit_uv, 0.0).rg);
    if (dot(hit_n, ssr.eye.xyz - hit_pos) < 0.0) hit_n = -hit_n;
    trust *= smoothstep(0.0, 0.1, -dot(hit_n, dir));

    // Fade toward the frame's edge instead of cutting off where the screen
    // stops having an answer.
    float2 edge = min(hit_uv, 1.0 - hit_uv);
    trust *= smoothstep(0.0, 0.08, min(edge.x, edge.y));
    if (trust <= 0.0) return 0.0;

    // The cone's half-width where the ray landed, in pixels at that depth.
    float travelled = length(hit_pos - world_pos);
    float hit_z = max(dot(hit_pos - ssr.eye.xyz, ssr.forward.xyz), ssr.depth.x);
    float radius_px = travelled * cone_tan / (hit_z * ssr.forward.w);
    float blur = log2(max(radius_px * CONE_TEXELS_PER_RADIUS, 1.0));
    // The pyramid averages a square of the screen around the hit, not what
    // the cone sees: past a few levels that square takes in the floor under
    // the object and the sky beside it, and the reflection turns to a flat
    // grey that darkens whatever it replaces. Capped, the rest of the spread
    // comes from the rays themselves.
    colour = scene_color.SampleLevel(hit_uv, min(blur, min(BLUR_MAX, ssr.params.z - 1.0))).rgb;
    return trust;
}


// Reversed-Z depth to distance along the view axis.
float linear_depth(float d) {
    float near = ssr.depth.x;
    float far = ssr.depth.y;
    return near * far / max(near + d * (far - near), 1e-12);
}

// A world point as (pixel x, pixel y, depth). Only called in front of the eye.
float3 to_screen(float3 p) {
    float4 clip = mul(ssr.view_proj, float4(p, 1.0));
    float3 ndc = clip.xyz / clip.w;
    return float3((ndc.x * 0.5 + 0.5) * ssr.size.x, (0.5 - ndc.y * 0.5) * ssr.size.y, ndc.z);
}

// How far a pixel's surface depth changes across one pixel, in raw depth. A
// plane's depth is affine across the screen, so it is the step to a neighbour:
// the smaller one on each axis, so a silhouette's jump is not taken for slope.
float depth_step(int2 px, float centre) {
    int2 last = int2(ssr.size.xy) - 1;
    float2 texel = ssr.size.zw;
    float l = hiz.SampleLevel((float2(clamp(px + int2(-1, 0), int2(0), last)) + 0.5) * texel, 0.0).r;
    float r = hiz.SampleLevel((float2(clamp(px + int2(1, 0), int2(0), last)) + 0.5) * texel, 0.0).r;
    float u = hiz.SampleLevel((float2(clamp(px + int2(0, -1), int2(0), last)) + 0.5) * texel, 0.0).r;
    float d = hiz.SampleLevel((float2(clamp(px + int2(0, 1), int2(0), last)) + 0.5) * texel, 0.0).r;
    return min(abs(l - centre), abs(r - centre)) + min(abs(u - centre), abs(d - centre));
}

// Marches the segment `from` -> `to` through the depth pyramid. Returns whether
// it hit, where in pixels, and how far to trust the hit.
bool trace(float3 from, float3 to, out float2 hit_px, out float confidence) {
    hit_px = float2(0.0);
    confidence = 1.0;
    float3 d = to - from;
    float span = max(abs(d.x), abs(d.y));
    // A ray along the line of sight covers no pixels; there is nothing to walk.
    if (span < 1.0) return false;

    // The exit side of a cell in each axis, and the reciprocal slope. A ray
    // running along a row or column never crosses a boundary in the other
    // axis, and its crossing must come out as "never" whichever side of the
    // boundary it starts on -- hence a zero slope and an added infinity, not a
    // large slope that the subtraction can turn negative.
    float2 toward = float2(d.x > 0.0 ? 1.0 : 0.0, d.y > 0.0 ? 1.0 : 0.0);
    bool2 flat = abs(d.xy) <= 1e-6;
    float2 inv = select(flat, float2(0.0), 1.0 / d.xy);
    float2 never = select(flat, float2(1e30), float2(0.0));
    // Far enough past a boundary to be in the next cell, in units of t.
    float nudge = 0.01 / span;

    int top = int(ssr.depth.z) - 1;
    int2 full = int2(ssr.size.xy);
    float thickness = ssr.depth.w;

    // Leave the pixel the ray starts in before testing anything.
    float2 cell0 = floor(from.xy);
    float2 exit0 = ((cell0 + toward) - from.xy) * inv + never;
    float t = max(min(exit0.x, exit0.y), 0.0) + nudge;
    int level = 0;

    for (int iter = 0; iter < MAX_STEPS; ++iter) {
        if (t >= 1.0) return false;
        float2 p = from.xy + d.xy * t;
        if (p.x < 0.0 || p.y < 0.0 || p.x >= ssr.size.x || p.y >= ssr.size.y) return false;

        float cell_size = float(1 << level);
        float2 cell = floor(p / cell_size);
        // An odd dimension's last row or column belongs to the last cell.
        int2 dims = max(full >> level, int2(1));
        float2 cell_uv = (float2(min(int2(cell), dims - 1)) + 0.5) / float2(dims);
        float nearest = hiz.SampleLevel(cell_uv, float(level)).r;

        float2 exit = ((cell + toward) * cell_size - from.xy) * inv + never;
        float t_exit = min(exit.x, exit.y);

        float z_in = from.z + d.z * t;
        float z_out = from.z + d.z * t_exit;
        // Reversed-Z: greater is nearer. Nearer than everything here, the
        // whole way across, means nothing in this cell can be hit.
        if (min(z_in, z_out) > nearest) {
            t = t_exit + nudge;
            level = min(level + 1, top);
            continue;
        }

        if (level > 0) {
            // Something in this cell may be in the way. Go to where the ray
            // reaches that depth and look closer.
            if (z_in > nearest) t += (nearest - z_in) / d.z;
            level -= 1;
            continue;
        }

        // Level 0: one pixel, one surface. A ray that crosses its depth here
        // has hit it. One that arrives already behind has either slipped
        // under the surface between two pixels -- a hit -- or is passing
        // behind an object the depth buffer cannot see around. Nothing tells
        // the two apart but distance, so trust falls off with it rather than
        // stopping at a line that neighbouring pixels land either side of.
        if (z_in > nearest) {
            hit_px = from.xy + d.xy * (t + (nearest - z_in) / d.z);
            return true;
        }
        //
        // Except a ray that runs nearly along a surface -- a mirror's view of
        // a wall at a grazing angle. The depth buffer holds one depth per
        // pixel, at its centre, so such a ray can stay in front of one
        // pixel's centre and arrive behind the next one's without a crossing
        // between them, and how far behind depends on where in the pixel it
        // fell. Grading trust by that turned the reflection into stripes, a
        // few pixels of full strength then a few of fading, column after
        // column. Behind by no more than the surface's own depth change over
        // a pixel is that slip, and a full hit.
        float slip = nearest - z_in;
        if (slip <= depth_step(int2(cell), nearest)) {
            hit_px = p;
            return true;
        }
        float surface_z = linear_depth(nearest);
        float behind = linear_depth(z_in) - surface_z;
        float allowed = thickness + surface_z * 0.01;
        if (behind <= allowed) {
            hit_px = p;
            confidence = 1.0 - smoothstep(0.25, 1.0, behind / allowed);
            return true;
        }
        t = t_exit + nudge;
    }
    return false;
}

