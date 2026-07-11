package main

import "core:c"
import "core:fmt"
import "core:math"
import ml "core:math/linalg/glsl"
import "core:strings"
import stbi "vendor:stb/image"

// ── Equirectangular HDRI environment (dome light) ───────────────────────────
//
// Direction <-> texel convention:
//   theta = acos(dir.y)          in [0, PI]   (theta = 0 -> +Y pole -> top row)
//   phi   = atan2(dir.z, dir.x)  in [-PI, PI]
//   u = (phi + PI) / (2*PI)      v = theta / PI
// The map is stored with a top-left origin (stbi, no flip), so v = 0 is the
// top row. Importance sampling uses a Pharr/PBRT piecewise-constant 2D
// distribution built from per-texel luminance weighted by sin(theta).

TWO_PI :: 6.28318530717958647692

env_luminance :: proc(r, g, b: f32) -> f32 {
	return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

// Rotate a direction about +Y by `angle` radians.
rotate_y :: proc(d: Vec3, angle: f64) -> Vec3 {
	if angle == 0.0 {
		return d
	}
	ca := math.cos(angle)
	sa := math.sin(angle)
	return Vec3{d.x * ca + d.z * sa, d.y, -d.x * sa + d.z * ca}
}

load_environment :: proc(
	path: string,
	rotation_degrees: f64,
	intensity: f64,
	allocator := context.allocator,
) -> (Environment, bool) {
	if len(path) == 0 {
		return Environment{}, false
	}

	width, height, channels: c.int
	stbi.set_flip_vertically_on_load(c.int(0)) // row 0 = top of image = +Y pole
	raw := stbi.loadf(strings.clone_to_cstring(path, allocator), &width, &height, &channels, 3)
	if raw == nil {
		fmt.eprintln("Failed to load HDRI:", path)
		return Environment{}, false
	}
	defer stbi.image_free(raw)

	w := int(width)
	h := int(height)
	if w <= 0 || h <= 0 {
		return Environment{}, false
	}

	pixels := make([]f32, w * h * 3, allocator)
	copy(pixels, ([^]f32)(raw)[:w * h * 3])

	env := Environment {
		width     = i32(w),
		height    = i32(h),
		pixels    = pixels,
		rotation  = math.to_radians(rotation_degrees),
		intensity = intensity,
		has_data  = true,
	}

	build_env_distribution(&env, allocator)

	fmt.println("Loaded HDRI:", path, w, "x", h, "intensity:", intensity)
	return env, true
}

destroy_environment :: proc(env: ^Environment) {
	if env.has_data {
		delete(env.pixels)
		delete(env.marginal_cdf)
		delete(env.conditional_cdf)
		env.has_data = false
	}
}

// Build the marginal (row) + conditional (per-row column) CDFs. Both CDFs are
// normalized to end at 1. `func_int` is the mean of the sin(theta)-weighted
// luminance over every texel and normalizes the pdf.
build_env_distribution :: proc(env: ^Environment, allocator := context.allocator) {
	w := int(env.width)
	h := int(env.height)

	env.conditional_cdf = make([]f32, h * (w + 1), allocator)
	env.marginal_cdf = make([]f32, h + 1, allocator)
	row_int := make([]f64, h, allocator)
	defer delete(row_int, allocator)

	for y in 0 ..< h {
		// sin(theta) at the row center; theta = PI * (y+0.5)/h.
		sin_theta := math.sin(math.PI * (f64(y) + 0.5) / f64(h))
		base := y * (w + 1)
		env.conditional_cdf[base] = 0.0
		acc: f64 = 0.0
		for x in 0 ..< w {
			pidx := (y * w + x) * 3
			lum := f64(env_luminance(env.pixels[pidx], env.pixels[pidx + 1], env.pixels[pidx + 2]))
			acc += lum * sin_theta
			env.conditional_cdf[base + x + 1] = f32(acc)
		}
		row_int[y] = acc / f64(w)
		// Normalize this row's CDF to [0, 1].
		if acc > 0.0 {
			inv := f32(1.0 / acc)
			for x in 1 ..= w {
				env.conditional_cdf[base + x] *= inv
			}
		} else {
			// Uniform fallback for a black row.
			for x in 1 ..= w {
				env.conditional_cdf[base + x] = f32(f64(x) / f64(w))
			}
		}
	}

	// Marginal CDF over rows.
	env.marginal_cdf[0] = 0.0
	macc: f64 = 0.0
	for y in 0 ..< h {
		macc += row_int[y]
		env.marginal_cdf[y + 1] = f32(macc)
	}
	env.func_int = macc / f64(h)
	if macc > 0.0 {
		inv := f32(1.0 / macc)
		for y in 1 ..= h {
			env.marginal_cdf[y] *= inv
		}
	} else {
		for y in 1 ..= h {
			env.marginal_cdf[y] = f32(f64(y) / f64(h))
		}
	}
}

// Binary-search a normalized CDF of `n+1` entries for the largest bucket `i`
// with cdf[i] <= xi. Returns the continuous position in [0,1] and the bucket.
@(private = "file")
sample_cdf :: proc(cdf: []f32, offset, n: int, xi: f64) -> (pos: f64, bucket: int) {
	lo := 0
	hi := n
	x := f32(xi)
	for lo + 1 < hi {
		mid := (lo + hi) / 2
		if cdf[offset + mid] <= x {
			lo = mid
		} else {
			hi = mid
		}
	}
	bucket = lo
	c0 := cdf[offset + bucket]
	c1 := cdf[offset + bucket + 1]
	du := 0.0
	if c1 > c0 {
		du = f64((x - c0) / (c1 - c0))
	}
	pos = (f64(bucket) + du) / f64(n)
	return
}

// Bilinear lookup of the environment radiance for a world-space direction.
env_lookup :: proc(env: ^Environment, dir: Vec3) -> Color {
	if !env.has_data {
		return Color{0, 0, 0}
	}
	// Undo the environment rotation: rotating the dome by +rotation is the
	// same as rotating the query direction by -rotation.
	d := rotate_y(dir, -env.rotation)
	d = ml.normalize(d)

	theta := math.acos(clamp(d.y, -1.0, 1.0))
	phi := math.atan2(d.z, d.x)
	u := (phi + math.PI) / TWO_PI
	v := theta / math.PI

	w := int(env.width)
	h := int(env.height)

	fx := u * f64(w) - 0.5
	fy := v * f64(h) - 0.5
	x0 := int(math.floor(fx))
	y0 := int(math.floor(fy))
	tx := fx - f64(x0)
	ty := fy - f64(y0)

	sample := proc(env: ^Environment, w, h, x, y: int) -> Color {
		xi := ((x % w) + w) % w // wrap horizontally
		yi := clamp(y, 0, h - 1) // clamp vertically at the poles
		pidx := (yi * w + xi) * 3
		return Color{f64(env.pixels[pidx]), f64(env.pixels[pidx + 1]), f64(env.pixels[pidx + 2])}
	}

	c00 := sample(env, w, h, x0, y0)
	c10 := sample(env, w, h, x0 + 1, y0)
	c01 := sample(env, w, h, x0, y0 + 1)
	c11 := sample(env, w, h, x0 + 1, y0 + 1)

	top := c00 * (1.0 - tx) + c10 * tx
	bot := c01 * (1.0 - tx) + c11 * tx
	col := top * (1.0 - ty) + bot * ty
	return col * env.intensity
}

// Importance-sample a direction from the environment. Returns the world-space
// direction, its radiance, and the solid-angle pdf.
env_sample :: proc(env: ^Environment, rng: ^Rng) -> (dir: Vec3, radiance: Color, pdf: f64) {
	if !env.has_data {
		return Vec3{0, 1, 0}, Color{0, 0, 0}, 0.0
	}
	w := int(env.width)
	h := int(env.height)

	xi1 := rng_f64(rng)
	xi2 := rng_f64(rng)

	v, row := sample_cdf(env.marginal_cdf, 0, h, xi1)
	u, _ := sample_cdf(env.conditional_cdf, row * (w + 1), w, xi2)

	theta := v * math.PI
	phi := u * TWO_PI - math.PI
	sin_theta := math.sin(theta)

	base := Vec3{sin_theta * math.cos(phi), math.cos(theta), sin_theta * math.sin(phi)}
	dir = rotate_y(base, env.rotation)

	if sin_theta <= 0.0 || env.func_int <= 0.0 {
		return dir, Color{0, 0, 0}, 0.0
	}

	// map pdf (in uv) = func / func_int, converted to solid angle.
	col := clamp(int(u * f64(w)), 0, w - 1)
	rowc := clamp(int(v * f64(h)), 0, h - 1)
	pidx := (rowc * w + col) * 3
	lum := f64(env_luminance(env.pixels[pidx], env.pixels[pidx + 1], env.pixels[pidx + 2]))
	map_pdf := lum * sin_theta / env.func_int
	pdf = map_pdf / (2.0 * math.PI * math.PI * sin_theta)

	radiance = env_lookup(env, dir)
	return
}

// Solid-angle pdf of sampling `dir` from the environment (for MIS).
env_pdf :: proc(env: ^Environment, dir: Vec3) -> f64 {
	if !env.has_data || env.func_int <= 0.0 {
		return 0.0
	}
	d := rotate_y(dir, -env.rotation)
	d = ml.normalize(d)
	theta := math.acos(clamp(d.y, -1.0, 1.0))
	sin_theta := math.sin(theta)
	if sin_theta <= 0.0 {
		return 0.0
	}
	phi := math.atan2(d.z, d.x)
	u := (phi + math.PI) / TWO_PI
	v := theta / math.PI

	w := int(env.width)
	h := int(env.height)
	col := clamp(int(u * f64(w)), 0, w - 1)
	row := clamp(int(v * f64(h)), 0, h - 1)
	pidx := (row * w + col) * 3
	lum := f64(env_luminance(env.pixels[pidx], env.pixels[pidx + 1], env.pixels[pidx + 2]))
	// pdf_uv = (lum * sinTheta) / func_int; pdf_solid = pdf_uv / (2*PI^2*sinTheta)
	// so the sinTheta cancels.
	return lum / (2.0 * math.PI * math.PI * env.func_int)
}
