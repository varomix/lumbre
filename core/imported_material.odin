package lumbre_core

// Renderer-side representation of a material after an importer has resolved
// its native shading graph. USD, glTF, and interactive frontends may differ in
// how they discover inputs, but all map through this one conversion.
Imported_Channel :: enum i32 { R, G, B, A }

Imported_Material :: struct {
	base_color, emission, transmission_color, specular_color: Color,
	roughness, metallic, ior, transmission, specular: f64,
	coat, coat_roughness, subsurface, subsurface_scale: f64,
	subsurface_color, subsurface_radius: Color,
	albedo_tex, normal_tex, emissive_tex: TextureMap,
	roughness_tex, metallic_tex: TextureMap,
	roughness_channel, metallic_channel: Imported_Channel,
	roughness_scale, roughness_bias, metallic_scale, metallic_bias: f64,
}

imported_material_to_principled :: proc(src: Imported_Material) -> Material {
	mat := Material{
		kind = .Principled, albedo = src.base_color, emission = src.emission,
		roughness = src.roughness, metallic = src.metallic, ir = src.ior,
		specular = src.specular, specular_tint = src.specular_color,
		spec_trans = src.transmission, clearcoat = src.coat,
		clearcoat_roughness = src.coat_roughness, subsurface = src.subsurface,
		subsurface_color = src.subsurface_color, subsurface_radius = src.subsurface_radius,
		subsurface_scale = src.subsurface_scale,
		albedo_tex = src.albedo_tex, normal_tex = src.normal_tex, emissive_tex = src.emissive_tex,
	}
	if mat.spec_trans > 0 { mat.albedo = src.transmission_color }
	ior_f0 := (mat.ir - 1.0) / (mat.ir + 1.0)
	mat.specular *= 12.5 * ior_f0 * ior_f0
	if mat.subsurface > 0 { mat.albedo = mat.subsurface_color * mat.subsurface }
	if src.roughness_tex.has_data || src.metallic_tex.has_data {
		if packed, ok := pack_imported_metallic_roughness(src); ok {
			mat.metallic_roughness_tex = packed
			mat.roughness = 1; mat.metallic = 1
		}
	}
	return normalize_material(mat)
}

pack_imported_metallic_roughness :: proc(src: Imported_Material) -> (TextureMap, bool) {
	rough_ok, metal_ok := src.roughness_tex.has_data, src.metallic_tex.has_data
	if !rough_ok && !metal_ok { return TextureMap{}, false }
	base := src.roughness_tex if rough_ok else src.metallic_tex
	out := make_texture(base.width, base.height); out.srgb = false
	fetch :: proc(tex: TextureMap, ok: bool, channel: Imported_Channel, scale, bias, fallback: f64, x, y, w, h: i32) -> u8 {
		if !ok { return u8(clamp(fallback, 0, 1) * 255 + 0.5) }
		sx, sy := x * tex.width / w, y * tex.height / h
		v := f64(tex.pixels[(int(sy) * int(tex.width) + int(sx)) * 4 + int(channel)]) / 255.0 * scale + bias
		return u8(clamp(v, 0, 1) * 255 + 0.5)
	}
	for y in 0 ..< out.height {
		for x in 0 ..< out.width {
			o := (int(y) * int(out.width) + int(x)) * 4
			out.pixels[o + 0] = 255
			out.pixels[o + 1] = fetch(src.roughness_tex, rough_ok, src.roughness_channel, src.roughness_scale, src.roughness_bias, src.roughness, x, y, out.width, out.height)
			out.pixels[o + 2] = fetch(src.metallic_tex, metal_ok, src.metallic_channel, src.metallic_scale, src.metallic_bias, src.metallic, x, y, out.width, out.height)
			out.pixels[o + 3] = 255
		}
	}
	return out, true
}
