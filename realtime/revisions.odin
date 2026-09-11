package lumbre_realtime

import "core:hash/xxhash"
import "core:slice"
import lc "../core"
import sdl "vendor:sdl3"

// Content revisions survive USD re-imports, unlike pointers. Hashing source
// geometry is linear but avoids flattening, sorting, transfer and mip work.
// These are process-local cache keys, not persistent asset identifiers.
fingerprint :: proc(value: $T, seed: u64 = 0) -> u64 {
	v := value
	return xxhash.XXH64(([^]u8)(rawptr(&v))[:size_of(T)], seed)
}
// What the uploaded vertices and batches depend on: which mesh each node draws,
// material assignment, and the triangles themselves. Placement is absent -- it
// lives in the instance buffer, under `instance_revision`.
geometry_revision :: proc(scene: ^lc.Scene) -> u64 {
	h := fingerprint(len(scene.materials))
	h = fingerprint(len(scene.nodes), h)
	for node in scene.nodes {
		h = fingerprint(node.mesh_idx, h)
		h = fingerprint(node.material_override_idx, h)
	}
	for mesh in scene.meshes {
		h = fingerprint(len(mesh.triangles), h)
		// Sharing decides how nodes group into instanced meshes.
		h = fingerprint(mesh.borrowed_triangles, h)
		h = xxhash.XXH64(slice.to_bytes(mesh.triangles), h)
	}
	for sphere in scene.spheres {
		h = fingerprint(sphere.center, h)
		h = fingerprint(sphere.radius, h)
	}
	return fingerprint(len(scene.spheres), h)
}
// Node placement: what the instance buffer and culling bounds depend on.
instance_revision :: proc(scene: ^lc.Scene) -> u64 {
	h := fingerprint(len(scene.nodes))
	for node in scene.nodes {
		h = fingerprint(node.local_transform, h)
		h = fingerprint(node.parent, h)
	}
	return h
}
texture_revision :: proc(tex: lc.TextureMap) -> u64 {
	if !tex.has_data { return 0 }
	h := fingerprint(tex.width)
	h = fingerprint(tex.height, h)
	h = fingerprint(tex.srgb, h)
	return xxhash.XXH64(tex.pixels, h)
}
scene_texture_revision :: proc(scene: ^lc.Scene) -> u64 {
	h: u64
	for mat in scene.materials {
		for tex in ([4]lc.TextureMap{mat.albedo_tex, mat.metallic_roughness_tex, mat.normal_tex, mat.emissive_tex}) {
			h = fingerprint(texture_revision(tex), h)
		}
	}
	for sphere in scene.spheres {
		mat := sphere.material
		for tex in ([4]lc.TextureMap{mat.albedo_tex, mat.metallic_roughness_tex, mat.normal_tex, mat.emissive_tex}) {
			h = fingerprint(texture_revision(tex), h)
		}
	}
	return h
}
environment_revision :: proc(env: lc.Environment) -> u64 {
	if !env.has_data { return 0 }
	h := fingerprint(env.width)
	h = fingerprint(env.height, h)
	return xxhash.XXH64(slice.to_bytes(env.pixels), h)
}

Texture_Entry :: struct { texture: ^sdl.GPUTexture, used: bool }
Texture_Cache :: map[u64]Texture_Entry
texture_cache_begin :: proc(cache: ^Texture_Cache) {
	for key, &entry in cache^ { entry.used = false }
}
texture_cache_prune :: proc(gpu: ^sdl.GPUDevice, cache: ^Texture_Cache) {
	stale := make([dynamic]u64, context.temp_allocator)
	for key, entry in cache^ { if !entry.used { append(&stale, key) } }
	for key in stale { sdl.ReleaseGPUTexture(gpu, cache[key].texture); delete_key(cache, key) }
}
texture_cache_destroy :: proc(gpu: ^sdl.GPUDevice, cache: ^Texture_Cache) {
	for _, entry in cache^ { sdl.ReleaseGPUTexture(gpu, entry.texture) }
	delete(cache^)
	cache^ = nil
}

// Refresh shader values without visiting any vertex or computing transforms.
scene_refresh_materials :: proc(s: ^Scene_GPU, scene: ^lc.Scene) {
	for &b in s.batches {
		index := int(b.material_index)
		if index >= 0 && index < len(scene.materials) {
			b.material = material_uniforms(scene.materials[index])
		} else if si := index - len(scene.materials); si >= 0 && si < len(scene.spheres) {
			b.material = material_uniforms(scene.spheres[si].material)
		}
	}
}
scene_bind_textures :: proc(gpu: ^sdl.GPUDevice, s: ^Scene_GPU, scene: ^lc.Scene, cache: ^Texture_Cache) -> bool {
	texture_cache_begin(cache)
	for &b in s.batches {
		index := int(b.material_index)
		mat: lc.Material
		if index >= 0 && index < len(scene.materials) { mat = scene.materials[index] }
		else if si := index - len(scene.materials); si >= 0 && si < len(scene.spheres) { mat = scene.spheres[si].material }
		b.albedo = cached_texture(gpu, mat.albedo_tex, cache, s.white)
		b.mr = cached_texture(gpu, mat.metallic_roughness_tex, cache, s.white)
		b.normal = cached_texture(gpu, mat.normal_tex, cache, s.flat_normal)
		b.emissive = cached_texture(gpu, mat.emissive_tex, cache, s.white)
		if b.albedo == nil || b.mr == nil || b.normal == nil || b.emissive == nil { return false }
	}
	return true
}
cached_texture :: proc(gpu: ^sdl.GPUDevice, tex: lc.TextureMap, cache: ^Texture_Cache, fallback: ^sdl.GPUTexture) -> ^sdl.GPUTexture {
	if !tex.has_data || len(tex.pixels) == 0 || tex.width <= 0 || tex.height <= 0 { return fallback }
	key := texture_revision(tex)
	if entry, found := cache[key]; found {
		entry.used = true; cache[key] = entry
		return entry.texture
	}
	texture := upload_texture(gpu, tex)
	if texture == nil { return nil }
	cache[key] = {texture, true}
	return texture
}
