package main

import "core:c"
import "core:fmt"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"
import stbi "vendor:stb/image"
import m "core:math/linalg/glsl"

// ── GPU data structs (packed for Metal) ──────────────────────────────────────

GPUMaterial :: struct {
	albedo:   [3]f32,
	_pad0:    f32,
	emission: [3]f32,
	_pad1:    f32,
	kind:     i32,
	fuzz:     f32,
	ir:       f32,
	roughness: f32,
	metallic:  f32,
	emission_strength: f32,
	_pad2:    [6]f32,
}

GPUSceneData :: struct {
	origin:            [4]f32,
	lower_left:        [4]f32,
	horizontal:        [4]f32,
	vertical:          [4]f32,
	u:                 [4]f32,
	v:                 [4]f32,
	lens_radius:       f32,
	image_width:       i32,
	image_height:      i32,
	samples_per_pixel: i32,
	max_depth:         i32,
	max_radiance:      f32,
	debug_mode:        i32,
	light_count:       i32,
	seed:              u32,
	_pad:              [3]f32,
}

GPUSphere :: struct {
	center:        [4]f32,
	radius:        f32,
	material_kind: i32,
	_fuzz:         f32,
	_ir:           f32,
	albedo:        [4]f32,
}

AxisAlignedBoundingBox :: struct {
	min: [3]f32,
	max: [3]f32,
}

// ── Sphere → mesh converter ─────────────────────────────────────────────────

build_icosphere :: proc(center: Vec3, radius: f64, material: Material, allocator := context.allocator) -> []Triangle {
	// Simple UV sphere with 32×16 segments — good enough for display
	segments_u := 32
	segments_v := 16
	tri_count := segments_u * segments_v * 2
	tris := make([]Triangle, tri_count, allocator)
	idx := 0

	for j in 0 ..< segments_v {
		lat0 := m.PI * f64(j) / f64(segments_v)
		lat1 := m.PI * f64(j + 1) / f64(segments_v)
		for i in 0 ..< segments_u {
			lon0 := 2.0 * m.PI * f64(i) / f64(segments_u)
			lon1 := 2.0 * m.PI * f64(i + 1) / f64(segments_u)

			v0 := center + radius * Vec3{m.sin(lat0) * m.cos(lon0), m.cos(lat0), m.sin(lat0) * m.sin(lon0)}
			v1 := center + radius * Vec3{m.sin(lat1) * m.cos(lon0), m.cos(lat1), m.sin(lat1) * m.sin(lon0)}
			v2 := center + radius * Vec3{m.sin(lat1) * m.cos(lon1), m.cos(lat1), m.sin(lat1) * m.sin(lon1)}
			v3 := center + radius * Vec3{m.sin(lat0) * m.cos(lon1), m.cos(lat0), m.sin(lat0) * m.sin(lon1)}

			n0 := m.normalize(v0 - center)
			n1 := m.normalize(v1 - center)
			n2 := m.normalize(v2 - center)
			n3 := m.normalize(v3 - center)

			tris[idx] = Triangle{v0, v1, v2, n0, n1, n2, {}, {}, {}, 0}
			idx += 1
			tris[idx] = Triangle{v0, v2, v3, n0, n2, n3, {}, {}, {}, 0}
			idx += 1
		}
	}
	return tris
}

// ── Main GPU render function ────────────────────────────────────────────────

Render_GPU_Buffers :: struct {
	device:             ^MTL.Device,
	cmd_queue:          ^MTL.CommandQueue,
	vertex_buffer:      ^MTL.Buffer,
	index_buffer:       ^MTL.Buffer,
	material_buffer:    ^MTL.Buffer,
	scene_buffer:       ^MTL.Buffer,
	output_buffer:      ^MTL.Buffer,
	accel_struct:       ^MTL.AccelerationStructure,
	pipeline:           ^MTL.ComputePipelineState,
	ift:                ^MTL.IntersectionFunctionTable,
	num_triangles:      i32,
}

render_gpu :: proc(
	scene: ^Scene,
	image_width, image_height: i32,
	samples_per_pixel, max_depth: i32,
	max_radiance: f64,
	file_output: cstring,
	debug_mode: i32 = 0,
) {
	device := MTL.CreateSystemDefaultDevice()
	assert(device != nil, "Metal device required")
	assert(bool(device->supportsRaytracing()), "Raytracing required")
	fmt.println("Device:", device->name()->odinString())

	cmd_queue := device->newCommandQueue()

	// Convert scene to GPU-friendly format
	all_triangles := make([dynamic]Triangle)
	materials := make([dynamic]Material)
	defer delete(all_triangles)
	defer delete(materials)

	// Process meshes
	for mesh in scene.meshes {
		for tri in mesh.triangles {
			append(&all_triangles, tri)
		}
	}

	// Add OBJ-loaded materials to the materials array
	for mat in scene.materials {
		append(&materials, mat)
	}

	// Process spheres — convert to icosphere meshes
	for sphere in scene.spheres {
		sphere_tris := build_icosphere(sphere.center, sphere.radius, sphere.material)
		tri_mat_idx := i32(len(materials))
		for i in 0 ..< len(sphere_tris) {
			sphere_tris[i].mat_idx = tri_mat_idx
			append(&all_triangles, sphere_tris[i])
		}
		delete(sphere_tris)
		append(&materials, sphere.material)
	}

	num_tris := i32(len(all_triangles))
	fmt.println("Triangles:", num_tris)

	if num_tris == 0 {
		fmt.eprintln("No geometry to render")
		return
	}

	// Assign materials per triangle and build buffers
	GPUTriVertex :: struct { pos: [3]f32, _pad: f32 }
	vertices := make([]GPUTriVertex, num_tris * 3)
	indices := make([]u32, num_tris * 3)
	gpu_materials := make([]GPUMaterial, num_tris)
	defer delete(vertices)
	defer delete(indices)
	defer delete(gpu_materials)

	for i in 0 ..< num_tris {
		tri := all_triangles[i]
		base := i * 3
		vertices[base + 0] = GPUTriVertex{[3]f32{f32(tri.v0.x), f32(tri.v0.y), f32(tri.v0.z)}, 0}
		vertices[base + 1] = GPUTriVertex{[3]f32{f32(tri.v1.x), f32(tri.v1.y), f32(tri.v1.z)}, 0}
		vertices[base + 2] = GPUTriVertex{[3]f32{f32(tri.v2.x), f32(tri.v2.y), f32(tri.v2.z)}, 0}
		indices[base + 0] = u32(base)
		indices[base + 1] = u32(base + 1)
		indices[base + 2] = u32(base + 2)

		// Resolve material
		mat_idx := tri.mat_idx
		mat: Material
		if mat_idx >= 0 && i32(mat_idx) < i32(len(materials)) {
			mat = materials[mat_idx]
		} else if len(scene.materials) > 0 && int(mat_idx) >= 0 && int(mat_idx) < len(scene.materials) {
			mat = scene.materials[mat_idx]
		} else if len(scene.meshes) > 0 {
			mat = scene.meshes[0].material
		}
		// Use default if nothing found
		kind_val := i32(0) // Lambertian
		switch mat.kind {
		case .Lambertian: kind_val = 0
		case .Metal: kind_val = 1
		case .Dielectric: kind_val = 2
		case .Principled: kind_val = 3
		case .Emissive: kind_val = 4
		}
		gpu_materials[i] = GPUMaterial {
			albedo   = {f32(mat.albedo.x), f32(mat.albedo.y), f32(mat.albedo.z)},
			_pad0    = 0,
			emission = {f32(mat.emission.x), f32(mat.emission.y), f32(mat.emission.z)},
			_pad1    = 0,
			kind     = kind_val,
			fuzz     = f32(mat.fuzz),
			ir       = f32(mat.ir),
			roughness = f32(mat.roughness),
			metallic  = f32(mat.metallic),
			emission_strength = f32(mat.emission_strength),
		}
	}

	// Build list of emissive triangle primitive IDs for direct light sampling
	light_prims := make([dynamic]u32)
	defer delete(light_prims)
	for i in 0 ..< num_tris {
		if gpu_materials[i].kind == 4 { // Emissive
			append(&light_prims, u32(i))
		}
	}
	fmt.println("Light triangles:", len(light_prims))
	fmt.println("SceneData size:", size_of(GPUSceneData), "light_count offset:", offset_of(GPUSceneData, light_count))
	fmt.println("light_prims:", light_prims[:])

	// Scene data
	cam := scene.camera
	aspect := f64(image_width) / f64(image_height)
	scene_data := GPUSceneData {
		origin            = {f32(cam.origin.x), f32(cam.origin.y), f32(cam.origin.z), 0},
		lower_left        = {f32(cam.lower_left_corner.x), f32(cam.lower_left_corner.y), f32(cam.lower_left_corner.z), 0},
		horizontal        = {f32(cam.horizontal.x), f32(cam.horizontal.y), f32(cam.horizontal.z), 0},
		vertical          = {f32(cam.vertical.x), f32(cam.vertical.y), f32(cam.vertical.z), 0},
		u                 = {f32(cam.u.x), f32(cam.u.y), f32(cam.u.z), 0},
		v                 = {f32(cam.v.x), f32(cam.v.y), f32(cam.v.z), 0},
		lens_radius       = f32(cam.lens_radius),
		image_width       = image_width,
		image_height      = image_height,
		samples_per_pixel = samples_per_pixel,
		max_depth         = max_depth,
		max_radiance      = f32(max_radiance),
		debug_mode        = debug_mode,
		light_count       = i32(len(light_prims)),
		seed              = 42,
	}
	fmt.println("scene_data.light_count:", scene_data.light_count)
	fmt.println("Material buffer size:", len(gpu_materials) * size_of(GPUMaterial), "bytes, count:", len(gpu_materials))

	// Create Metal buffers
	vertex_buffer := device->newBufferWithSlice(vertices[:], MTL.ResourceStorageModeShared)
	index_buffer := device->newBufferWithSlice(indices[:], MTL.ResourceStorageModeShared)
	material_buffer := device->newBufferWithSlice(gpu_materials[:], MTL.ResourceStorageModeShared)
	light_buffer := device->newBufferWithSlice(light_prims[:], MTL.ResourceStorageModeShared)
	scene_slice := ([^]byte)(&scene_data)[:size_of(GPUSceneData)]
	scene_buffer := device->newBufferWithBytes(scene_slice, MTL.ResourceStorageModeShared)

	pixel_count := int(image_width) * int(image_height)
	output_buffer := device->newBufferWithLength(
		NS.UInteger(pixel_count * size_of([4]f32)),
		MTL.ResourceStorageModeShared,
	)

	// Load shader
	msl_source := #load("shaders/raytrace.metal", string)
	src := NS.String.alloc()->initWithOdinString(msl_source)
	opts := MTL.CompileOptions.alloc()->init()
	opts->setFastMathEnabled(true)
	opts->setLanguageVersion(.Version3_0)

	library, err := device->newLibraryWithSource(src, opts)
	if err != nil {
		error_str := err->localizedDescription()->odinString()
		fmt.eprintln("Shader compilation failed:", error_str)
		return
	}

	kernel_func := library->newFunctionWithName(NS.AT("raytraceKernel"))
	assert(kernel_func != nil, "kernel function not found")

	desc := MTL.ComputePipelineDescriptor.alloc()->init()
	desc->setComputeFunction(kernel_func)

	pipeline, p_err := MTL.Device_newComputePipelineStateWithDescriptorWithReflection(
		device, desc, MTL.PipelineOption{}, nil,
	)
	if p_err != nil {
		fmt.eprintln("Pipeline creation failed:", p_err->localizedDescription()->odinString())
		return
	}

	// Build triangle acceleration structure
	fmt.println("Building acceleration structure...")

	tri_geom := MTL.AccelerationStructureTriangleGeometryDescriptor.alloc()->init()
	tri_geom->setVertexBuffer(vertex_buffer)
	tri_geom->setVertexStride(16) // float3 + pad
	tri_geom->setIndexBuffer(index_buffer)
	tri_geom->setIndexType(.UInt32)
	tri_geom->setTriangleCount(NS.UInteger(num_tris))

	prim_desc := MTL.PrimitiveAccelerationStructureDescriptor.alloc()->init()
	geometries := [?]^NS.Object{auto_cast tri_geom}
	geom_array := NS.Array.alloc()->initWithObjects(raw_data(geometries[:]), 1)
	prim_desc->setGeometryDescriptors(geom_array)

	sizes := device->accelerationStructureSizesWithDescriptor(prim_desc)
	fmt.println("  AS size:", sizes.accelerationStructureSize)

	as := device->newAccelerationStructureWithSize(NS.UInteger(sizes.accelerationStructureSize))
	scratch := device->newBufferWithLength(
		NS.UInteger(sizes.buildScratchBufferSize),
		MTL.ResourceStorageModeShared,
	)

	cmd_buf := cmd_queue->commandBuffer()
	as_encoder := cmd_buf->accelerationStructureCommandEncoder()
	as_encoder->buildAccelerationStructure(as, prim_desc, scratch, 0)
	as_encoder->endEncoding()
	cmd_buf->commit()
	cmd_buf->waitUntilCompleted()
	fmt.println("  Done.")

	// Dispatch compute
	fmt.println("Rendering...")

	dispatch_buf := cmd_queue->commandBuffer()
	encoder := dispatch_buf->computeCommandEncoder()

	encoder->setComputePipelineState(pipeline)
	encoder->setBuffer(scene_buffer, 0, 0)
	encoder->setBuffer(material_buffer, 0, 1)
	encoder->setBuffer(output_buffer, 0, 2)
	encoder->setAccelerationStructure(as, 3)
	// Buffer 4: vertex data for normal calculation (packed float3 per vertex)
	encoder->setBuffer(vertex_buffer, 0, 4)
	// Buffer 5: index buffer
	encoder->setBuffer(index_buffer, 0, 5)
	encoder->setBuffer(light_buffer, 0, 6)

	tg_size := MTL.Size{width = 16, height = 8, depth = 1}
	grid_size := MTL.Size{
		width  = NS.Integer(image_width),
		height = NS.Integer(image_height),
		depth  = 1,
	}
	encoder->dispatchThreads(grid_size, tg_size)
	encoder->endEncoding()
	dispatch_buf->commit()
	dispatch_buf->waitUntilCompleted()
	fmt.println("  Done.")

	// Readback + write
	fmt.println("Writing", file_output)

	output_data := output_buffer->contentsAsSlice([][4]f32)
	pixels := make([]u8, pixel_count * 3)
	defer delete(pixels)

	for i in 0 ..< pixel_count {
		r := u8(clamp(output_data[i][0] * 255.0, 0.0, 255.0))
		g := u8(clamp(output_data[i][1] * 255.0, 0.0, 255.0))
		b := u8(clamp(output_data[i][2] * 255.0, 0.0, 255.0))
		pixels[i * 3 + 0] = r
		pixels[i * 3 + 1] = g
		pixels[i * 3 + 2] = b
	}

	stbi.flip_vertically_on_write(true)
	ok := stbi.write_png(
		file_output,
		c.int(image_width),
		c.int(image_height),
		3,
		raw_data(pixels),
		c.int(image_width * 3),
	)
	if ok == 0 {
		fmt.eprintln("Failed to write PNG")
		return
	}
	fmt.println("Wrote", file_output)
}
