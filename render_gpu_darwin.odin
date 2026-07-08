package main

import "core:c"
import "core:fmt"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"
import stbi "vendor:stb/image"

SphereGPU :: struct {
	center:        [4]f32,
	radius:        f32,
	material_kind: i32,
	fuzz:          f32,
	ir:            f32,
	albedo:        [4]f32,
}

AxisAlignedBoundingBox :: struct {
	min: [3]f32,
	max: [3]f32,
}

SceneData :: struct {
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
	seed:              u32,
	_pad:              [2]f32,
}

render_gpu :: proc(
	world: []Sphere,
	camera: Camera,
	image_width, image_height: i32,
	samples_per_pixel, max_depth: i32,
	file_output: cstring,
) {
	spheres_gpu := make([dynamic]SphereGPU, len(world))
	defer delete(spheres_gpu)

	for s, i in world {
		kind := i32(0)
		switch s.material.kind {
		case .Lambertian:
			kind = 0
		case .Metal:
			kind = 1
		case .Dielectric:
			kind = 2
		}
		spheres_gpu[i] = SphereGPU {
			center        = {f32(s.center.x), f32(s.center.y), f32(s.center.z), 0},
			radius        = f32(s.radius),
			material_kind = kind,
			fuzz          = f32(s.material.fuzz),
			ir            = f32(s.material.ir),
			albedo        = {
				f32(s.material.albedo.x),
				f32(s.material.albedo.y),
				f32(s.material.albedo.z),
				0,
			},
		}
	}

	boxes := make([dynamic]AxisAlignedBoundingBox, len(world))
	defer delete(boxes)
	for s, i in world {
		r := [3]f32{f32(s.radius), f32(s.radius), f32(s.radius)}
		c := [3]f32{f32(s.center.x), f32(s.center.y), f32(s.center.z)}
		boxes[i] = AxisAlignedBoundingBox {
			min = c - r,
			max = c + r,
		}
	}

	scene_data := SceneData {
		origin            = {f32(camera.origin.x), f32(camera.origin.y), f32(camera.origin.z), 0},
		lower_left        = {
			f32(camera.lower_left_corner.x),
			f32(camera.lower_left_corner.y),
			f32(camera.lower_left_corner.z),
			0,
		},
		horizontal        = {
			f32(camera.horizontal.x),
			f32(camera.horizontal.y),
			f32(camera.horizontal.z),
			0,
		},
		vertical          = {
			f32(camera.vertical.x),
			f32(camera.vertical.y),
			f32(camera.vertical.z),
			0,
		},
		u                 = {f32(camera.u.x), f32(camera.u.y), f32(camera.u.z), 0},
		v                 = {f32(camera.v.x), f32(camera.v.y), f32(camera.v.z), 0},
		lens_radius       = f32(camera.lens_radius),
		image_width       = image_width,
		image_height      = image_height,
		samples_per_pixel = samples_per_pixel,
		max_depth         = max_depth,
		seed              = 42,
	}

	pool := NS.scoped_autoreleasepool()
	defer pool->drain()

	device := MTL.CreateSystemDefaultDevice()
	assert(device != nil, "Metal device required")
	assert(bool(device->supportsRaytracing()), "Raytracing required")
	fmt.println("Device:", device->name()->odinString())

	sphere_buffer := device->newBufferWithSlice(spheres_gpu[:], MTL.ResourceStorageModeShared)
	bbox_buffer := device->newBufferWithSlice(boxes[:], MTL.ResourceStorageModeShared)
	scene_slice := ([^]byte)(&scene_data)[:size_of(SceneData)]
	scene_buffer := device->newBufferWithBytes(scene_slice, MTL.ResourceStorageModeShared)

	pixel_count := int(image_width) * int(image_height)
	output_buffer := device->newBufferWithLength(
		NS.UInteger(pixel_count * size_of([4]f32)),
		MTL.ResourceStorageModeShared,
	)

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

	intersection_func := library->newFunctionWithName(NS.AT("sphereIntersection"))
	assert(intersection_func != nil, "intersection function not found")

	desc := MTL.ComputePipelineDescriptor.alloc()->init()
	desc->setComputeFunction(kernel_func)

	linked_functions := MTL.LinkedFunctions.linkedFunctions()
	funcs := [?]^NS.Object{auto_cast intersection_func}
	funcs_array := NS.Array.alloc()->initWithObjects(raw_data(funcs[:]), 1)
	linked_functions->setFunctions(funcs_array)
	desc->setLinkedFunctions(linked_functions)

	pipeline, p_err := MTL.Device_newComputePipelineStateWithDescriptorWithReflection(
		device,
		desc,
		MTL.PipelineOption{},
		nil,
	)
	if p_err != nil {
		fmt.eprintln("Pipeline creation failed:", p_err->localizedDescription()->odinString())
		return
	}

	handle := pipeline->functionHandleWithFunction(intersection_func)
	assert(handle != nil, "function handle is nil")

	ift_desc := MTL.IntersectionFunctionTableDescriptor.alloc()->init()
	ift_desc->setFunctionCount(1)
	ift := pipeline->newIntersectionFunctionTable(ift_desc)
	MTL.IntersectionFunctionTable_setFunction(ift, handle, 0)
	MTL.IntersectionFunctionTable_setBuffer(ift, sphere_buffer, 0, 5)

	fmt.println("Building acceleration structure...")

	bbox_geom := MTL.AccelerationStructureBoundingBoxGeometryDescriptor.alloc()->init()
	bbox_geom->setBoundingBoxBuffer(bbox_buffer)
	bbox_geom->setBoundingBoxCount(NS.UInteger(len(boxes)))
	bbox_geom->setBoundingBoxStride(NS.UInteger(size_of(AxisAlignedBoundingBox)))

	prim_desc := MTL.PrimitiveAccelerationStructureDescriptor.alloc()->init()

	geometries := [?]^NS.Object{auto_cast bbox_geom}
	geom_array := NS.Array.alloc()->initWithObjects(raw_data(geometries[:]), 1)
	prim_desc->setGeometryDescriptors(geom_array)

	sizes := device->accelerationStructureSizesWithDescriptor(prim_desc)
	fmt.println("  AS size:", sizes.accelerationStructureSize)

	as := device->newAccelerationStructureWithSize(NS.UInteger(sizes.accelerationStructureSize))
	scratch := device->newBufferWithLength(
		NS.UInteger(sizes.buildScratchBufferSize),
		MTL.ResourceStorageModeShared,
	)

	cmd_queue := device->newCommandQueue()
	cmd_buf := cmd_queue->commandBuffer()
	as_encoder := cmd_buf->accelerationStructureCommandEncoder()
	as_encoder->buildAccelerationStructure(as, prim_desc, scratch, 0)
	as_encoder->endEncoding()
	cmd_buf->commit()
	cmd_buf->waitUntilCompleted()

	fmt.println("  Done.")

	fmt.println("Rendering...")

	dispatch_buf := cmd_queue->commandBuffer()
	encoder := dispatch_buf->computeCommandEncoder()

	encoder->setComputePipelineState(pipeline)
	encoder->setBuffer(scene_buffer, 0, 0)
	encoder->setBuffer(sphere_buffer, 0, 1)
	encoder->setBuffer(output_buffer, 0, 2)
	encoder->setAccelerationStructure(as, 3)
	encoder->setIntersectionFunctionTable(ift, 4)
	encoder->setBuffer(sphere_buffer, 0, 5)

	tg_size := MTL.Size {
		width  = 16,
		height = 8,
		depth  = 1,
	}
	grid_size := MTL.Size {
		width  = NS.Integer(image_width),
		height = NS.Integer(image_height),
		depth  = 1,
	}
	encoder->dispatchThreads(grid_size, tg_size)
	encoder->endEncoding()
	dispatch_buf->commit()
	dispatch_buf->waitUntilCompleted()

	fmt.println("  Done.")

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
