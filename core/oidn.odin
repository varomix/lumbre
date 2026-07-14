package lumbre_core

import "core:c"
import "core:dynlib"
import "core:fmt"
import "core:strings"

// Minimal dynamic binding for the stable OpenImageDenoise C API. Keeping OIDN
// dynamically loaded means a normal `odin build .` remains self-contained;
// deployments opt in by installing libOpenImageDenoise and its model files.
OIDN_DEVICE_TYPE_CPU :: c.int(1)
OIDN_FORMAT_FLOAT3   :: c.int(3)
OIDN_ERROR_NONE      :: c.int(0)

OIDN_New_Device             :: #type proc "c" (device_type: c.int) -> rawptr
OIDN_Release_Device         :: #type proc "c" (device: rawptr)
OIDN_Commit_Device          :: #type proc "c" (device: rawptr)
OIDN_Get_Device_Error       :: #type proc "c" (device: rawptr, message: ^cstring) -> c.int
OIDN_New_Filter             :: #type proc "c" (device: rawptr, filter_type: cstring) -> rawptr
OIDN_Release_Filter         :: #type proc "c" (filter: rawptr)
OIDN_Set_Shared_Filter_Image :: #type proc "c" (
	filter: rawptr, name: cstring, data: rawptr, format: c.int,
	width, height, byte_offset, pixel_byte_stride, row_byte_stride: uintptr,
)
OIDN_Set_Filter_Bool        :: #type proc "c" (filter: rawptr, name: cstring, value: c.bool)
OIDN_Commit_Filter          :: #type proc "c" (filter: rawptr)
OIDN_Execute_Filter         :: #type proc "c" (filter: rawptr)

OIDN_API :: struct {
	new_device:               OIDN_New_Device,
	release_device:           OIDN_Release_Device,
	commit_device:            OIDN_Commit_Device,
	get_device_error:         OIDN_Get_Device_Error,
	new_filter:               OIDN_New_Filter,
	release_filter:           OIDN_Release_Filter,
	set_shared_filter_image:  OIDN_Set_Shared_Filter_Image,
	set_filter_bool:          OIDN_Set_Filter_Bool,
	commit_filter:            OIDN_Commit_Filter,
	execute_filter:           OIDN_Execute_Filter,
}

oidn_load_symbol :: proc(lib: dynlib.Library, $T: typeid, name: string) -> (T, bool) {
	ptr, found := dynlib.symbol_address(lib, name)
	return cast(T)ptr, found
}

oidn_load :: proc(path: string) -> (dynlib.Library, OIDN_API, bool) {
	lib, loaded := dynlib.load_library(path)
	if !loaded {
		return nil, {}, false
	}

	api: OIDN_API
	ok: bool
	defer if !ok { dynlib.unload_library(lib) }
	api.new_device, ok = oidn_load_symbol(lib, OIDN_New_Device, "oidnNewDevice"); if !ok { return nil, {}, false }
	api.release_device, ok = oidn_load_symbol(lib, OIDN_Release_Device, "oidnReleaseDevice"); if !ok { return nil, {}, false }
	api.commit_device, ok = oidn_load_symbol(lib, OIDN_Commit_Device, "oidnCommitDevice"); if !ok { return nil, {}, false }
	api.get_device_error, ok = oidn_load_symbol(lib, OIDN_Get_Device_Error, "oidnGetDeviceError"); if !ok { return nil, {}, false }
	api.new_filter, ok = oidn_load_symbol(lib, OIDN_New_Filter, "oidnNewFilter"); if !ok { return nil, {}, false }
	api.release_filter, ok = oidn_load_symbol(lib, OIDN_Release_Filter, "oidnReleaseFilter"); if !ok { return nil, {}, false }
	api.set_shared_filter_image, ok = oidn_load_symbol(lib, OIDN_Set_Shared_Filter_Image, "oidnSetSharedFilterImage"); if !ok { return nil, {}, false }
	api.set_filter_bool, ok = oidn_load_symbol(lib, OIDN_Set_Filter_Bool, "oidnSetFilterBool"); if !ok { return nil, {}, false }
	api.commit_filter, ok = oidn_load_symbol(lib, OIDN_Commit_Filter, "oidnCommitFilter"); if !ok { return nil, {}, false }
	api.execute_filter, ok = oidn_load_symbol(lib, OIDN_Execute_Filter, "oidnExecuteFilter"); if !ok { return nil, {}, false }
	return lib, api, true
}

oidn_error :: proc(api: OIDN_API, device: rawptr) -> string {
	message: cstring
	code := api.get_device_error(device, &message)
	if code == OIDN_ERROR_NONE {
		return ""
	}
	return strings.clone_from_cstring(message) if message != nil else fmt.tprintf("OIDN error %d", code)
}

// Denoises a linear HDR beauty buffer in-place. The renderer owns all three
// buffers; Float3 + a 16-byte pixel stride deliberately skips their alpha lane.
oidn_denoise :: proc(beauty, albedo, normal: [][4]f32, width, height: i32, library_path: string = "") -> bool {
	if width <= 0 || height <= 0 || len(beauty) != len(albedo) || len(beauty) != len(normal) {
		fmt.eprintln("OIDN: invalid image or auxiliary-buffer dimensions")
		return false
	}

	paths := make([dynamic]string)
	defer delete(paths)
	if library_path != "" {
		append(&paths, library_path)
	}
	append(&paths, "lib/darwin/oidn/libOpenImageDenoise.dylib", "libOpenImageDenoise.dylib", "/opt/homebrew/lib/libOpenImageDenoise.dylib")

	lib: dynlib.Library
	api: OIDN_API
	loaded := false
	for path in paths {
		lib, api, loaded = oidn_load(path)
		if loaded { break }
	}
	if !loaded {
		fmt.eprintln("OIDN is enabled but libOpenImageDenoise.dylib was not found. Set LUMBRE_OIDN_LIBRARY or install it under lib/darwin/oidn/.")
		return false
	}
	defer dynlib.unload_library(lib)

	device := api.new_device(OIDN_DEVICE_TYPE_CPU)
	if device == nil {
		fmt.eprintln("OIDN: could not create the CPU device")
		return false
	}
	defer api.release_device(device)
	api.commit_device(device)
	if err := oidn_error(api, device); err != "" {
		fmt.eprintln("OIDN device error:", err)
		return false
	}

	filter := api.new_filter(device, "RT")
	if filter == nil {
		fmt.eprintln("OIDN: could not create the RT filter:", oidn_error(api, device))
		return false
	}
	defer api.release_filter(filter)

	pixel_stride := uintptr(size_of([4]f32))
	row_stride := uintptr(width) * pixel_stride
	api.set_shared_filter_image(filter, "color", raw_data(beauty), OIDN_FORMAT_FLOAT3, uintptr(width), uintptr(height), 0, pixel_stride, row_stride)
	api.set_shared_filter_image(filter, "albedo", raw_data(albedo), OIDN_FORMAT_FLOAT3, uintptr(width), uintptr(height), 0, pixel_stride, row_stride)
	api.set_shared_filter_image(filter, "normal", raw_data(normal), OIDN_FORMAT_FLOAT3, uintptr(width), uintptr(height), 0, pixel_stride, row_stride)
	api.set_shared_filter_image(filter, "output", raw_data(beauty), OIDN_FORMAT_FLOAT3, uintptr(width), uintptr(height), 0, pixel_stride, row_stride)
	api.set_filter_bool(filter, "hdr", true)
	api.set_filter_bool(filter, "cleanAux", true)
	api.commit_filter(filter)
	api.execute_filter(filter)
	if err := oidn_error(api, device); err != "" {
		fmt.eprintln("OIDN filter error:", err)
		return false
	}
	return true
}
