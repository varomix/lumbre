// Lumbre's Houdini 21 Hydra plugin entry point.
//
// This delegate hosts Lumbre's renderer inside the Houdini process through the
// narrow C ABI in `houdini/lumbre_bridge/lumbre_bridge.h`. The bridge dylib
// imports only Lumbre's USD-free `core`, so nothing here pulls in Lumbre's
// vendored OpenUSD alongside Houdini's own. The delegate:
//   * loads the bridge dylib and creates one persistent Lumbre context;
//   * uploads synced Hydra meshes as world-space triangles;
//   * derives the camera and resolution from Hydra's render-pass state;
//   * renders on the CPU and publishes the framebuffer into Hydra's color AOV.
// If the bridge cannot be loaded it falls back to a recognizable diagnostic
// gradient so the plugin still proves discovery and lifecycle.

#include "pxr/pxr.h"
#include "pxr/base/tf/registryManager.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/type.h"
#include "pxr/imaging/hd/camera.h"
#include "pxr/imaging/hd/changeTracker.h"
#include "pxr/imaging/hd/driver.h"
#include "pxr/imaging/hd/light.h"
#include "pxr/imaging/hd/mesh.h"
#include "pxr/imaging/hd/meshUtil.h"
#include "pxr/imaging/hd/renderDelegate.h"
#include "pxr/imaging/hd/renderBuffer.h"
#include "pxr/imaging/hd/renderIndex.h"
#include "pxr/imaging/hd/renderPass.h"
#include "pxr/imaging/hd/renderPassState.h"
#include "pxr/imaging/hd/rendererPlugin.h"
#include "pxr/imaging/hd/rendererPluginRegistry.h"
#include "pxr/imaging/hd/resourceRegistry.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hgi/hgi.h"
#include "pxr/imaging/hgi/blitCmds.h"
#include "pxr/imaging/hgi/blitCmdsOps.h"
#include "pxr/imaging/hgi/texture.h"
#include "pxr/imaging/hgi/tokens.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/vt/array.h"

#include "../lumbre_bridge/lumbre_bridge.h"

#include <dlfcn.h>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

// ---------------------------------------------------------------------------
// Bridge dylib loader
// ---------------------------------------------------------------------------

// Function-pointer view of the C ABI. Resolved once via dlopen/dlsym so the
// plugin has no link-time dependency on the Odin dylib; a missing bridge is a
// graceful fallback, not a load failure.
struct LumbreBridge {
    void *handle = nullptr;

    uint32_t (*abi_version)(void) = nullptr;
    LumbreBridgeContext (*create)(void) = nullptr;
    void (*destroy)(LumbreBridgeContext) = nullptr;
    int (*replace_triangles)(LumbreBridgeContext, const LumbreBridgeTriangle *, int32_t) = nullptr;
    int (*replace_lights)(LumbreBridgeContext, const LumbreBridgeLight *, int32_t) = nullptr;
    int (*set_resolution)(LumbreBridgeContext, int32_t, int32_t) = nullptr;
    int (*set_quality)(LumbreBridgeContext, int32_t, int32_t) = nullptr;
    int (*set_camera)(LumbreBridgeContext, const float[3], const float[3], const float[3], float) = nullptr;
    int (*render)(LumbreBridgeContext) = nullptr;
    int (*framebuffer_size)(LumbreBridgeContext, int32_t *, int32_t *) = nullptr;
    int (*read_rgba_f32)(LumbreBridgeContext, float *, int32_t, int32_t) = nullptr;
    int (*write_png)(LumbreBridgeContext, const char *) = nullptr;

    bool IsValid() const { return handle && create && render && read_rgba_f32; }

    bool Load() {
        // Try the sibling install location first, then a bare name so a
        // DYLD-visible copy is also found.
        const char *candidates[] = {
            "@rpath/libLumbreBridge.dylib",
            "libLumbreBridge.dylib",
        };
        for (const char *name : candidates) {
            handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (handle) break;
        }
        if (!handle) {
            std::fprintf(stderr, "Lumbre: could not dlopen libLumbreBridge.dylib (%s)\n",
                         dlerror());
            return false;
        }
        abi_version = reinterpret_cast<decltype(abi_version)>(dlsym(handle, "lumbre_bridge_abi_version"));
        create = reinterpret_cast<decltype(create)>(dlsym(handle, "lumbre_bridge_create"));
        destroy = reinterpret_cast<decltype(destroy)>(dlsym(handle, "lumbre_bridge_destroy"));
        replace_triangles = reinterpret_cast<decltype(replace_triangles)>(dlsym(handle, "lumbre_bridge_replace_triangles"));
        replace_lights = reinterpret_cast<decltype(replace_lights)>(dlsym(handle, "lumbre_bridge_replace_lights"));
        set_resolution = reinterpret_cast<decltype(set_resolution)>(dlsym(handle, "lumbre_bridge_set_resolution"));
        set_quality = reinterpret_cast<decltype(set_quality)>(dlsym(handle, "lumbre_bridge_set_quality"));
        set_camera = reinterpret_cast<decltype(set_camera)>(dlsym(handle, "lumbre_bridge_set_camera"));
        render = reinterpret_cast<decltype(render)>(dlsym(handle, "lumbre_bridge_render"));
        framebuffer_size = reinterpret_cast<decltype(framebuffer_size)>(dlsym(handle, "lumbre_bridge_framebuffer_size"));
        read_rgba_f32 = reinterpret_cast<decltype(read_rgba_f32)>(dlsym(handle, "lumbre_bridge_read_rgba_f32"));
        write_png = reinterpret_cast<decltype(write_png)>(dlsym(handle, "lumbre_bridge_write_png"));

        if (!IsValid()) {
            std::fputs("Lumbre: bridge dylib is missing required symbols\n", stderr);
            return false;
        }
        if (abi_version && abi_version() != LUMBRE_HOUDINI_BRIDGE_ABI_VERSION) {
            std::fprintf(stderr, "Lumbre: bridge ABI mismatch (dylib %u, plugin %u)\n",
                         abi_version(), LUMBRE_HOUDINI_BRIDGE_ABI_VERSION);
            return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Render buffer
// ---------------------------------------------------------------------------

// CPU render buffer. Format-generic: Houdini allocates several AOVs (color as
// Float32Vec4, depth as Float32, etc.) through this class, so Allocate must
// accept whatever format Hydra asks for -- rejecting depth left the viewport
// with nothing to composite against. Pixels are read back by Houdini through
// the CPU Map() path (HUSD_RenderBuffer).
class HdLumbreRenderBuffer final : public HdRenderBuffer {
public:
    explicit HdLumbreRenderBuffer(SdfPath const &id, Hgi *hgi)
        : HdRenderBuffer(id), _hgi(hgi) {
        std::fprintf(stderr, "Lumbre: buffer created id='%s' this=%p\\n",
                     id.GetText(), static_cast<void *>(this));
    }

    bool Allocate(
        GfVec3i const &dimensions,
        HdFormat format,
        bool multiSampled) override {
        if (dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[2] != 1) {
            _Deallocate();
            return false;
        }
        _dimensions = dimensions;
        _format = format;
        _multiSampled = multiSampled;
        _texelSize = HdDataSizeOfFormat(format);
        const size_t bytes = static_cast<size_t>(dimensions[0]) * dimensions[1] * _texelSize;
        _DestroyGpuTexture();
        _buffer.assign(bytes, 0);
        _mappers.store(0, std::memory_order_relaxed);
        _gpuDirty = true;
        _converged = false;
        std::fprintf(stderr,
                     "Lumbre: buffer Allocate id='%s' this=%p dims=%dx%dx%d fmt=%d texelBytes=%zu totalBytes=%zu msaa=%d color=%d\\n",
                     GetId().GetText(), static_cast<void *>(this), dimensions[0], dimensions[1],
                     dimensions[2], int(format), _texelSize, bytes, int(multiSampled),
                     int(format == HdFormatFloat32Vec4));
        return true;
    }

    unsigned int GetWidth() const override { return _dimensions[0]; }
    unsigned int GetHeight() const override { return _dimensions[1]; }
    unsigned int GetDepth() const override { return _dimensions[2]; }
    HdFormat GetFormat() const override { return _format; }
    bool IsMultiSampled() const override { return _multiSampled; }
    void *Map() override {
        if (_buffer.empty()) {
            _Debug("Map", nullptr, _mappers.load(std::memory_order_relaxed));
            return nullptr;
        }
        const int count = _mappers.fetch_add(1, std::memory_order_acq_rel) + 1;
        _Debug("Map", _buffer.data(), count);
        return _buffer.data();
    }
    void Unmap() override {
        // Hydra's presentation task maps the buffer before copying it. Keep
        // this state accurate: reporting false here causes HUSD to discard a
        // valid CPU color AOV as though it had no readable backing store.
        int count = _mappers.load(std::memory_order_acquire);
        while (count > 0 &&
               !_mappers.compare_exchange_weak(count, count - 1,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {}
        _Debug("Unmap", _buffer.empty() ? nullptr : _buffer.data(),
               _mappers.load(std::memory_order_acquire));
    }
    bool IsMapped() const override {
        const int count = _mappers.load(std::memory_order_acquire);
        _Debug("IsMapped", _buffer.empty() ? nullptr : _buffer.data(), count);
        return count != 0;
    }
    void Resolve() override {
        _Debug("Resolve", _buffer.empty() ? nullptr : _buffer.data(),
               _mappers.load(std::memory_order_acquire));
    }
    bool IsConverged() const override {
        _Debug("IsConverged", _buffer.empty() ? nullptr : _buffer.data(),
               _mappers.load(std::memory_order_acquire));
        return _converged;
    }
    VtValue GetResource(bool multiSampled) const override {
        // Houdini supplies HgiNull to this CPU delegate. Returning no GPU
        // resource makes HUSD_RenderBuffer map and copy this buffer directly.
        const int event = _debugEvents.fetch_add(1, std::memory_order_relaxed);
        if (event < 16) {
            std::fprintf(stderr,
                         "Lumbre: buffer GetResource id='%s' this=%p msaa=%d -> empty (CPU Map path)\\n",
                         GetId().GetText(), static_cast<const void *>(this), int(multiSampled));
        }
        return VtValue();
    }
    void SetConverged(bool converged) { _converged = converged; }
    void MarkGpuDirty() { _gpuDirty = true; }

    // Color (Float32Vec4) pixel access for the render pass. Null for other AOVs.
    float *ColorData() {
        if (_format != HdFormatFloat32Vec4 || _buffer.empty()) return nullptr;
        return reinterpret_cast<float *>(_buffer.data());
    }
    float *DepthData() {
        if (_format != HdFormatFloat32 || _buffer.empty()) return nullptr;
        return reinterpret_cast<float *>(_buffer.data());
    }
    bool IsColor() const { return _format == HdFormatFloat32Vec4; }

    void FillDiagnosticGradient() {
        float *px = ColorData();
        const int width = _dimensions[0];
        const int height = _dimensions[1];
        if (!px || width <= 0 || height <= 0) return;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(width - 1 > 0 ? width - 1 : 1);
                const float v = static_cast<float>(y) / static_cast<float>(height - 1 > 0 ? height - 1 : 1);
                float *p = px + (static_cast<size_t>(y) * width + x) * 4;
                p[0] = 0.12f + 0.88f * u; p[1] = 0.025f + 0.18f * v;
                p[2] = 0.035f + 0.32f * (1.0f - u); p[3] = 1.0f;
            }
        }
        _converged = true;
    }

protected:
    void _Deallocate() override {
        std::fprintf(stderr, "Lumbre: buffer Deallocate id='%s' this=%p bytes=%zu\\n",
                     GetId().GetText(), static_cast<void *>(this), _buffer.size());
        _buffer.clear();
        _dimensions = GfVec3i(0);
        _format = HdFormatInvalid;
        _mappers.store(0, std::memory_order_relaxed);
        _DestroyGpuTexture();
        _gpuDirty = true;
        _converged = false;
    }

private:
    // The viewport can poll these methods every frame. Log only the first
    // handful of calls per buffer so a single viewport interaction is useful
    // without flooding Houdini's console.
    void _Debug(const char *operation, const void *data, int mapperCount) const {
        const int event = _debugEvents.fetch_add(1, std::memory_order_relaxed);
        if (event < 16) {
            std::fprintf(stderr,
                         "Lumbre: buffer %s id='%s' this=%p data=%p mapped=%d dims=%ux%u fmt=%d converged=%d\\n",
                         operation, GetId().GetText(), static_cast<const void *>(this), data,
                         mapperCount, GetWidth(), GetHeight(), int(_format), int(_converged));
        }
    }

    void _DestroyGpuTexture() const {
        if (_hgi) {
            if (_gpuTexture) _hgi->DestroyTexture(&_gpuTexture);
            if (_gpuTextureMSAA) _hgi->DestroyTexture(&_gpuTextureMSAA);
        }
    }

    void _CreateGpuTextures() {
        _DestroyGpuTexture();
        if (!_hgi || !IsColor() || _buffer.empty()) return;

        HgiTextureDesc desc;
        desc.debugName = "HdLumbre color AOV";
        desc.usage = HgiTextureUsageBitsShaderRead | HgiTextureUsageBitsColorTarget;
        desc.format = HgiFormatFloat32Vec4;
        desc.dimensions = _dimensions;
        desc.sampleCount = HgiSampleCount1;
        _gpuTexture = _hgi->CreateTexture(desc);

        desc.debugName = "HdLumbre color AOV MSAA";
        desc.sampleCount = HgiSampleCount4;
        _gpuTextureMSAA = _hgi->CreateTexture(desc);
        std::fprintf(stderr,
                     "Lumbre: buffer GPU allocate id='%s' api='%s' texture=%p msaaTexture=%p\\n",
                     GetId().GetText(), _hgi->GetAPIName().GetText(),
                     static_cast<void *>(_gpuTexture.Get()),
                     static_cast<void *>(_gpuTextureMSAA.Get()));
    }

    void _UploadGpuTextures() const {
        if (!_gpuDirty || !_hgi || !_gpuTexture || !_gpuTextureMSAA || _buffer.empty()) return;

        HgiBlitCmdsUniquePtr blit = _hgi->CreateBlitCmds();
        HgiTextureCpuToGpuOp op;
        op.cpuSourceBuffer = _buffer.data();
        op.bufferByteSize = _buffer.size();
        op.gpuDestinationTexture = _gpuTexture;
        blit->CopyTextureCpuToGpu(op);
        op.gpuDestinationTexture = _gpuTextureMSAA;
        blit->CopyTextureCpuToGpu(op);
        _hgi->SubmitCmds(blit.get(), HgiSubmitWaitTypeWaitUntilCompleted);
        _gpuDirty = false;
        std::fprintf(stderr, "Lumbre: buffer GPU upload id='%s' texture=%p msaaTexture=%p bytes=%zu\\n",
                     GetId().GetText(), static_cast<void *>(_gpuTexture.Get()),
                     static_cast<void *>(_gpuTextureMSAA.Get()), _buffer.size());
    }

    GfVec3i _dimensions = GfVec3i(0);
    HdFormat _format = HdFormatInvalid;
    bool _multiSampled = false;
    bool _converged = false;
    Hgi *_hgi = nullptr;
    mutable HgiTextureHandle _gpuTexture;
    mutable HgiTextureHandle _gpuTextureMSAA;
    mutable bool _gpuDirty = true;
    std::atomic<int> _mappers{0};
    mutable std::atomic<int> _debugEvents{0};
    size_t _texelSize = 0;
    std::vector<uint8_t> _buffer;
};

// ---------------------------------------------------------------------------
// Shared geometry registry (delegate-owned; touched by mesh sync + render)
// ---------------------------------------------------------------------------

// A minimal render-param that hands synced meshes a pointer to the shared
// geometry registry on the delegate.
class HdLumbreRenderDelegate;

class LumbreRenderParam final : public HdRenderParam {
public:
    explicit LumbreRenderParam(HdLumbreRenderDelegate *owner) : _owner(owner) {}
    HdLumbreRenderDelegate *Owner() const { return _owner; }
private:
    HdLumbreRenderDelegate *_owner;
};

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

class HdLumbreMesh final : public HdMesh {
public:
    explicit HdLumbreMesh(SdfPath const &id)
        : HdMesh(id) {}

    HdDirtyBits GetInitialDirtyBitsMask() const override {
        return HdChangeTracker::AllSceneDirtyBits;
    }

    void Sync(
        HdSceneDelegate *sceneDelegate,
        HdRenderParam *renderParam,
        HdDirtyBits *dirtyBits,
        TfToken const &) override;

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override {
        return bits;
    }
    void _InitRepr(TfToken const &, HdDirtyBits *) override {}
};

// Hydra represents USD lights as sprims.  This small adapter translates their
// transform and standard HdLightTokens into the bridge's world-space format.
class HdLumbreLight final : public HdLight {
public:
    HdLumbreLight(SdfPath const &id, TfToken const &typeId) : HdLight(id), _typeId(typeId) {}
    HdDirtyBits GetInitialDirtyBitsMask() const override { return HdLight::AllDirty; }
    void Sync(HdSceneDelegate *sceneDelegate, HdRenderParam *renderParam,
              HdDirtyBits *dirtyBits) override;
private:
    TfToken _typeId;
};

// ---------------------------------------------------------------------------
// Render pass
// ---------------------------------------------------------------------------

class HdLumbreRenderPass final : public HdRenderPass {
public:
    HdLumbreRenderPass(HdRenderIndex *index, HdRprimCollection const &collection,
                       HdLumbreRenderDelegate *owner)
        : HdRenderPass(index, collection)
        , _owner(owner) {
        std::fputs("Lumbre: Hydra render pass created\n", stderr);
    }

    bool IsConverged() const override { return _execCount > 8; }

private:
    void _Execute(
        HdRenderPassStateSharedPtr const &renderPassState,
        TfTokenVector const &) override;

    HdLumbreRenderDelegate *_owner;
    mutable int _execCount = 0;
};

// ---------------------------------------------------------------------------
// Render delegate
// ---------------------------------------------------------------------------

class HdLumbreRenderDelegate final : public HdRenderDelegate {
public:
    explicit HdLumbreRenderDelegate(HdRenderSettingsMap const &settings)
        : HdRenderDelegate(settings)
        , _resourceRegistry(std::make_shared<HdResourceRegistry>())
        , _renderParam(std::make_unique<LumbreRenderParam>(this)) {
        if (_bridge.Load()) {
            _context = _bridge.create();
            std::fprintf(stderr, "Lumbre: bridge context %s\n",
                         _context ? "created" : "creation failed");
        } else {
            std::fputs("Lumbre: falling back to diagnostic gradient (no bridge)\n", stderr);
        }
    }

    ~HdLumbreRenderDelegate() override {
        if (_context && _bridge.destroy) {
            _bridge.destroy(_context);
            _context = nullptr;
        }
    }

    HdRenderParam *GetRenderParam() const override { return _renderParam.get(); }

    const TfTokenVector &GetSupportedRprimTypes() const override { return _rprimTypes; }
    const TfTokenVector &GetSupportedSprimTypes() const override { return _sprimTypes; }
    const TfTokenVector &GetSupportedBprimTypes() const override { return _bprimTypes; }

    void SetDrivers(HdDriverVector const &drivers) override {
        _hgi = nullptr;
        for (HdDriver *driver : drivers) {
            if (driver && driver->name == HgiTokens->renderDriver &&
                driver->driver.IsHolding<Hgi *>()) {
                _hgi = driver->driver.UncheckedGet<Hgi *>();
                break;
            }
        }
        std::fprintf(stderr, "Lumbre: SetDrivers count=%zu hgi=%p\\n",
                     drivers.size(), static_cast<void *>(_hgi));
    }

    HdResourceRegistrySharedPtr GetResourceRegistry() const override {
        return _resourceRegistry;
    }

    HdRenderPassSharedPtr CreateRenderPass(
        HdRenderIndex *index,
        HdRprimCollection const &collection) override {
        return std::make_shared<HdLumbreRenderPass>(index, collection, this);
    }

    HdInstancer *CreateInstancer(HdSceneDelegate *, SdfPath const &) override { return nullptr; }
    void DestroyInstancer(HdInstancer *) override {}

    HdRprim *CreateRprim(TfToken const &typeId, SdfPath const &id) override {
        std::fprintf(stderr, "Lumbre: CreateRprim type='%s' id='%s'\\n",
                     typeId.GetText(), id.GetText());
        if (typeId != HdPrimTypeTokens->mesh) {
            std::fprintf(stderr, "Lumbre: rejecting unsupported rprim type='%s'\\n",
                         typeId.GetText());
            return nullptr;
        }
        return new HdLumbreMesh(id);
    }
    void DestroyRprim(HdRprim *rprim) override {
        if (rprim) {
            std::fprintf(stderr, "Lumbre: DestroyRprim id='%s'\\n",
                         rprim->GetId().GetText());
            RemoveGeometry(rprim->GetId());
            delete rprim;
        }
    }

    // The camera sprim is required: without it Hydra never syncs the viewport
    // camera, and the render-pass state's world-to-view matrix stays identity
    // (camera stuck at the origin). HdCamera's built-in implementation is enough
    // — the render pass reads its matrices off the render-pass state.
    HdSprim *CreateSprim(TfToken const &typeId, SdfPath const &id) override {
        if (typeId == HdPrimTypeTokens->camera) return new HdCamera(id);
        if (typeId == HdPrimTypeTokens->rectLight || typeId == HdPrimTypeTokens->diskLight ||
            typeId == HdPrimTypeTokens->sphereLight || typeId == HdPrimTypeTokens->cylinderLight ||
            typeId == HdPrimTypeTokens->distantLight || typeId == HdPrimTypeTokens->simpleLight) {
            std::fprintf(stderr, "Lumbre: CreateSprim light type='%s' id='%s'\\n",
                         typeId.GetText(), id.GetText());
            return new HdLumbreLight(id, typeId);
        }
        return nullptr;
    }
    HdSprim *CreateFallbackSprim(TfToken const &typeId) override {
        if (typeId == HdPrimTypeTokens->camera) return new HdCamera(SdfPath::EmptyPath());
        return nullptr;
    }
    void DestroySprim(HdSprim *sprim) override {
        if (sprim) RemoveLight(sprim->GetId());
        delete sprim;
    }

    HdBprim *CreateBprim(TfToken const &typeId, SdfPath const &id) override {
        if (typeId != HdPrimTypeTokens->renderBuffer) return nullptr;
        std::fprintf(stderr, "Lumbre: CreateBprim renderBuffer id='%s' hgi=%p\\n",
                     id.GetText(), static_cast<void *>(_hgi));
        return new HdLumbreRenderBuffer(id, _hgi);
    }
    HdBprim *CreateFallbackBprim(TfToken const &) override { return nullptr; }
    void DestroyBprim(HdBprim *bprim) override { delete bprim; }

    void CommitResources(HdChangeTracker *) override { _resourceRegistry->Commit(); }

    HdAovDescriptor GetDefaultAovDescriptor(TfToken const &name) const override {
        if (name == HdAovTokens->color) {
            return HdAovDescriptor(HdFormatFloat32Vec4, false, VtValue(GfVec4f(0.0f)));
        }
        if (name == HdAovTokens->depth) {
            // Solaris composites renderer output with the viewport using this
            // buffer.  A far-plane depth preserves the beauty AOV where
            // Lumbre has not implemented per-pixel depth yet.
            return HdAovDescriptor(HdFormatFloat32, false, VtValue(1.0f));
        }
        return HdRenderDelegate::GetDefaultAovDescriptor(name);
    }

    // --- geometry registry, used by mesh sync and the render pass ---

    void SetGeometry(SdfPath const &id, std::vector<LumbreBridgeTriangle> tris) {
        std::lock_guard<std::mutex> lock(_geomMutex);
        std::fprintf(stderr, "Lumbre: SetGeometry id='%s' triangles=%zu\\n",
                     id.GetText(), tris.size());
        _geometry[id] = std::move(tris);
        _geometryDirty = true;
    }

    void RemoveGeometry(SdfPath const &id) {
        std::lock_guard<std::mutex> lock(_geomMutex);
        if (_geometry.erase(id) > 0) {
            std::fprintf(stderr, "Lumbre: RemoveGeometry id='%s'\\n", id.GetText());
            _geometryDirty = true;
        }
    }

    // Upload the accumulated geometry to the bridge if it changed since the
    // last upload. Returns the total triangle count now resident.
    size_t UploadGeometryIfDirty() {
        std::lock_guard<std::mutex> lock(_geomMutex);
        size_t total = 0;
        for (auto const &kv : _geometry) total += kv.second.size();
        if (!_geometryDirty) return total;

        std::vector<LumbreBridgeTriangle> flat;
        flat.reserve(total);
        for (auto const &kv : _geometry) {
            flat.insert(flat.end(), kv.second.begin(), kv.second.end());
        }
        std::fprintf(stderr, "Lumbre: UploadGeometry meshes=%zu triangles=%zu bridge=%d context=%d\\n",
                     _geometry.size(), flat.size(), int(_bridge.replace_triangles != nullptr),
                     int(_context != nullptr));
        if (_context && _bridge.replace_triangles && !flat.empty()) {
            const int uploaded = _bridge.replace_triangles(_context, flat.data(),
                                                            static_cast<int32_t>(flat.size()));
            std::fprintf(stderr, "Lumbre: UploadGeometry bridge result=%d\\n", uploaded);
        } else if (flat.empty()) {
            std::fputs("Lumbre: UploadGeometry skipped: no mesh triangles received from Hydra\\n", stderr);
        }
        _geometryDirty = false;
        return total;
    }

    void SetLight(SdfPath const &id, LumbreBridgeLight const &light) {
        std::lock_guard<std::mutex> lock(_lightMutex);
        _lights[id] = light;
        _lightsDirty = true;
    }
    void RemoveLight(SdfPath const &id) {
        std::lock_guard<std::mutex> lock(_lightMutex);
        if (_lights.erase(id)) _lightsDirty = true;
    }
    size_t UploadLightsIfDirty() {
        std::lock_guard<std::mutex> lock(_lightMutex);
        if (!_lightsDirty) return _lights.size();
        std::vector<LumbreBridgeLight> flat;
        flat.reserve(_lights.size());
        for (auto const &entry : _lights) flat.push_back(entry.second);
        const int uploaded = (_context && _bridge.replace_lights)
            ? _bridge.replace_lights(_context, flat.empty() ? nullptr : flat.data(), int32_t(flat.size())) : 0;
        std::fprintf(stderr, "Lumbre: UploadLights count=%zu bridge result=%d\\n", flat.size(), uploaded);
        _lightsDirty = false;
        return flat.size();
    }

    LumbreBridge const &Bridge() const { return _bridge; }
    LumbreBridgeContext Context() const { return _context; }
    Hgi *GetHgi() const { return _hgi; }

private:
    static const TfTokenVector _emptyTypes;
    static const TfTokenVector _rprimTypes;
    static const TfTokenVector _sprimTypes;
    static const TfTokenVector _bprimTypes;

    HdResourceRegistrySharedPtr _resourceRegistry;
    std::unique_ptr<LumbreRenderParam> _renderParam;
    Hgi *_hgi = nullptr;

    LumbreBridge _bridge;
    LumbreBridgeContext _context = nullptr;

    std::mutex _geomMutex;
    std::map<SdfPath, std::vector<LumbreBridgeTriangle>> _geometry;
    bool _geometryDirty = false;
    std::mutex _lightMutex;
    std::map<SdfPath, LumbreBridgeLight> _lights;
    bool _lightsDirty = true;
};

const TfTokenVector HdLumbreRenderDelegate::_emptyTypes;
const TfTokenVector HdLumbreRenderDelegate::_rprimTypes = {
    HdPrimTypeTokens->mesh,
};
const TfTokenVector HdLumbreRenderDelegate::_sprimTypes = {
    HdPrimTypeTokens->camera,
    HdPrimTypeTokens->rectLight, HdPrimTypeTokens->diskLight,
    HdPrimTypeTokens->sphereLight, HdPrimTypeTokens->cylinderLight,
    HdPrimTypeTokens->distantLight, HdPrimTypeTokens->simpleLight,
};
const TfTokenVector HdLumbreRenderDelegate::_bprimTypes = {
    HdPrimTypeTokens->renderBuffer,
};

// ---------------------------------------------------------------------------
// Mesh sync implementation
// ---------------------------------------------------------------------------

void HdLumbreMesh::Sync(
    HdSceneDelegate *sceneDelegate,
    HdRenderParam *renderParam,
    HdDirtyBits *dirtyBits,
    TfToken const &) {
    SdfPath const &id = GetId();
    _UpdateVisibility(sceneDelegate, dirtyBits);
    _UpdateInstancer(sceneDelegate, dirtyBits);
    UpdateRenderTag(sceneDelegate, renderParam);
    if (*dirtyBits & HdChangeTracker::DirtyMaterialId) {
        SetMaterialId(sceneDelegate->GetMaterialId(id));
    }

    auto *param = static_cast<LumbreRenderParam *>(renderParam);
    HdLumbreRenderDelegate *owner = param ? param->Owner() : nullptr;

    const bool topoOrXformDirty =
        HdChangeTracker::IsTopologyDirty(*dirtyBits, id) ||
        HdChangeTracker::IsPrimvarDirty(*dirtyBits, id, HdTokens->points) ||
        HdChangeTracker::IsTransformDirty(*dirtyBits, id);

    std::fprintf(stderr,
                 "Lumbre: MeshSync id='%s' dirty=0x%x visible=%d owner=%d geometryDirty=%d\\n",
                 id.GetText(), unsigned(*dirtyBits), int(IsVisible()), int(owner != nullptr),
                 int(topoOrXformDirty));

    if (owner && topoOrXformDirty && IsVisible()) {
        VtValue pointsValue = sceneDelegate->Get(id, HdTokens->points);
        HdMeshTopology topology = sceneDelegate->GetMeshTopology(id);
        GfMatrix4d xform = sceneDelegate->GetTransform(id);

        std::fprintf(stderr,
                     "Lumbre: MeshSync input id='%s' pointsType='%s' faces=%zu indices=%zu\\n",
                     id.GetText(), pointsValue.GetTypeName().c_str(),
                     topology.GetFaceVertexCounts().size(),
                     topology.GetFaceVertexIndices().size());

        if (pointsValue.IsHolding<VtVec3fArray>()) {
            VtVec3fArray points = pointsValue.UncheckedGet<VtVec3fArray>();

            HdMeshUtil meshUtil(&topology, id);
            VtVec3iArray triIndices;
            VtIntArray primParams;
            meshUtil.ComputeTriangleIndices(&triIndices, &primParams);

            std::vector<LumbreBridgeTriangle> tris;
            tris.reserve(triIndices.size());
            for (GfVec3i const &t : triIndices) {
                if (t[0] < 0 || t[1] < 0 || t[2] < 0 ||
                    static_cast<size_t>(t[0]) >= points.size() ||
                    static_cast<size_t>(t[1]) >= points.size() ||
                    static_cast<size_t>(t[2]) >= points.size()) {
                    continue;
                }
                // World-space positions.
                GfVec3d p0 = xform.Transform(GfVec3d(points[t[0]]));
                GfVec3d p1 = xform.Transform(GfVec3d(points[t[1]]));
                GfVec3d p2 = xform.Transform(GfVec3d(points[t[2]]));
                // Flat face normal (beauty-first; smooth normals land later).
                GfVec3d n = GfCross(p1 - p0, p2 - p0);
                double len = n.GetLength();
                if (len > 1e-12) n /= len;

                LumbreBridgeTriangle tri;
                const GfVec3d verts[3] = {p0, p1, p2};
                for (int v = 0; v < 3; ++v) {
                    tri.positions[v * 3 + 0] = static_cast<float>(verts[v][0]);
                    tri.positions[v * 3 + 1] = static_cast<float>(verts[v][1]);
                    tri.positions[v * 3 + 2] = static_cast<float>(verts[v][2]);
                    tri.normals[v * 3 + 0] = static_cast<float>(n[0]);
                    tri.normals[v * 3 + 1] = static_cast<float>(n[1]);
                    tri.normals[v * 3 + 2] = static_cast<float>(n[2]);
                }
                tris.push_back(tri);
            }
            owner->SetGeometry(id, std::move(tris));
        } else {
            std::fprintf(stderr,
                         "Lumbre: MeshSync skipped id='%s': points are not VtVec3fArray\\n",
                         id.GetText());
        }
    } else if (owner && !IsVisible()) {
        std::fprintf(stderr, "Lumbre: MeshSync removing invisible id='%s'\\n", id.GetText());
        owner->RemoveGeometry(id);
    } else if (!topoOrXformDirty) {
        std::fprintf(stderr, "Lumbre: MeshSync skipped id='%s': no topology/points/transform dirtiness\\n",
                     id.GetText());
    }

    *dirtyBits = HdChangeTracker::Clean;
}

void HdLumbreLight::Sync(
    HdSceneDelegate *sceneDelegate,
    HdRenderParam *renderParam,
    HdDirtyBits *dirtyBits) {
    auto *param = static_cast<LumbreRenderParam *>(renderParam);
    HdLumbreRenderDelegate *owner = param ? param->Owner() : nullptr;
    if (!owner || !sceneDelegate || !dirtyBits) return;

    const SdfPath &id = GetId();
    const bool changed = (*dirtyBits & HdLight::AllDirty) != 0;
    if (!changed) return;

    auto scalar = [&](TfToken const &token, float fallback) {
        VtValue value = sceneDelegate->GetLightParamValue(id, token);
        if (value.IsHolding<float>()) return value.UncheckedGet<float>();
        if (value.IsHolding<double>()) return float(value.UncheckedGet<double>());
        return fallback;
    };
    auto color = [&](TfToken const &token, GfVec3f fallback) {
        VtValue value = sceneDelegate->GetLightParamValue(id, token);
        if (value.IsHolding<GfVec3f>()) return value.UncheckedGet<GfVec3f>();
        if (value.IsHolding<GfVec3d>()) {
            GfVec3d c = value.UncheckedGet<GfVec3d>();
            return GfVec3f(float(c[0]), float(c[1]), float(c[2]));
        }
        return fallback;
    };
    const GfMatrix4d xform = sceneDelegate->GetTransform(id);
    const GfVec3d center = xform.ExtractTranslation();
    GfVec3d xAxis(xform[0][0], xform[0][1], xform[0][2]);
    GfVec3d yAxis(xform[1][0], xform[1][1], xform[1][2]);
    GfVec3d zAxis(xform[2][0], xform[2][1], xform[2][2]);
    const double xScale = std::max(xAxis.GetLength(), 1e-6);
    const double yScale = std::max(yAxis.GetLength(), 1e-6);
    const double zScale = std::max(zAxis.GetLength(), 1e-6);
    xAxis /= xScale; yAxis /= yScale; zAxis /= zScale;

    const GfVec3f c = color(HdLightTokens->color, GfVec3f(1.0f));
    const float exposure = scalar(HdLightTokens->exposure, 0.0f);
    const float power = scalar(HdLightTokens->intensity, 1.0f) * std::pow(2.0f, exposure);
    LumbreBridgeLight light{};
    light.position[0] = float(center[0]); light.position[1] = float(center[1]); light.position[2] = float(center[2]);
    light.direction[0] = float(-zAxis[0]); light.direction[1] = float(-zAxis[1]); light.direction[2] = float(-zAxis[2]);
    light.intensity[0] = c[0] * power; light.intensity[1] = c[1] * power; light.intensity[2] = c[2] * power;

    if (_typeId == HdPrimTypeTokens->rectLight) {
        light.kind = 0;
        const double width = scalar(HdLightTokens->width, 1.0f) * xScale;
        const double height = scalar(HdLightTokens->height, 1.0f) * yScale;
        const GfVec3d u = xAxis * width, v = yAxis * height;
        const GfVec3d corner = center - 0.5 * (u + v);
        for (int i = 0; i < 3; ++i) { light.position[i] = float(corner[i]); light.u[i] = float(u[i]); light.v[i] = float(v[i]); }
    } else if (_typeId == HdPrimTypeTokens->diskLight) {
        light.kind = 3;
        light.radius = scalar(HdLightTokens->radius, 0.5f) * float((xScale + yScale) * 0.5);
    } else if (_typeId == HdPrimTypeTokens->sphereLight) {
        light.kind = 1;
        light.radius = scalar(HdLightTokens->radius, 0.5f) * float((xScale + yScale + zScale) / 3.0);
    } else if (_typeId == HdPrimTypeTokens->cylinderLight) {
        light.kind = 4;
        light.radius = scalar(HdLightTokens->radius, 0.5f) * float((xScale + yScale) * 0.5);
        light.height = scalar(HdLightTokens->length, 1.0f) * float(zScale);
    } else if (_typeId == HdPrimTypeTokens->distantLight) {
        light.kind = 7;
        light.angular_radius = 0.5f * scalar(HdLightTokens->angle, 0.0f) * float(M_PI / 180.0);
    } else {
        light.kind = 5; // simpleLight is a point light in Hydra.
    }
    owner->SetLight(id, light);
    std::fprintf(stderr, "Lumbre: LightSync id='%s' type='%s' kind=%d power=%.3f\\n",
                 id.GetText(), _typeId.GetText(), light.kind, power);
    *dirtyBits = HdChangeTracker::Clean;
}

// ---------------------------------------------------------------------------
// Render pass implementation
// ---------------------------------------------------------------------------

void HdLumbreRenderPass::_Execute(
    HdRenderPassStateSharedPtr const &renderPassState,
    TfTokenVector const &) {
    HdRenderPassAovBindingVector const &bindings = renderPassState->GetAovBindings();

    // Locate the color render buffer Hydra wants us to fill.
    HdLumbreRenderBuffer *colorBuffer = nullptr;
    HdLumbreRenderBuffer *depthBuffer = nullptr;
    for (HdRenderPassAovBinding const &binding : bindings) {
        HdRenderBuffer *renderBuffer = binding.renderBuffer;
        if (!renderBuffer && !binding.renderBufferId.IsEmpty()) {
            renderBuffer = dynamic_cast<HdRenderBuffer *>(
                GetRenderIndex()->GetBprim(HdPrimTypeTokens->renderBuffer,
                                            binding.renderBufferId));
        }
        if (binding.aovName == HdAovTokens->color) {
            colorBuffer = dynamic_cast<HdLumbreRenderBuffer *>(renderBuffer);
        } else if (binding.aovName == HdAovTokens->depth) {
            depthBuffer = dynamic_cast<HdLumbreRenderBuffer *>(renderBuffer);
        }
    }
    if (!colorBuffer) {
        return;
    }

    const int width = static_cast<int>(colorBuffer->GetWidth());
    const int height = static_cast<int>(colorBuffer->GetHeight());
    if (width <= 0 || height <= 0) {
        return;
    }

    LumbreBridge const &bridge = _owner->Bridge();
    LumbreBridgeContext ctx = _owner->Context();

    if (!bridge.IsValid() || !ctx) {
        colorBuffer->FillDiagnosticGradient();
        return;
    }

    // Camera from the render-pass state. View space looks down -Z with +Y up.
    GfMatrix4d view = renderPassState->GetWorldToViewMatrix();
    GfMatrix4d proj = renderPassState->GetProjectionMatrix();
    GfMatrix4d camToWorld = view.GetInverse();

    GfVec3d eye = camToWorld.ExtractTranslation();
    GfVec3d fwd(-camToWorld[2][0], -camToWorld[2][1], -camToWorld[2][2]);
    GfVec3d up(camToWorld[1][0], camToWorld[1][1], camToWorld[1][2]);
    fwd.Normalize();
    up.Normalize();
    GfVec3d target = eye + fwd;

    // Vertical FOV from the projection matrix: proj[1][1] = 1/tan(fovY/2).
    double fovY = 40.0;
    if (std::fabs(proj[1][1]) > 1e-9) {
        fovY = 2.0 * std::atan(1.0 / proj[1][1]) * 180.0 / M_PI;
        fovY = std::fabs(fovY);
    }

    const float originF[3] = {float(eye[0]), float(eye[1]), float(eye[2])};
    const float targetF[3] = {float(target[0]), float(target[1]), float(target[2])};
    const float upF[3] = {float(up[0]), float(up[1]), float(up[2])};

    const size_t triCount = _owner->UploadGeometryIfDirty();
    const size_t lightCount = _owner->UploadLightsIfDirty();
    if (bridge.set_resolution) bridge.set_resolution(ctx, width, height);
    if (bridge.set_camera) bridge.set_camera(ctx, originF, targetF, upF, float(fovY));
    bool rendered = bridge.render && bridge.render(ctx);

    // Read the framebuffer into the mapped Float32Vec4 buffer.
    bool readOk = false;
    if (float *dst = colorBuffer->ColorData()) {
        if (bridge.read_rgba_f32 &&
            bridge.read_rgba_f32(ctx, dst, width, height)) {
            readOk = true;
        } else {
            colorBuffer->FillDiagnosticGradient();
        }
    }
    // GetResource() is queried by Solaris after this pass. Mark the Hgi texture
    // stale so the next request receives the image just written above.
    colorBuffer->MarkGpuDirty();
    colorBuffer->SetConverged(true);
    if (depthBuffer) {
        if (float *depth = depthBuffer->DepthData()) {
            std::fill(depth, depth + static_cast<size_t>(depthBuffer->GetWidth()) * depthBuffer->GetHeight(), 1.0f);
        }
        depthBuffer->SetConverged(true);
    }
    ++_execCount;

    // Escape hatch: set LUMBRE_DUMP_PNG=/path.png to write the first rendered
    // frame to disk, confirming the renderer independently of the viewport.
    if (readOk && bridge.write_png) {
        static std::atomic_bool dumped{false};
        if (const char *dumpPath = std::getenv("LUMBRE_DUMP_PNG")) {
            if (!dumped.exchange(true)) {
                bool ok = bridge.write_png(ctx, dumpPath) != 0;
                std::fprintf(stderr, "Lumbre: dumped frame to %s -> %s\n",
                             dumpPath, ok ? "ok" : "FAILED");
            }
        }
    }

    // Diagnostics: the first few executes print the AOV bindings, the derived
    // camera, and a sampled pixel from the buffer AFTER the write. This tells us
    // whether the image data is real (so any blank viewport is a presentation
    // issue) or flat/black (a camera or lighting issue).
    static std::atomic_int frame{0};
    int f = frame.fetch_add(1);
    if (f < 5) {
        std::fprintf(stderr,
            "Lumbre[%d]: buf=%dx%d tris=%zu lights=%zu rendered=%d readOk=%d aovBindings=%zu\n",
            f, width, height, triCount, lightCount, int(rendered), int(readOk), bindings.size());
        for (HdRenderPassAovBinding const &b : bindings) {
            std::fprintf(stderr, "Lumbre[%d]:   aov='%s' buf=%p fmt=%d\n",
                f, b.aovName.GetText(), static_cast<void *>(b.renderBuffer),
                b.renderBuffer ? int(b.renderBuffer->GetFormat()) : -1);
        }
        std::fprintf(stderr,
            "Lumbre[%d]: eye=(%.3f %.3f %.3f) target=(%.3f %.3f %.3f) up=(%.3f %.3f %.3f) fovY=%.2f\n",
            f, eye[0], eye[1], eye[2], target[0], target[1], target[2],
            up[0], up[1], up[2], fovY);
        if (float *dst = colorBuffer->ColorData()) {
            auto sample = [&](int x, int y) {
                size_t i = (size_t(y) * width + x) * 4;
                std::fprintf(stderr, "Lumbre[%d]:   px(%d,%d)=(%.3f %.3f %.3f %.3f)\n",
                    f, x, y, dst[i], dst[i + 1], dst[i + 2], dst[i + 3]);
            };
            sample(width / 2, height / 2);
            sample(width / 4, height / 4);
            sample(0, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

class HdLumbreRendererPlugin final : public HdRendererPlugin {
public:
    HdRenderDelegate *CreateRenderDelegate() override {
        return new HdLumbreRenderDelegate({});
    }
    HdRenderDelegate *CreateRenderDelegate(
        HdRenderSettingsMap const &settings) override {
        return new HdLumbreRenderDelegate(settings);
    }
    void DeleteRenderDelegate(HdRenderDelegate *delegate) override {
        delete delegate;
    }
    bool IsSupported(bool) const override { return true; }
};

TF_REGISTRY_FUNCTION(TfType) {
    HdRendererPluginRegistry::Define<HdLumbreRendererPlugin>();
}

PXR_NAMESPACE_CLOSE_SCOPE
