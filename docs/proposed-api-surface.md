# Proposed API Surface

This document sketches a compute-first graphics API that can be implemented on
top of `libkfd`. The API follows the "No Graphics API" direction: shaders are
ordinary freestanding C/C++ code compiled by Clang for AMDGPU, root arguments
are user-defined structs passed by GPU pointer, and fixed-function graphics
features are implemented by optional device-side libraries.

The first backend target is AMDGPU through KFD compute queues. Hardware texture
sampling may be added later through AMD image instructions and descriptors, but
the MVP should work with raw buffers only.

## Goals

- Make compute dispatch, GPU memory, synchronization, and presentation small and
  explicit.
- Use normal C/C++ structs as the binding model instead of descriptor sets,
  root signatures, or reflected resource layouts.
- Keep device libraries optional and source-level: users include the headers
  they need and link them into the GPU ELF.
- Support a software graphics pipeline in compute: rasterization, depth, blend,
  texture sampling, lightmaps, sprites, and post-processing.
- Keep the API backend-neutral enough that a future DRM graphics backend could
  be added without changing application code.

## Non-Goals

- No initial hardware rasterization backend.
- No Vulkan/D3D-style PSO compatibility layer.
- No dynamic GPU linker in the MVP. Device code is statically linked into a
  single AMDGPU ELF.
- No general libc or C++ runtime on the GPU unless provided by an explicit
  device library.

## Backend Mapping

| API concept | Initial `libkfd` implementation |
|-------------|----------------------------------|
| `GpuContext` | `kfd::Context` + selected `kfd::Device` |
| `GpuQueue` | `kfd::ComputeQueue` and `kfd::SDMAQueue` |
| `GpuBuffer` | `kfd::Buffer` |
| `GpuKernel` | `kfd::Executable` + `kfd::Kernel` |
| `GpuFence` | `kfd::Signal` |
| `GpuCommandBuffer` | Immediate queue submission first, optional userland batch later |
| `GpuSurface` | DMA-BUF exported buffer plus platform presenter |

## Core Types

```cpp
struct GpuContext;
struct GpuDevice;
struct GpuQueue;
struct GpuBuffer;
struct GpuKernel;
struct GpuModule;
struct GpuFence;
struct GpuSurface;
struct GpuPresenter;

struct GpuDim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};
```

## Memory API

```cpp
enum GpuMemoryType {
  GPU_MEMORY_UPLOAD,
  GPU_MEMORY_DEVICE,
  GPU_MEMORY_READBACK,
};

enum GpuMemoryFlags : uint32_t {
  GPU_MEMORY_WRITABLE = 1u << 0,
  GPU_MEMORY_EXECUTABLE = 1u << 1,
  GPU_MEMORY_COHERENT = 1u << 2,
  GPU_MEMORY_UNCACHED = 1u << 3,
  GPU_MEMORY_EXPORTABLE = 1u << 4,
};

GpuBuffer *gpuMalloc(GpuDevice *dev, size_t size, size_t alignment,
                     GpuMemoryType type, uint32_t flags);
void gpuFree(GpuBuffer *buffer);

void *gpuBufferCpu(GpuBuffer *buffer);
void *gpuBufferGpu(GpuBuffer *buffer);

void gpuMemcpy(GpuQueue *queue, void *dst_gpu, const void *src_gpu,
               size_t bytes);
void gpuMemset32(GpuQueue *queue, void *dst_gpu, uint32_t value,
                 size_t bytes);
```

MVP memory types map as follows:

- `GPU_MEMORY_UPLOAD`: host-visible GTT.
- `GPU_MEMORY_DEVICE`: VRAM.
- `GPU_MEMORY_READBACK`: host-visible coherent memory optimized for CPU reads,
  initially also GTT.

## Kernel And Module API

```cpp
GpuModule *gpuLoadModule(GpuDevice *dev, const void *elf, size_t elf_size);
void gpuUnloadModule(GpuModule *module);

GpuKernel *gpuFindKernel(GpuModule *module, const char *name);

struct GpuDispatchDesc {
  GpuKernel *kernel;
  void *root_gpu;
  GpuDim3 grid;
  GpuDim3 block;
  uint32_t dynamic_lds = 0;
  uint32_t private_segment_size = 0;
};

void gpuDispatch(GpuQueue *queue, const GpuDispatchDesc *desc);
void gpuSignal(GpuQueue *queue, GpuFence *fence);
void gpuWait(GpuQueue *queue, GpuFence *fence, uint64_t value);
```

The root pointer is a GPU address to a user-defined C/C++ struct. The runtime
does not inspect the struct layout.

Example device-side shape:

```cpp
struct alignas(16) ClearArgs {
  uint32_t *dst;
  uint32_t value;
  uint32_t width;
  uint32_t height;
};

extern "C" __attribute__((amdgpu_kernel))
void clear_kernel(const ClearArgs *args);
```

The exact kernel attribute/macro should be hidden behind a small public device
header, for example `gpu/kernel.h`.

## Synchronization API

```cpp
GpuFence *gpuCreateFence(GpuContext *ctx, uint64_t initial_value);
void gpuDestroyFence(GpuFence *fence);

void gpuResetFence(GpuFence *fence, uint64_t value);
void gpuWaitFenceCpu(GpuFence *fence, uint64_t value, uint64_t timeout_ns);

void gpuBarrier(GpuQueue *queue, uint32_t barrier_flags);
```

Initial barriers map to conservative `ACQUIRE_MEM` / cache flush packets on the
compute queue and SDMA GCR flushes where needed. Later versions can split this
into more precise execution and cache scopes.

## Presentation API

```cpp
struct GpuSurfaceDesc {
  uint32_t width;
  uint32_t height;
  GpuFormat format;
};

GpuSurface *gpuCreateSurface(GpuDevice *dev, const GpuSurfaceDesc *desc);
void gpuDestroySurface(GpuSurface *surface);

GpuPresenter *gpuCreatePresenterX11(GpuSurface *surface, void *display,
                                    uint32_t window);
void gpuPresent(GpuPresenter *presenter, GpuSurface *surface);
```

The initial implementation should reuse the `computetoy` model: render into an
exportable linear buffer, export it as DMA-BUF, import it into the window system
through DRI3/Present or Wayland dmabuf, then present.

## Device Libraries

Device libraries are normal C/C++ headers and source files compiled into the GPU
ELF:

- `gpu/kernel.h`: kernel attributes, thread/block ID helpers.
- `gpu/math.h`: scalar/vector math helpers.
- `gpu/memory.h`: atomics, coherent loads/stores, barriers.
- `gpu/image.h`: raw image load/store helpers.
- `gpu/raster.h`: software rasterization helpers.
- `gpu/quake.h`: Quake-specific span, palette, lightmap, and particle helpers.

The build system should support:

```sh
clang --target=amdgcn--amdhsa -mcpu=gfx1100 -nogpulib -nostdlibinc \
  -I path/to/device/include -O2 shader.cpp -o shader.elf
```

## MVP Cut

The MVP API should include only:

- Context/device/queue creation.
- `gpuMalloc`, `gpuFree`, `gpuMemcpy`.
- ELF module load and kernel lookup.
- Root pointer dispatch.
- Fence wait/signal.
- Linear `XRGB8888` surface allocation and presentation.

Everything else can be layered as device libraries or added after Quake can draw
frames through the API.

## Reference: Original Prototype API

This is the original prototype API surface from Sebastian Aaltonen's "No
Graphics API" article, included here as a reference target. It is pseudocode,
not a drop-in C header: it uses default arguments, placeholder ellipses, and
types such as `Span`, `ByteSpan`, and `uvec3` that are not defined here.

Our MVP C API is intentionally smaller and Quake-friendly. The purpose of this
reference is to keep the long-term feature target visible while we build the
compute-first backend incrementally.

```cpp
// Opaque handles
struct GpuPipeline;
struct GpuTexture;
struct GpuDepthStencilState;
struct GpuBlendState;
struct GpuQueue;
struct GpuCommandBuffer;
struct GpuSemaphore;

// Enums
enum MEMORY { MEMORY_DEFAULT, MEMORY_GPU, MEMORY_READBACK };
enum CULL { CULL_CCW, CULL_CW, CULL_ALL, CULL_NONE };
enum DEPTH_FLAGS { DEPTH_READ = 0x1, DEPTH_WRITE = 0x2 };
enum OP { OP_NEVER, OP_LESS, OP_EQUAL, OP_LESS_EQUAL, OP_GREATER, OP_NOT_EQUAL, OP_GREATER_EQUAL, OP_ALWAYS };
enum BLEND { BLEND_ADD, BLEND_SUBTRACT, BLEND_REV_SUBTRACT, BLEND_MIN, BLEND_MAX };
enum FACTOR { FACTOR_ZERO, FACTOR_ONE, FACTOR_SRC_COLOR, FACTOR_DST_COLOR, FACTOR_SRC_ALPHA, ... };
enum TOPOLOGY { TOPOLOGY_TRIANGLE_LIST, TOPOLOGY_TRIANGLE_STRIP, TOPOLOGY_TRIANGLE_FAN };
enum TEXTURE { TEXTURE_1D, TEXTURE_2D, TEXTURE_3D, TEXTURE_CUBE, TEXTURE_2D_ARRAY, TEXTURE_CUBE_ARRAY };
enum FORMAT { FORMAT_NONE, FORMAT_RGBA8_UNORM, FORMAT_D32_FLOAT, FORMAT_RG11B10_FLOAT, FORMAT_RGB10_A2_UNORM, ... };
enum USAGE_FLAGS { USAGE_SAMPLED, USAGE_STORAGE, USAGE_COLOR_ATTACHMENT, USAGE_DEPTH_STENCIL_ATTACHMENT, ... };
enum STAGE { STAGE_TRANSFER, STAGE_COMPUTE, STAGE_RASTER_COLOR_OUT, STAGE_PIXEL_SHADER, STAGE_VERTEX_SHADER, ... };
enum HAZARD_FLAGS { HAZARD_DRAW_ARGUMENTS = 0x1, HAZARD_DESCRIPTORS = 0x2, HAZARD_DEPTH_STENCIL = 0x4 };
enum SIGNAL { SIGNAL_ATOMIC_SET, SIGNAL_ATOMIC_MAX, SIGNAL_ATOMIC_OR, ... };

// Structs
struct Stencil
{
    OP test = OP_ALWAYS;
    OP failOp = OP_KEEP;
    OP passOp = OP_KEEP;
    OP depthFailOp = OP_KEEP;
    uint8 reference = 0;
};

struct GpuDepthStencilDesc
{
    DEPTH_FLAGS depthMode = 0;
    OP depthTest = OP_ALWAYS;
    float depthBias = 0.0f;
    float depthBiasSlopeFactor = 0.0f;
    float depthBiasClamp = 0.0f;
    uint8 stencilReadMask = 0xff;
    uint8 stencilWriteMask = 0xff;
    Stencil stencilFront;
    Stencil stencilBack;
};

struct GpuBlendDesc
{
    BLEND colorOp = BLEND_ADD;
    FACTOR srcColorFactor = FACTOR_ONE;
    FACTOR dstColorFactor = FACTOR_ZERO;
    BLEND alphaOp = BLEND_ADD;
    FACTOR srcAlphaFactor = FACTOR_ONE;
    FACTOR dstAlphaFactor = FACTOR_ZERO;
    uint8 colorWriteMask = 0xf;
};

struct ColorTarget {
    FORMAT format = FORMAT_NONE;
    uint8 writeMask = 0xf;
};

struct GpuRasterDesc
{
    TOPOLOGY topology = TOPOLOGY_TRIANGLE_LIST;
    CULL cull = CULL_NONE;
    bool alphaToCoverage = false;
    bool supportDualSourceBlending = false;
    uint8 sampleCount = 1;
    FORMAT depthFormat = FORMAT_NONE;
    FORMAT stencilFormat = FORMAT_NONE;
    Span<ColorTarget> colorTargets = {};
    GpuBlendDesc* blendstate = nullptr; // optional embedded blend state
};

struct GpuTextureDesc
{
    TEXTURE type = TEXTURE_2D;
    uint32x3 dimensions;
    uint32 mipCount = 1;
    uint32 layerCount = 1;
    uint32 sampleCount = 1;
    FORMAT format = FORMAT_NONE;
    USAGE_FLAGS usage = 0;
};

struct GpuViewDesc
{
    FORMAT format = FORMAT_NONE;
    uint8 baseMip = 0;
    uint8 mipCount = ALL_MIPS;
    uint16 baseLayer = 0;
    uint16 layerCount = ALL_LAYERS;
};

struct GpuTextureSizeAlign { size_t size; size_t align; };
struct GpuTextureDescriptor { uint64[4] data; };

// Memory
void* gpuMalloc(size_t bytes, MEMORY memory = MEMORY_DEFAULT);
void* gpuMalloc(size_t bytes, size_t align, MEMORY memory = MEMORY_DEFAULT);
void gpuFree(void *ptr);
void* gpuHostToDevicePointer(void *ptr);

// Textures
GpuTextureSizeAlign gpuTextureSizeAlign(GpuTextureDesc desc);
GpuTexture gpuCreateTexture(GpuTextureDesc desc, void* ptrGpu);
GpuTextureDescriptor gpuTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc);
GpuTextureDescriptor gpuRWTextureViewDescriptor(GpuTexture texture, GpuViewDesc desc);

// Pipelines
GpuPipeline gpuCreateComputePipeline(ByteSpan computeIR);
GpuPipeline gpuCreateGraphicsPipeline(ByteSpan vertexIR, ByteSpan pixelIR, GpuRasterDesc desc);
GpuPipeline gpuCreateGraphicsMeshletPipeline(ByteSpan meshletIR, ByteSpan pixelIR, GpuRasterDesc desc);
void gpuFreePipeline(GpuPipeline pipeline);

// State objects
GpuDepthStencilState gpuCreateDepthStencilState(GpuDepthStencilDesc desc);
GpuBlendState gpuCreateBlendState(GpuBlendDesc desc);
void gpuFreeDepthStencilState(GpuDepthStencilState state);
void gpuFreeBlendState(GpuBlendState state);

// Queue
GpuQueue gpuCreateQueue(/* DEVICE & QUEUE CREATION DETAILS OMITTED */);
GpuCommandBuffer gpuStartCommandRecording(GpuQueue queue);
void gpuSubmit(GpuQueue queue, Span<GpuCommandBuffer> commandBuffers);

// Semaphores
GpuSemaphore gpuCreateSemaphore(uint64 initValue);
void gpuWaitSemaphore(GpuSemaphore sema, uint64 value);
void gpuDestroySemaphore(GpuSemaphore sema);

// Commands
void gpuMemCpy(GpuCommandBuffer cb, void* destGpu, void* srcGpu);
void gpuCopyToTexture(GpuCommandBuffer cb, void* destGpu, void* srcGpu, GpuTexture texture);
void gpuCopyFromTexture(GpuCommandBuffer cb, void* destGpu, void* srcGpu, GpuTexture texture);

void gpuSetActiveTextureHeapPtr(GpuCommandBuffer cb, void *ptrGpu);

void gpuBarrier(GpuCommandBuffer cb, STAGE before, STAGE after, HAZARD_FLAGS hazards = 0);
void gpuSignalAfter(GpuCommandBuffer cb, STAGE before, void *ptrGpu, uint64 value, SIGNAL signal);
void gpuWaitBefore(GpuCommandBuffer cb, STAGE after, void *ptrGpu, uint64 value, OP op, HAZARD_FLAGS hazards = 0, uint64 mask = ~0);

void gpuSetPipeline(GpuCommandBuffer cb, GpuPipeline pipeline);
void gpuSetDepthStencilState(GpuCommandBuffer cb, GpuDepthStencilState state);
void gpuSetBlendState(GpuCommandBuffer cb, GpuBlendState state);

void gpuDispatch(GpuCommandBuffer cb, void* dataGpu, uvec3 gridDimensions);
void gpuDispatchIndirect(GpuCommandBuffer cb, void* dataGpu, void* gridDimensionsGpu);

void gpuBeginRenderPass(GpuCommandBuffer cb, GpuRenderPassDesc desc);
void gpuEndRenderPass(GpuCommandBuffer cb);

void gpuDrawIndexedInstanced(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, uint32 indexCount, uint32 instanceCount);
void gpuDrawIndexedInstancedIndirect(GpuCommandBuffer cb, void* vertexDataGpu, void* pixelDataGpu, void* indicesGpu, void* argsGpu);
void gpuDrawIndexedInstancedIndirectMulti(GpuCommandBuffer cb, void* dataVxGpu, uint32 vxStride, void* dataPxGpu, uint32 pxStride, void* argsGpu, void* drawCountGpu);

void gpuDrawMeshlets(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, uvec3 dim);
void gpuDrawMeshletsIndirect(GpuCommandBuffer cb, void* meshletDataGpu, void* pixelDataGpu, void *dimGpu);
```
