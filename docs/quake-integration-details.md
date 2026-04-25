# Quake Integration Details

This document outlines an MVP path for running classic Quake through the
compute-first API described in `docs/proposed-api-surface.md`.

The main idea is to avoid hardware rasterization entirely. Quake already has a
software-renderer architecture, so the first backend should preserve Quake's CPU
visibility and span-oriented pipeline while moving selected stages to AMDGPU
compute kernels.

## MVP Definition

The MVP is complete when:

- Quake runs unmodified game logic and asset loading on the CPU.
- Frames are presented through the new API, not through SDL/OpenGL/Vulkan.
- The visible game image is produced by GPU compute kernels.
- World surfaces, sprites, alias models, particles, palette conversion, and
  depth/visibility are correct enough to play the first level.
- The backend uses `libkfd` compute, SDMA, signals, and DMA-BUF presentation.

## Strategy

Start from Quake's software renderer, not from a triangle API. The software
renderer already decomposes the scene into work that is friendly to compute:

- BSP traversal and visibility.
- Surface lists.
- Spans.
- Mipmapped indexed textures.
- Lightmaps.
- Sprites and particles.
- Alias model triangles.
- 8-bit framebuffer plus palette.

Keep CPU-side scene traversal first. Move pixel-producing work to GPU in stages.

## High-Level Frame Flow

```text
CPU game + visibility
  -> build frame jobs
  -> upload jobs/textures/lightmaps/palette as needed
  -> dispatch GPU render kernels
  -> dispatch palette/format conversion kernel
  -> present DMA-BUF surface
```

The first version should use double or triple buffered frame resources:

- Root argument buffer.
- Job buffers.
- 8-bit indexed framebuffer.
- Z/depth buffer.
- `XRGB8888` present surface.
- Fence for frame completion.

## Stage 0: CPU Renderer With GPU Present

Goal: prove windowing, DMA-BUF presentation, frame pacing, and basic API shape.

Implementation:

- Let Quake render normally into its CPU 8-bit framebuffer.
- Upload the 8-bit framebuffer and palette to GPU memory.
- Dispatch a palette conversion kernel into an exportable `XRGB8888` surface.
- Present that surface through the `computetoy`-style presenter.

This stage does not accelerate rendering, but it validates the end-to-end path.

## Stage 1: Palette Conversion Kernel

Inputs:

```cpp
struct alignas(16) PaletteConvertArgs {
  const uint8_t *src_indexed;
  const uint32_t *palette_xrgb;
  uint32_t *dst_xrgb;
  uint32_t width;
  uint32_t height;
  uint32_t src_pitch;
  uint32_t dst_pitch_pixels;
};
```

Kernel behavior:

- One thread per output pixel.
- Load an 8-bit palette index.
- Load `XRGB8888` palette entry.
- Store to the present surface.

This is the first permanent kernel and should become a test fixture for the API.

## Stage 2: GPU Span Renderer

Goal: move Quake world surface drawing to GPU while keeping CPU BSP traversal and
surface/span generation.

CPU responsibilities:

- Run normal Quake visibility and clipping.
- Generate compact span jobs.
- Upload span jobs to GPU.

GPU responsibilities:

- Draw spans into an 8-bit framebuffer.
- Read texture mips and lightmaps from raw buffers.
- Apply Quake-style fixed-point stepping.
- Optionally update depth.

Suggested span job:

```cpp
struct alignas(16) QuakeSpanJob {
  uint32_t y;
  uint32_t x0;
  uint32_t count;
  uint32_t texture_id;

  int32_t s;
  int32_t t;
  int32_t ds_dx;
  int32_t dt_dx;

  uint32_t lightmap_id;
  int32_t light_s;
  int32_t light_t;
  int32_t dlight_s_dx;
  int32_t dlight_t_dx;
};
```

The exact fields should follow the renderer path we port. The important part is
to keep jobs linear and append-only so the CPU can fill them cheaply.

## Stage 3: World Surfaces

The world renderer should initially use raw buffer sampling:

- Texture data: Quake mip levels in GPU buffers.
- Lightmaps: uploaded or generated as byte/intensity buffers.
- Surface metadata: one compact GPU struct per surface or texture.
- Output: 8-bit framebuffer, then palette conversion.

Avoid hardware image descriptors for the MVP. Quake's texture model is simple
enough that raw pointer loads are sufficient and easier to debug.

## Stage 4: Sprites And Particles

Sprites and particles can be implemented as separate kernels after world spans:

- Sprite jobs: screen rectangle, texture pointer/id, fixed-point UV steps,
  transparent index.
- Particle jobs: screen position, size, color index, optional depth.

The first particle version can be one thread per particle with small square
writes. Later versions can bin particles by tile to reduce write contention.

## Stage 5: Alias Models

Alias models need triangle rasterization. Keep this simple first:

- CPU transforms and clips vertices.
- CPU emits triangle jobs.
- GPU rasterizes triangles into the 8-bit framebuffer and depth buffer.

Suggested triangle job:

```cpp
struct alignas(16) QuakeTriangleJob {
  float x0, y0, z0;
  float x1, y1, z1;
  float x2, y2, z2;
  float u0, v0;
  float u1, v1;
  float u2, v2;
  uint32_t texture_id;
  uint32_t flags;
};
```

This does not need to match modern GPU raster rules. It only needs to match
Quake closely enough for alias models.

## Fixed-Function Features To Emulate

| Feature | MVP implementation |
|---------|--------------------|
| Viewport/scissor | Bounds checks in kernels |
| Texture sampling | Raw buffer lookup into Quake mips |
| Filtering | Nearest first, optional bilinear later |
| Depth | Software depth buffer in GPU memory |
| Blending | Explicit compute writes |
| Transparency | Quake transparent palette index rules |
| Lightmaps | Raw buffer lookup and table/math combine |
| Render target | Linear 8-bit framebuffer plus XRGB present surface |
| Presentation | DMA-BUF import/present |

## Data Upload Policy

Static data:

- BSP surfaces.
- Texture mips.
- Alias model frames.
- Sprite textures.
- Palette tables.

Dynamic per-frame data:

- Root arguments.
- Span jobs.
- Sprite jobs.
- Particle jobs.
- Alias triangle jobs.
- Updated lightmaps, if needed.

Use persistent GPU buffers for static data. Use a ring/bump upload buffer for
per-frame jobs. The API should expose enough raw pointers that Quake data
structures can contain GPU addresses after upload.

## Synchronization

The MVP should use a conservative frame boundary:

1. CPU fills upload/job buffers for frame N.
2. CPU dispatches render kernels.
3. CPU dispatches palette conversion.
4. GPU signals frame fence.
5. Presenter displays the surface after the fence.

Fine-grained overlap can come later. Correctness and debuggability matter more
than latency in the first version.

## Repository Integration

Suggested layout:

```text
tools/quake/
  CMakeLists.txt
  main.cpp
  backend_gpu.cpp
  backend_gpu.h
  shaders/
    palette.c
    spans.c
    sprites.c
    triangles.c
```

Device support headers can live in:

```text
include/gpu/
  kernel.h
  math.h
  quake.h
```

The first tool should not vendor Quake source into this repository unless that
is a deliberate decision. Prefer a backend adapter that can be pointed at an
external Quake tree or a small standalone Quake-frame trace renderer.

## Recommended First Milestone

Build a standalone `quake_frame` sample before touching the full game:

- Load or synthesize a Quake-like 8-bit framebuffer and palette.
- Convert it on the GPU.
- Present it.
- Then add a span-job test scene with one textured wall.

Once that works, integrating into Quake becomes mostly a data plumbing problem
instead of a GPU bring-up problem.

## Open Questions

- Which Quake source port should be the integration base?
- Should the MVP preserve exact software-renderer pixel output or accept visual
  differences?
- Should the first playable version keep CPU-generated spans, or should surface
  setup move to GPU earlier?
- Which presentation backends are required first: X11 DRI3, Wayland dmabuf, or
  direct KMS?
- Should the API remain C-compatible, or is a C++ host API acceptable for the
  first prototype?
