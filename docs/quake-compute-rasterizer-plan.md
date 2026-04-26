# Quake Compute Rasterizer Plan

This document describes a from-scratch compute software renderer for classic
Quake on top of `libkfd`.

The goal is not to port WinQuake's software renderer function by function. The
goal is to preserve the classic 8-bit Quake look while using a GPU-friendly work
model from the beginning.

## Lessons From The Span Experiment

The incremental span path proved the end-to-end plumbing:

- Quake can present through `libkfd`.
- GPU palette conversion and DMA-BUF presentation work.
- Batched GPU span dispatches are stable enough to benchmark.

It also showed that patching the existing software renderer piece by piece is
not a good long-term architecture:

- Quake spans are generated late in the CPU pipeline, after much of the useful
  structure has been lost.
- Per-surface texture packing and per-frame span metadata are expensive.
- Moving small CPU renderer fragments to GPU creates synchronization and upload
  overhead instead of exposing large coherent work.
- CPU-style 8-pixel subdivision only helps if the whole work shape is designed
  around it. Translating it directly into a GPU lane loop destroys occupancy.

The next renderer should submit scene-level draw work, not CPU span internals.

## Target Architecture

Keep Quake game code and visibility on the CPU. Move raster ownership to GPU.

```text
CPU game + visibility
  -> visible world/entity command buffers
  -> persistent GPU scene resources
  -> GPU transform/setup/bin/raster passes
  -> 8-bit indexed framebuffer
  -> palette/scale/present kernel
  -> DMA-BUF present
```

The CPU should submit compact draw commands and dirty resource updates. The GPU
should own transform, setup, rasterization, texture lookup, light lookup, depth,
and final palette conversion.

## Visual Contract

The renderer should preserve the classical Quake look by design:

- 8-bit paletted final game image.
- Quake palette and colormap lighting.
- Nearest texture sampling by default.
- Quake mip levels or a close screen-space mip policy.
- Low-resolution indexed framebuffer with nearest/integer-style scaling.
- Explicit classic rules for sky, water turbulence, sprites, particles, and
  transparent palette indices.

It does not need to preserve WinQuake's exact inner loops.

## Persistent GPU Resources

Static or mostly-static resources should live on GPU:

- Texture atlas containing Quake texture mip levels as 8-bit indices.
- Lightmap atlas/pages as 8-bit light values or compact intensity values.
- Palette table: 256 entries of XRGB.
- Colormap table for classic light-to-palette mapping.
- World surface metadata.
- World polygon vertices and indices.
- Alias model frames and skin textures.
- Sprite textures.
- 8-bit indexed framebuffer.
- Depth buffer.
- XRGB present surfaces.

Per-frame uploads should be limited to:

- Frame constants.
- Visible surface IDs or compact draw commands.
- Entity instance data.
- Dynamic light/lightmap updates.
- Particles.

Avoid per-frame texture repacking and avoid CPU framebuffer readback.

## Frame Pipeline

### 1. CPU Visibility And Commands

The CPU keeps:

- BSP/PVS traversal.
- Frustum checks.
- Surface visibility list construction.
- Entity list construction.
- Animation frame selection.
- Dynamic light decisions.

The CPU emits command buffers:

```c
struct WorldSurfaceCmd {
	uint surface_id;
	uint flags;
};

struct AliasModelCmd {
	uint model_id;
	uint frame_id;
	uint skin_id;
	float transform[12];
	uint lighting;
};

struct SpriteCmd {
	uint sprite_id;
	uint frame_id;
	float origin[3];
	uint flags;
};
```

These are renderer commands, not graphics API commands. They describe Quake
scene objects in a compact form.

### 2. GPU Setup

GPU setup turns visible commands into screen-space primitives:

- Transform vertices.
- Project to screen.
- Clip or conservatively reject offscreen primitives.
- Compute plane/edge equations.
- Compute texture/lightmap gradients.
- Select mip level.
- Write primitive records for rasterization.

For the MVP, some setup may remain on CPU if it shortens implementation. The
important rule is that the CPU should not generate final spans or pixels.

### 3. Tiled Binning

Raster work should be binned into screen tiles, for example `8x8` or `16x16`.

```text
primitive records
  -> tile primitive lists
  -> one or more workgroups per tile
```

This gives the GPU coherent work and limits depth/color write contention.

Start simple:

- CPU or GPU clears tile counters.
- Each primitive appends itself to all overlapping tiles.
- Overflow falls back or uses a secondary overflow list.

Optimize later:

- Prefix-summed tile lists.
- Hierarchical bins.
- Separate world/entity/particle bins.

### 4. Tiled Raster

One workgroup owns a tile. Threads cover pixels within the tile and iterate the
tile's primitive list.

For each covered pixel:

- Evaluate primitive coverage.
- Evaluate depth.
- Compare/update software depth.
- Compute texture coordinates.
- Sample 8-bit texture.
- Sample lightmap or lighting value.
- Apply Quake colormap/palette rule.
- Store 8-bit output index.

The first implementation can use global memory for depth/color. Later versions
can stage tile color/depth in LDS if profiling shows it helps.

### 5. Resolve And Present

The indexed framebuffer remains the renderer's canonical output.

Final pass:

- Scale/letterbox the indexed framebuffer.
- Apply palette lookup to XRGB.
- Composite UI/top overlays if still produced by CPU.
- Write to exportable DMA-BUF present surface.

## Rendering Order

Suggested pass order:

1. Clear indexed framebuffer and depth.
2. Opaque world surfaces.
3. Sky surfaces.
4. Water/turbulent surfaces.
5. Alias models.
6. Sprites.
7. Particles.
8. Transparent/cutout surfaces.
9. UI/console/HUD.
10. Palette resolve and present.

The MVP can keep UI and some special effects on the old CPU path while the world
path is proven.

## MVP Scope

The first meaningful milestone should be opaque world rendering:

- CPU submits visible world surface IDs.
- GPU renders world polygons into an 8-bit framebuffer.
- Texture atlas and static lightmap atlas are persistent.
- Palette resolve/present remains GPU-side.
- Alias models, sprites, particles, water, sky, and UI can remain fallback paths.

Success criteria:

- `e1m1` world surfaces render with recognizable Quake texturing and lighting.
- No per-frame texture repacking.
- No per-frame framebuffer readback.
- Stable timedemo benchmark at 640x480.
- Output is close enough to WinQuake to compare visually.

## Data Structures

### Surface Metadata

```c
struct GpuSurface {
	uint first_vertex;
	uint vertex_count;
	uint texture_id;
	uint lightmap_id;
	uint flags;
	float plane[4];
	float tex_s[4];
	float tex_t[4];
	float light_s[4];
	float light_t[4];
};
```

### Primitive Record

```c
struct GpuPrimitive {
	uint surface_id;
	uint first_vertex;
	uint vertex_count;
	uint texture_id;
	uint lightmap_id;
	uint flags;
	float depth_bias;
};
```

### Texture Metadata

```c
struct GpuTexture {
	uint mip_offset[4];
	uint width[4];
	uint height[4];
	uint flags;
};
```

### Tile List

```c
struct TileHeader {
	uint offset;
	uint count;
	uint overflow;
};
```

The exact layout should be tuned for AMDGPU memory access, but all buffers should
remain plain C-compatible arrays.

## Lighting Model

Preserve Quake's indexed lighting:

- Texture sample returns an 8-bit palette index.
- Lightmap/dynamic light selects a colormap row or light level.
- `colormap[light_level][texel]` returns final 8-bit index.

This avoids drifting into RGB lighting while still letting the GPU do the work.

Initial simplification:

- Static lightmap only.
- No dynamic lightmap updates.
- Fullbright texels handled through existing Quake rules.

Then add:

- Dirty lightmap page uploads.
- Dynamic lights.
- Animated light styles.

## Mip Policy

Classic Quake's mip choice is part of the look.

Start with a screen-space approximation:

- Compute approximate texture derivatives during setup.
- Select one of Quake's existing mip levels.
- Sample nearest from that mip.

Do not add filtering by default. Bilinear filtering can be an optional debug mode
later, but it should not define the baseline look.

## Special Surfaces

### Sky

Implement as a separate pass with classic sky texture behavior. It can use tile
rasterization but a different shader path.

### Water

Implement turbulent UV perturbation in the raster kernel or a dedicated water
kernel. Preserve nearest indexed sampling.

### Transparent And Cutout

Keep explicit rules:

- `255` transparent index where applicable.
- Depth test/write behavior matching Quake's expectations.
- Palette/colormap blending for classic translucent effects if enabled.

## Alias Models

Alias models should use the same tiled rasterizer once the world path is stable.

Plan:

- CPU selects animation frame and pose.
- GPU or CPU transforms alias vertices.
- Emit triangle primitives.
- Raster triangles with affine texture mapping first.
- Apply Quake-style model lighting.

Exact software edge rules are less important than preserving silhouette, palette,
nearest skin sampling, and lighting style.

## Sprites And Particles

Sprites:

- Emit screen-aligned quads.
- Use transparent index handling.
- Bin/raster like other primitives.

Particles:

- Start as one job per particle writing a small square.
- Later bin by tile to reduce scattered writes.

## Performance Priorities

Prioritize architectural wins:

- Persistent resident resources.
- No per-frame texture repacking.
- No CPU span generation.
- No GPU framebuffer readback.
- Coherent tile-local raster work.
- Large dispatches with enough occupancy.
- Minimal CPU/GPU synchronization.

Avoid spending time on:

- Per-pixel reciprocal micro-optimizations.
- Recreating WinQuake inner loops on GPU.
- Tiny dispatches per surface.
- CPU-generated spans or chunks as the long-term representation.

## Implementation Roadmap

### Stage A: Resource Builder

- Build GPU texture atlas from Quake textures and mips.
- Build GPU world surface metadata.
- Build static lightmap atlas.
- Keep resources persistent across frames.

### Stage B: World Command Buffer

- Emit visible world surface IDs from the existing BSP/PVS path.
- Upload one compact command buffer per frame.
- Keep old renderer available as fallback.

### Stage C: Minimal World Raster

- Raster world polygons directly, without tile binning if necessary.
- Write 8-bit framebuffer and depth.
- Support texture atlas and static lightmap sampling.
- Present through existing palette conversion path.

### Stage D: Tiled Raster

- Add tile bins.
- Dispatch one or more workgroups per tile.
- Move world raster to tile lists.
- Compare perf against Stage C.

### Stage E: Classic Features

- Sky.
- Water turbulence.
- Animated light styles and dirty lightmaps.
- Transparent/cutout surfaces.

### Stage F: Entities

- Alias models.
- Sprites.
- Particles.

### Stage G: Cleanup

- Remove old span experiment from the main path.
- Keep debug cvars for fallback/comparison only.
- Add perf counters for command counts, primitive counts, tile occupancy, and GPU
  pass timings.

## Open Questions

- Should clipping be CPU-side for the first world MVP, or should setup generate
  conservative screen bounds and let raster kernels reject pixels?
- Is `8x8` or `16x16` the better tile size for target AMD hardware?
- Should depth be fixed-point to match Quake-style `z` behavior or floating
  initially for simpler bring-up?
- How much exactness is required for edge fill rules before visual differences
  become noticeable?
- Should UI remain CPU-rendered into an indexed overlay, or should it become
  another GPU pass early?

## Recommendation

Start with persistent world resources and a GPU-owned world raster path. Do not
continue optimizing CPU span offload. The span experiment should remain as proof
that `libkfd` presentation and simple compute kernels work, but the production
renderer should use scene-level commands and tiled compute rasterization.
