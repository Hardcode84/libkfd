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

## Prior Art Baseline

The renderer is a tile-based deferred renderer implemented with compute kernels.
Its closest references are:

- NVIDIA `cudaraster` (Laine and Karras, HPG 2011): a full CUDA graphics
  pipeline with multi-level binning.
- TU Graz `cuRE` (Kenzel, Kerbl, Schmalstieg, Steinberger, SIGGRAPH 2018): a
  later GPU compute graphics pipeline with streaming stages and bounded memory.
- Larrabee/OpenSWR/llvmpipe/mobile TBDRs: production examples of tile-based
  software or hardware rasterization.

The Quake renderer should borrow the broad architecture, not the full generality.
Quake has large polygons, cheap fixed shading, no MSAA requirement, no arbitrary
shader programs, and CPU-side BSP/PVS visibility. This puts it in the
large-triangle, cheap-shader TBDR regime, not the Nanite micro-polygon regime
and not the voxel/ray-search regime.

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

Raster work should be binned into `16x16` screen tiles for the MVP.

```text
primitive records
  -> tile primitive lists
  -> one or more workgroups per tile
```

`16x16` is the default because it maps well to AMD wave32 execution:
256 pixels per tile gives enough waves per tile to keep occupancy healthy while
keeping tile-local metadata small. `8x8` can be useful later as a fine level
under a coarse hierarchy, cudaraster-style, but it should not be the only MVP
tile size.

This gives the GPU coherent work and limits depth/color write contention.

Start simple:

- CPU or GPU clears tile counters.
- Each primitive appends itself to all overlapping tiles.
- Overflow falls back or uses a secondary overflow list.

Optimize later:

- Prefix-summed tile lists.
- Coarse -> fine hierarchical bins.
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

### 5. Depth, Color, And Races

AMDGPU has 32-bit and 64-bit atomics, not byte atomics. The renderer must not
pretend an 8-bit framebuffer can be updated atomically by overlapping
primitives.

Use two policies:

- Opaque world MVP: exploit CPU BSP/PVS ordering and tile-local primitive order.
  Raster world surfaces in deterministic order within each tile. Use a normal
  8-bit color store and a depth test/update policy that preserves the selected
  world ordering. Validate this visually and with debug modes.
- General entities and later mixed-order paths: use a packed 64-bit atomic path
  when correctness needs race-free depth and color. Pack depth plus color or
  primitive ID into one value, following the Nanite-style
  `atomicMax/atomicMin(depth_and_payload)` pattern.

Keep the indexed framebuffer as the canonical color target. If the packed atomic
path stores IDs instead of color, shade those pixels in a later resolve pass only
for the paths that need it. Do not turn the whole Quake world renderer into a
visibility-buffer pipeline unless profiling proves it is needed.

### 6. HiZ

Add a per-tile hierarchical depth summary early.

For the MVP, a simple `16x16` tile depth bound is enough:

- Setup/raster records conservative depth bounds per tile.
- Tile raster tests the bound before walking the full primitive list.
- Opaque world surfaces can combine BSP order with tile depth bounds to reject
  obvious hidden work.

This should be treated as part of the world-raster architecture, not as a later
micro-optimization.

### 7. Resolve And Present

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

### Tile Depth

```c
struct TileDepth {
	uint nearest_depth;
	uint farthest_depth;
};
```

Depth can start as `uint32_t` fixed-point or float-bit ordered depth, whichever
gets the MVP running fastest. Exact WinQuake Z behavior is less important than a
stable ordering policy and a debug mode that makes disagreements visible.

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
- Coherent `16x16` tile-local raster work.
- Explicit color/depth race policy.
- Per-tile HiZ/depth bounds.
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
- Choose and validate the opaque-world color/depth ordering policy.

### Stage D: Tiled Raster

- Add `16x16` tile bins.
- Dispatch one or more workgroups per tile.
- Move world raster to tile lists.
- Add per-tile depth bounds / HiZ.
- Compare perf against Stage C.

### Stage E: Hierarchical Binning

- Add a coarse bin level above `16x16` tiles if large world polygons make
  primitive x tile counts expensive.
- Use bounded overflow lists from the start.
- Treat cudaraster/cuRE as the reference shape for coarse -> fine binning.

### Stage F: Classic Features

- Sky.
- Water turbulence.
- Animated light styles and dirty lightmaps.
- Transparent/cutout surfaces.

### Stage G: Entities

- Alias models.
- Sprites.
- Particles.
- Use the packed atomic path for entity/transparent cases that cannot rely on
  opaque world ordering.

### Stage H: Cleanup

- Remove old span experiment from the main path.
- Keep debug cvars for fallback/comparison only.
- Add perf counters for command counts, primitive counts, tile occupancy, and GPU
  pass timings.

## Open Questions

- Should clipping be CPU-side for the first world MVP, or should setup generate
  conservative screen bounds and let raster kernels reject pixels?
- Can opaque world color stores safely rely on BSP/tile-local order, or do some
  maps/surfaces require packed atomics even for world rendering?
- Should depth be fixed-point or float-bit ordered for the first MVP?
- How much exactness is required for edge fill rules before visual differences
  become noticeable?
- Should UI remain CPU-rendered into an indexed overlay, or should it become
  another GPU pass early?

## Recommendation

Start with persistent world resources and a GPU-owned world raster path. Do not
continue optimizing CPU span offload. The span experiment should remain as proof
that `libkfd` presentation and simple compute kernels work, but the production
renderer should use scene-level commands and tiled compute rasterization.
