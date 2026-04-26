# Quake Compute Rasterizer Prior Art And Architectural Critique

This document complements `quake-compute-rasterizer-plan.md` by placing the
proposed architecture in the context of existing compute-oriented software
rendering work, and by identifying where the plan is well-founded, where it
borrows wisely, and where it under-specifies decisions that prior art has
already settled.

It is not a survey paper. The goal is to extract concrete lessons that apply to
a from-scratch GPU-compute software renderer for classic Quake on `libkfd`.

## 1. The Architectural DNA Of The Proposed Renderer

Strip the Quake-specific vocabulary and the proposal is a Tile-Based Deferred
Renderer implemented as GPU compute kernels:

- Persistent GPU resources (atlases, surface metadata).
- CPU front-end producing scene-level primitive commands.
- Atomic-append per-tile binning.
- One workgroup per tile evaluating coverage, sampling textures, writing tile
  output.
- Final resolve pass converting the tile output to the present format.

It is not a Nanite-style micro-polygon renderer, not a voxel/ray-search
renderer, and not a vis-buffer plus deferred shading pipeline. It sits in the
same family as Larrabee, OpenSWR, llvmpipe, and mobile TBDR GPUs, and inherits
most directly from the NVIDIA `cudaraster` (2011) and TU Graz `cuRE` (2018)
GPU-compute graphics pipelines.

## 2. Why The Regime Determines What Prior Art Applies

Most modern compute-rasterization research targets one of two extremes:

- Micro-polygon regime: millions of triangles, sub-pixel sized, complex
  shading. Examples: Nanite, REYES, modern cluster renderers.
- Voxel/ray-search regime: static scenes baked into sparse octrees, ray-search
  per pixel. Examples: GigaVoxels, ESVO, Unlimited Detail.

Quake is in neither regime. Its characteristics are:

- A few thousand visible polygons per frame, mostly very large (entire BSP wall
  faces).
- BSP traversal yields back-to-front leaf order for free, plus precomputed PVS.
- Per-pixel shading is two memory lookups and a colormap rule (~5 ALU ops).
- Output is 8-bit indexed, single render target, no PBR, no multi-light.
- Static world plus a separate small entity list.

This is the classic large-triangle, cheap-shader regime. The relevant prior art
is software TBDR — first on CPU (§3), then realized as GPU compute (§4).

## 3. The TBDR Lineage (CPU-Side)

The plan reinvents pieces of a well-explored architecture:

- **Larrabee** (Intel, 2008–2010). Forsyth's SIGGRAPH 2008 talk and his blog
  series describe a 32x32 fine tile, two-level binning (coarse 256x256 then
  fine 32x32), tile-resident color/depth in last-level cache, atomic-append
  primitive lists, software pipeline running across many x86+SIMD cores.
  Larrabee never shipped as a GPU but the architecture survived in everything
  below.
- **Intel OpenSWR** (2016–2020). Larrabee's TBDR shape rebuilt as a Mesa
  Gallium driver, multi-threaded, AVX2, production-quality, still in Mesa.
- **Mesa `llvmpipe`**. 64x64 macro-tiles with 4x4 sub-tile rasterization,
  JIT-compiled shaders. The closest production analog to what the plan is
  building, just on CPU.
- **Mobile TBDR** (PowerVR Series 1+ from 1996, Mali Midgard onward, Apple
  GPUs). Commercial deployment of "tile color stays in tile memory until
  end-of-tile resolve". Tile-resident 8-/16-bit color writing to system memory
  only at end-of-tile is exactly the plan's "8-bit indexed framebuffer ->
  palette resolve at present" pattern.
- **Microsoft WARP** (2009+). Full DX10/11 software rasterizer, tile-based,
  multi-threaded.
- **Pixomatic** (Michael Abrash and RAD, 2003). Quake-era SIMD software raster.
  The Black Book Quake chapters provide direct context for what the plan is
  preserving.

Concrete lessons from this lineage:

1. Tile sizes converged at 16x16 to 32x32 across nearly all of these projects,
   not 8x8.
2. Two-level (coarse -> fine) binning becomes necessary as soon as primitives
   commonly span many fine tiles.
3. Tile-resident color/depth in fast memory always pays off; mobile TBDRs are
   built around exactly this.
4. Hierarchical Z is cheap and effective. Mali, PowerVR, and llvmpipe all do
   it.

## 4. Cudaraster And cuRE: Graphics Pipelines As Pure GPU Compute

The most directly comparable prior work is two papers that together represent
the GPU-side answer to "implement a graphics pipeline in compute". Both are
open-sourced, both target the same problem space the Quake plan does, and both
predate Nanite by a decade or more. The plan should treat these as primary
references.

### 4.1 Cudaraster (Laine And Karras, NVIDIA, HPG 2011)

Samuli Laine and Tero Karras's *High-Performance Software Rasterization on
GPUs* (HPG 2011) from NVIDIA Research, released as the open-source
`cudaraster` project (Google Code archive; GitHub mirror at
`github.com/ap1/cudaraster`).

- Implements a complete graphics pipeline — triangle setup through ROP —
  entirely in CUDA compute, with no fixed-function raster.
- Four-stage chunker-style pipeline: triangle setup -> bin raster -> coarse
  raster -> fine raster.
- Three binning levels: bin (very coarse), coarse (~32x32), fine (~8x8).
- Obeys API ordering, guarantees hole-free rasterization, supports MSAA.
- Within 2-8x of hardware raster on the GPUs of its day.
- Cited by every subsequent GPU-compute graphics pipeline paper.

### 4.2 cuRE (Kenzel, Kerbl, Schmalstieg, Steinberger, TU Graz, SIGGRAPH 2018)

Michael Kenzel, Bernhard Kerbl, Dieter Schmalstieg, and Markus Steinberger's
*A High-Performance Software Graphics Pipeline Architecture for the GPU*
(ACM TOG / SIGGRAPH 2018), released as the `cuRE` project under MIT license.

The successor to cudaraster, addressing what 2011 left unsolved:

- Fully-concurrent, multi-stage, streaming design with no per-stage barriers.
- Dynamic load balancing across stages (built on Steinberger's earlier
  *Softshell: Dynamic Scheduling on GPUs*, TOG 2012).
- Bounded memory consumption — cudaraster could blow up under adversarial
  inputs.
- Vertex reuse across primitives sharing vertices.
- Primitive order preservation.
- Screen-space derivatives of dependent variables.
- Tested on >100 real video game scenes.
- Within one order of magnitude of the hardware pipeline (so ~3-10x slower
  depending on scene).

### 4.3 Lessons That Transfer To The Quake Plan

- **Multi-level binning is the right shape.** Bin -> coarse -> fine. The
  plan's "tile binning" is the *fine* level only; production-quality work
  needs at least the coarse level above it. Quake-sized polygons (a wall
  covering ~50 fine tiles) are exactly the regime that motivated this
  hierarchy in cudaraster.
- **Queue-based work distribution between stages.** Each stage emits primitive
  records into a queue consumed by the next. The plan's "primitive records ->
  tile primitive lists -> workgroups per tile" is the same shape, just less
  general.
- **Concurrent streaming beats barrier-per-stage at scale** (cuRE vs
  cudaraster). The Quake MVP can ship with stage barriers (simpler); plan for
  concurrent streaming if frame time matters later.
- **Bounded memory** is a real constraint cudaraster ducked. Per-tile lists
  with overflow handling — already in the plan — is the right approach; size
  the overflow conservatively from the start.
- **Performance baseline.** Cudaraster: 2-8x slower than HW. cuRE: within one
  order of magnitude. A Quake-specific design with a fixed pipeline, no MSAA,
  no API ordering, BSP-pre-sorted, and trivial shading should land at the
  favorable end of this envelope or better, on the relevant subset of work.
- **Reference implementations exist.** cuRE is open, MIT-licensed, modern, and
  demonstrably runs hundreds of real game scenes. Treat it as the primary
  reference for "how do you stage a compute raster across multiple binning
  levels in production".

### 4.4 What Does Not Transfer

- Both target general triangles with arbitrary pixel/vertex shaders. Quake's
  pipeline is fixed.
- Both preserve API draw order. Quake does not need this for the opaque world
  (BSP order suffices).
- Both support MSAA. Quake does not.
- cuRE handles vertex reuse across primitives. Quake's per-surface vertex
  layout makes reuse less important.

### 4.5 FreePipe And Softshell: Earlier And Adjacent Work

For completeness:

- **FreePipe** (Liu, Huang, Liu, Wu, I3D 2010) was the first significant
  CUDA-based software graphics pipeline, focused on multi-fragment effects.
  Both cudaraster and cuRE cite it as predecessor work. Less directly
  applicable to the Quake plan than the above.
- **Softshell** (Steinberger et al., TOG 2012) provided the dynamic GPU
  scheduling foundation later used by cuRE. OS-style scheduling on GPU work —
  dynamic priorities, work cancellation, pause/resume. Of indirect relevance:
  useful if the renderer ever grows non-trivial scheduling needs (e.g.
  async upload of dynamic lightmap pages overlapped with raster).

## 5. Nanite Is Not The Ancestor But Contributes Two Specific Ideas

Nanite (Karis, SIGGRAPH 2021 *Nanite: A Deep Dive*) targets a different regime:
sub-pixel triangles where hardware raster wastes 75%+ of pixel-shader threads
on partially-covered 2x2 quads. Nanite ships two raster paths: hardware raster
for large triangles, software (compute) raster for small ones. The software
path uses:

- Persistent threads pulling work from a queue, instead of static
  workgroup-per-primitive. Tolerates wildly variable per-cluster cost.
- 64-bit `atomicMax((~depth << 32) | tri_id)` writing depth and visibility id
  in one race-free op.
- Visibility buffer: defer all material shading to a screen-space pass over
  `(tri_id, barys)`.

Two of these ideas are worth borrowing into Quake-land:

- The 64-bit atomic packing trick is the cleanest solution to the
  byte-framebuffer race problem (see §8.3). On AMD GFX10+ this is a single
  instruction.
- Persistent threads only win when per-tile work varies by 100x+. Quake's
  modest primitive counts do not justify the extra complexity. Skip.

The vis-buffer model is wrong for Quake. Shading is two lookups; deferring it
costs a separate full-screen pass for no win. Forward shading per tile is
correct.

## 6. Unlimited Detail (Euclideon): Cautionary Footnote

Euclideon, founded by Bruce Dell in Brisbane around 2010, demoed an "Atom
Engine" / "Unlimited Detail" renderer 2010–2014 with "unlimited polygon counts"
claims. The actual technology was a CPU sparse-voxel-octree ray-search
renderer. The marketing was inflated, the gaming engine never shipped at the
promised quality, and the company pivoted to Holoverse location-based VR
(~2016) and later to underground laser-projection visualization. They still
exist; the original gaming claim collapsed.

The legitimate research in the same direction lives elsewhere: Crassin's
GigaVoxels (2009), Laine and Karras's Efficient Sparse Voxel Octrees (NVIDIA,
2010), Atomontage. These target large static scenes — CAD, scientific
visualization, geological — where their tradeoffs win.

Lesson for the Quake plan: do not go this way. Projection plus tile raster is
the right regime for static-geometry low-poly games with moving entities.
Voxel/ray-search loses on dynamic geometry, animation, and the fact that GPU
rasterization performance kept improving faster than search-renderer companies
could ship.

## 7. Intel's Occlusion Line: Three Projects, One Big Lesson

- **Software Occlusion Culling sample** (Intel, 2013). Rasterize bounding boxes
  into a low-res depth buffer with AVX2, query before submitting full draws.
- **Masked Software Occlusion Culling** (Andersson and Hasselgren, JCGT 2015).
  Hierarchical Z plus per-tile coverage masks. ~10x faster than plain software
  raster. Adopted at Frostbite, Naughty Dog ICE, and multiple AAA studios.
- **OpenSWR** (covered in §3).

The big lesson is HiZ. A coarse depth pre-pass — even just per-tile `Z_max` —
kills overdraw cheaply. Quake's BSP gives back-to-front leaf order for free;
combining BSP-ordered traversal with a per-tile `Z_max` HiZ would skip large
amounts of work in the world-raster kernel without any visibility data
structure changes. The plan does not mention HiZ. It should.

## 8. Architectural Critique Applied To The Plan

With these lineages as ammunition, here is where the plan is right, where it
is silent, and where it is wrong.

### 8.1 Tile Size: 8x8 As The Only Level Is Unusually Small

Larrabee 32x32, OpenSWR 16x16, llvmpipe 64x64 with 4x4 sub-tile, Mali 16x16,
PowerVR 32x32. Cudaraster uses 8x8 fine tiles but *under* a coarse level. The
plan proposes 8x8 or 16x16 as the only binning level.

On AMD RDNA wave32, 16x16 = 256 pixels = 8 waves per tile = good occupancy;
8x8 = 64 pixels = 2 waves = marginal. As the only level, default to 16x16. As
the fine level under a coarse hierarchy (the cudaraster shape), 8x8 is
defensible.

### 8.2 Forward Shading Inside The Tile: Correct

Vis-buffer would over-engineer trivially cheap Quake shading. Keep forward
shading inside the tile.

### 8.3 The Byte-Framebuffer Atomicity Problem: The Gap

The plan glosses this. AMDGPU has 32-bit and 64-bit atomics, no byte-level
atomics. Three real options:

- Pack 4 bytes into a u32 and use `atomicCAS` loops for byte updates. Correct,
  slow under contention.
- Separate u32 depth buffer with `atomicMin`, byte color buffer with
  non-atomic store. Races on overlapping coverage by different primitives.
  Acceptable for Quake's opaque world *because BSP back-to-front order already
  orders draws* — exploit it. Entities use a different policy.
- Nanite-style 64-bit `atomicMax((~depth << 32) | (color << 24) | tri_id)`
  writes depth + color + id in one race-free op. Works on GFX10+. Probably the
  cleanest answer.

The plan should pick one explicitly. This is the single biggest unspecified
architectural question.

### 8.4 Atomic-Append Tile Binning With Overflow List: Fine For MVP

Standard Larrabee/cudaraster shape. Add a coarse level above the fine level
when primitive x tile counts grow — cuRE's three-level streaming pipeline is
the reference shape.

### 8.5 Software Depth: Under-Specified

Use a u32 atomic-min depth buffer separate from color. Float depth is fine for
bring-up; fixed-point Z only matters if matching WinQuake's exact Z behavior is
a requirement.

### 8.6 Missing HiZ Pre-Pass: Free Win

Add a per-tile `Z_max`, written during setup, tested before tile raster.
Cheap, big win, no PVS changes. Combines naturally with BSP order.

### 8.7 Exploit BSP Order

The plan treats every primitive as if it might overlap with any other. For
opaque world surfaces, BSP guarantees front-to-back-ish leaf order; opaque
world raster can write color *without* color atomics if dispatch order matches
BSP order and the depth policy is "first write wins". Halves the atomic
pressure on the dominant path.

### 8.8 CPU Keeps BSP/PVS: Correct

GPU-driven cluster culling (Nanite, Wronski/Frostbite *GPU-Driven Rendering
Pipelines*) wins at million-cluster scale. Quake has hundreds. CPU traversal
stays cheaper.

### 8.9 Persistent Atlases: Correct

Mip pyramids for a Quake level fit in single-digit MB. Trivial residency
problem. Right call.

### 8.10 Workgroup-Per-Tile Vs Persistent Threads

Workgroup-per-tile is correct for Quake's modest variance. Persistent threads
(Nanite-style queue pulling, cuRE-style streaming) only win when per-tile work
varies by 100x+. Skip for MVP; revisit if profiling demands.

### 8.11 Vertex Transform Location

Plan defers this to MVP discretion. ~10K visible vertices per frame is ~120 KB
upload; CPU transform is trivially fine. Move to GPU when entity counts grow.

## 9. Recommended Reading

In priority order:

- Michael Kenzel, Bernhard Kerbl, Dieter Schmalstieg, Markus Steinberger,
  *A High-Performance Software Graphics Pipeline Architecture for the GPU*
  (ACM TOG / SIGGRAPH 2018, the cuRE paper). The single most directly
  comparable prior work; modern, MIT-licensed source available.
- Samuli Laine, Tero Karras, *High-Performance Software Rasterization on GPUs*
  (HPG 2011, the cudaraster paper). Foundational; reads cleanly.
- Tom Forsyth, *Larrabee: A Many-Core x86 Architecture for Visual Computing*
  (SIGGRAPH 2008). Intellectual ancestor of all of the above.
- Brian Karis, *Nanite: A Deep Dive* (SIGGRAPH 2021). For the 64-bit atomic
  trick and the software/hardware raster split rationale.
- Magnus Andersson, Jon Hasselgren, *Masked Software Occlusion Culling*
  (JCGT 2015). HiZ plus coverage masks done right.
- Markus Steinberger et al., *Softshell: Dynamic Scheduling on GPUs*
  (TOG 2012). Background for cuRE's streaming model.
- Christopher Burns, Warren Hunt, *The Visibility Buffer* (JCGT 2013).
  Option to keep in mind if shading ever grows.
- Mesa `llvmpipe` source (`src/gallium/drivers/llvmpipe/`). Closest production
  CPU-side TBDR analog.
- Ulrich Haar, Sebastian Aaltonen, *GPU-Driven Rendering Pipelines*
  (SIGGRAPH 2015). Already implicit in `proposed-api-surface.md`'s heritage.
- Fabian Giesen, *A Trip Through The Graphics Pipeline* blog series. The
  underlying mental model for all of the above.
- Michael Abrash, *Graphics Programming Black Book*, Quake-renderer chapters.
  Contextualizes what the plan is preserving.
- Fang Liu, Meng-Cheng Huang, Xue-Hui Liu, En-Hua Wu, *FreePipe* (I3D 2010).
  The earliest CUDA software graphics pipeline; mostly historical interest.

## 10. Summary

The proposed architecture is a Larrabee-shape software TBDR realized as GPU
compute. Its most direct ancestors are NVIDIA `cudaraster` (2011) and TU Graz
`cuRE` (2018) — both open-source, both implementing full graphics pipelines in
pure CUDA, both within an order of magnitude of hardware raster. The plan is
right about the regime (large polys, cheap shading, tile binning, persistent
atlases, forward in-tile shading) and right to ignore Nanite-style
micro-polygon and voxel-search lineages as primary inspirations.

Its under-specified spots are:

- Tile size as a single level (should be 16x16, not 8x8; 8x8 is fine as the
  fine level under a coarse level, cudaraster-style).
- Byte-framebuffer atomicity (likely best solved Nanite-style with 64-bit
  packed atomics, or by exploiting BSP order to avoid color atomics
  altogether).
- Missing HiZ pre-pass (essentially free given BSP order).
- Single-level binning will not scale; cuRE's streaming three-level pipeline
  is the production target.

Borrow architecture from cuRE and cudaraster. Borrow the 64-bit atomic trick
from Nanite. Borrow HiZ from Masked Software Occlusion Culling. Ignore the
voxel-search lineage entirely.
