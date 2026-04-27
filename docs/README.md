# Streaming Software Rasterizer — Design Docs

A standalone graphics-pipeline demo running on AMDGPU via `libkfd`.
The host streams transformed triangles (rotating Utah teapot or
similar simple model) to a GPU-resident megakernel that bins,
rasterizes, and shades them, then flips to a dma-buf-shared
swapchain. Throughout the docs the working scene is referred to as
the *demo workload* (~6 K triangles per frame, 1080p, 60 fps target).

The four documents in this directory are design notes for that
pipeline. None is implemented yet; they are the blueprint.

## Reading Order

1. `cure-streaming-queues.md` — vocabulary. Reference for the cuRE
   streaming-queue primitives (`MultiIndexQueue`, `ProgressQueue`)
   that the pipeline proposal borrows from. Read this first.
2. `bin-ownership-pipeline-proposal.md` — the **pipeline design**.
   A persistent megakernel of `N` symmetric workgroups; each WG
   picks distribute-mode (drain host queue, scatter primitives into
   per-tile queues) or render-mode (drain a tile, rasterize) per
   iteration via dynamic ownership locks. Single binary host queue
   lock shared between host and GPU; 3-state per-tile locks that
   let distributor pushes and renderer rasterizes overlap.
3. `frame-completion-detection.md` — how the host detects
   end-of-frame against a *persistent* megakernel, since cuRE's
   "kernel-per-draw-call exit" does not generalize. Recommends a
   bulk per-frame counter with per-WG SGPR accumulator; documents
   both polling and KFD-signal interrupt-driven host-wait paths.

## What's Out Of Scope

These docs cover the **rasterization pipeline** (binning,
rasterization, framebuffer write, frame completion). They do not
cover:

- Vertex transform on the host (assume a small CPU-side T&L step).
- Texture sampling (deferred — start with flat-shaded triangles).
- Present path (assume the dma-buf swapchain pattern already
  exercised by `tools/computetoy`).
- Demo content beyond "rotating teapot or similar" (asset format,
  mesh loading, animation are downstream concerns).

## Glossary

- **WG** — workgroup. The CTA-equivalent on AMDGPU; runs on a single
  CU with `W` waves resident.
- **Wave** — a hardware wavefront; 32 lanes on RDNA, 64 on GFX9.
- **Coarse bin** — a screen-space rectangle (typically 32×32 to
  128×128 pixels), the unit of work assignment to a WG.
- **Fine tile** — a sub-rectangle of a coarse bin (typically 8×8 or
  16×16), the unit of wave-level rasterization within a WG.
- **B** — number of coarse bins covering the framebuffer.
- **N** — number of resident WGs (≈ hardware occupancy).
- **B/N** — design-determining ratio. The bin-ownership design is
  robust at `B/N ≥ 16` and breaks down below `B/N < 4`.
- **Megakernel** — a single long-running compute kernel that
  encapsulates all pipeline stages, contrasted with one
  kernel-per-stage.
- **Persistent WG** — a WG that does not exit when its first work
  item finishes; it loops back and pulls more work from queues until
  a global termination flag is set.
- **Fine-grained SVM** — host-coherent shared virtual memory used
  for the host↔GPU ring. Required for the streaming-from-host
  pattern.
