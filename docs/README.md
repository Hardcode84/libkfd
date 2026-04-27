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
   that the two pipeline proposals borrow from. Read this first.
2. `streaming-pipeline-proposal.md` — the **primary candidate
   pipeline shape**. One distributor WG with wave-disjoint coarse-bin
   partitioning; renderer WGs as multi-consumer of bin queues. Roles
   are fixed at WG-launch time. Best fit when `B/N < 16`.
3. `bin-ownership-pipeline-proposal.md` — **alternative pipeline
   shape**. All WGs symmetric; each iteration picks distribute-mode
   or render-mode dynamically via per-tile locks. Best fit when
   `B/N ≥ 16` (denser scenes, smaller tiles, 4K).
4. `frame-completion-detection.md` — orthogonal to the choice
   between (2) and (3). How the host detects end-of-frame against a
   *persistent* megakernel, since cuRE's "kernel-per-draw-call exit"
   does not generalize. Recommends a bulk per-frame counter with
   per-WG SGPR accumulator; documents both polling and KFD-signal
   interrupt-driven host-wait paths.

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
- **B/N** — design-determining ratio. Drives the choice between the
  streaming and bin-ownership shapes.
- **Megakernel** — a single long-running compute kernel that
  encapsulates all pipeline stages, contrasted with one
  kernel-per-stage.
- **Persistent WG** — a WG that does not exit when its first work
  item finishes; it loops back and pulls more work from queues until
  a global termination flag is set.
- **Fine-grained SVM** — host-coherent shared virtual memory used
  for the host↔GPU ring. Required for the streaming-from-host
  pattern.
