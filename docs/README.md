# Streaming Software Rasterizer — Design Docs

A standalone graphics-pipeline demo running on AMDGPU via `libkfd`.
The host streams transformed triangles (rotating Utah teapot or
similar simple model, **procedurally checker-textured**) to a
GPU-resident megakernel that bins, rasterizes, and shades them, then
flips to a dma-buf-shared swapchain. Throughout the docs the working
scene is referred to as the *demo workload* (~6 K triangles per
frame, 1080p, 60 fps target).

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
   lock shared between host and GPU; per-tile `queue_lock` (brief)
   and `render_lock` (long) plus an `in_ready` flag and a global
   MPMC `ready` ring of tile IDs. Distributors push tile IDs into
   the ring on each empty→non-empty transition; renderers pop in
   O(1) without scanning. See §1.1 for what changed since the first
   prototype.
3. `frame-completion-detection.md` — how the host detects
   end-of-frame against a *persistent* megakernel, since cuRE's
   "kernel-per-draw-call exit" does not generalize. Recommends a
   bulk per-frame counter with per-WG SGPR accumulator; documents
   both polling and KFD-signal interrupt-driven host-wait paths.
4. `headless-testing.md` — running the kernel in CI / on a
   display-less dev box. Same kernel binary as the windowed demo;
   readback replaces flip. Defines test categories (pixel-equality,
   invariants, concurrency stress, frame-completion, termination,
   microbenchmarks) and the single `headless_runner` driver they
   share.

## What's In Scope

These docs cover the **rasterization pipeline** end-to-end:

- Host transforms each frame's vertices to clip space and packs
  triangles (with per-vertex UV) into batches.
- Streaming binning, rasterization, depth test, framebuffer write
  on the GPU.
- Per-pixel **procedural checkerboard** sampling against
  interpolated UVs — `((u_int >> 3) ^ (v_int >> 3)) & 1` selects
  between two colors. No texture memory; the "texture" is a
  closed-form expression in the rasterizer. This validates UV
  interpolation and per-pixel shading hooks without coupling the
  pipeline to texture cache or sampler state.
- Frame completion detection, host-GPU wake.

## What's Out Of Scope

- Vertex transform on the host beyond a small CPU-side T&L step
  (no skinning, animation rigs, LOD).
- Sampled texture maps (the checker is procedural). A real texture
  cache + sampler is a downstream extension.
- **Designing** a new present path. The demo *reuses* the dma-buf
  swapchain pattern already exercised by `tools/computetoy` once
  windowed mode lands at the end of stage 1
  (`bin-ownership-pipeline-proposal.md` §9). Stages 0 and the
  headless test runs of stage 1+ never touch a swapchain — they
  read the framebuffer back to host memory instead
  (`headless-testing.md` §2).
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
- **W** — number of waves per WG.
- **C** — per-tile queue capacity (entries). Recommended default
  256 (`bin-ownership-pipeline-proposal.md` §5.3).
- **P_max** — primitive-store capacity (primitives buffered in
  fine-grained SVM, frame-resident).
- **N_F** — primitive count of frame F (used by the per-frame
  ack protocol, `frame-completion-detection.md` §5).
- **NUM_INFLIGHT_FRAMES** — host's frame-ring depth (typically 2–4),
  unrelated to `N`.
- **B/N** — design-determining ratio. Three regimes:
  - `B/N ≥ 16` — robust; renderer pops the ready ring and almost
    never re-pushes on `render_lock` contention.
  - `4 ≤ B/N < 16` — works; the ready ring still serves all renderers
    in O(1) but `render_lock` re-pushes become measurable
    (`bin-ownership-pipeline-proposal.md` §6.2).
  - `B/N < 4` — breaks down; fixed-role partitioning (a few
    distributor WGs, rest renderers) is the better trade.

  The 1080p teapot demo lands at `B/N ≈ 25` with 32×32 bins.
- **Megakernel** — a single long-running compute kernel that
  encapsulates all pipeline stages, contrasted with one
  kernel-per-stage.
- **Persistent WG** — a WG that does not exit when its first work
  item finishes; it loops back and pulls more work from queues until
  a global termination flag is set.
- **Fine-grained SVM** — host-coherent shared virtual memory used
  for the host↔GPU ring. Required for the streaming-from-host
  pattern.
