# Headless Testing

The pipeline must be runnable **without a display** — no DRM
modesetting, no dma-buf swapchain, no compositor. Every component
described in `bin-ownership-pipeline-proposal.md` is exercised by
piping primitives through the kernel and reading the framebuffer
back to host memory. This is the configuration that runs in CI,
on remote AMDGPU dev boxes, and during single-step debugging.

**Performance and budget numbers in this document are
targets / order-of-magnitude estimates**, not measurements. The
~1 ms PCIe readback, the < 5 s / 30 s CI budgets, and the ≤ 10 ms
termination bound (§3.5) are policy/design targets that real
runs will calibrate against.

## 1. Why Headless First

The presentation path (window, swapchain, vsync, flip) is
orthogonal to everything we are designing:

- The kernel writes pixels into a buffer. The buffer's *consumer*
  (compositor vs. test harness) is irrelevant to kernel behaviour.
- KFD compute queues are display-independent. `tools/computetoy`
  uses the dma-buf swapchain pattern, but the kernel itself does
  not require it.
- Headless runs are deterministic in the ways that matter for
  testing (no vsync jitter, no compositor scheduling) while still
  exercising the same kernel binary.
- CI machines typically have an AMDGPU but no display server.

The demo's eventual windowed mode is a thin wrapper on top of the
headless path: replace the readback step with a flip.

## 2. Output Capture

The framebuffer (color + depth) is allocated in VRAM, same as for
the windowed path. After the kernel signals frame completion
(`frame-completion-detection.md` §5), the host issues a VRAM→host
copy of the color buffer and either:

1. Hashes / checksums it and compares to a stored value.
2. Writes it to a PPM/PNG for visual inspection.
3. Walks pixels and applies invariant predicates.

Color buffer at 1080p is 8 MB (BGRA8); copy time ≈ 1 ms over PCIe.
For invariant tests that only inspect a handful of pixels, do a
narrow region copy instead of the full buffer.

```c
// Sketch. The exact API lives behind libkfd's existing buffer
// types; only the framebuffer view is test-specific.
KFD_CHECK(fb_color.copy_to_host(host_pixels, /*region*/ {0, 0, W, H}));
```

## 3. Test Categories

### 3.1 Pixel-Equality Tests (Golden Images)

Render a fixed scene with a deterministic camera, compare the
result to a stored reference image. Useful for regression detection
across kernel changes.

**Caveats**:
- GPU rasterizers are *not* bit-deterministic across drivers and
  hardware revisions. Floating-point edge functions, fma fusion,
  and rounding mode can flip a pixel on a triangle edge.
- Compare with a tolerance: per-channel ε of 1–2 LSB and a max
  pixel-mismatch percentage of ~0.05 %.
- Store the golden image alongside the test binary, not in the
  kernel source tree, so kernel changes that *intentionally* alter
  output don't churn the source tree.

Best applied to **synthetic scenes** the test harness controls
end-to-end (a single triangle, a 4-triangle quad, a few overlapping
quads), not to the full teapot.

### 3.2 Invariant Tests

Inspect the output without a reference image. Examples:

- **Coverage**: render a triangle whose screen-space bounding box
  is exactly tile-aligned; check that every pixel inside the
  triangle is non-background, every pixel outside is background.
- **Z-test**: render two overlapping triangles at different depths;
  check that only the closer triangle's color is present in the
  overlap region.
- **UV interpolation / checker**: render a screen-aligned quad
  spanning the full UV range `(0,0) → (1,1)`; check that the
  resulting framebuffer matches the procedural checker pattern
  exactly. The checker is closed-form (§2.6.1 of bin-ownership)
  so this *is* deterministic.
- **Scissor / clip**: render a triangle that extends off-screen;
  check that no pixel was written outside the framebuffer bounds.

These tests target the rasterizer math, not the binning protocol.

### 3.3 Concurrency Stress Tests

The bin-ownership locks (§2.2 of bin-ownership) are the hardest
part of the pipeline to validate. A pixel-equality test will not
catch a missed wakeup or a rare lock corner case — for example a
ready-ring entry that's pushed but never popped because of an
`in_ready` race, or a renderer that loses primitives across the
empty-drain exit.

Approach: run the pipeline for `S` seconds with a synthetic load
designed to hit lock contention:

- **Hot tile**: all primitives target a single tile, so
  distributors push to that tile's `queue_lock` while one renderer
  holds its `render_lock` every iteration. Verifies that
  distributor pushes and renderer rasterization overlap (§2.5)
  without dropping primitives, and that the `in_ready` clear
  inside `queue_lock` re-publishes the tile to the ready ring
  whenever the renderer's empty drain races a fresh push.
- **Hot host queue**: many small batches at high frequency, so
  the host and GPU contend on `host_queue_lock` constantly.
  Verifies the layered backoff protocol (§2.7).
- **Adversarial sizes**: batch sizes that don't divide evenly into
  the host queue's free space, forcing partial pushes and host-side
  retry loops.
- **Renderer churn**: many renderer WGs popping the same hot
  tile's ID from the ready ring within the narrow re-push window
  (§6.2). Verifies that `try_claim(&render_lock)` failures
  re-push the tile rather than leaking work.

Build the kernel with **debug counters** in fine-grained SVM:
incremented on contended-acquire (per lock kind), on `wg_backoff`
calls per tier, on tile-queue overflow, on `try_claim(&render_lock)`
failures, and on `ready.push`/`ready.pop` returning false. After
the run the host reads the counters and compares to expected
ranges (`hot tile` should show non-zero contended `queue_lock`;
a `cold tile` baseline should show ~zero).

For correctness invariants — *no two renderers ever rasterize the
same tile concurrently*, *every primitive pushed by the host is
counted in `frame_ack[].rendered` exactly once* — the kernel can
write per-tile "in-rasterize" flags and per-frame-id "rasterized"
counters that assert in the renderer that they transition
0→1→0 cleanly. The assertion writes to a global "trap log"
buffer that the host inspects after the run.

### 3.4 Frame-Completion Tests

Submit `F` frames of known primitive count, verify that the
`rendered` counter (`frame-completion-detection.md` §5) reaches
`expected` for every frame and that frames are signalled in order.

Variants:
- **Empty frames**: `expected = 0`. Should signal immediately.
- **One-primitive frames**: smallest unit, exercises the
  per-WG-SGPR accumulator flush.
- **Frame straddling host backoff**: queue is full when the host
  tries to seal `expected`. Verifies that the seal-after-push
  ordering (§5.3 of frame-completion-detection) is respected.

### 3.5 Termination Tests

Set the `terminate` flag (§7.1 of bin-ownership) and verify the
kernel exits within a bounded time. The bound is independent of
the host heartbeat threshold (§7.3 of bin-ownership; that's
hang-detection, not exit-latency): a WG sees `terminate` on the
*next* main-loop iteration and exits as soon as it has no claimed
tile. Worst case is therefore one in-flight rasterize completion
(estimated ~50 μs per coarse tile, §2.5) plus one `wg_backoff` cap
(~2 μs, §2.8). Test target: **≤ 10 ms** of wall time between
`terminate = 1` and dispatch fence signal, with a generous margin
over the per-WG worst case to absorb dispatch-fence-completion
latency. Variants:

- Terminate from idle: no work in flight.
- Terminate mid-frame: half a frame's primitives are queued. The
  kernel should still exit; remaining primitives are dropped (the
  test does not check the framebuffer in this case).
- Oversubscribed launch: `M > N` WGs queued, only `N` resident.
  Late-launched WGs should observe `terminate=1` immediately and
  fast-exit (§7.2).

### 3.6 Microbenchmarks

Not strictly tests, but use the same harness:

- Distribute throughput: synthetic batches with one primitive per
  batch, all targeting one tile. Measures distributor-only path.
- Render throughput: pre-populate tile queues, set
  `host_queue_lock` to "drained", measure render-only steady state.
- Host-overlap: vary the host's per-batch CPU work; measure GPU
  utilization. Verifies that the streaming ring keeps the GPU fed.

Microbenchmarks are run gated (not in the standard `ctest` suite)
because their runtime is variable and they need a quiescent system.

## 4. Test Driver Shape

A single `headless_runner` binary, parameterized:

```
headless_runner --scene=<file>          # primitive list, JSON or binary
                --frames=<N>            # how many frames to render
                --capture=<dir>         # dump framebuffers as PPM
                --golden=<file>         # compare against golden hash
                --duration=<seconds>    # for stress tests
                --counters=<file>       # dump debug counters
                --tile=<size>           # 32 | 64 | 128
                --resolution=<WxH>      # default 1920x1080
                --mode=<stage0|stage1|stage2|stage3>
```

`--mode` selects the bring-up stage from `bin-ownership-pipeline-proposal.md`
§9. The runner has stage-specific init and per-frame loops that
share the kernel binary but differ in how the host feeds it and
how it waits for completion.

### 4.1 Stage 0 (non-persistent, single-shot per frame)

1. Init libkfd (`kfd::Compute`), allocate VRAM framebuffer.
2. Per frame:
   1. Build the entire frame's primitive list in a VRAM upload
      buffer.
   2. Dispatch the (non-persistent) megakernel; wait on the
      dispatch fence.
   3. Optionally read the framebuffer to host, hash, or dump.

No fine-grained-SVM ring, no `terminate` flag, no `frame_ack`
counters. This is the smoke runner for stages 0.x and validates
the bin-ownership locks (§3.1, §3.2 above).

### 4.2 Stage 1+ (persistent, with `terminate` and per-frame ack)

1. Init libkfd, allocate fine-grained SVM regions (host queue
   batch ring, primitive store, locks, `frame_ack` slots,
   counters), allocate VRAM framebuffer.
2. Load the kernel binary, dispatch the persistent megakernel.
3. Per frame:
   1. Build batches.
   2. Push via the host protocol (§2.7 of bin-ownership for
      stage 2+; a simple `memcpy` upload for stage 1).
   3. Wait for completion (§5 of frame-completion-detection;
      polling for stage 1–2, KFD-signal interrupt for stage 3).
   4. Optionally read the framebuffer to host, hash, or dump.
4. Set `terminate=1`, wait for the kernel to drain, validate
   counters and trap log.

Tests are thin wrappers around the runner: a scene file, expected
hash, expected counter ranges, mode flag. New tests don't need new
binaries.

## 5. Determinism and Tolerance

What is deterministic across runs on the same GPU:
- Procedural checker shading (§2.6.1 of bin-ownership), given
  identical interpolated UVs.
- Triangle setup math, *if* the host produces bit-identical inputs.
- Frame ordering (frame N completes before frame N+1).

What is **not** deterministic:
- Pixel-exact output across kernel changes that touch the
  rasterizer's edge functions, depth interpolation, or fma usage.
- Cross-GPU comparison (GFX9 vs RDNA may produce different LSBs).
- Lock contention timing (debug counters are *order-of-magnitude*
  invariants, not exact values).

Tests pick the right tolerance per category:
- Pixel-equality: hash with channel-LSB tolerance, mismatch ≤ 0.05 %.
- Invariant: exact (the invariants are deterministic by
  construction).
- Concurrency: ranges, not exact values.

## 6. CI Integration

Requirements:
- AMDGPU + KFD-capable kernel (CI runner has the GPU).
- No display server needed; the runner uses `kfd::Compute` only.
- Set a short watchdog heartbeat threshold (`--heartbeat-ms=50`)
  so a hang fails fast instead of waiting for the OS watchdog.
- Trap log dumped on test failure for post-mortem.

Per-PR test budget:
- Smoke + invariants: < 5 s total. Run on every PR.
- Concurrency stress: 30 s. Run on every PR.
- Frame-completion: < 5 s. Run on every PR.
- Microbenchmarks: opt-in label. Run on perf-affecting PRs.

## 7. libkfd Extensions Required

Most of what's needed already exists in libkfd or is listed in
`bin-ownership-pipeline-proposal.md` §7.6:

- Async dispatch + persistent kernel launch (already in §7.6).
- Fine-grained SVM allocation (already in §7.6).
- Helper PM4 queue with `WAIT_REG_MEM` + `RELEASE_MEM` (already
  in §7.6, via `frame-completion-detection.md` §5.8).
- VRAM→host copy with explicit fence (probably exists via
  `kfd::ComputeQueue::copy_to_host` or equivalent; verify).

Strictly *new* for headless tests:
- A lightweight "trap log" buffer convention: ring of
  `(wg_id, wave_id, code, payload)` entries the kernel writes on
  invariant violation. No new libkfd API; just a documented
  pattern using fine-grained SVM.

## 8. Reading

- `bin-ownership-pipeline-proposal.md` §7.6 — libkfd
  extensions consumed by both the runtime and the test harness.
- `frame-completion-detection.md` §5 — frame-end detection
  used by the per-frame loop.
- `tools/computetoy/` — existing dma-buf swapchain example;
  contrast with the headless runner.
