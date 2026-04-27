# Frame Completion Detection

How does the host know that all primitives of frame F have actually
been rasterized into the framebuffer, so it can flip / present?

This is an underdefined area in `bin-ownership-pipeline-proposal.md`.
That proposal assumes a persistent megakernel that lives across many
frames; the kernel does not exit at frame boundaries, so there is no
`cudaDeviceSynchronize`- or fence-on-dispatch-complete equivalent.
Frame completion has to be detected *inside* the live pipeline and
*signalled out* to the host.

This document:

1. Reviews how cuRE solved the same problem (§1).
2. Explains why we cannot just copy cuRE (§2).
3. Enumerates variants (§3) with trade-offs.
4. Picks a recommendation (§4) and shows how it integrates with the
   bin-ownership pipeline (§5).

## 1. What cuRE Does

cuRE's answer is the simplest possible one: **one megakernel
invocation per draw call**. The persistence is across pipeline stages
within a single draw, not across draws.

From `source/cure/PipelineKernel.cpp`:

```cpp
void PipelineKernel::prepare() const
{
  preparenewprimitive({ 1U, 1U, 1U }, { 1024U, 1U, 1U }, 0U,
                      nullptr, rasterizer_count);
}

int PipelineKernel::launch() const
{
  kernel({ static_cast<unsigned>(rasterizer_count), 1U, 1U },
         { warps_per_block * 32U, 1U, 1U }, 0U, nullptr);
  return rasterizer_count;
}
```

`prepare()` initializes `geometryProducingBlocksCount = rasterizer_count`
in the device-side global. `launch()` dispatches the megakernel.

The megakernel's main loop (from `source/cure/pipeline/Pipeline.cuh`):

```cpp
extern "C" __device__ int geometryProducingBlocksCount;

while (runstate[0] || runstate[1])
{
  if (runstate[0] && !runstate[4]) {
    if (!GeometryStage::run(...)) {            // input exhausted
      atomicSub(&geometryProducingBlocksCount, 1);
      runstate[0] = false;                     // this WG done with geometry
    }
  }
  // try to acquire a rasterizer; spin-wait while
  // geometryProducingBlocksCount != 0 OR rasterizers still have work
  ...
}
```

What happens:

1. Each WG starts with a static partition of the input vertex range to
   process.
2. As a WG exhausts its input, it decrements
   `geometryProducingBlocksCount` (from `rasterizer_count` to 0).
3. The kernel keeps running while either some WG is still producing
   geometry OR some WG still owns a rasterizer with non-empty queues.
4. When both conditions are false, the loop exits and the kernel
   returns.
5. The host's `cuLaunchKernel` (or its async equivalent) completes
   when the kernel exits.

So the "frame complete" signal is **kernel completion itself**,
detected by the CUDA driver at the dispatch level. There is no
in-kernel ack needed because the kernel is the frame.

The `ProgressQueue` watermark (described in `cure-streaming-queues.md`
§3.2) is a *separate* mechanism for *primitive ordering* within a
draw call (so that a downstream stage can preserve API draw order
without serializing producers). It is not an end-of-frame signal. It
fires per primitive completion of an upstream stage, not per frame.

### 1.1 Implications

- The kernel runs for ~ms-scale wall time per frame (one whole frame).
- All work is bounded by the input range known at launch.
- A new frame = a new kernel launch.
- The host cannot stream new primitives into a running cuRE kernel.

This is the opposite of what we want for libkfd. Our streaming /
bin-ownership designs explicitly want host-streamable primitives into
a long-lived megakernel.

## 2. Why We Cannot Just Copy cuRE

Both libkfd proposals make the megakernel persistent across frames so
that:

- Per-frame kernel re-launch overhead is paid once per app run, not
  per frame.
- LDS / register state is preserved across frames (e.g., HiZ mip
  caches, tile-resident state).
- Host can stream sub-frame work without waiting for prior frames.
- The host queue (§2.2.1 of bin-ownership proposal) is the work
  source, not a static input range; the WGs do not know how much
  input there will be ahead of time.

Consequences:

- `geometryProducingBlocksCount`-style "I'm done with my static
  partition" does not apply: WGs do not have a static partition. They
  pull from a shared queue.
- Kernel exit ≠ frame end. Kernel exit happens only at app shutdown
  via the `terminate` flag.
- The host has no `cuLaunchKernel`-completion signal to wait on.

We need an in-kernel mechanism that:

- Fires once per frame (not per primitive, not per draw call).
- Tells the host "every primitive you submitted for frame F has
  reached the framebuffer".
- Survives WGs that never see a primitive of F (a tile not touched by
  F should not block the signal).
- Survives WGs that take vastly different times (some tiles trivial,
  some hot).
- Adds at most a few atomics per frame, not per primitive.

## 3. Variants

Notation:

- `F` = frame id.
- `N_F` = total primitive count submitted for frame F (host-known).
- `B` = number of coarse bins / tiles.

### 3.1 V1: Per-Primitive Ack

```c
// SVM
_Atomic uint32_t rendered_count[NUM_INFLIGHT_FRAMES];

// GPU renderer, after rasterizing one primitive of frame F:
atomic_fetch_add_explicit(&rendered_count[F % NUM_INFLIGHT_FRAMES], 1,
                          memory_order_release);

// Host
host_set(&expected_count[F % NUM_INFLIGHT_FRAMES], N_F);
while (atomic_load(&rendered_count[F % NUM_INFLIGHT_FRAMES]) < N_F)
  cpu_relax();
flip();
```

Cost: one atomic per primitive. At 100k prims/frame this is 100k
PCIe-coherent atomics on a single hot location → ~100 ms at PCIe
round-trip rates. **Disqualified for performance.**

Mitigations: shard the counter (`rendered_count[F][shard]`), reduce
on host. Even sharded, it's per-primitive overhead with no
amortization. Skip.

### 3.2 V2: Bulk Per-Frame Counter (Drain-Granularity Ack)

Instead of one atomic per primitive, one atomic per *tile drain*. A
renderer drains a tile queue into LDS, then rasterizes K primitives.
Group those K by frame_id; emit one `atomicAdd` per frame group:

```c
struct FrameAck {
  _Atomic uint32_t expected;   // host writes after pushing all of F
  _Atomic uint32_t rendered;   // GPU bulk-increments
  uint32_t         frame_id;   // sanity check (host-set)
};

FrameAck frame_ack[NUM_INFLIGHT_FRAMES];   // fine-grained SVM

// Distributor side: each pushed primitive carries frame_id F.
// Tile queue entry: { prim_id, frame_id }.

// Renderer, after rasterizing K_F primitives of frame F from a single
// drained batch:
atomic_fetch_add(&frame_ack[F % N_F_INFLIGHT].rendered, K_F);

// Host, after pushing all N_F primitives of F:
atomic_store(&frame_ack[F % N_F_INFLIGHT].expected, N_F);
while (atomic_load(&frame_ack[F % N_F_INFLIGHT].rendered) < N_F)
  cpu_relax_or_yield();
flip();
```

Cost: one atomic per tile drain × number of frames mixed in that
drain. For the demo workload (rotating teapot, ~6 K triangles per
frame) each drain mixes ~1 frame, so 1 atomic per drain ≈ B
drains/frame ≈ a few hundred atomics/frame on a single hot location.
~hundreds of μs of contention if all WGs hit the same counter — still
acceptable but borderline.

Mitigations:

- **Per-WG accumulator**: each WG accumulates `K_F` per frame in a
  per-WG SGPR register; flushes to global `rendered` only when the WG
  changes frame focus or when accumulator exceeds a threshold (e.g.,
  256). Reduces atomic count by ~256×.
- **Per-WG accumulator + watermark flush**: WG observes `expected`
  set, then flushes its accumulator. This guarantees the host's wait
  loop terminates.

Storage: `NUM_INFLIGHT_FRAMES × 16 B = 64 B` for 4 in-flight frames.
Negligible.

### 3.3 V3: EOF Marker In Host Queue

Host pushes a special "end-of-frame F" batch as the last item for F:

```c
struct StreamBatch {
  uint8_t  kind;          // PRIM_BATCH or EOF_F
  uint8_t  frame_id;
  uint16_t prim_count;
  uint32_t prim_ids[];
};

// Distributor pop:
if (batch.kind == EOF_F) {
  // Forward to all tile queues? Or just signal?
  ...
}
```

The marker tells the *distributor* "no more host-queue prims for F".
But this does not say anything about *rasterization* completion. A
prim popped at t=0 may still be rasterizing at t=10 ms; an EOF
popped at t=1 ms says nothing about that.

So EOF-in-host-queue alone is insufficient. It needs to be combined
with one of:

- A counter (V2 atop EOF: EOF triggers the host to set
  `expected = N_F` and start polling).
- An EOF-propagation through tile queues (V4).

EOF-as-trigger-for-V2 is exactly what V2 does already in a different
shape — the marker is just `atomic_store(&expected, N_F)`. The
sentinel marker buys nothing extra. **Subsumed by V2.**

### 3.4 V4: EOF Propagated To Every Tile Queue

The distributor, on popping an EOF F batch, pushes an EOF F sentinel
into every one of the B tile queues. Each tile queue is
FIFO-ordered, so a renderer that drains the tile sees EOF F only
*after* all prims of F that targeted this tile.

```c
// Distributor on EOF F:
for (uint32_t t = 0; t < B; ++t) {
  // claim TILE_QLOCK on tile t, push EOF F entry, release
}

// Renderer drain:
while (drain_one(&entry)) {
  if (entry.kind == EOF_F) {
    // This tile is fully rasterized for F.
    atomic_fetch_add(&tile_eof_acks[F % N_F_INFLIGHT], 1);
    continue;
  }
  rasterize(entry);
}

// Host:
while (atomic_load(&tile_eof_acks[F % N_F_INFLIGHT]) < B)
  cpu_relax();
```

Cost:

- B tile-queue pushes per frame just for EOF (≈ 8160 at 1080p / 32×32).
  Each push is a `TILE_QLOCK` acquire/release plus 8–16 B of queue
  data. At ~100 ns per push that's ~800 μs of distribute-mode work
  per frame — too much.
- B atomic increments at frame end (one per tile renderer that sees
  EOF). Same cost as ~1 atomic per drain in V2 but with more
  contention (all renderers ack at once at frame end).
- Tile queue capacity must accommodate the EOF entry without
  overflowing.

Improvements:

- Push EOF only to tiles that received at least one prim of F. Per
  frame the distributor maintains a `frame_touched[B/32]` bitmap;
  EOF push iterates only set bits.
- Use a per-tile `tile_last_frame` field instead of a queue entry:
  distributor writes `tile_last_frame[t] = F` on every push to tile
  t. Renderer reads it after drain; if `tile_last_frame[t] == F` and
  queue empty, this tile is ack'd for F. (This is V5.)

Pure V4 is too expensive due to EOF push amplification. V4 with
touch bitmap is about as cheap as V2 but more complex.

### 3.5 V5: Per-Tile Frame Watermark

```c
uint32_t          tile_last_pushed_frame[B];   // distributor write
_Atomic uint32_t  tile_last_drained_frame[B];  // renderer write

// Distributor, when pushing a prim of frame F to tile t:
tile_last_pushed_frame[t] = max(tile_last_pushed_frame[t], F);

// Renderer, after fully rasterizing a drain batch from tile t whose
// max prim frame_id was F_drained:
atomic_fetch_max(&tile_last_drained_frame[t], F_drained);

// Host, to test "frame F complete":
//   for every tile t: tile_last_drained_frame[t] >= F  OR  t was
//   never touched by F.
```

The "never touched by F" branch is the hard part: the host does not
know which tiles F touched without GPU help. Two options:

- **GPU-side touch tracking**: distributor sets a bit in
  `frame_touched[F % N_F_INFLIGHT][B/32]` when pushing a prim of F to
  tile t. Host iterates set bits; for each set bit, polls
  `tile_last_drained_frame[t] >= F`. Storage: `N_F_INFLIGHT ×
  B / 8` bytes — ~4 KB at 1080p / 32×32 / 4 frames.
- **Conservative all-tiles**: host treats all B tiles as touched.
  This converts to: `min(tile_last_drained_frame[]) >= F`. Wrong for
  scenes where some tiles never receive a prim — they will never tick
  forward and the wait loop never exits.

The conservative-all variant has a hard bug. The touch-bitmap
variant works but is V6 essentially.

Cost (touch-bitmap variant):

- Distributor: 1 atomic-or per pushed prim per tile (already paying
  one atomic for the queue push; one more bit-set on average).
- Renderer: 1 `atomicMax` per drain (cheap; one cache line).
- Host: scans B / 32 dwords (~256 dwords at 1080p / 32×32) at flip
  time and polls. Cheap.

Compared to V2 + per-WG accumulator: V5 is harder to make per-frame
clear (the bitmap must be reset before a frame can reuse the slot)
but more diagnostically rich (you can tell *which* tile is
straggling).

### 3.6 V6: Touch-Bitmap (V4/V5 Hybrid)

Already described as part of V4 and V5. As a standalone proposal:

```c
// SVM
_Atomic uint64_t frame_touched[N_F_INFLIGHT][B / 64];  // bitmap of touched tiles
_Atomic uint64_t frame_drained[N_F_INFLIGHT][B / 64];  // bitmap of drained tiles

// Distributor, pushing prim of F to tile t:
atomic_fetch_or(&frame_touched[F % N_F_INFLIGHT][t / 64], 1ULL << (t % 64));

// Renderer, draining a tile t whose queue's last entry's frame was F_max:
// (after rasterize, when this tile's queue is empty for frames < F_max)
atomic_fetch_or(&frame_drained[F_max % N_F_INFLIGHT][t / 64], 1ULL << (t % 64));

// Host: frame F complete when frame_touched[F] == frame_drained[F] for all dwords.
```

Subtle: a tile may be touched multiple times across frames, so the
"drained for F" bit must mean "queue empty *and* last drained prim
had frame ≥ F". This is handled by combining with V5's
`tile_last_drained_frame` and only setting the drained bit when the
tile's watermark crosses F.

Storage: 2 × N_F_INFLIGHT × B / 8 bytes = ~2 KB at 1080p / 32×32 / 4
frames. Negligible.

Pro: precise, per-tile diagnostics, no per-prim atomics on a hot
location.

Con: more state to maintain, more atomic ops at distribute and drain
time, host-side check is O(B / 64) bitmap compare instead of one
counter compare. Implementation complexity higher than V2.

### 3.7 V7: Quiescence Detection

Host writes "I want F done", every WG checks every iteration: "is the
host queue empty AND every tile queue empty AND is no WG currently
rendering?". The first WG to observe quiescence sets a "frame done"
flag.

Problems:

- "Every tile queue empty" is O(B) reads per check. Hot.
- "No WG currently rendering" requires a global active-WG counter.
- False positives: queues can be transiently empty while a renderer
  is mid-rasterize.
- Synchronizing the check across B tile queues without barrier
  primitives is awkward.

Sound versions of this are essentially counter-based (V2/V6) wearing
a different costume. Not pursued.

### 3.8 V8: Fence-On-Dispatch (Inapplicable)

KFD/HSA fences fire on dispatch packet completion. In a persistent
megakernel the dispatch never completes until app shutdown, so this
is nothing more than a shutdown signal. Inapplicable to per-frame
detection.

The bin-ownership proposal's §7.1 `terminate` flag is in this
family: it flips for app exit, not for per-frame completion.

## 4. Recommendation

**V2 (bulk per-frame counter) with per-WG accumulator** as the
baseline, optionally extended to V6 (touch-bitmap) if per-tile
diagnostics or partial-frame presentation are needed.

Rationale:

- Cost: ~B atomics on a single 64-bit counter per frame, with per-WG
  accumulation reducing this by ~10× to a few dozen atomics per
  frame.
- Storage: ~64 B per in-flight frame.
- No protocol change to tile queues (they keep the same entry
  format; one extra `frame_id` byte per entry).
- Single host-side wait condition: `rendered >= expected`.
- Trivial to add a frame-id sanity check.
- Easy to migrate to V6 later: V2 is a strict subset of the
  bookkeeping V6 needs.

V6 buys two things V2 does not:

- Tells the host *which* tile is straggling (debug aid).
- Permits an "early flip" optimization where a partial frame can be
  presented if some tiles are known done.

Neither is required for a first implementation. Defer V6 to a later
optimization stage.

## 5. Integration With The Bin-Ownership Pipeline

This section maps V2 onto the bin-ownership pipeline as currently
specified in `bin-ownership-pipeline-proposal.md`.

### 5.1 New State

```c
// fine-grained SVM, host+GPU shared
struct FrameAck {
  _Atomic uint32_t expected;   // host: total prims pushed for this frame
  _Atomic uint32_t rendered;   // GPU: bulk-incremented per drain
  uint32_t         frame_id;   // sanity (host writes once per use)
  uint32_t         _pad;       // cache-line padding
};

FrameAck frame_ack[NUM_INFLIGHT_FRAMES];   // typically 2–4

#define FRAME_SLOT(F) ((F) % NUM_INFLIGHT_FRAMES)
```

Sized at 4 in-flight frames: 256 B. Cache-line aligned to avoid
false sharing between slots.

Reuse rule: slot `i` may hold frame F only if the previous user of
slot `i` (frame F − NUM_INFLIGHT_FRAMES) is fully complete (i.e., its
host wait returned). The host enforces this by not starting frame F
until frame F − NUM_INFLIGHT_FRAMES has been flipped.

### 5.2 Tile Queue Entry Format Change

```c
// before
struct TileEntry { uint32_t prim_id; };

// after
struct TileEntry { uint32_t prim_id; uint8_t frame_id; uint8_t _pad[3]; };
```

`frame_id` is the low 8 bits of F. NUM_INFLIGHT_FRAMES ≤ 4 makes 8
bits redundantly safe for hundreds of frames between wraps. The
extra 4 B per entry adds to tile queue memory (§6.1 of bin-ownership
proposal) — multiplies the per-entry size from 4 to 8 B. At C = 256
this is 1 KB → 2 KB per tile, 16 MB → 32 MB total at 1080p / 32×32.
Acceptable but worth noting in the memory budget.

Alternative: pack `frame_id` into the high bits of `prim_id` (24-bit
prim id × 8-bit frame id). Avoids the size doubling at the cost of
prim-id range. With ≤ 16 M prims/frame this is fine for any realistic
scene.

### 5.3 Host Side (Polling Wait)

Per `bin-ownership-pipeline-proposal.md` §2.7 the host pushes batches
under `host_queue_lock`. The frame-complete machinery layers on top.
This subsection shows the polling host wait; §5.8 describes an
interrupt-driven alternative that uses a KFD `Signal`.

```c
void host_render_frame(uint32_t F, StreamBatch *batches, size_t n_batches) {
  uint32_t slot = FRAME_SLOT(F);

  // Ensure prior occupant of this slot is done (back-pressure).
  while (atomic_load(&frame_ack[slot].rendered) <
         atomic_load(&frame_ack[slot].expected)) {
    cpu_relax_or_yield();
  }

  // Reset the slot for F.
  atomic_store(&frame_ack[slot].rendered, 0);
  atomic_store(&frame_ack[slot].expected, 0);  // 0 = "not yet sealed"
  frame_ack[slot].frame_id = F;
  atomic_thread_fence(memory_order_release);

  // Push everything (each batch tagged with F at compose time).
  size_t total_prims = 0;
  for (size_t i = 0; i < n_batches; ++i) {
    host_push_n(&batches[i], 1);              // §2.7 of bin-ownership
    total_prims += batches[i].prim_count;
  }

  // Seal: tell the GPU the final count.
  atomic_store(&frame_ack[slot].expected, total_prims);

  // Wait for completion.
  while (atomic_load(&frame_ack[slot].rendered) < total_prims) {
    cpu_relax_or_yield();                     // same backoff ladder
  }

  flip();                                     // dma-buf present
}
```

Note that `expected = 0` during push is intentional: until sealed,
the wait predicate `rendered < expected` is trivially false, so any
GPU bulk-increment that lands before the seal is harmless — it just
advances `rendered` further than `expected = 0`. The wait loop only
becomes meaningful after the seal.

### 5.4 GPU Distributor Side

`distribute_batch()` per `bin-ownership-pipeline-proposal.md` §2.4
already reads each prim out of the popped batch and pushes it to a
tile queue. The only change is to forward `batch.frame_id` into each
`TileEntry`:

```c
// in distribute_batch, phase 1 (LDS bucketing):
for (lane in batch.prim_ids) {
  uint32_t tile = aabb_to_tile(prim);
  uint32_t slot = atomicAdd(&lds_bucket_count[tile], 1);
  lds_bucket[tile][slot] = (TileEntry){
    .prim_id  = prim,
    .frame_id = batch.frame_id,
  };
}
// phase 2 push to tile queues unchanged.
```

No new atomics. No new failure modes.

### 5.5 GPU Renderer Side

`render(tile)` per `bin-ownership-pipeline-proposal.md` §2.5 has
three phases: drain into LDS, transition lock, rasterize. After
phase 3 (rasterize complete) we add the bulk ack:

```c
__device__ void render(uint32_t tile) {
  __shared__ uint32_t lds_count;
  __shared__ TileEntry lds_entries[MAX_TILE_PRIMS];

  drain_tile_queue(tile, lds_entries, &lds_count);          // phase 1
  render_drain_done(tile);                                   // phase 2

  rasterize_tile(tile, lds_entries, lds_count);              // phase 3

  // Phase 4: ack frame counts.
  // Most drains are single-frame; the per-WG accumulator flushes the
  // common case in zero atomics-per-prim.
  ack_drain(lds_entries, lds_count);

  render_release(tile);
}
```

```c
// per-WG, in registers / SGPRs
struct WgAcc {
  uint8_t  frame_id;          // -1 = empty
  uint32_t count;
};

WgAcc wg_acc;

__device__ void ack_drain(TileEntry *entries, uint32_t n) {
  // Fast path: all entries share frame_id.
  if (n == 0) return;
  uint8_t F0 = entries[0].frame_id;
  bool same = true;
  for (uint32_t i = 1; i < n; ++i) {
    if (entries[i].frame_id != F0) { same = false; break; }
  }
  if (same) {
    if (wg_acc.frame_id == F0) {
      wg_acc.count += n;
    } else {
      flush_acc();              // emit one atomicAdd to global
      wg_acc.frame_id = F0;
      wg_acc.count = n;
    }
    if (wg_acc.count >= ACC_FLUSH_THRESHOLD) flush_acc();
    return;
  }
  // Slow path: multi-frame drain (rare).
  flush_acc();                  // empty the accumulator first
  for (uint32_t i = 0; i < n; ++i) {
    uint8_t F = entries[i].frame_id;
    if (wg_acc.frame_id == F) {
      ++wg_acc.count;
    } else {
      flush_acc();
      wg_acc.frame_id = F;
      wg_acc.count = 1;
    }
  }
  flush_acc();
}

__device__ void flush_acc(void) {
  if (wg_acc.count == 0) return;
  atomic_fetch_add(&frame_ack[FRAME_SLOT(wg_acc.frame_id)].rendered,
                   wg_acc.count);
  wg_acc.count = 0;
}
```

WG_ACC_FLUSH_THRESHOLD ≈ 256 keeps host-visible advance fine-grained
enough that the host wait loop sees timely progress (worst case the
host waits for one full batch's worth of accumulator ≈ tens of μs).

### 5.6 Periodic Forced Flush

The accumulator only flushes when it overflows or when a different
frame is observed. A WG that drained one tile of frame F, then never
sees another prim of F, will hold its accumulator forever. Two
defenses:

- **End-of-iteration flush on observed seal**: on each main-loop
  iteration the WG checks whether `expected[F % N_F_INFLIGHT]` was
  set; if yes and `wg_acc.frame_id == F`, flush. This guarantees the
  host wait loop terminates.
- **Heartbeat-piggybacked flush**: if a WG has had no productive
  iteration in K consecutive iterations (the same condition that
  drives §2.8 GPU backoff), flush. This caps tail latency on quiet
  WGs.

The seal-observed flush is mandatory; the heartbeat flush is a tail-
latency optimization.

### 5.7 Cost Summary

Per frame at demo-shape (P ≈ 6 K triangles, B ≈ 8 K tiles at 1080p /
16×16, ~hundreds of drains):

| Site | Atomics | Memory | Notes |
|---|---|---|---|
| Host seal | 1 store on `expected` | 0 | per frame |
| Host wait | ~10 loads on `rendered` | 0 | until ack |
| Distributor | 0 new | +4 B/entry or shared in prim_id | tag passthrough only |
| Renderer ack | ~hundreds of atomicAdds | 0 | bulk; SGPR accumulator amortizes per-prim |
| Frame slot reset | 2 atomic stores | 0 | per frame |

Total new GPU-host PCIe atomic traffic per frame: a few hundred
`atomicAdd` to a single hot 64-bit location. At ~1 μs PCIe atomic
latency this adds up to ~0.5–1 ms of *aggregate* atomic cost across
WGs running in parallel — but per-WG it's amortized by the
accumulator, so wallclock cost is dominated by atomic *throughput*
which AMD GFX10/11 can sustain at a few hundred Mops/s on L2-coherent
traffic. **Host-visible wallclock cost: < 100 μs/frame.**

### 5.8 Interrupt-Driven Host Wait via KFD Signal

The §5.3 wait loop pins one host CPU thread spinning on
`frame_ack[slot].rendered`. This is fine for short frames at high fps
but wastes CPU when frames are long (≥ 1 ms) or when the host has
other work it could do. AMDGPU's KFD interface supports interrupt-
driven wakeup; libkfd already wraps it.

#### Primitives

`include/libkfd/event.h` exposes `kfd::Event`:

> "At context initialization we register an event page with the
> kernel. Each event is assigned a slot index. To signal an event,
> the GPU writes any value other than ~0 to that slot and fires an
> interrupt. The interrupt handler will then wake any event waiting
> on that index."

`include/libkfd/signal.h` bundles a `kfd::Event` with a 64-bit GPU-
writable fence value and offers `Signal::wait(cond, value, timeout_ns,
spin_ns)` that "spins first then falls back to an interrupt-driven
wait" — exactly the polling-then-block pattern we want for the host.

`ComputeQueue::signal(Signal&)` (`include/libkfd/queue.h:185`) and
`ComputeQueue::wait_reg_mem` (`:228`) submit the PM4 packets that
make this work end-to-end.

#### The persistent-megakernel wrinkle

The standard `ComputeQueue::dispatch(kernel, cfg, kernarg, signal)`
overload queues a `RELEASE_MEM` packet *after* the dispatch on the
same queue. The CP fires the signal at end-of-pipe, i.e., after the
dispatch retires. With a persistent megakernel that retires only at
app shutdown, **a same-queue RELEASE_MEM never fires until shutdown**.

The fix is to put the signaling on a **separate compute queue** that
runs only PM4 packets and no dispatch. Its CP can independently
process `WAIT_REG_MEM + RELEASE_MEM` while the megakernel is
running on the first queue.

```text
Compute queue Q1:           [dispatch persistent megakernel; runs forever]
Compute queue Q2 (helper):  WAIT_REG_MEM(&rendered, GTE, N_F)
                            RELEASE_MEM(INT_SEL=2, signal)
                            // host appends another pair per next frame
```

Q2's CP stalls on the WAIT_REG_MEM until the megakernel's bulk-ack
atomics push `rendered` past `N_F`, then the RELEASE_MEM fires the
KFD interrupt, which wakes the host's `Signal::wait()`.

#### Per-frame protocol

```c
// At app init, once.
auto helper_q = ctx.create_compute_queue(...);                 // Q2
kfd::Signal frame_signal[NUM_INFLIGHT_FRAMES];
for (auto &s : frame_signal) s = kfd::Signal::create(ctx, /*initial*/ 1).value();

void host_render_frame_blocking(uint32_t F,
                                StreamBatch *batches, size_t n_batches) {
  uint32_t slot = FRAME_SLOT(F);

  // Wait for prior occupant (now interrupt-driven, not spin).
  KFD_CHECK(frame_signal[slot].wait(Condition::EQ, 0,
                                    /*timeout_ns*/ UINT64_MAX,
                                    /*spin_ns*/ 100'000));
  KFD_CHECK(frame_signal[slot].reset(/*value*/ 1));

  // Same setup as §5.3.
  atomic_store(&frame_ack[slot].rendered, 0);
  atomic_store(&frame_ack[slot].expected, 0);
  frame_ack[slot].frame_id = F;
  atomic_thread_fence(memory_order_release);

  size_t total_prims = 0;
  for (size_t i = 0; i < n_batches; ++i) {
    host_push_n(&batches[i], 1);
    total_prims += batches[i].prim_count;
  }
  atomic_store(&frame_ack[slot].expected, total_prims);  // GPU seal flush

  // Build the per-frame WAIT_REG_MEM + RELEASE_MEM pair on the helper
  // queue. WAIT_REG_MEM stalls Q2's CP until rendered >= total_prims;
  // RELEASE_MEM then decrements the fence and fires the event.
  KFD_CHECK(helper_q.wait_reg_mem(&frame_ack[slot].rendered,
                                  Condition::GTE,
                                  static_cast<uint32_t>(total_prims)));
  KFD_CHECK(helper_q.signal(frame_signal[slot]));

  // Block until the interrupt fires. spin_ns lets us catch sub-ms
  // frames without a syscall round-trip.
  KFD_CHECK(frame_signal[slot].wait(Condition::EQ, 0,
                                    /*timeout_ns*/ 100'000'000,
                                    /*spin_ns*/ 100'000));
  flip();
}
```

#### What changes vs §5.3

| Concern | §5.3 polling | §5.8 interrupt |
|---|---|---|
| Host CPU during wait | ~1 thread pinned at 100% | ~0 (kernel-blocked) |
| Wakeup latency, sub-ms frame | < 1 μs | spin window catches it (~100 μs) |
| Wakeup latency, multi-ms frame | poll cadence | interrupt + ctx switch (~10 μs) |
| Per-frame host work | 0 packets | 2 PM4 packets (~32 B) submitted |
| KFD events used | 0 | NUM_INFLIGHT_FRAMES |
| Extra GPU resources | 0 | 1 helper compute queue |
| Fits if host has other work | poorly | well |

#### What stays unchanged

- Tile queue entry format (§5.2) — still carries `frame_id`.
- Distributor side (§5.4) — passes `frame_id` through to tile entries.
- Renderer side (§5.5) — same `ack_drain` + per-WG accumulator. The
  GPU writes to `frame_ack[slot].rendered` exactly the same way; the
  helper queue just *observes* the writes and signals.
- §5.6 seal-observed flush — still needed. The GPU does not see the
  WAIT_REG_MEM threshold; it reacts to `expected` being set by the
  host. Without seal-observed flush, a quiet WG holding K prims of F
  in its accumulator can stall the helper queue's WAIT_REG_MEM
  forever (host's wait would then time out at 100 ms above).

#### Caveats

1. **KFD events are finite per process** — typical limit ~4096.
   Reusing one `Signal` per swapchain slot via `reset()` keeps usage
   bounded at NUM_INFLIGHT_FRAMES.
2. **WAIT_REG_MEM reference is encoded into the packet.** Each frame
   needs a fresh packet pair; the threshold `N_F` is baked in. Cost
   is negligible (~32 B + one `submit()` call per frame).
3. **Helper queue scheduling.** Q2 does no compute; its CP needs to
   be admitted to a CU's scheduler but uses ~no compute resources
   beyond ring buffer storage. SDMA queues could in principle run
   the same packets — review `lib/queue.cpp` and `packets/sdma.h`
   for support on the target gen if compute-queue contention proves
   problematic.
4. **The `expected` field is no longer the host wait predicate**, but
   it is still the **GPU seal-flush trigger** (§5.6). Keep it.
5. **Spin window tuning.** `Signal::wait(..., spin_ns=100'000)` keeps
   ~100 μs of CPU busy-wait before the syscall fallback. Tune up
   (sub-ms frames common) or down (long frames, host has other
   work).
6. **Interrupt routing for sendmsg.** Compute kernels can
   `s_sendmsg sendmsg(MSG_INTERRUPT)` directly — libkfd's trap
   handler does this in `lib/device/trap_handler.S:450` to deliver
   exception payloads. KFD only routes those interrupts back to user
   space through the trap handler's CWSR header path
   (`include/libkfd/abi.h:297`). It is **not** a generic "wake any
   KFD event from compute kernel" mechanism; that is what RELEASE_MEM
   is for.

#### When to use which

- **Polling (§5.3)** for early bring-up, simplicity, and at high fps
  where the host has no other work.
- **Interrupt-driven (§5.8)** when host CPU usage matters (laptop /
  battery), when frames are long (debug builds, large scenes), or
  when the host needs the CPU for game logic in parallel with the
  GPU. Same on-GPU machinery, just a different host wait shape.

The two are not mutually exclusive: ship §5.3 first, swap in §5.8
when measurements justify the helper-queue plumbing.

## 6. Open Questions

1. **NUM_INFLIGHT_FRAMES policy.** 2 is the minimum for double-
   buffered presentation; 3–4 lets the host stay slightly ahead of
   the GPU. Demo presentation can use the simpler dma-buf swapchain
   already exercised by `tools/computetoy` (depth 2). Revisit if the
   demo gains a triple-buffered present path.
2. **Initial frame**. F = 0 is a special case: no prior occupant of
   the slot, so the back-pressure check at §5.3 must not block on
   uninitialized state. Initialize all slots with
   `expected = rendered = 0` at megakernel startup; the predicate
   `0 < 0` is false, so the loop falls through. Verify.
3. **Aborted frames.** If the host decides mid-frame to abort frame
   F (e.g., resize), the GPU may still have prims of F in queues. We
   either drain them (cheap; ~ms) or set a per-frame `aborted` flag
   that the renderer checks before rasterizing. The latter saves ms
   of work for large frames; the former is simpler. Defer.
4. **Frame-mixing within a tile drain.** §5.5's slow path assumes
   rare multi-frame drains. With NUM_INFLIGHT_FRAMES = 4 and the
   overlap-mode scheduler, this can in principle happen often if F
   and F+1 push prims to the same tile in close succession. Measure
   before deciding whether to optimize.
5. **Memory ordering on `expected` seal vs `rendered` increment.**
   `rendered` updates must not be reordered after the host's
   `expected` read. `memory_order_release` on the GPU's
   `atomic_fetch_add` and `memory_order_acquire` on the host's
   `atomic_load(&rendered)` is sufficient for the wait predicate.
   The seal write needs `memory_order_release` paired with the host's
   acquire-load of `expected` from the GPU side (none — only the host
   reads `expected`). So just `memory_order_relaxed` may suffice for
   the seal store; `release` is conservative and free.
6. **Wait shape: polling vs interrupt-driven.** §5.3 polls
   `frame_ack[slot].rendered` from the host; §5.8 attaches a helper
   PM4 queue that fires a KFD `Signal` when `rendered` crosses
   `expected`. Decision drivers: host CPU budget, frame duration,
   debug ergonomics. Default to §5.3 in the first cut and revisit
   once the megakernel is stable. SDMA queues vs a second compute
   queue for the helper path is its own measurement.
7. **Helper-queue placement on multi-GPU / multi-process.** A second
   compute queue per process scales with NUM_PROCESSES; KFD admits
   them via the HWS scheduler. If contention against the megakernel
   queue for CP cycles becomes measurable, evaluate moving the
   WAIT_REG_MEM + RELEASE_MEM pair to an SDMA queue.

## 7. Reading

- `cure-streaming-queues.md` §3.2 (this repo) — `ProgressQueue`
  watermark for primitive ordering, contrasted with frame
  completion.
- `bin-ownership-pipeline-proposal.md` §2.7 — host push protocol that
  this proposal layers on top.
- `bin-ownership-pipeline-proposal.md` §7.1 — `terminate` flag
  (app-shutdown, not per-frame).
- `headless-testing.md` §3.4 — frame-completion test cases that
  validate this protocol (empty frames, one-primitive frames,
  frame straddling host backoff).
- `include/libkfd/event.h` — KFD event RAII wrapper used by §5.8.
- `include/libkfd/signal.h` — `Signal::wait(spin_then_block)` used by
  §5.8.
- `include/libkfd/queue.h` (`signal()`, `wait_reg_mem()`,
  `dispatch(..., Signal&)`) — the PM4 plumbing.
- `include/libkfd/packets/pm4.h` (`RELEASE_MEM`, `WAIT_REG_MEM`,
  `IntSel`) — packet definitions.
- `lib/device/trap_handler.S` — `s_sendmsg sendmsg(MSG_INTERRUPT)`
  use site, for context on why compute kernels do not directly fire
  KFD events.
- Kenzel et al., *A High-Performance Software Graphics Pipeline
  Architecture for the GPU*, SIGGRAPH 2018 — cuRE paper.
- `source/cure/pipeline/Pipeline.cuh` and
  `source/cure/PipelineKernel.cpp` — cuRE's kernel-per-draw model.
