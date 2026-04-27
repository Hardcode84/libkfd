# Streaming Pipeline Proposal

Status: design proposal. Not implemented. Targets a standalone graphics
demo on AMDGPU via libkfd — a host app streaming transformed triangles
(rotating Utah teapot or similar) at 60 fps to a GPU-resident
megakernel that bins, rasterizes, and shades them.

Companion to `cure-streaming-queues.md`, which describes the primitives this
design borrows from. Read that one first for the queue protocol vocabulary.

## 1. Why The Streaming Shape

A naive baseline for a software rasterizer dispatches each pipeline
stage as its own compute kernel:

```text
host: dispatch tile_bin     (sized for worst case scene)
host: dispatch world_raster (waits on tile_bin's fence)
host: dispatch resolve
```

Costs that the streaming proposal targets:

- No overlap between host CPU work and GPU stages within a frame. The host
  transforms vertices, packs primitives, then dispatches; the GPU runs;
  the host blocks on the fence, then submits next frame.
- No overlap between pipeline stages. `world_raster` cannot start until
  `tile_bin` has finished completely, even though the first tiles' bins are
  ready long before the last tile's.
- Worst-case-sized intermediates. Per-tile primitive lists, counts, and
  HiZ extents are all sized for the maximum primitive count the
  renderer must handle.
- Per-dispatch overhead. A KFD compute dispatch is on the order of μs of
  latency before the kernel starts executing. Frequent dispatches amortize
  poorly.

The streaming proposal collapses the multi-dispatch model into a single
long-lived megakernel and feeds it primitives via a shared host↔GPU ring
buffer.

## 2. Architecture

### 2.1 High-Level Shape

```text
Host                                GPU (one megakernel)
────                                ─────────────────────────────────────
┌──────────────┐                    ┌─────────────────────────────────────┐
│ scene walk   │                    │ Distributor WG (wg_id == 0)         │
│ + batching   │ ─push─►   ┌────────┤  W waves per WG                     │
└──────────────┘           │        │  wave 0 polls global queue          │
                           │        │  s_barrier + LDS broadcast          │
                  ┌────────▼─────┐  │  each wave handles disjoint bin set │
                  │ Global queue │  └─────────────────────────────────────┘
                  │ fine-grained │              │
                  │ SVM ring     │              ▼   single producer wave per bin
                  │ (SPSC)       │     ┌────────────────┐
                  └──────────────┘     │ Coarse bin     │
                                       │ queues (SPMC)  │
                                       │ (one per bin)  │
                                       └────────┬───────┘
                                                │   multiple renderer WGs steal
                                                ▼
                                       ┌─────────────────────────────────────┐
                                       │ Renderer WGs (wg_id > 0)            │
                                       │  W waves per WG                     │
                                       │  wave 0 picks a coarse bin          │
                                       │  s_barrier + LDS broadcast          │
                                       │  each wave drains one fine tile     │
                                       └─────────────────────────────────────┘
                                                │   wave-disjoint pixel writes
                                                ▼
                                       framebuffer (no atomics)
```

Two structural choices distinguish this from a naive multi-distributor
design:

- **One distributor WG, `W` waves**. The wave-level decomposition inside the
  WG provides the parallelism that K distributor WGs would in a
  multi-distributor design, while making each bin queue *single-producer*.
  See §2.6.
- **Each renderer WG owns a coarse bin, one fine tile per wave**. The
  framebuffer is written exclusively per wave per fine tile, with no
  inter-wave atomics on color or depth. See §2.7.

### 2.2 Lifecycles

**Per dispatch** (one dispatch may span one frame, or many):

1. Host allocates fine-grained coherent SVM for the global queue and on-GPU
   memory for bin queues, framebuffer, vertex/index/texture data.
2. Host launches the megakernel with `N` workgroups (oversubscribed past
   hardware occupancy) of `W` waves each.
3. Host begins pushing primitive batches into the global queue.
4. GPU workgroups, persistent, loop:
   - Distributor WG (`wg_id == 0`): wave 0 pops a batch from the global
     queue, broadcasts via LDS, all `W` waves fan out to disjoint bin sets.
   - Renderer WGs (`wg_id > 0`): wave 0 picks a non-empty coarse bin,
     broadcasts the bin's primitive list via LDS, all `W` waves rasterize
     their assigned fine tile against that list.
5. Host signals end-of-stream. Workgroups observe and exit. Dispatch fence
   completes. Host reads the framebuffer.

**Per workgroup**, role is fixed by `wg_id`:

```text
if wg_id == 0:    role = DISTRIBUTOR
else:             role = RENDERER
```

### 2.3 Wave-Level Role Within A Workgroup

Both distributor and renderer WGs use the same intra-WG pattern: **wave 0
performs the queue-touching work** (a global atomic op or coherent load),
**broadcasts the result via LDS**, and **all `W` waves cooperatively process
the fetched unit of work**.

```c
__shared__ StreamBatch lds_batch;
__shared__ int32_t lds_status;  // 0 = continue, 1 = exit, -1 = retry

if (wave_id == 0 && lane_id == 0) {
  lds_status = -1;
  for (int attempt = 0; attempt < SPIN_LIMIT; ++attempt) {
    if (atomic_load(&terminate)) { lds_status = 1; break; }
    if (try_pop(&lds_batch))     { lds_status = 0; break; }
  }
}
__builtin_amdgcn_s_barrier();

if (lds_status == 1) return;
if (lds_status == -1) continue;

process_share(wave_id, &lds_batch);
```

`s_barrier` is workgroup-scoped on AMDGPU and requires all waves in the WG
to participate. This is acceptable because `W` is small (2/4/8) and the
barrier cost is dominated by the polling latency it gates.

While wave 0 spins on an empty queue, waves 1..W-1 are stalled at the
barrier. They hold register and LDS state on the CU. This is benign in
steady state but wasteful during host stalls. The cooperative polling
pattern (bounded `SPIN_LIMIT`, periodic `terminate` re-check) bounds the
stall and ensures clean termination.

Approximate cost model:

| Operation | Cost |
|---|---|
| `atomic_load` on coherent SVM (PCIe) | 500 ns – 1 μs |
| `atomic_load` on VRAM | ~50 ns |
| `s_barrier` workgroup-scoped | ~10 cycles |
| LDS read/write | ~20 cycles |

The barrier and LDS broadcast are essentially free relative to the queue
poll itself.

### 2.4 Data Flow Per Batch

A "batch" is the atomic unit of work passed through the queues. Concretely a
batch is a fixed-stride record:

```c
struct StreamBatch {
  uint32_t kind;             // PRIMITIVE_BATCH or END_OF_STREAM
  uint32_t prim_count;       // number of primitives in this batch
  uint32_t prim_offset;      // index into the global primitive store
  uint32_t reserved;
};
```

Batches contain *indices* into a separate primitive-data store, not the
primitive data itself. This mirrors `MultiIndexQueue` in cuRE: the queue is
narrow, the data is wide and read by index. A batch that represents 256
triangles costs 16 bytes in the queue, 256 × `sizeof(qr_raster_triangle)` in
the primitive store.

### 2.5 The Global Queue Protocol

Single-producer (host), single-consumer (wave 0 lane 0 of the distributor
WG). Bounded ring of `SIZE` slots, `SIZE` a compile-time power of two.

**Slot states**: each slot holds either a valid batch or a sentinel
`UNUSED = 0xFFFFFFFF` in `kind`. The sentinel doubles as the memory-ordering
checkpoint, exactly as in `MultiIndexQueue`.

The single-producer-single-consumer property removes the need for atomic
updates on `back` and `front`: each side keeps its own index in CPU/GPU
local state. The slot-publish / slot-drain pattern via the `kind` sentinel
is what synchronizes them across the PCIe boundary.

**Producer (host, single thread)**:

```c
uint32_t pos = back & (SIZE - 1);
while (atomic_load(&queue[pos].kind) != UNUSED)
  cpu_relax();
queue[pos].prim_count  = ...;
queue[pos].prim_offset = ...;
atomic_thread_fence(memory_order_release);
atomic_store(&queue[pos].kind, BATCH);
back += 1;
```

**Consumer (wave 0 lane 0 of distributor WG)**:

```c
uint32_t pos = front & (SIZE - 1);
StreamBatch local;
do {
  local.kind = atomic_load(&queue[pos].kind);
} while (local.kind == UNUSED);
local.prim_count  = queue[pos].prim_count;
local.prim_offset = queue[pos].prim_offset;
atomic_thread_fence(memory_order_acquire);
atomic_store(&queue[pos].kind, UNUSED);
front += 1;
```

This is the SPSC variant of the `cure-streaming-queues.md` `MultiIndexQueue`
protocol, except the producer side runs on the host CPU and the atomic ops
on `kind` traverse PCIe.

If multi-threaded primitive submission is later needed on the host side,
add an `atomic_fetch_add(&back, 1)` on the host side. The consumer side
stays single-threaded.

### 2.6 Coarse Bin Queues (Wave-Disjoint Partitioning)

One queue per coarse bin. On-GPU memory.

Within the single distributor WG, bins are partitioned by wave: wave `w`
owns bins where `bin_id % W == w`. Each bin has exactly one producer wave,
running once per popped global batch. This makes each bin queue
**single-producer**, the largest simplification over the K-distributor
model.

Renderers may consume from any bin queue (multi-consumer), so bin queues
are SPMC. The sentinel-based slot reuse pattern still applies; the consumer
side does an `atomicAdd(&front)` to reserve a slot, then spins on the
sentinel. The producer side does **not** need atomic `back` updates — only
one wave writes, and within a wave the lane-0 leader manages `back` in
scalar registers.

Bin queues store batch identifiers, not primitive indices directly. A
distributor receives one host-batch of (e.g.) 256 triangles. Each wave
AABB-tests the batch's primitives against its assigned bins and emits, per
bin, *one* sub-batch of (id, mask) where the mask indicates which
primitives in the source batch fall in this bin. This keeps the bin-queue
traffic at one entry per (host-batch × bin) rather than one per
(primitive × bin), and the AABB test parallelizes across the wave's lanes.

Geometry: with `W` waves and `B` total coarse bins, each wave handles
`B / W` bins per popped global batch. Distribution work per host-batch is
bounded by `prim_count × bins_per_wave / wave_lanes` cycles per wave.

### 2.7 Renderer Work Selection And Wave-Per-Fine-Tile

A renderer WG owns one coarse bin at a time. The coarse bin is subdivided
into `W` fine tiles, one per wave. Each wave rasterizes its assigned fine
tile.

Geometry table (target: 16×16 fine tile per wave, multiple passes per
wave to cover it):

| Wave size | W | Fine tile | Coarse bin |
|---|---|---|---|
| 64 (GFX9; RDNA wave64 opt-in) | 4 | 16×16 (256 px, 4 passes) | 32×32 (1024 px) |
| 32 (RDNA default) | 4 | 8×16 (128 px, 4 passes) | 16×32 (512 px) |

Renderer loop (per-WG view):

```c
for (;;) {
  __shared__ uint32_t lds_bin;
  __shared__ uint32_t lds_prim_count;
  __shared__ uint32_t lds_prim_indices[MAX_BIN_PRIMS];
  __shared__ int32_t  lds_status;

  if (wave_id == 0 && lane_id == 0) {
    if (atomic_load(&terminate)) {
      lds_status = 1;
    } else if (try_pick_bin(&lds_bin, lds_prim_indices, &lds_prim_count)) {
      lds_status = 0;
    } else {
      lds_status = -1;
    }
  }
  __builtin_amdgcn_s_barrier();

  if (lds_status ==  1) return;
  if (lds_status == -1) { backoff(); continue; }

  rasterize_fine_tile(lds_bin, lds_prim_indices, lds_prim_count, wave_id);
}
```

`rasterize_fine_tile(bin, prims, count, wave_id)` reads the shared
primitive list (in LDS) and writes only its disjoint pixel region.
**No inter-wave atomics on the framebuffer.** No inter-wave barrier within
the bin — each wave drains its fine tile to completion and the WG re-enters
its top-level loop. Imbalance becomes "fast wave loops back faster" rather
than "fast wave blocks at barrier".

Bin selection (`try_pick_bin`) candidates, simplest to most invasive:

- **Hash on `wg_id`**: each renderer is sticky to one bin or a small set of
  bins. Cache-friendly, can starve sparse bins.
- **Round-robin scan**: scan all bins for a non-empty queue. Cheap,
  reasonable baseline.
- **Counter-based steal**: maintain per-bin fill counters; renderers prefer
  highest fill. Best load balance, costs an atomic per pick.

For the demo workload (rotating teapot, ~6 K triangles per frame),
the round-robin scan is sufficient.

### 2.8 Termination

A single `terminate` flag in fine-grained SVM:

```c
atomic_uint32_t terminate;  // 0 == active, 1 == draining
```

Host sequence at end of frame (or end of stream):

1. Push the last real batch.
2. Push an `END_OF_STREAM` batch, *or* set `terminate = 1`. Either works.
3. Wait on the dispatch fence.

Workgroup exit conditions, as enforced by the wave-level pattern in §2.3:

- **Distributor WG**: wave 0 pops a batch, sees `END_OF_STREAM` →
  atomically sets `terminate = 1` (idempotent), broadcasts via LDS,
  barrier, all waves observe and exit. The bin queues are *not* drained by
  the distributor; renderers handle remaining work after observing the
  flag.
- **Renderer WG**: at top of loop, wave 0 checks `terminate` *and* finds
  no non-empty bin queue → broadcast exit signal via LDS, barrier, all
  waves exit. The bin-queue empty check after `terminate` is what ensures
  rendering completes before exit.

The distributor does *not* need to broadcast `END_OF_STREAM` into bin
queues; the global flag is sufficient and avoids a fan-out write across
all bins.

This implies a kernel-per-frame model: the host's dispatch fence
*is* the frame-complete signal. If the megakernel needs to persist
across multiple frames (lower per-frame launch cost, preserved
LDS/SGPR state), the `END_OF_STREAM` flag becomes a per-app-shutdown
signal and per-frame completion needs an in-kernel ack mechanism.
See `frame-completion-detection.md` for variants and a recommended
shape (V2: bulk per-frame counter with per-WG accumulator).

### 2.9 Oversubscribed Launch And Fast-Exit

Launch `N` workgroups where `N > hardware_occupancy`. The hardware
dispatcher fills CUs to capacity; the rest queue.

In steady state only `hardware_occupancy` WGs run; the surplus is idle.

When `terminate = 1` and resident WGs begin exiting, the queued surplus is
admitted onto freed CUs. They run their loop once, observe `terminate`, exit
on first iteration. This is the cleanup path that ensures every dispatched
WG eventually completes, which is required for the dispatch fence to
signal.

Without oversubscription, persistent WGs that exited cleanly leave the CU
idle until dispatch end. With oversubscription, the surplus drains the
dispatch quickly. This is purely a fence-completion mechanism; it does not
add steady-state parallelism.

`N` should be sized so that surplus WGs do not blow out the PM4 dispatch
size. AMDGPU's grid limits are large; this is not a real constraint.

## 3. Comparison To Alternatives

### 3.1 Stage-Per-Dispatch Baseline

| Concern | Stage-per-dispatch | Streaming proposal |
|---|---|---|
| Pipeline stage overlap | None (barrier-per-stage) | Full (megakernel) |
| Intermediate sizing | Worst-case per-tile arrays | Bounded queues, backpressure |
| Host overlap with GPU | None within frame | Full (host pushes during dispatch) |
| Per-frame dispatch count | 3–4 | 1 |
| Bin-counter atomics | Per-tile counters/overflows | None (wave-disjoint single producer) |
| Framebuffer atomics | None (per-pixel ownership) | None (wave-disjoint fine-tile ownership) |
| Queue counter atomics | None | Few (SPMC bin-queue `front` only) |
| Termination | Implicit (kernel exits) | Explicit flag in fine-grained SVM |
| Implementation cost | Trivial | Significant |

### 3.2 cuRE-Style (Upfront Upload)

cuRE pushes the entire frame's primitives into VRAM before dispatch, then
runs a megakernel that streams them through on-GPU queues only. No host↔GPU
queue.

Pros relative to the streaming proposal:

- No PCIe atomics on the hot path. All queue traffic is on-GPU.
- Simpler ordering: the producer is the geometry-fetch stage in the same
  megakernel, not the host.
- No host watchdog needed; if the host crashes mid-frame, the GPU finishes
  what was uploaded and exits.

Cons:

- Latency: host CPU work and GPU rendering are sequenced per frame. No
  intra-frame overlap.
- Memory: full frame's primitives must fit in VRAM at frame start.
- Less amenable to variable-rate scenes.

For the rotating-teapot demo specifically, cuRE-style would be the
simpler trade — the host's per-frame work is tiny (one model matrix
update, ~6 K vertex transforms) and the primitive payload fits in
<1 MB. The streaming proposal earns its complexity when the host
pipeline is heavy (skinning, animation, complex culling, LOD) — it is
designed as a *demonstrator* of the host-streaming shape, with the
teapot as a stand-in for richer workloads.

### 3.3 Hybrid (Bulk Upload, Streaming Within The Frame)

A useful middle ground: host uploads the entire frame's primitives upfront
(no streaming from the host) but the GPU pipeline is still the
megakernel-with-bin-queues design (streaming within the GPU). This captures
the on-GPU benefits — bounded intermediates, stage overlap, single dispatch
— without the PCIe atomic cost or host-watchdog complexity.

This is essentially what cuRE does. Worth flagging as a deliberate fallback
position if the full streaming design proves too costly.

## 4. Open Issues

The proposal is not implementation-ready. Specific decisions and risks:

### 4.1 PCIe Atomic Cost (Severity: high)

PCIe atomic round-trip on consumer AMDGPU cards is in the
**~500ns–1μs** range, versus ~50ns for VRAM-local atomics. This is the
single biggest cost driver.

Implication: batch size must be large enough that dispatch atomic overhead
is < 10% of per-batch processing time. At 1μs reserve and ~50ns per
triangle, batch ≥ 256 triangles is the floor. 1024 is more comfortable.

Cross-root-complex atomics (multi-socket hosts, certain chipsets) may not
support PCIe atomics at all and silently fall back to non-atomic emulation.
Validate on target platforms.

### 4.2 Producer-Consumer Memory Ordering (Severity: high)

Bare `atomicInc(&back)` followed by data write is racy. The protocol must
be *one of*:

- Sentinel-based slot reuse (cuRE pattern, this doc's choice).
- Per-slot version counter.
- Release/acquire fence sequence with explicit `__threadfence_system()` on
  the GPU side.

Pick one and stay with it. Mixing produces hard-to-debug visibility bugs.

### 4.3 Single-Distributor Throughput Cap (Severity: medium)

The proposal uses one distributor WG, which is a deliberate trade-off:
wave-disjoint bin partitioning eliminates cross-WG atomics on bin queues
at the cost of capping distribution throughput at one WG's worth of
parallelism.

Rough budget on RDNA2 6700 XT (40 CUs, ~80 resident WGs at typical
occupancy):

- W=4 waves, batch=256 prims, ~16 bins/wave (with 64 coarse bins)
- Per-batch distribution: 256 prims × 16 bins / 64 lanes = 64 cycles per
  wave, executed in parallel across waves → ~64 cycles total
- At 2 GHz, ~32 ns of pure distribution work per batch
- Plus the global-queue atomic: 500 ns – 1 μs

PCIe atomic dominates by 20–30×. Distribution is not the bottleneck.

This story breaks if:

- Per-batch processing time drops (smaller batches; integrated APU with no
  PCIe boundary; MI300-class coherent fabric where atomics are cheap).
- Bin count rises sharply (e.g., 1024 bins, 256 per wave → 256 cycles per
  primitive per wave → ~65K cycles per batch ≈ 33 μs). Distribution then
  rivals atomic latency.
- Scene complexity rises beyond demo-shape (denser triangles, higher
  per-batch primitive counts).

If the cap binds, scale by sharding bins across multiple distributor WGs
(WG 0 handles bins 0..127, WG 1 handles 128..255, etc.) while preserving
wave-disjoint partitioning within each WG. This re-introduces some
cross-WG atomics on the global queue front pointer but keeps bin queue
producers single-wave. Treat as an escape hatch, not a default.

### 4.4 Bin Queue Sizing (Severity: low)

With single-producer per bin and no cross-WG contention, sizing reduces
to a steady-state question: how far ahead can the producer wave run
before consumers (renderer WGs) catch up? Backpressure handles overflow
correctly (the producer wave spins on `slot != UNUSED`), but if the queue
is too small the steady-state is "producer wave blocked" → distributor
WG stalls → host blocks pushing into the global queue.

Initial rule of thumb: `bin_queue_depth ≥ 16` slots. Tune per scene
profile.

### 4.5 Renderer Wave-Tile Imbalance (Severity: medium)

Static "wave_id → fine_tile" assignment within a coarse bin gives each
wave equal *area* but not equal *work*. A fine tile at the bin edge often
has fewer primitives than the center; the wave assigned to the dense tile
runs longer.

If waves were synchronized at a barrier between bins (one bin per
barrier-pass), the slowest wave would bound the bin-completion time.
The proposal's "no inter-tile barrier within a bin" choice (§2.7) avoids
this: each wave drains its fine tile, then the WG re-enters its top-level
loop. Imbalance becomes "fast wave loops back faster", which is harmless.

If profiling shows persistent imbalance even without the inter-tile
barrier (one WG repeatedly drawing the work, others idle), mitigations:

- **Per-WG dynamic tile claim**: replace static `wave_id → fine_tile` with
  an LDS atomic counter. Each wave grabs the next un-claimed fine tile.
  Costs one LDS atomic per claim, balances real load.
- **Smaller fine tiles**: if fine tiles are small enough (8×8, 4×4),
  variance averages out across many tiles per bin.

The static assignment is the right starting point; revisit only if data
demands it.

### 4.6 Host Watchdog (Severity: medium)

If the host crashes after dispatch but before pushing `END_OF_STREAM`, the
GPU spins forever. KFD's watchdog will eventually kill the queue (typical
timeout: low single-digit seconds depending on platform/scheduling
priority), but the user-visible behavior is bad.

Defensive options:

- **Heartbeat counter** in fine-grained SVM. Host bumps every N ms. A
  designated monitor WG (one of the distributors) checks; staleness past
  threshold → set `terminate = 1`.
- **Time-based timeout**. AMDGPU exposes `s_memrealtime`. WG idles for X
  seconds → set `terminate = 1`.

Heartbeat is more responsive and cheaper to implement.

### 4.7 KFD Watchdog Compatibility (Severity: medium)

A megakernel that runs for an entire frame at 60 fps (~16 ms) is well below
KFD's hang detection threshold. Megakernels that run longer (multi-frame
batching, scene streaming with no end-of-stream marker) approach the
threshold. Make sure progress is observable: any active queue traffic is
sufficient.

If multi-second kernels are required, queue priority configuration in KFD
exists for this case. Document the requirement; do not silently rely on it.

### 4.8 Wave Size Portability (Severity: medium)

`W` waves per WG and the per-wave fine-tile geometry interact with the
target arch's wave size. AMDGPU GFX9 is wave64 only; GFX10/11/12 (RDNA)
defaults to wave32 in compute with wave64 available as opt-in.

Three portability strategies:

1. **Compile wave64 across the board**. Wastes lanes on RDNA's
   scalar-heavy paths but keeps the WG/bin/tile geometry constant across
   archs. Simplest.
2. **Per-arch `W` tuning**. The demo's CMake already follows the
   `libkfd` cross-compile-per-AMDGPU-arch pattern (see
   `tools/computetoy` for an example); `W` and the fine-tile geometry
   can be per-arch macros.
3. **Hold coarse-bin pixel area constant**, let wave size dictate
   fine-tile shape. Per-arch fine-tile geometry, per-arch coarse-bin
   geometry. Most work to specify; preserves both wave occupancy and
   cache behavior.

The third is the most honest. The first is the right starting point.

## 5. `libkfd` Extensions Required

The current `libkfd::ComputeQueue` API is dispatch-then-wait:

```cpp
kfd_gpu_dispatch(ctx, kernel, dispatch_cfg, kernarg, fence);
kfd_gpu_fence_wait(fence, 0, UINT64_MAX);
```

The streaming proposal needs the host to keep producing into the SVM ring
*while* the dispatch is in flight. Concretely:

### 5.1 Async Dispatch

```cpp
kfd_gpu_dispatch_async(ctx, kernel, dispatch_cfg, kernarg, &fence_out);
// returns immediately; fence_out is a handle the host can poll/wait
```

Already implementable as a thin wrapper over the existing PM4 submission
that does not call `kfd_gpu_fence_wait`.

### 5.2 Fine-Grained Coherent SVM Allocation

`libkfd::Memory` supports fine-grained allocations (per existing module
shape). The streaming proposal needs:

- Allocation flag specifying coherent (no explicit cache flush) and
  fine-grained.
- Both CPU pointer and GPU pointer surfaced from the same handle.
- Atomic-safe semantics validated on target hardware.

If this is already there, the proposal can build on top with no `libkfd`
work. If not, this is the most important missing primitive.

### 5.3 Fence Polling

`kfd_gpu_fence_wait_with_timeout(fence, timeout_ns)` so the host can
interleave queue production with periodic fence checks instead of blocking.

### 5.4 Watchdog Disable / Priority Boost

For long-running megakernels, an opt-in path to extended-priority queues or
disabled watchdog. KFD has the controls; `libkfd` needs to expose them.

## 6. Concrete Refinement (If Implemented)

The minimal shippable instantiation, in priority order:

1. **WG count and roles**: 1 distributor WG (`wg_id == 0`), rest renderers
   (`wg_id > 0`). Hard split, compile-time.
2. **Waves per WG**: `W = 4` to start. Compile wave64 across all archs for
   geometry consistency (per §4.8 strategy 1).
3. **Wave-disjoint partitioning**: distributor wave `w` owns bins where
   `bin_id % W == w`; renderer wave `w` owns fine tile `w` within the
   current coarse bin.
4. **Queue protocol**: sentinel-based slot reuse. Global queue is SPSC
   (one host producer, one GPU consumer wave), no atomics on
   `back`/`front`. Bin queues are SPMC (one producer wave, multiple
   renderer WGs); only the consumer side needs an atomic on `front`.
5. **Wave-level pattern**: wave 0 polls + LDS broadcast + `s_barrier`. All
   waves cooperate on the broadcasted unit of work. Cooperative polling
   with bounded `SPIN_LIMIT` and periodic `terminate` re-check.
6. **Renderer fine-tile drain**: each wave drains its fine tile to
   completion, then the WG re-enters its top-level loop. No inter-wave
   barrier within a bin.
7. **Termination**: single global `terminate` flag in fine-grained SVM.
   No marker broadcast into bin queues.
8. **Batch size**: 256 primitives floor, 1024 target.
9. **Oversubscription**: launch `N = 2 × hw_occupancy` WGs. Fast-exit on
   `terminate`.
10. **Host watchdog**: heartbeat counter in fine-grained SVM, monitored by
    the distributor WG (one designated wave within it).
11. **`libkfd` extensions**: async dispatch + fine-grained SVM (if
    missing) + timeout-based fence wait + watchdog-priority opt-in.

## 7. Demo Staging

A reasonable path for the rotating-teapot demo:

1. Stand up the stage-per-dispatch baseline first (§3.1). One kernel
   bins triangles, one rasterizes, one resolves. Fastest to a
   pixel-on-screen.
2. Collapse to the **hybrid (bulk upload + on-GPU streaming)** shape
   from §3.3. Adds the megakernel and on-GPU queues without the
   host↔GPU PCIe-atomic complexity. Validates the streaming-queue
   primitives, distributor/renderer split, and frame-completion
   machinery (`frame-completion-detection.md`).
3. If the demo is ever extended to a host pipeline heavy enough to
   benefit (skinning, animation, complex culling), promote to full
   host-streaming as described in §2.

The `bin-ownership-pipeline-proposal.md` is an alternative to this
proposal that sacrifices the wave-disjoint distributor in exchange for
dynamic role selection at WG granularity. The two are not stackable;
they live behind a runtime selector based on `B/N`.

## 8. Reading

- `cure-streaming-queues.md` — primitive vocabulary used throughout.
- `bin-ownership-pipeline-proposal.md` — alternative pipeline shape
  with dynamic per-WG role selection.
- `frame-completion-detection.md` — host-side frame-end detection
  and KFD-signal-driven wait that this proposal layers on top of.
- Steinberger et al., *Whippletree: Task-based Scheduling of Dynamic
  Workloads on the GPU*, TOG 2014. Provides the formal model for
  role-selection-on-WG-id with shared queues.
- Kenzel et al., *A High-Performance Software Graphics Pipeline Architecture
  for the GPU*, SIGGRAPH 2018 (cuRE). Reference implementation of the
  on-GPU streaming half.
