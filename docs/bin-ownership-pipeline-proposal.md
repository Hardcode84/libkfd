# Bin-Ownership Pipeline Proposal

Status: design proposal. Alternative to `streaming-pipeline-proposal.md`.

The streaming proposal partitions work *structurally*: one distributor WG
with wave-disjoint bin partitioning, and N-1 renderer WGs as multi-consumer
of bin queues. Roles are fixed at WG-launch time by `wg_id`.

This proposal partitions work *dynamically*: every WG is symmetric and,
each iteration, picks one of two modes:

1. **Distribute mode**: drain the host queue, populate the relevant tile
   queues with the batch's primitives.
2. **Render mode**: drain a single tile queue, rasterize it.

Mutual exclusion across WGs is enforced via two kinds of ownership locks:

- One **host queue lock** (binary), ensuring at most one WG is
  distributing at a time.
- B **tile locks** (3-state, two independent bits — §2.2.2), one per
  coarse bin. The two bits separately track *queue access* (single-
  writer/reader) and *rasterization* (single-renderer-per-tile) so a
  distributor can push new primitives to a tile while a renderer is
  rasterizing it.

The active distributor identity rotates as WGs release the host queue
lock and other WGs re-claim it. Each tile cycles through queue-access
and rendering states independently and can overlap them.

Read `streaming-pipeline-proposal.md` first for shared vocabulary
(megakernel, persistent WGs, fine-grained SVM, batch records, oversubscribed
launch, host-side ring buffer, watchdog). This document only describes
what differs.

## 1. Why This Shape

The streaming proposal works cleanly when:

- Coarse bin count `B` is moderate.
- Per-bin workload is roughly uniform.
- Producer/consumer balance is steady within a frame.

It struggles when:

- `B` is much larger than `N` (resident WG count). Wave-disjoint
  partitioning under-uses parallelism: only one distributor WG of `W`
  waves works on the entire bin set.
- Per-bin work variance is high. A few hot bins bottleneck while idle
  WGs cannot help.
- Producer load is bursty. The fixed distributor WG saturates while
  renderer WGs idle, or vice versa.

The two-mode lock-based model addresses these by making any WG capable
of either role per iteration. The active distributor identity rotates
dynamically, and the renderer pool absorbs idle capacity.

Concrete numbers (RDNA2 6700 XT, ~80 resident WGs at typical occupancy):

| Render res | Coarse bin size | B | B/N |
|---|---|---|---|
| 1080p | 32×32 | ~2 K  | 25  |
| 1080p | 64×64 | ~510  | 6   |
| 1080p | 128×128 | ~135 | 1.7 |
| 4K | 32×32 | ~8 K  | 100 |
| 4K | 64×64 | ~2 K  | 25  |
| 4K | 128×128 | ~510 | 6   |

The model is robust at `B/N ≥ 16` (random tile claim is essentially
uncontended). It works at `4 ≤ B/N ≤ 16` with picking heuristics. It
breaks down below `B/N < 4`.

For 1080p Quake-shape, the streaming proposal is the simpler trade. For
4K or higher-density scenes, this proposal is.

## 2. Architecture

### 2.1 High-Level Shape

```text
Host                                GPU (one megakernel)
────                                ─────────────────────────────────────
┌──────────────┐                    ┌─────────────────────────────────────┐
│ scene walk   │                    │ N symmetric WGs (no role split)     │
│ + batching   │ ─push─►            │  loop:                              │
└──────────────┘                    │    if try_claim(host_queue_lock):   │
                    ┌──────────┐    │      distribute() {                 │
                    │ Host     │    │        pop batch                    │
                    │ queue    │ ◄──┤        bucket by tile in LDS        │
                    │ + lock   │    │        for each bucket:             │
                    └──────────┘    │          claim tile_lock briefly,   │
                                    │          push, release              │
                                    │      }                              │
                                    │      release host_queue_lock        │
                                    │    else:                            │
                                    │      tile = pick_tile()             │
                                    │      if try_claim(tile_lock[tile]): │
                                    │        render(tile)                 │
                                    │        release tile_lock[tile]      │
                                    └─────────────────────────────────────┘
                                                ▲
                                                │
                                       ┌────────┴────────┐
                                       │ Per-tile state  │
                                       │  - queue        │
                                       │  - lock         │
                                       │ (B of these)    │
                                       └─────────────────┘
                                                │
                                                ▼
                                       framebuffer
                                       (per-tile exclusive write)
```

Differences from `streaming-pipeline-proposal.md`:

- The host queue is gated by a single binary lock; only one WG drains
  it at a time. The active distributor identity rotates dynamically.
- Each tile is gated by a 3-state lock (§2.2.2): a queue-access bit
  (push or drain) and an independent rendering bit. Distributor pushes
  and renderer rasterization can overlap on the same tile; only the
  brief queue-access window is mutually exclusive.
- WG roles are dynamic per iteration, not fixed by `wg_id`.

### 2.2 The Two Lock Types

The host queue lock is a plain binary lock. Tile locks are 3-state, with
two independent ownership bits.

```c
__device__ uint32_t host_queue_lock;   // 0 = UNLOCKED, 1 = LOCKED
__device__ uint32_t tile_lock[B];      // 2-bit field, see below
```

#### 2.2.1 Host queue lock (binary)

```c
__device__ bool try_claim(uint32_t *lock) {
  return atomicCAS(lock, UNLOCKED, LOCKED) == UNLOCKED;
}

__device__ void release(uint32_t *lock) {
  atomic_thread_fence(memory_order_release);
  atomicExch(lock, UNLOCKED);
}
```

#### 2.2.2 Tile lock (3-state, two bits)

A tile is touched by two kinds of operations with different semantics:
*queue access* (distributor pushing, renderer draining) and *framebuffer
access* (renderer rasterizing). Queue access is brief (~μs); framebuffer
access is long (10K–100K cycles). The two need different mutual-exclusion
rules:

- Queue access is mutually exclusive with all other queue access on
  this tile (single-writer/single-reader on the queue's pointers).
- Framebuffer access is mutually exclusive with other framebuffer
  access on this tile (no two renderers on the same tile).
- Queue access is *not* mutually exclusive with framebuffer access
  on this tile (distributor pushing while a renderer rasterizes is
  fine — the new primitives just wait for the next render pass).

A binary lock conflates these. Using a binary lock and "hold through
rasterize" forces distributors to wait for rasterization to finish.
Using a binary lock and "release before rasterize" lets a second
renderer claim the tile and clobber the first renderer's pixel writes.

The 3-state design encodes the two concerns in two independent bits:

```c
#define TILE_QLOCK    0x1u    // queue access in progress (push or drain)
#define TILE_RENDER   0x2u    // a renderer owns rasterization
```

The four reachable values:

| Value | Bits  | Meaning                                                  |
|-------|-------|----------------------------------------------------------|
| 0x0   | 00    | UNLOCKED — fully idle                                    |
| 0x1   | 01    | queue access (push or drain), no rasterizer              |
| 0x2   | 10    | RENDERING (rasterize in progress, queue idle)            |
| 0x3   | 11    | RENDERING + queue access (push during rasterize)         |

State transitions, all single CAS or atomic-bit op:

| Operation                      | Transition           |
|--------------------------------|----------------------|
| Distributor start push         | `0x0→0x1` or `0x2→0x3` |
| Distributor end push           | `atomicAnd(~0x1)`    |
| Renderer start (drain)         | `0x0→0x1` only       |
| Renderer drain → rasterize     | `0x1→0x2`            |
| Renderer end rasterize         | `atomicAnd(~0x2)`    |

The releases use `atomicAnd` (clear-bit) rather than `atomicExch` so
each side clears only its own bit and preserves the other side's
ownership. `atomicAnd` is also cheaper on AMDGPU than `atomicExch`
because it does not need to return a value.

```c
__device__ bool dist_try_push_lock(uint32_t tile) {
  // Acquire TILE_QLOCK from UNLOCKED or RENDERING. Two CAS attempts max.
  if (atomicCAS(&tile_lock[tile], 0x0, 0x1) == 0x0) return true;
  if (atomicCAS(&tile_lock[tile], 0x2, 0x3) == 0x2) return true;
  return false;
}

__device__ void dist_release_push_lock(uint32_t tile) {
  atomic_thread_fence(memory_order_release);
  atomicAnd(&tile_lock[tile], ~TILE_QLOCK);
}

__device__ bool render_try_acquire(uint32_t tile) {
  // Need fully UNLOCKED; will not start a render pass during another's.
  return atomicCAS(&tile_lock[tile], 0x0, TILE_QLOCK) == 0x0;
}

__device__ void render_drain_done(uint32_t tile) {
  // Drain finished, switch from queue-locked to rendering-only.
  // CAS form so we never write 0x2 if someone snuck in.
  atomicCAS(&tile_lock[tile], TILE_QLOCK, TILE_RENDER);
}

__device__ void render_release(uint32_t tile) {
  atomic_thread_fence(memory_order_release);
  atomicAnd(&tile_lock[tile], ~TILE_RENDER);
}
```

Distributor cost: at most 2 CAS to acquire + 1 atomicAnd to release.
Renderer cost: 1 CAS to drain-acquire + 1 CAS to switch + 1 atomicAnd
to release. The renderer's extra CAS is amortized against the entire
rasterize phase and is negligible.

**Critical**: none of these acquire calls spin. On failure the WG
immediately makes another scheduling decision — switch modes, try a
different tile, or back off briefly. Spinning on a single lock wastes
the WG's time when alternative work is plentiful.

**Lock ordering for deadlock freedom**:

- Distributor holds `host_queue_lock` first, then briefly acquires
  `TILE_QLOCK` on one tile to push, releases it, may then acquire
  `TILE_QLOCK` on another tile, etc. Never holds queue access on two
  tiles simultaneously.
- Renderer holds only the `tile_lock` bits of one tile, never touches
  `host_queue_lock`. The acquire-drain → switch-to-render sequence
  uses the same tile's lock without nesting.

The lock-acquisition graph has no cycle. Trivially deadlock-free.

**Memory ordering**: the release fence before `atomicAnd(~TILE_RENDER)`
publishes the renderer's framebuffer writes to the next renderer that
acquires this tile. The release fence before `atomicAnd(~TILE_QLOCK)`
publishes queue updates to the next queue accessor. These are the same
ordering requirements as a binary lock release — only the bit cleared
differs.

### 2.3 Workgroup Loop With Mode Dispatch

```c
for (;;) {
  if (atomic_load(&terminate)) break;

  // §3 covers alternative mode-selection policies.
  if (try_claim(&host_queue_lock)) {
    distribute();                         // §2.4
    release(&host_queue_lock);
  } else {
    uint32_t tile = pick_tile();          // §4
    if (tile == NO_TILE) { backoff(); continue; }
    if (render_try_acquire(tile)) {
      render(tile);                       // §2.5; releases TILE_RENDER inside
    }
    // CAS-fail: tile is busy (queue access or another renderer);
    // just loop and try a different tile next iteration.
  }
}
```

The wave-level pattern from `streaming-pipeline-proposal.md` §2.3 still
applies inside `distribute()` and `render()`: wave 0 of the WG performs
the queue-touching operations, broadcasts via LDS, all `W` waves
cooperate on the broadcasted unit of work.

### 2.4 Distribute Mode

The distributor processes one batch from the host queue per
`distribute()` call. It buckets primitives by target tile in LDS first
(no global atomics), then pushes each bucket under that tile's lock.

```c
__device__ void distribute() {
  StreamBatch batch;
  if (!pop_host_queue(&batch)) return;       // queue empty

  // Phase 1: bucket-by-tile in LDS (parallel across waves, no atomics
  //          on tile queues yet)
  __shared__ uint32_t lds_bucket_count[B_SUBSET];
  __shared__ uint32_t lds_bucket_prim[B_SUBSET][BUCKET_DEPTH];
  bucket_by_tile(&batch, lds_bucket_count, lds_bucket_prim);

  // Phase 2: push each non-empty bucket under its tile's queue-lock bit
  uint32_t pending = num_active_buckets(lds_bucket_count);
  while (pending > 0) {
    for (uint32_t b = 0; b < B_SUBSET; ++b) {
      if (lds_bucket_count[b] == 0) continue;
      uint32_t tile = bucket_to_tile(b);
      if (dist_try_push_lock(tile)) {
        push_bucket_to_tile_queue(tile,
                                  lds_bucket_prim[b],
                                  lds_bucket_count[b]);
        dist_release_push_lock(tile);
        lds_bucket_count[b] = 0;
        --pending;
      }
    }
    if (pending > 0) backoff();              // retry contended buckets
  }
}
```

`dist_try_push_lock` succeeds whenever the tile is UNLOCKED (`0x0`) or
RENDERING-only (`0x2`). It fails only when another WG is already
mid-push or mid-drain on this tile — a window of a few μs at most. The
distributor effectively never waits on rasterization.

Phase 1 parallelizes across the WG's `W` waves: each wave handles a
disjoint subset of primitives, AABB-tests each against all bins, writes
hits into LDS buckets indexed by tile. Bucket-counter updates use LDS
atomics (~20 cycles), much cheaper than global.

Phase 2 lock acquisitions per batch: one for the host queue plus one per
distinct target tile. For a 256-primitive batch hitting ~16 tiles on
average, that's 17 acquisitions per batch — bounded and predictable.

If a tile's queue-lock bit fails (another distributor or a renderer is
draining it), distributor moves on to other buckets in this round, then
comes back. Bounded by the longest queue-access window (drain or push) —
a few μs. Crucially, this is *not* bounded by rasterization time; the
3-state lock lets a distributor push during rasterize.

`B_SUBSET` is the number of tiles the distributor handles per batch. If
the batch's primitives can hit any of B tiles, bucketing naively
requires B LDS slots, which scales poorly. Two practical mitigations:

- **LDS hash buckets**: bucket into `B_SUBSET = 64` LDS slots indexed by
  `tile_id % 64`. Each slot holds a small list of (tile, prims) pairs.
  Bounded LDS, slight collision handling.
- **Per-batch tile set**: precompute the set of tiles the batch's
  primitives can hit (upper bound by primitive AABBs). For
  Quake-shape batches this is typically 8–32 distinct tiles.

The second is preferable for moderate batches.

### 2.5 Render Mode

A render pass on tile `t` proceeds in three lock phases mapping directly
to the 3-state protocol in §2.2.2:

1. Acquire `TILE_QLOCK` on `t` (CAS from `0x0`).
2. Drain the tile queue into LDS, then transition `TILE_QLOCK → TILE_RENDER`
   (CAS `0x1 → 0x2`).
3. Rasterize from LDS while holding only `TILE_RENDER`. Release at the
   end (`atomicAnd(~TILE_RENDER)`).

```c
__device__ void render(uint32_t tile) {
  // Phase 1: acquire the queue lock to safely drain.
  // Caller already verified render_try_acquire(tile) succeeded.

  __shared__ uint32_t lds_prim_count;
  __shared__ uint32_t lds_prim_indices[MAX_TILE_PRIMS];
  drain_tile_queue(tile, lds_prim_indices, &lds_prim_count);

  // Phase 2: switch to RENDERING-only. Distributors may now push to
  // this tile's queue concurrently with the rasterize below.
  render_drain_done(tile);

  // Phase 3: rasterize from LDS. New pushes accumulate in the tile
  // queue and are rendered by the next claim of this tile.
  rasterize_tile(tile, lds_prim_indices, lds_prim_count);

  // Release the rendering bit. atomicAnd preserves a concurrent
  // distributor's TILE_QLOCK bit if any.
  render_release(tile);
}
```

While this WG is in phase 3, the tile state is `0x2` (or `0x3` if a
distributor has the queue lock). Other renderers see neither `0x0`
nor `0x1` and so cannot start a fresh render pass on this tile —
preventing the framebuffer race that would happen if both rasterized
into `t`'s pixels at once. Distributors see "not queue-locked" in the
`0x2` state and can acquire the queue-lock bit to push, raising the
state to `0x3` briefly, then back to `0x2`.

The blocking timing is asymmetric and that's the whole point:

- Distributor blocked on this tile: only during another WG's drain or
  another distributor's push (~μs).
- Other renderer blocked on this tile: through this renderer's entire
  phase (drain + rasterize ≈ 50 μs).
- This renderer blocked on its own tile: only during another
  distributor's concurrent push (~μs, and only at queue-pointer
  manipulation; the LDS-resident primitive list is unaffected).

New primitives pushed during a render pass are rendered in the *next*
claim of this tile. For opaque "first closer wins" depth this is
correct (the depth test handles ordering). For blended draws or
alpha-test, the across-pass ordering needs to match the host's submit
order — same caveat that applies to any tile that is rendered more
than once per frame.

The wave-per-fine-tile rasterization decomposition from
`streaming-pipeline-proposal.md` §2.7 applies inside `rasterize_tile`,
optionally with the LDS-resident sub-tile claim pattern in §2.6 below.

### 2.6 Within-WG Sub-Tile Ownership

Inside `rasterize_tile`, the W waves can either statically partition
fine sub-tiles (one wave per sub-tile, as in
`streaming-pipeline-proposal.md` §2.7) or dynamically claim sub-tiles
via LDS atomics:

```c
__shared__ uint32_t lds_subtile_claim[NUM_SUBTILES];

uint32_t my_subtile = (uint32_t)-1;
for (uint32_t off = 0; off < NUM_SUBTILES; ++off) {
  uint32_t idx = (wave_id * STRIDE + off) % NUM_SUBTILES;
  if (lane_id == 0) {
    if (atomicCAS(&lds_subtile_claim[idx], UNCLAIMED, CLAIMED)
        == UNCLAIMED) {
      my_subtile = idx;
    }
  }
  my_subtile = __builtin_amdgcn_readfirstlane(my_subtile);
  if (my_subtile != (uint32_t)-1) break;
}
if (my_subtile == (uint32_t)-1) return;        // none left

rasterize_subtile(tile, my_subtile);
```

LDS atomicCAS is ~20 cycles, much cheaper than the global atomic for
the WG-level claim. With 16 sub-tiles and 4 waves, average claims per
wave is 4 plus ~1 retry on collision. Total ~100 cycles of LDS atomic
per wave per coarse bin — negligible against rasterization work.

**No inter-wave atomics on the framebuffer**: each wave writes to its
claimed sub-tile exclusively, the same property as the streaming
proposal's static wave-per-fine-tile assignment, but with dynamic load
balance.

## 3. Mode Selection Strategy

The "try distribute first, fall back to render" pattern in §2.3 is the
baseline. It tends to keep exactly one WG in distribute mode at any
time (whichever wins the host queue lock), with the rest rendering.
Mode rotation happens naturally as WGs release and re-claim the host
queue lock between iterations.

### 3.1 Greedy Distribute (Baseline)

```c
if (try_claim(&host_queue_lock)) { distribute(); ... }
else                              { render(...); }
```

Active distributor keeps re-claiming as long as the host queue has
work. Maximizes distribution throughput. Suitable for steady-state.

### 3.2 Demand-Based Distribute

Check host queue depth (`back - front`) before attempting the host
queue lock. Avoids unnecessary CAS contention on the lock when the
host has nothing pending.

```c
uint32_t depth = host_queue_back - host_queue_front;
if (depth > 0 && try_claim(&host_queue_lock)) { distribute(); ... }
else                                          { render(...); }
```

Particularly useful when the host produces in bursts with idle gaps
between them.

### 3.3 Rotating Distribute

After a WG finishes a `distribute()`, it skips the host queue claim on
the next iteration (sets a per-WG flag). Rotates the distributor
identity across WGs to spread the LDS bucketing cost. Useful only if
distribute mode imposes meaningful per-WG overhead beyond the work
itself; usually unnecessary.

### 3.4 Reserved-Distributor Pool

Dedicate a subset of WGs (e.g., `wg_id % K == 0`) to distribute only;
others to render only. Re-introduces a soft role split as a tuning
hint, not a structural constraint. Distributor pool can be small (1–4
WGs) since distribution is light per batch.

If you go this far, the streaming proposal is probably the better
design — it provides the same shape with cleaner semantics.

### 3.5 Recommendation

Greedy distribute as baseline. Move to demand-based if the host queue is
often empty and CAS contention shows up in profiling.

## 4. Tile Selection Strategy

When a WG decides to render, it picks which tile to claim. With B tiles
and `B >> N`, simple strategies suffice.

### 4.1 Stride-Offset Linear Scan (Recommended Baseline)

```c
__device__ uint32_t pick_tile() {
  uint32_t start = (wg_id * 2654435761u) % B;   // Knuth multiplicative hash
  for (uint32_t off = 0; off < B; ++off) {
    uint32_t tile = (start + off) % B;
    // Renderer needs fully-UNLOCKED state (0x0); reject anything else.
    if (atomic_load_relaxed(&tile_lock[tile]) == 0x0 &&
        atomic_load_relaxed(&tile_pending[tile]) > 0) {
      return tile;            // candidate; render_try_acquire still races
    }
  }
  return NO_TILE;
}
```

The hash gives each WG a different starting offset; initial claims are
spread across the tile space. With `B/N > 16` the first slot examined
is almost always free. The relaxed loads before the CAS are hints only
— actual ownership is decided by `render_try_acquire`'s CAS (§2.2.2).

`tile_pending[tile]` is a coarse counter incremented when distributor
pushes to a tile and decremented when renderer drains. Skipping empty
tiles avoids claiming a lock just to find no work.

Expected cost: 2 loads + 1 CAS per pick on success. ~80 ns total.

### 4.2 Workload-Aware

Use `tile_pending[tile]` to scan for the highest-workload free tile.
Better load balance, costs an extra atomic per push. Use only when
profiling shows naive picking causes imbalance.

### 4.3 Two-Phase Home + Steal

Each WG has a home range `[wg_id × B/N, (wg_id+1) × B/N)`. Tries home
first, falls back to global stride-offset scan if all home tiles are
locked or empty.

Trades some load balance for spatial locality: home tiles are spatially
close, so texture/lightmap caches stay hot across iterations on the
same WG.

### 4.4 Z-Order Locality

Map `tile_id` to a Morton-coded spatial position. Home ranges in
Z-order ensure adjacent tiles share texture cache lines. Significant
memory-bandwidth win on big scenes; complicates tile-id arithmetic.

### 4.5 Recommendation

Stride-offset linear scan with `tile_pending` skip as baseline. Add
workload-aware (4.2) only if profiling shows imbalance. Add Z-order
(4.4) if memory bandwidth becomes the bottleneck.

## 5. Comparison To Streaming Pipeline Proposal

| Concern | Streaming (wave-disjoint) | Bin-ownership (this) |
|---|---|---|
| Distributor count | 1 WG (fixed) | At most 1 WG at a time (rotating) |
| Tile queue producer | Single wave (structural) | WG holding `TILE_QLOCK` |
| Tile queue consumer | Multi WG (SPMC, atomic on `front`) | WG holding `TILE_QLOCK` (drain phase) |
| Tile queue mutual exclusion | None needed (single-producer + SPMC consumer) | Per-tile 3-state lock |
| Concurrent push during rasterize | N/A | Yes (distributor + renderer overlap) |
| Bin counter atomics | None | None |
| Framebuffer atomics | None | None (per-tile renderer-exclusive) |
| Per-iteration global atomics | Few (queue counters) | 2 CAS + 1 atomicAnd per phase |
| Global queue model | SPSC (1 host, 1 wave 0) | SPSC (1 host, 1 GPU lock holder) |
| Best regime | `B/N` moderate, balanced | `B/N >> 1`, variable, bursty |
| Failure mode | Single-distributor cap binds | Renderer claim contention on hot tiles |
| Implementation complexity | Lower | Higher |

Both designs preserve "exactly one writer per tile queue at any moment"
— wave-disjoint achieves it structurally, ownership-lock achieves it
dynamically. The framebuffer is atomic-free in both.

The two designs are not stackable. Choosing one means committing to its
WG-loop structure.

## 6. Memory Footprint

The pipeline's global memory cost is dominated by **per-tile queue state**.
Everything else is fixed-overhead or scales with primitive count rather
than tile count. Numbers below are at a 32-bit `tile_lock` and a 32-bit
primitive-id queue entry; see §6.3 for sizing the queue capacity `C`.

### 6.1 Per-Tile State

Each coarse tile carries a fixed-size record:

| Field | Bytes | Where it lives |
|---|---|---|
| `tile_lock[i]` (3-state, §2.2.2) | 4 | VRAM |
| `tile_pending[i]` (depth hint) | 4 | VRAM |
| `tile_queue.head` | 4 | VRAM |
| `tile_queue.tail` | 4 | VRAM |
| `tile_queue.data[C]` (uint32 prim ids) | 4·C | VRAM |
| **Total** | **16 + 4·C** | VRAM |

Each queue entry is a `uint32_t` primitive index; actual primitive
screen-space data lives once in the streaming primitive store (§6.5),
not duplicated per tile.

The tile arrays live in **device-local VRAM**, not fine-grained SVM:
only GPU WGs touch them, the host never reads or writes. PCIe-coherent
SVM is reserved for the host queue, the `terminate` flag, and the
heartbeat counter (§6.5).

### 6.2 Total Tile-State Memory

Tile count `B = ceil(W/T) × ceil(H/T)`:

| Resolution | T=32 | T=64 | T=128 |
|---|---|---|---|
| 1280×720 | 920 | 240 | 60 |
| 1920×1080 | 2 040 | 510 | 135 |
| 2560×1440 | 3 600 | 920 | 240 |
| 3840×2160 | 8 160 | 2 040 | 510 |

Total tile-state memory at common queue capacities, `B · (16 + 4·C)`:

| Resolution | Tile | B | C=128 | C=256 | C=512 | C=1024 |
|---|---|---|---|---|---|---|
| 1080p | 32 | 2 040 | 1.03 MB | 2.02 MB | 4.02 MB | 8.00 MB |
| 1080p | 64 | 510   | 0.26 MB | 0.51 MB | 1.00 MB | 2.00 MB |
| 1080p | 128 | 135  | 0.07 MB | 0.13 MB | 0.27 MB | 0.53 MB |
| 1440p | 32 | 3 600 | 1.81 MB | 3.57 MB | 7.09 MB | 14.1 MB |
| 1440p | 64 | 920   | 0.46 MB | 0.91 MB | 1.81 MB | 3.61 MB |
| 4K    | 32 | 8 160 | 4.11 MB | 8.09 MB | 16.1 MB | 32.0 MB |
| 4K    | 64 | 2 040 | 1.03 MB | 2.02 MB | 4.02 MB | 8.00 MB |
| 4K    | 128 | 510  | 0.26 MB | 0.51 MB | 1.00 MB | 2.00 MB |

All comfortably below 1% of an 8 GB GPU at any sane configuration. The
4K 32×32 with C=1024 (32 MB) is the only entry approaching "noticeable",
and still negligible against framebuffer + textures.

### 6.3 Sizing C

Queue capacity needs to absorb the largest plausible burst between
drains. Three regimes to consider:

- **Per-push burst.** Distribute mode pushes one LDS bucket per
  `TILE_QLOCK` acquire. Bucket cap is `BUCKET_DEPTH` (typical 32–64
  prims). One push can never exceed this.
- **Multi-push between drains.** While a tile is in `TILE_RENDER`
  (rasterizing, ~50 μs), distributors can push concurrently. With one
  active distributor and ~10 μs per push, that's ≤5 pushes ≤320 prims
  per tile during a single render pass.
- **Aggregate hot-tile load.** For Quake-shape (P = 10K prims/frame,
  f ≈ 4 tile hits per prim), uniform distribution yields `f·P/B ≈ 20`
  prims per tile per frame at 1080p 32×32. Hot tiles (sky dome, large
  brush) carry 10–20× the average → 200–400 prims/tile/frame, drained
  across multiple render passes.

Practical recommendations:

| Workload | C |
|---|---|
| Quake at 1080p (light) | 128 |
| Quake at 1440p / 4K | 256 |
| Hot-tile-heavy or >50K prims/frame | 512 |
| Worst-case headroom (no overflow tolerance) | 1024 |

Start at **C=256**. Profile queue-full events; drop to 128 if observed
max depth stays well under, raise to 512 if overflow protection becomes
a concern.

### 6.4 Overflow Policy

When a tile queue is full, three choices with different memory cost:

1. **Block-on-full** (distributor spins or yields). Zero extra memory;
   potentially adds latency. The default assumed by §2.4.
2. **Spill to global overflow pool**. One per-frame VRAM region (e.g.,
   4 MB) holds primitives that didn't fit; a fallback rasterization
   pass walks the spill list against affected tiles. Cheap memory,
   complex code path.
3. **Bounded drop with re-rasterize.** `B/8` byte bitmap of "skipped
   tiles" plus the spill pool; a re-pass at frame end fixes them.
   Acceptable for opaque depth-sorted geometry.

The bin-ownership proposal assumes (1) by default. If profiling shows
distributor stalls on full queues are frequent, (2) is the natural
extension.

### 6.5 Other Memory Categories

Independent of `B` and `C`, but required by the pipeline:

| Buffer | Sizing | Where |
|---|---|---|
| Host queue (batch ring) | 256 batches × ~256 B header ≈ 64 KB | fine SVM |
| Primitive store | P_max × ~64 B; 50K × 64 B = 3.2 MB | fine SVM |
| `host_queue_lock` | 4 B | fine SVM |
| `terminate` flag | 4 B | fine SVM |
| Heartbeat counter | 4 B | fine SVM |
| Coarse depth (HiZ, optional) | B × 8 B | VRAM |
| Framebuffer (color + depth) | W·H · 8 B | VRAM |
| LDS scratch (per WG) | ~16 KB / WG | LDS (on-chip) |

Fine-grained SVM total is under 5 MB even with a generous primitive
store — important because fine-grained SVM atomics cross PCIe and are
significantly more expensive than VRAM atomics.

If the streaming proposal's HiZ extension is adopted, the per-tile
`B × 8 B` is negligible (64 KB at 4K 32×32).

### 6.6 Total Pipeline Footprint

Sum at the recommended **C=256** default:

| Resolution | Tile | Tile state | Other (fine SVM) | Other (VRAM, FB) | Total |
|---|---|---|---|---|---|
| 1080p | 32 | 2.02 MB | ~3.5 MB | ~16 MB | ~22 MB |
| 1440p | 32 | 3.57 MB | ~3.5 MB | ~28 MB | ~35 MB |
| 4K | 32 | 8.09 MB | ~3.5 MB | ~63 MB | ~75 MB |
| 4K | 64 | 2.02 MB | ~3.5 MB | ~63 MB | ~69 MB |

The framebuffer dominates at high resolutions; tile state is a small
contribution. The `B/N ≥ 16` recommendation in §1 (favoring smaller
tiles for fewer renderer-claim collisions, §7.2) costs only a few MB
of extra VRAM — there is no memory pressure to push toward larger
tiles.

## 7. Open Issues

### 7.1 Distributor Push Contention On Hot Tiles (Severity: low)

The 3-state lock decouples push from rasterize, so distributors are
*not* blocked on rasterization. They only contend with another
distributor pushing to the same tile or a renderer in its drain phase
— both ~μs windows. With `B >> N` and per-batch bucket scattering,
contention is rare; with hot-tile workloads (many primitives target
the same coarse bin), distributor's phase-2 retry loop may revisit the
same bucket several times.

Mitigations if profiling shows this matters:

- Sharded host queues (§7.7) — multiple parallel distributors push to
  different shards in parallel.
- Larger `B` — finer bins reduce per-bin push frequency.

### 7.2 Renderer Claim Contention (Severity: medium)

Other WGs that want to render a tile already in `TILE_RENDER` state
fail their CAS (`render_try_acquire` requires fully UNLOCKED) and pick
a different tile. With `B >> N` they find one within 1–2 picks. With
`B/N < 4`, the failure rate climbs; many WGs may scan multiple tiles
before finding one in state `0x0`.

This is the binding constraint on the bin-ownership design's lower
bound for `B/N`. Rasterization time (10K–100K cycles) dominates
TILE_RENDER hold time, so the population of "free" tiles cycles slowly.

Mitigation: keep `B/N` large by sizing coarse bins small. Profile
`render_try_acquire` failure rate; if > 5%, increase B.

### 7.3 Distributor Starvation (Severity: low)

In greedy-distribute mode (§3.1), only one WG is distributing at a
time, so the rest are rendering. Distribution is single-threaded but
typically light enough to keep up.

If the host produces faster than one WG can distribute, host queue
fills, host stalls pushing. Diagnose by monitoring host queue depth.
Mitigations: sharded host queues, larger batch size, or move to the
streaming proposal if the bottleneck is consistent.

### 7.4 Deadlock Avoidance (Severity: low)

Lock ordering rule (§2.2): distributor holds host_queue_lock then at
most one tile_lock; renderer holds only tile_lock; no cycles. Trivially
deadlock-free.

### 7.5 Choosing B (Severity: high)

`B` is the central tuning parameter:

- Too small: renderers cannot find UNLOCKED tiles; `render_try_acquire`
  CAS-failure rate climbs; effective parallelism collapses (§7.2).
- Too large: per-pixel overhead from finer bin granularity, tile-state
  cache pressure, more queue traffic, larger `tile_lock[]` array (see
  §6.2 for footprint at each B).

Initial guidance:

| Render res | Recommended bin size |
|---|---|
| 1080p | 32×32 |
| 1440p | 32×32 |
| 4K | 32×32 or 64×64 |

Profile lock contention and per-bin work distribution; tune. The
memory cost of "more bins" is negligible at any sane resolution
(§6.2), so the practical lower bound on T comes from §7.2 (renderer
claim contention) rather than memory pressure.

### 7.6 Fairness (Severity: low)

`atomicCAS` provides no fairness guarantee. In practice with persistent
WGs, `B >> N`, and stride-offset hashing, all WGs make forward
progress. If a future workload shows starvation, consider ticket locks.

### 7.7 Sharded Host Queues (Severity: enhancement)

If one distributor cannot keep up with host production, shard the host
queue into K parallel sub-queues, each with its own lock. Host writes
to sub-queue `(prim_id % K)`. WGs claim a random sub-queue lock. Allows
up to K parallel distributors at the cost of host-side push complexity.

Worth implementing only if §7.3 measurably bites.

### 7.8 Lock Holder Eviction (Severity: low)

A WG that claims a lock and gets descheduled holds the lock
indefinitely. With persistent megakernels and resident WGs (no
preemption mid-kernel on AMDGPU compute queues), this should not
happen. If KFD ever introduces compute preemption, this design becomes
fragile and would need a watchdog-based lock release.

## 8. Shared Infrastructure With Streaming Pipeline Proposal

The following are unchanged from `streaming-pipeline-proposal.md` and
are not re-described here:

- Megakernel + persistent WGs + oversubscribed launch with fast-exit
  (§2.9 there).
- Fine-grained coherent SVM allocation requirements.
- Global `terminate` flag in fine-grained SVM.
- Host watchdog (heartbeat counter).
- KFD watchdog compatibility.
- Wave size portability (compile wave64 across all archs as default).
- `libkfd` extensions: async dispatch, fine-grained SVM, fence polling
  with timeout, watchdog/priority controls.

The wave-level role within a WG (§2.3 there) — wave 0 polls + LDS
broadcast + `s_barrier` — is also the same pattern, applied here to
"pick mode" + "process under lock" rather than "pop a batch" +
"distribute".

## 9. Concrete Refinement (If Implemented)

The minimal shippable instantiation, in priority order:

1. **WG count**: `N = 2 × hw_occupancy` symmetric WGs. No role
   distinction by `wg_id`.
2. **Waves per WG**: `W = 4`, wave64.
3. **Locks**: `host_queue_lock` (binary, single uint32_t) and
   `tile_lock[B]` (3-state, two-bit field per tile, §2.2.2), both in
   fine-grained SVM. CAS acquire, no spin on failure.
4. **Mode selection**: greedy distribute (§3.1) as baseline; switch to
   demand-based (§3.2) if host queue often empty.
5. **Distribute mode**: bucket by tile in LDS (per-batch tile set),
   then push each bucket under that tile's `TILE_QLOCK` bit (acquired
   from either UNLOCKED or RENDERING). Phase-2 retries contended tiles.
6. **Render mode**: 3-phase protocol (§2.5) — acquire from UNLOCKED,
   drain into LDS, transition to TILE_RENDER, rasterize, release. New
   pushes during rasterize are rendered next claim of this tile.
7. **Tile selection**: stride-offset linear scan with `tile_pending`
   skip (§4.1).
8. **Within-WG sub-tile**: LDS atomic-CAS sub-tile claim (§2.6), 16
   sub-tiles per coarse bin, 4 waves.
9. **Bin size**: 32×32 to start. 64×64 for 4K if `B/N ≥ 16` still
   holds. Avoid 128×128 unless `B/N ≥ 8` after measurement. Memory
   footprint at each option is in §6.2; queue capacity sizing is in
   §6.3.
10. **Termination**: `terminate` flag in fine-grained SVM (per
    `streaming-pipeline-proposal.md` §2.8).
11. **Oversubscription and fast-exit**: as in
    `streaming-pipeline-proposal.md` §2.9.
12. **Host watchdog**: heartbeat counter; any WG can check on its
    iteration.

## 10. Where This Slots In The Existing Roadmap

`quake-compute-rasterizer-todo.md` Stage F (hierarchical binning) lands
either of:

- The streaming proposal (`streaming-pipeline-proposal.md`).
- This proposal (bin-ownership locks).
- Something simpler that pre-dates either (Stage F as currently
  scoped: add a coarse level above 16×16 fine tiles, still single
  dispatch, still barrier-per-stage).

The two proposals are not stackable. If both are implemented they live
behind a runtime selector that picks based on `B/N`:

- `B/N < 8` → streaming proposal.
- `B/N ≥ 16` → bin-ownership.
- `8 ≤ B/N < 16` → either; benchmark.

A reasonable staging:

1. Land Stage F as currently described (additive, low risk).
2. If targeting 4K or high-density scenes, prototype this proposal.
3. If targeting 1080p Quake-shape only, prototype the streaming
   proposal.
4. Cross-pollinate: both designs share `libkfd` extensions and the
   wave-level intra-WG patterns.

## 11. Reading

- `streaming-pipeline-proposal.md` — the structurally-partitioned
  alternative. Required reading for shared infrastructure (§8 of this
  doc lists what carries over).
- `cure-streaming-queues.md` — the queue primitive vocabulary,
  including the SPMC variant of `MultiIndexQueue`.
- `quake-compute-rasterizer-prior-art.md` — lineage and why
  cuRE-shaped designs are the right reference.
- `quake-compute-rasterizer-plan.md` — current architecture being
  replaced.
- Steinberger et al., *Whippletree: Task-based Scheduling of Dynamic
  Workloads on the GPU*, TOG 2014. The dynamic-task model this
  proposal generalizes from.
- Chase, Lev, *Dynamic Circular Work-Stealing Deque*, SPAA 2005. The
  classical work-stealing primitive that informs the within-WG
  sub-tile claim pattern.
