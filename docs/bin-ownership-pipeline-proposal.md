# Pipeline Proposal: Bin-Ownership Megakernel

Status: design proposal. Not implemented. The canonical pipeline
shape for the streaming-rasterizer demo on AMDGPU + libkfd.

The pipeline is one persistent compute megakernel running `N`
symmetric workgroups. Each WG, on each iteration, picks one of two
modes:

1. **Distribute mode**: pop a primitive batch from the host↔GPU
   ring, scatter the batch's primitives into per-tile queues.
2. **Render mode**: drain one tile's queue, rasterize the
   primitives, write the framebuffer slice for that tile.

Mutual exclusion is enforced via two kinds of ownership locks:

- One **host queue lock** (binary), shared between the host and GPU
  WGs. Whichever entity (the host pushing, or a distributor WG
  popping) holds the flag has exclusive access to the host queue's
  pointers and slots. At most one party touches the queue at a time.
- B **tile locks** (3-state, two independent bits — §2.2.2), one per
  coarse bin. The two bits separately track *queue access* (single-
  writer/reader) and *rasterization* (single-renderer-per-tile) so a
  distributor can push new primitives to a tile while a renderer is
  rasterizing it.

The active distributor identity rotates as WGs release the host queue
lock and other WGs (or the host) re-claim it. Each tile cycles
through queue-access and rendering states independently and can
overlap them.

Companion docs:

- `cure-streaming-queues.md` — vocabulary for streaming queue
  primitives borrowed from cuRE.
- `frame-completion-detection.md` — how the host detects end-of-
  frame against the persistent megakernel.

## 1. Why This Shape

The bin-ownership shape is robust against three failure modes that
fixed-role designs (one distributor WG, N−1 renderer WGs) struggle
with:

- **`B` much larger than `N` (resident WG count).** A single
  distributor WG cannot keep up with binning when there are many
  bins; renderers idle waiting for fan-out. Bin-ownership lets every
  WG distribute when the host queue has work and render otherwise.
- **High per-bin work variance.** A few hot bins bottleneck while
  idle renderer WGs cannot help. Bin-ownership lets idle WGs claim
  any unrendered tile.
- **Bursty producer load.** A fixed distributor WG saturates while
  renderer WGs idle, or vice versa. Bin-ownership absorbs bursts by
  letting any WG distribute.

`B` and `N` define a design-determining ratio. Concrete numbers
(RDNA2 6700 XT, ~80 resident WGs at typical occupancy):

| Render res | Coarse bin size | B | B/N |
|---|---|---|---|
| 1080p | 32×32 | ~2 K  | 25  |
| 1080p | 64×64 | ~510  | 6   |
| 1080p | 128×128 | ~135 | 1.7 |
| 4K | 32×32 | ~8 K  | 100 |
| 4K | 64×64 | ~2 K  | 25  |
| 4K | 128×128 | ~510 | 6   |

The model is robust at `B/N ≥ 16` (random tile claim is essentially
uncontended). It works at `4 ≤ B/N ≤ 16` with picking heuristics
(§4). It breaks down below `B/N < 4` — at that point a fixed-role
partitioning scheme (one or a few WGs as distributors, the rest as
renderers) becomes the better trade. The 1080p teapot demo lands at
`B/N ≈ 25` with 32×32 bins, comfortably inside the robust regime.

## 2. Architecture

### 2.1 High-Level Shape

```text
Host                                GPU (one megakernel)
────                                ─────────────────────────────────────
┌──────────────┐                    ┌─────────────────────────────────────┐
│ scene walk   │                    │ N symmetric WGs (no role split)     │
│ + batching   │ ◄─acq/rel host_q─► │  loop:                              │
│ push under   │ ─CAS lock─►        │    batch = NULL                     │
│ host_q_lock  │                    │    if try_claim(host_queue_lock):   │
└──────────────┘                    │      batch = pop_host_queue()       │
                    ┌──────────┐    │      release(host_queue_lock)       │
                    │ Host     │ ◄──┤    if batch:                        │
                    │ queue    │    │      distribute_batch() {           │
                    │ +shared  │    │        bucket by tile in LDS        │
                    │ lock     │    │        for each bucket:             │
                    └──────────┘    │          claim tile_lock briefly,   │
                                    │          push, release              │
                                    │      }                              │
                                    │    else:                            │
                                    │      tile = pick_tile()             │
                                    │      if render_try_acquire(tile):   │
                                    │        render(tile)                 │
                                    │        release tile_lock[tile]      │
                                    └─────────────────────────────────────┘
                                                ▲
                                                │
                                       ┌────────┴────────┐
                                       │ Per-tile state  │
                                       │  - queue        │
                                       │  - lock (3-state)│
                                       │ (B of these)    │
                                       └─────────────────┘
                                                │
                                                ▼
                                       framebuffer
                                       (per-tile exclusive write)
```

Key properties:

- The host queue is gated by a single binary lock that is **shared
  between the host and GPU WGs**. The host acquires it to push; a WG
  acquires it to pop. Whichever side wins the CAS owns the queue for
  one operation; the lock-holder identity rotates dynamically. Pops
  release the lock immediately so multiple WGs can run
  `distribute_batch` concurrently after popping (§2.3).
- Each tile is gated by a 3-state lock (§2.2.2): a queue-access bit
  (push or drain) and an independent rendering bit. Distributor pushes
  and renderer rasterization can overlap on the same tile; only the
  brief queue-access window is mutually exclusive.
- WG roles are dynamic per iteration, not fixed by `wg_id`.

### 2.2 The Two Lock Types

The host queue lock is a plain binary lock, **placed in fine-grained
SVM** so both the host and GPU WGs can CAS it. Tile locks are 3-state
with two independent ownership bits, in **VRAM** (GPU-only).

```c
// fine-grained SVM (host + GPU access)
volatile _Atomic uint32_t host_queue_lock;     // 0 = UNLOCKED, 1 = LOCKED

// VRAM (GPU-only)
__device__ uint32_t tile_lock[B];              // 2-bit field, see below
```

#### 2.2.1 Host queue lock (binary, host+GPU shared)

```c
// fine-grained SVM, addressable from both host and GPU
volatile _Atomic uint32_t host_queue_lock;     // 0 = UNLOCKED, 1 = LOCKED

__device__ /* or host */
bool try_claim(uint32_t *lock) {
  return atomicCAS(lock, UNLOCKED, LOCKED) == UNLOCKED;
}

__device__ /* or host */
void release(uint32_t *lock) {
  atomic_thread_fence(memory_order_release);
  atomicExch(lock, UNLOCKED);
}
```

This lock is in fine-grained SVM — both the host and GPU WGs target it
with `atomicCAS`/`atomicExch`. On the host side these compile to
appropriate `__atomic_compare_exchange` / `__atomic_exchange` builtins
with PCIe-coherent semantics. Acquire-side memory ordering is on the
CAS itself; the matching release fence ensures payload writes are
visible to the next acquirer regardless of which side it's on.

Whichever side holds the flag has exclusive access to the queue's
`front` and `back` pointers and to the queue slots that it touches
during one push or one pop. Neither side spins while holding the lock
— if the work cannot be completed (host: queue full; GPU: queue empty),
the lock is released first and the wait happens unlocked (§2.7).

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

The lock-hold window for `host_queue_lock` is kept as small as
possible — only the actual pop. Bucketing and tile pushes happen
*after* releasing it, so the host's competing pushes are not blocked
by the WG's distribution work.

```c
uint32_t attempts = 0;                    // §2.8 backoff counter
for (;;) {
  if (atomic_load(&terminate)) break;

  // §3 covers alternative mode-selection policies.
  StreamBatch batch;
  bool got_batch = false;
  if (try_claim(&host_queue_lock)) {
    got_batch = pop_host_queue(&batch);
    release(&host_queue_lock);            // release ASAP — host may be waiting
  }

  if (got_batch) {
    distribute_batch(&batch);             // §2.4; no host_queue_lock held
    attempts = 0;                         // productive: reset backoff
  } else {
    // Either CAS failed (host or another distributor holds the lock)
    // or queue was empty. Fall through to render mode.
    uint32_t tile = pick_tile();          // §4
    if (tile == NO_TILE) { wg_backoff(&attempts); continue; }   // §2.8
    if (render_try_acquire(tile)) {
      render(tile);                       // §2.5; releases TILE_RENDER inside
      attempts = 0;
    }
    // CAS-fail: tile is busy (queue access or another renderer);
    // just loop and try a different tile next iteration.
  }
}
```

`pop_host_queue` returns false if the queue is empty; that's the
fast-out path for an idle frame. When the host has work pending, the
typical WG iteration is: CAS-acquire (~50 ns) + pop (~few hundred ns,
one cache line) + release (~50 ns) ≈ 1 μs of lock hold. That's also
the maximum window the host has to wait per pop.

Inside `distribute_batch()` and `render()` the WG uses the standard
wave-level cooperation pattern: wave 0 performs the queue-touching
operations (CAS, dequeue head, etc.), writes results into LDS,
`s_barrier`s, and all `W` waves of the WG then cooperate on the
broadcasted unit of work. Lane 0 of wave 0 specifically performs the
atomic CAS on locks; the rest of the wave is a no-op until the LDS
broadcast.

### 2.4 Distribute Mode

The distributor processes one batch (already popped under
`host_queue_lock` per §2.3) per `distribute_batch()` call. It buckets
primitives by target tile in LDS first (no global atomics), then
pushes each bucket under that tile's queue-lock bit.

```c
__device__ void distribute_batch(StreamBatch *batch) {
  // Phase 1: bucket-by-tile in LDS (parallel across waves, no atomics
  //          on tile queues yet)
  __shared__ uint32_t lds_bucket_count[B_SUBSET];
  __shared__ uint32_t lds_bucket_prim[B_SUBSET][BUCKET_DEPTH];
  bucket_by_tile(batch, lds_bucket_count, lds_bucket_prim);

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
    if (pending > 0) wg_backoff(&attempts);  // §2.8; retry contended buckets
  }
}
```

Phases 1 and 2 do not touch `host_queue_lock`. Multiple WGs can be in
`distribute_batch()` concurrently — each works on its own popped batch
and pushes to tile queues using per-tile locks (§2.2.2). The only
serialized point is the pop itself.

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
  primitives can hit (upper bound by primitive AABBs). For typical
  demo batches (rotating teapot, ~6 K triangles split into batches of
  256) this is 8–32 distinct tiles per batch.

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

`rasterize_tile` subdivides the coarse bin into `W` fine sub-tiles
(one per wave) and lets each wave rasterize its assigned sub-tile.
This decomposition is the key to atomic-free framebuffer writes:
each wave owns a disjoint pixel region, so depth tests, blends, and
color writes need no cross-wave synchronization. Geometry table:

| Wave size | W | Fine sub-tile | Coarse bin |
|---|---|---|---|
| 64 (GFX9; RDNA wave64 opt-in) | 4 | 16×16 (256 px, 4 passes) | 32×32 (1024 px) |
| 32 (RDNA default) | 4 | 8×16 (128 px, 4 passes) | 16×32 (512 px) |

§2.6 below describes both static (one wave per sub-tile) and dynamic
(LDS-CAS-claimed) sub-tile assignment.

### 2.6 Within-WG Sub-Tile Ownership

Inside `rasterize_tile`, the W waves can either statically partition
fine sub-tiles (one wave per sub-tile) or dynamically claim sub-tiles
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

### 2.7 Host-Side Push Protocol

The host walks the scene, builds batches in fine-grained SVM, and
pushes each completed batch to the host queue. The push competes with
GPU distributor WGs for `host_queue_lock` (§2.2.1) — the lock is
shared, not GPU-internal. Whichever side wins the CAS owns the queue
for one operation.

The push protocol has three steps: **acquire, push as much as fits,
release**. If not everything fits, back off (without holding the
lock) and retry from acquire.

```c
void host_push_n(StreamBatch *batches, size_t n) {
  size_t pushed = 0;
  uint32_t spins = 0, yields = 0;
  struct timespec ts = { 0, 1000 };   // sleep tier starts at 1 μs

  while (pushed < n) {
    if (try_claim(&host_queue_lock)) {
      // Step 2: lock held — push as many batches as fit.
      while (pushed < n && queue_has_space(&q)) {
        push_unlocked(&q, &batches[pushed++]);
      }
      // Step 3: release before any waiting. Holding the lock during
      // backoff would deadlock — distributor WGs need this lock to
      // pop and free slots.
      release(&host_queue_lock);

      if (pushed >= n) return;        // all done
      // else: queue had less space than we needed; fall through to
      // backoff so the GPU can drain.
    }
    // Either CAS failed (someone else holds the lock right now) or
    // we partially pushed and queue is full. Same backoff ladder:
    if (spins < HOST_SPIN_LIMIT) {
      cpu_relax(); ++spins;
    } else if (yields < HOST_YIELD_LIMIT) {
      sched_yield(); ++yields;
    } else {
      nanosleep(&ts, NULL);
      ts.tv_nsec = MIN(ts.tv_nsec * 2, 100000);   // cap at 100 μs
    }
  }
}
```

`push_unlocked` writes one batch's payload at `q->slots[q->back & MASK]`
and stores `back+1` to `q->back`. Both operations happen under the
lock, so no fencing is required between the payload write and the
back-pointer store *for ordering against other writers*. The release
fence built into `release(&host_queue_lock)` (§2.2.1) is what
publishes the new `back` and the payload bytes to the next acquirer.

`queue_has_space(&q)` is a relaxed comparison against `q->front`
(updated by GPU distributors under the same lock). Since both `front`
and `back` are touched only under the lock, no atomic load is needed
*for ordering*; the lock acquire's CAS provides the happens-before
relationship with the previous releaser's writes to either pointer.

#### Critical rules

1. **Never hold the lock during backoff.** Distributor WGs need the
   same lock to drain the queue. Sleeping with the lock held is a
   deadlock: the host waits for space, the GPU can't make space.
   The protocol above releases unconditionally before any wait.
2. **Push as much as fits in one acquire.** Bulk push amortizes the
   CAS+release cost across many batches. The lock-hold window is
   `O(K · memcpy)` for K batches — typically a few μs per batch, so a
   bulk push of 4–8 batches stays under the GPU's typical
   distribute-mode hold time.
3. **Release before checking "all done".** Step 3 above releases first,
   then evaluates `pushed >= n`. Even on the success path, the lock
   doesn't extend past the writes.

#### Backoff budgets

| Tier | Limit | Rationale |
|---|---|---|
| spin | 256 | ~1 μs/pause × 256 ≈ 256 μs, comfortably above typical GPU drain |
| yield | 32 | covers OS scheduler hiccups (sibling thread runs) |
| sleep | unbounded | progressive 1 μs → 100 μs; bounded by watchdog (§7) |

The `cpu_relax` hint matters. On x86 it tells the CPU to relax SMT
scheduling (frees pipeline slots for the SMT sibling) and inserts a
several-cycle delay that reduces L1/L2 cache traffic during the spin.
On ARM `yield` is a similar hint. A tight busy-loop without it
saturates the cache-coherence fabric and slows down the GPU's PCIe
atomic operations on adjacent fine-grained SVM regions — including
the very `host_queue_lock` we are trying to acquire.

#### Why the protocol does not separate "lock contention" from "queue full"

The two failure modes look identical from the host's vantage point:
either the CAS-acquire fails (somebody else holds the lock) or the
acquire succeeds but only some of the pending batches fit. Both
recover the same way — release if held, back off, retry. Treating
them as one keeps the loop tight and avoids per-failure-mode tuning.

Profiling can still distinguish them by counting partial pushes vs
CAS misses; the streaming pipeline's instrumentation hooks (heartbeat
counters per WG) extend naturally to host-side counters.

#### Sharded variant (§6.7)

If multiple host threads concurrently produce batches, the lock-based
push above scales poorly: one host_queue_lock serializes all of them
*and* every distributor WG. Sharded host queues (§6.7) give each
shard its own lock and slot range. A host thread picks a shard
(round-robin or hashed by primitive id), tries the protocol above on
that shard's lock, and falls back to the next shard on contention or
queue-full instead of just sleeping. This turns CAS contention and
queue-full into work-stealing-style producer-side distribution.

### 2.8 GPU-Side Backoff

Several places in the WG loop need to back off when work is
unavailable: phase-2 retry on contended `tile_lock` in
`distribute_batch` (§2.4), `pick_tile()` returning `NO_TILE` (§2.3),
and the case where both `host_queue_lock` and a `tile_lock` cannot be
claimed in the same iteration. The right primitive on AMDGPU is
**`s_sleep`**, exposed by clang as `__builtin_amdgcn_s_sleep(simm16)`.

#### What s_sleep does

`s_sleep N` halts wave issue for approximately `64·N + jitter` cycles.
The wave stays resident — VGPRs, SGPRs, LDS, and EXEC are preserved.
Other waves on the same SIMD can issue freely while this one sleeps,
which is exactly the scheduling hint we want.

| simm16 | Cycles | Wall time @ 2 GHz |
|---|---|---|
| 0 | 1–64 | < 32 ns |
| 1 | 64–128 | 32–64 ns |
| 4 | 256–320 | ~150 ns |
| 16 | 1 024–1 088 | ~520 ns |
| 64 | 4 096–4 160 | ~2 μs |

(simm16 is 7-bit unsigned on GFX9+, so 0–127, max ~4 μs.)

`s_sleep` is the GPU equivalent of x86 `pause` with a tunable delay.
It does **not**:
- release the WG (still resident on a CU),
- yield the CU to another WG,
- wait for outstanding memory ops to complete,
- release any locks the caller may hold.

It does:
- hint to the SIMD scheduler that this wave is idle, freeing issue
  slots for co-resident waves;
- insert enough delay for an in-flight cache-line invalidation or
  PCIe atomic round-trip to complete before the next CAS.

This matters: a tight CAS-spin without `s_sleep` slams the cache
coherence fabric and slows down the very PCIe atomics it is racing
on, including the host's `host_queue_lock` acquire.

#### Progressive backoff

```c
__device__ void wg_backoff(uint32_t *attempts) {
  uint32_t n = *attempts;
  if      (n <  4)  __builtin_amdgcn_s_sleep(1);     //  ~64 cycles
  else if (n < 16)  __builtin_amdgcn_s_sleep(4);     //  ~250 ns
  else if (n < 64)  __builtin_amdgcn_s_sleep(16);    //  ~520 ns
  else              __builtin_amdgcn_s_sleep(64);    //  ~2 μs (cap)
  if (n < 256) ++(*attempts);
}
```

The `attempts` counter is per-WG (lives in SGPRs) and resets to 0
whenever the WG completes a productive iteration — a successful pop,
a successful tile push, or a successful render. Sustained "no work"
escalates the sleep up to the ~2 μs cap.

```c
uint32_t attempts = 0;
for (;;) {
  if (atomic_load(&terminate)) break;
  if (try_pop_or_render()) { attempts = 0; continue; }
  wg_backoff(&attempts);
}
```

The cap matters: capping at ~2 μs keeps the WG responsive to new host
pushes (which arrive at ≤ host queue latency, ~μs), and keeps total
backoff time well under the watchdog threshold (§7.4, typically 1–10 ms).

#### Where to call wg_backoff

| Site | Backoff? | Rationale |
|---|---|---|
| §2.3 main loop CAS-fail on `host_queue_lock`, no tile claimable | yes | fully idle |
| §2.3 main loop CAS-fail on `host_queue_lock`, tile claimable | no | render is productive |
| §2.3 `pick_tile` returns `NO_TILE` | yes, before retry | no work this iteration |
| §2.4 phase-2 retry loop with `pending > 0` | yes, between scans | tile_lock contention |
| §2.5 `render_try_acquire` fails | no | just pick another tile next iter |

#### Wave-scope semantics

`s_sleep` is a scalar instruction: it executes on the SIMD's scalar
unit, halts the entire wave (all 32 or 64 lanes) for the configured
duration, and is independent of EXEC. Calling it once from wave 0
(the pollster wave per §2.3) puts that wave to sleep; the other
waves in the WG should be parked on an LDS `s_barrier` waiting for
work-broadcast and incur no additional backoff cost.

#### What about `s_setprio`, kernel exit, or `s_endpgm`?

- `s_setprio` (wave scheduling priority) is non-portable across GFX
  gens and easy to misuse — a wave that sets prio=0 and then sleeps
  can starve indefinitely on some archs. Avoid.
- Kernel exit defeats the megakernel: re-dispatch costs μs of fence +
  scheduler overhead and loses LDS state. The whole point of
  persistent WGs is to avoid this loop.
- `s_endpgm` only on the termination path (§7.1 termination); never
  as a backoff.

#### Comparison with host-side backoff

| Concern | Host (§2.7) | GPU (this section) |
|---|---|---|
| Primitive | `cpu_relax` / `pause` | `s_sleep` |
| Tier 2 | `sched_yield` | longer `s_sleep` |
| Tier 3 | `nanosleep` (progressive, μs → 100 μs) | `s_sleep(64)` cap (~2 μs per call) |
| Releases register state | n/a (just stack) | no — wave stays resident |
| Releases the lock | yes (mandatory) | yes (mandatory) |
| Wakeup mechanism | OS scheduler / cache coherence | SIMD scheduler / cache coherence |
| Per-call cap | unbounded (`nanosleep`) | ~2 μs (`s_sleep` simm16 max) |

The per-call cap matters: on the GPU there is no "wait until X
happens" primitive cheap enough to run in the megakernel. `s_sleep`
maxes out at ~2 μs and the WG must re-poll. That's fine because the
megakernel design assumes WGs are checking for work continuously
anyway. Total time spent in backoff is bounded by the watchdog
heartbeat threshold (§7.3); a WG that spends 100% of its time sleeping
will eventually trip the watchdog because it never bumps its
heartbeat counter — the heartbeat increment should happen only on
productive iterations.

## 3. Mode Selection Strategy

The "try pop first, fall back to render" pattern in §2.3 is the
baseline. The `host_queue_lock` is held only across the pop (~1 μs);
multiple WGs can be in `distribute_batch` concurrently after popping,
each operating on a different batch. Mode rotation happens naturally
as WGs and the host alternate as lock holders.

### 3.1 Greedy Pop-Then-Distribute (Baseline)

```c
StreamBatch batch; bool got = false;
if (try_claim(&host_queue_lock)) {
  got = pop_host_queue(&batch);
  release(&host_queue_lock);
}
if (got) distribute_batch(&batch);
else     render(...);
```

The WG always tries to pop. If the queue is empty or the lock is held
by the host or another distributor, fall through to render. Maximises
distribution throughput; suitable for steady state.

### 3.2 Demand-Based Pop

Check host queue depth (`back - front`) before attempting the lock.
Avoids unnecessary CAS contention against the host's push when the
host has nothing pending.

```c
uint32_t depth = host_queue_back - host_queue_front;   // relaxed read
StreamBatch batch; bool got = false;
if (depth > 0 && try_claim(&host_queue_lock)) {
  got = pop_host_queue(&batch);
  release(&host_queue_lock);
}
if (got) distribute_batch(&batch);
else     render(...);
```

Particularly useful when the host produces in bursts with idle gaps
between them.

### 3.3 Rotating Pop

After a WG finishes a `distribute_batch()`, it skips the host queue
claim on the next iteration (sets a per-WG flag). Rotates the
distributor identity across WGs to spread the LDS bucketing cost.
Useful only if distribute mode imposes meaningful per-WG overhead
beyond the work itself; usually unnecessary.

### 3.4 Reserved-Distributor Pool

Dedicate a subset of WGs (e.g., `wg_id % K == 0`) to distribute only;
others to render only. Re-introduces a soft role split as a tuning
hint, not a structural constraint. Distributor pool can be small (1–4
WGs) since distribution is light per batch.

Defeats the point of dynamic role selection, but useful as a
diagnostic tool: if a fixed split outperforms the dynamic one,
profiling will quickly show which side (distribute or render) is
bottlenecking.

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

## 5. Memory Footprint

The pipeline's global memory cost is dominated by **per-tile queue state**.
Everything else is fixed-overhead or scales with primitive count rather
than tile count. Numbers below are at a 32-bit `tile_lock` and a 32-bit
primitive-id queue entry; see §5.3 for sizing the queue capacity `C`.

### 5.1 Per-Tile State

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
screen-space data lives once in the streaming primitive store (§5.5),
not duplicated per tile.

The tile arrays live in **device-local VRAM**, not fine-grained SVM:
only GPU WGs touch them, the host never reads or writes. PCIe-coherent
SVM is reserved for the host queue, the `terminate` flag, and the
heartbeat counter (§5.5).

### 5.2 Total Tile-State Memory

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

### 5.3 Sizing C

Queue capacity needs to absorb the largest plausible burst between
drains. Three regimes to consider:

- **Per-push burst.** Distribute mode pushes one LDS bucket per
  `TILE_QLOCK` acquire. Bucket cap is `BUCKET_DEPTH` (typical 32–64
  prims). One push can never exceed this.
- **Multi-push between drains.** While a tile is in `TILE_RENDER`
  (rasterizing, ~50 μs), distributors can push concurrently. With one
  active distributor and ~10 μs per push, that's ≤5 pushes ≤320 prims
  per tile during a single render pass.
- **Aggregate hot-tile load.** For demo-shape (P ≈ 6 K triangles per
  frame, f ≈ 4 tile hits per prim), uniform distribution yields
  `f·P/B ≈ 12` prims per tile per frame at 1080p 32×32. Foreground
  tiles can carry 10–20× the average → 120–240 prims/tile/frame,
  drained across multiple render passes.

Practical recommendations:

| Workload | C |
|---|---|
| Demo (teapot, ~6 K triangles) at 1080p | 128 |
| Larger demo scenes / 1440p / 4K | 256 |
| Hot-tile-heavy or >50K prims/frame | 512 |
| Worst-case headroom (no overflow tolerance) | 1024 |

Start at **C=256**. Profile queue-full events; drop to 128 if observed
max depth stays well under, raise to 512 if overflow protection becomes
a concern.

### 5.4 Overflow Policy

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

### 5.5 Other Memory Categories

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

If a hierarchical-Z extension is added (per-tile `(min, max)` depth
in VRAM, used to skip primitives that fail the tile-Z test before
binning), the per-tile `B × 8 B` is negligible (64 KB at 4K 32×32).

### 5.6 Total Pipeline Footprint

Sum at the recommended **C=256** default:

| Resolution | Tile | Tile state | Other (fine SVM) | Other (VRAM, FB) | Total |
|---|---|---|---|---|---|
| 1080p | 32 | 2.02 MB | ~3.5 MB | ~16 MB | ~22 MB |
| 1440p | 32 | 3.57 MB | ~3.5 MB | ~28 MB | ~35 MB |
| 4K | 32 | 8.09 MB | ~3.5 MB | ~63 MB | ~75 MB |
| 4K | 64 | 2.02 MB | ~3.5 MB | ~63 MB | ~69 MB |

The framebuffer dominates at high resolutions; tile state is a small
contribution. The `B/N ≥ 16` recommendation in §1 (favoring smaller
tiles for fewer renderer-claim collisions, §6.2) costs only a few MB
of extra VRAM — there is no memory pressure to push toward larger
tiles.

## 6. Open Issues

### 6.1 Distributor Push Contention On Hot Tiles (Severity: low)

The 3-state lock decouples push from rasterize, so distributors are
*not* blocked on rasterization. They only contend with another
distributor pushing to the same tile or a renderer in its drain phase
— both ~μs windows. With `B >> N` and per-batch bucket scattering,
contention is rare; with hot-tile workloads (many primitives target
the same coarse bin), distributor's phase-2 retry loop may revisit the
same bucket several times.

Mitigations if profiling shows this matters:

- Sharded host queues (§6.7) — multiple parallel distributors push to
  different shards in parallel.
- Larger `B` — finer bins reduce per-bin push frequency.

### 6.2 Renderer Claim Contention (Severity: medium)

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

### 6.3 Distributor Starvation (Severity: low)

`host_queue_lock` is the single serialization point for popping
batches. Multiple WGs can run `distribute_batch` concurrently after
popping (§2.3, §2.4) — the bottleneck is purely the pop CAS, not the
distribution work itself. A pop is ~1 μs, so the lock supports ~10⁶
pops/s in the steady state.

Concrete failure modes:

- **Host outruns CAS throughput.** If the host produces batches
  faster than ~1 M/s, the queue fills and the host's push backs off
  through §2.7's tiered protocol. Diagnose by monitoring host queue
  depth and how often the host crosses into tier 2/3 of the backoff.
- **Bursty contention.** Many WGs hammering the lock at once causes
  CAS misses; only one wins per cycle. Demand-based pop (§3.2) helps
  by skipping the CAS when `back == front`.

Crossing into tier 3 (sleep) regularly is the symptom that even with
parallel distribute the GPU consumer cannot keep up — typically
because individual `distribute_batch` calls are slow (large batches,
hot tiles). Mitigations: sharded host queues (§6.7), larger batch
size (fewer push round-trips per primitive), or fall back to a
fixed-role distributor pool (§3.4) if the bottleneck is consistent.

### 6.4 Deadlock Avoidance (Severity: low)

Lock ordering rule (§2.2): distributor holds host_queue_lock then at
most one tile_lock; renderer holds only tile_lock; no cycles. Trivially
deadlock-free.

### 6.5 Choosing B (Severity: high)

`B` is the central tuning parameter:

- Too small: renderers cannot find UNLOCKED tiles; `render_try_acquire`
  CAS-failure rate climbs; effective parallelism collapses (§6.2).
- Too large: per-pixel overhead from finer bin granularity, tile-state
  cache pressure, more queue traffic, larger `tile_lock[]` array (see
  §5.2 for footprint at each B).

Initial guidance:

| Render res | Recommended bin size |
|---|---|
| 1080p | 32×32 |
| 1440p | 32×32 |
| 4K | 32×32 or 64×64 |

Profile lock contention and per-bin work distribution; tune. The
memory cost of "more bins" is negligible at any sane resolution
(§5.2), so the practical lower bound on T comes from §6.2 (renderer
claim contention) rather than memory pressure.

### 6.6 Fairness (Severity: low)

`atomicCAS` provides no fairness guarantee. In practice with persistent
WGs, `B >> N`, and stride-offset hashing, all WGs make forward
progress. If a future workload shows starvation, consider ticket locks.

### 6.7 Sharded Host Queues (Severity: enhancement)

If one distributor cannot keep up with host production, shard the host
queue into K parallel sub-queues, each with its own lock. Host writes
to sub-queue `(prim_id % K)`. WGs claim a random sub-queue lock. Allows
up to K parallel distributors at the cost of host-side push complexity.

Worth implementing only if §6.3 measurably bites.

### 6.8 Lock Holder Eviction (Severity: low)

A WG that claims a lock and gets descheduled holds the lock
indefinitely. With persistent megakernels and resident WGs (no
preemption mid-kernel on AMDGPU compute queues), this should not
happen. If KFD ever introduces compute preemption, this design becomes
fragile and would need a watchdog-based lock release.

### 6.9 Frame Completion Detection (Severity: medium)

The pipeline as described does not yet specify how the host knows
that all primitives of frame F have been rasterized so it can flip
the framebuffer. cuRE side-steps this by making the megakernel
per-draw, but our persistent megakernel needs an in-kernel ack. See
`frame-completion-detection.md` for variants and the recommended
shape (V2: bulk per-frame counter with per-WG accumulator). Integration
points:

- §2.4 (distributor): forward `batch.frame_id` into each `TileEntry`.
- §2.5 (renderer): add a phase-4 `ack_drain` that flushes a per-WG
  SGPR accumulator into a global `frame_ack[F % N].rendered`.
- §2.7 (host push): host seals the frame by storing
  `frame_ack[F % N].expected = N_F` after pushing all batches.
- Tile queue entry size grows from 4 B to 8 B (or pack `frame_id`
  into high bits of `prim_id`); revisit §5.1 memory tables.

## 7. Persistence Infrastructure

### 7.1 Termination

A single `terminate` flag in fine-grained SVM:

```c
atomic_uint32_t terminate;  // 0 == active, 1 == draining
```

Host sequence at end of stream (app shutdown):

1. Push the last real batch.
2. Push an `END_OF_STREAM` batch *or* set `terminate = 1`. Either
   works; `terminate = 1` is cheaper.
3. Wait on the dispatch fence.

Workgroup exit conditions:

- Any WG that observes `terminate == 1` *and* finds no work in either
  mode (host queue empty, no `tile_pending[t] > 0`) broadcasts an
  exit signal via LDS, `s_barrier`s, all waves exit.

The bin-queue-empty check after `terminate` ensures all rendering
completes before exit. The distributor does not need to broadcast
`END_OF_STREAM` into tile queues; the global flag is sufficient and
avoids a fan-out write across all bins.

For per-frame (rather than per-app) completion against this
persistent megakernel, see `frame-completion-detection.md`. The
`terminate` flag handles app shutdown only; per-frame ack is a
separate counter mechanism.

### 7.2 Oversubscribed Launch And Fast-Exit

Launch `N` workgroups where `N > hardware_occupancy`. The hardware
dispatcher fills CUs to capacity; the rest queue.

In steady state only `hardware_occupancy` WGs run; the surplus is
idle. When `terminate = 1` and resident WGs begin exiting, the
queued surplus is admitted onto freed CUs. They run their loop once,
observe `terminate`, exit on first iteration. This is the cleanup
path that ensures every dispatched WG eventually completes, which is
required for the dispatch fence to signal.

Without oversubscription, persistent WGs that exited cleanly leave
the CU idle until dispatch end. With oversubscription, the surplus
drains the dispatch quickly. This is purely a fence-completion
mechanism; it does not add steady-state parallelism.

`N` should be sized so that surplus WGs do not blow out the PM4
dispatch size. AMDGPU's grid limits are large; this is not a real
constraint.

### 7.3 Host Watchdog

A heartbeat counter in fine-grained SVM, bumped by every WG on each
*productive* iteration (work was found, not a backoff). The host
periodically polls the heartbeat from a separate thread; if it has
not advanced in `T` ms (typical: 1 s) the host concludes the GPU is
hung and tears down via `kfd::ComputeQueue::reset()`.

Iterations that go through the GPU backoff ladder (§2.8) deliberately
do *not* bump the heartbeat — a fully idle pipeline (no host pushes,
no tile work) has every WG sleeping in `s_sleep`, but the heartbeat
stops advancing so the watchdog fires. To avoid false positives
during legitimately quiet periods, the host can suspend the watchdog
when it stops pushing (e.g., between frames in lockstep mode).

### 7.4 KFD Watchdog Compatibility

KFD has its own GPU-side watchdog that reset hangs the kernel after
a threshold (typically 1–10 s, kernel-tunable). For a persistent
megakernel that legitimately runs for the duration of an app, this
threshold must be raised or the queue must be opted into a
non-watchdog priority class. libkfd's `kfd::ComputeQueue` exposes
the priority controls; the demo uses HIGH priority by default.

### 7.5 Wave Size Portability

`W` waves per WG and the per-wave fine-tile geometry interact with
the target arch's wave size. AMDGPU GFX9 is wave64 only; GFX10/11/12
(RDNA) defaults to wave32 in compute with wave64 available as opt-in.

Three portability strategies:

1. **Compile wave64 across the board**. Wastes lanes on RDNA's
   scalar-heavy paths but keeps the WG/bin/tile geometry constant
   across archs. Simplest. Default for the demo.
2. **Per-arch `W` tuning**. The demo's CMake follows the
   libkfd cross-compile-per-AMDGPU-arch pattern (see
   `tools/computetoy` for an example); `W` and the fine-tile
   geometry can be per-arch macros.
3. **Hold coarse-bin pixel area constant**, let wave size dictate
   fine-tile shape. Per-arch fine-tile geometry, per-arch coarse-bin
   geometry. Most work to specify; preserves both wave occupancy
   and cache behavior.

The third is the most honest. The first is the right starting point.

### 7.6 libkfd Extensions Required

The current `libkfd::ComputeQueue` API is dispatch-then-wait. The
persistent megakernel needs the host to keep producing into the SVM
ring *while* the dispatch is in flight. Concretely:

- **Async dispatch**: `kfd_gpu_dispatch_async(...)` returns
  immediately with a fence handle the host can poll. Already
  implementable as a thin wrapper over the existing PM4 submission
  that does not call `kfd_gpu_fence_wait`.
- **Fine-grained coherent SVM**. `libkfd::Memory` supports
  fine-grained allocations; the demo needs the coherent
  (no-explicit-flush) flag, both CPU and GPU pointer surfaced from
  the same handle, and atomic-safe semantics validated on target
  hardware.
- **Fence polling with timeout**: `kfd_gpu_fence_wait_with_timeout`
  so the host can interleave queue production with periodic fence
  checks instead of blocking.
- **Watchdog / priority control**: opt-in path to extended-priority
  queues or disabled watchdog. KFD has the controls; libkfd needs
  to expose them.
- **Helper PM4 queue with `WAIT_REG_MEM` + `RELEASE_MEM`**: see
  `frame-completion-detection.md` §5.8 for the interrupt-driven
  host-wait pattern this enables.

If any of these are already in libkfd, they need no work; this is a
checklist for what the demo depends on.

## 8. Concrete Refinement (If Implemented)

The minimal shippable instantiation, in priority order:

1. **WG count**: `N = 2 × hw_occupancy` symmetric WGs. No role
   distinction by `wg_id`.
2. **Waves per WG**: `W = 4`, wave64.
3. **Locks**: `host_queue_lock` (binary, single uint32_t) in
   fine-grained SVM (host+GPU shared, §2.2.1); `tile_lock[B]`
   (3-state, two-bit field per tile, §2.2.2) in VRAM (GPU-only). CAS
   acquire, no spin on failure.
4. **Mode selection**: greedy pop-then-distribute (§3.1) as baseline;
   switch to demand-based pop (§3.2) if host queue often empty.
5. **Distribute mode**: pop one batch under `host_queue_lock`, release
   immediately (§2.3); then bucket by tile in LDS (per-batch tile set)
   and push each bucket under that tile's `TILE_QLOCK` bit (acquired
   from either UNLOCKED or RENDERING). Phase-2 retries contended tiles.
6. **Render mode**: 3-phase protocol (§2.5) — acquire from UNLOCKED,
   drain into LDS, transition to TILE_RENDER, rasterize, release. New
   pushes during rasterize are rendered next claim of this tile.
7. **Tile selection**: stride-offset linear scan with `tile_pending`
   skip (§4.1).
8. **Within-WG sub-tile**: LDS atomic-CAS sub-tile claim (§2.6), 16
   sub-tiles per coarse bin, 4 waves.
9. **Backoff** (§2.7 host, §2.8 GPU). Host: spin 256 × `cpu_relax`,
   then 32 × `sched_yield`, then progressive `nanosleep` 1 μs → 100 μs.
   GPU: progressive `s_sleep(1) → s_sleep(4) → s_sleep(16) → s_sleep(64)`
   on attempt counter at thresholds 4 / 16 / 64; reset on productive
   work.
10. **Bin size**: 32×32 to start. 64×64 for 4K if `B/N ≥ 16` still
    holds. Avoid 128×128 unless `B/N ≥ 8` after measurement. Memory
    footprint at each option is in §5.2; queue capacity sizing is in
    §5.3.
11. **Termination**: `terminate` flag in fine-grained SVM (§7.1).
12. **Oversubscription and fast-exit**: §7.2.
13. **Host watchdog**: heartbeat counter; any WG can check on its
    iteration (§7.3).
14. **libkfd extensions**: async dispatch + fine-grained SVM +
    fence polling with timeout + helper PM4 queue for frame-end
    interrupt (§7.6).

## 9. Demo Staging

A reasonable path for the rotating-teapot demo:

1. **Stage 0: Single-shot upload + fixed kernel-per-frame.** Host
   uploads the entire frame's primitives to VRAM, dispatches a
   non-persistent megakernel that drains the upload, exits. Same
   pipeline kernel as the final design but without the host↔GPU
   ring or persistent loop. Fastest to a pixel-on-screen and
   exercises the bin-ownership locks and 3-state per-tile
   protocol.
2. **Stage 1: Persistent megakernel with terminate flag.** Add the
   `terminate` flag (§7.1) and oversubscription (§7.2). The
   megakernel runs across multiple frames; per-frame completion
   is gated on a separate counter (`frame-completion-detection.md`
   §5).
3. **Stage 2: Host↔GPU streaming ring.** Replace the upload with
   a fine-grained-SVM ring that the host pushes to during frame N
   while the GPU is rendering frame N (§2.7 host push protocol).
   Earns the host-overlap-with-GPU benefit.
4. **Stage 3: Interrupt-driven host wait.** Replace the polling
   host wait with a helper PM4 queue (`frame-completion-detection.md`
   §5.8). Frees the host CPU for non-rendering work.

Stages can be implemented in order; each is incrementally testable.

## 10. Reading

- `cure-streaming-queues.md` — the queue primitive vocabulary,
  including the SPMC variant of `MultiIndexQueue`.
- `frame-completion-detection.md` — host-side frame-end detection
  and KFD-signal-driven wait, complementary to this proposal.
- Kenzel, Kerbl, Steinberger, Schmalstieg, *A High-Performance
  Software Graphics Pipeline Architecture for the GPU*, SIGGRAPH
  2018. The cuRE paper; the streaming-queue primitive vocabulary
  comes from here.
- Steinberger et al., *Whippletree: Task-based Scheduling of Dynamic
  Workloads on the GPU*, TOG 2014. The dynamic-task model this
  proposal generalizes from.
- Chase, Lev, *Dynamic Circular Work-Stealing Deque*, SPAA 2005. The
  classical work-stealing primitive that informs the within-WG
  sub-tile claim pattern.
