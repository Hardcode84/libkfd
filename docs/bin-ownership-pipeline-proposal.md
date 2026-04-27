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

Mutual exclusion is enforced via three kinds of ownership locks plus
one global ready ring:

- One **host queue lock** (binary), shared between the host and GPU
  WGs. Whichever entity (the host pushing, or a distributor WG
  popping) holds the flag has exclusive access to the host queue's
  pointers and slots. At most one party touches the queue at a time.
- B **per-tile `queue_lock`s** (binary), one per coarse bin. Held only
  while a distributor appends a bucket to that tile's primitive buffer
  or a renderer drains it into LDS. Hold time is short (~hundreds of
  ns).
- B **per-tile `render_lock`s** (binary), one per coarse bin. Held by
  a renderer for the entire drain-rasterize phase on that tile. Hold
  time is long (tens of µs). The two binary locks are independent: a
  distributor can append to a tile while a renderer is rasterizing the
  prims drained earlier.
- One global **`ready` ring** of tile IDs. A distributor that pushes
  to a tile that was empty appends the tile ID; a renderer pops a tile
  ID instead of scanning. Pop is O(1). A per-tile `in_ready` flag
  coalesces duplicate enqueues and is cleared inside `queue_lock` on
  the renderer's way out.

The active distributor identity rotates as WGs release the host queue
lock and other WGs (or the host) re-claim it. Each tile's
`queue_lock` / `render_lock` / `in_ready` state evolves independently;
distribution and rasterization on the same tile can overlap.

This design replaces an earlier 3-state per-tile lock plus
whole-screen `pick_tile()` scan; see §1.1 for why the change.

Companion docs:

- `cure-streaming-queues.md` — vocabulary for streaming queue
  primitives borrowed from cuRE.
- `frame-completion-detection.md` — how the host detects end-of-
  frame against the persistent megakernel.

**Performance numbers in this document are order-of-magnitude
estimates** derived from AMDGPU instruction latencies, typical
PCIe/SVM atomic round-trips, and per-operation arithmetic — not
measurements. Anything labelled "~", "≈", or "typical" is a
back-of-envelope figure; treat it as a sanity check, not a
specification. Concrete measurements will replace these as the
demo is brought up.

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
uncontended). It works at `4 ≤ B/N ≤ 16` with the ready ring as the
work-discovery primitive (§4). It breaks down below `B/N < 4` — at
that point a fixed-role partitioning scheme (one or a few WGs as
distributors, the rest as renderers) becomes the better trade. The
1080p teapot demo lands at `B/N ≈ 25` with 32×32 bins, comfortably
inside the robust regime.

### 1.1 What Changed Since The First Prototype

The first cut of this design (committed prior to 2026-04) used a
**3-state per-tile lock** (`TILE_QLOCK | TILE_RENDER` bits) and a
**whole-screen `pick_tile()` scan**: render-mode WGs walked all B
tiles looking for a fully-UNLOCKED non-empty one, then CAS'd to
acquire. We prototyped that shape and found two real problems:

- **Renderer wakeup is O(B) per pick.** At 1080p / 32×32, B ≈ 2 040
  tiles. Most of them are empty most of the time. A renderer ate a
  full ~100 ns per tile probed before finding work — a few µs per
  pick on a busy frame, an order of magnitude more on a sparse one.
  Aggregated across N ≈ 80 renderer WGs the scan dominated.
- **The 3-state lock is hard to reason about.** Two independent bits
  give four reachable states, six legal transitions, and three
  separate CAS callsites. Verifying that the renderer's "drain → keep
  RENDER, drop QLOCK" CAS does not race with a concurrent distributor
  push, that no thread observes the impossible `0x3 → 0x0` transition,
  and that the `atomicAnd(~bit)` releases preserve memory ordering
  took more proof per kB of code than the rest of the kernel
  combined.

This revision replaces both:

| Old design                        | This revision                           |
|-----------------------------------|-----------------------------------------|
| `pick_tile()` scans all B tiles   | `ready.pop()` returns any non-empty tile in O(1) |
| 3-state lock (2 bits, 4 values)   | Two independent binary locks (`queue_lock`, `render_lock`) |
| `tile_pending` counter per tile   | `in_ready` flag per tile (1 bit's worth) |
| Renderer must find UNLOCKED state | Renderer always tries to acquire `render_lock` on the popped tile; CAS-fail re-pushes |

The two fast-path properties the user originally asked for fall out
directly:

1. **Distributor finds the tile to push to in O(1).** It already
   knows the target tile from coarse binning; it acquires that tile's
   `queue_lock` (one CAS), appends the bucket via `memcpy`, releases
   (one atomicExch). On the empty→non-empty transition it CASes
   `in_ready` and pushes the tile ID into the ready ring.
2. **Renderer finds any non-empty tile in O(1).** It pops a tile ID
   from the global ready ring. No scan. If `render_lock` is contended
   by another renderer, the renderer re-pushes the tile ID and
   returns — bounded by a single CAS plus a single ring push.

§2.2 details the lock semantics; §2.5 walks through the renderer
drain-rasterize loop and the in_ready handover that prevents lost
wakeups.

## 2. Architecture

### 2.1 High-Level Shape

```text
Host                                GPU (one megakernel)
────                                ─────────────────────────────────────
┌──────────────┐                    ┌─────────────────────────────────────┐
│ scene walk   │                    │ N symmetric WGs (no role split)     │
│ + batching   │ ◄─acq/rel host_q─► │  loop:                              │
│ push under   │ ─CAS lock─►        │    if try_claim(host_queue_lock):   │
│ host_q_lock  │                    │      batch = pop_host_queue()       │
└──────────────┘                    │      release(host_queue_lock)       │
                    ┌──────────┐    │      if batch:                      │
                    │ Host     │ ◄──┤        distribute_batch() {         │
                    │ queue    │    │          bucket by tile in LDS      │
                    │ +shared  │    │          for each non-empty bucket: │
                    │ lock     │    │            acquire queue_lock       │
                    └──────────┘    │            append, was_empty?       │
                                    │            release queue_lock       │
                                    │            if was_empty:            │
                                    │              CAS in_ready 0→1       │
                                    │              ready.push(tile_id)    │
                                    │        }                            │
                                    │    elif ready.pop(&T):              │
                                    │      render(T)  // drain-rasterize  │
                                    └─────────────────────────────────────┘
                                                ▲                ▲
                                                │                │
                          ┌─────────────────────┴───┐  ┌─────────┴────────┐
                          │ Per-tile state          │  │ Global `ready`   │
                          │  buf[CAP], head         │  │ ring of tile IDs │
                          │  queue_lock (binary)    │◄─┤ MPMC: distribs   │
                          │  render_lock (binary)   │  │ push, renderers  │
                          │  in_ready (1 bit)       │  │ pop. Coalesced   │
                          │ (B of these)            │  │ via in_ready.    │
                          └─────────────────────────┘  └──────────────────┘
                                                │
                                                ▼
                                       framebuffer
                                       (per-tile exclusive write)
```

(`host_q_lock` in the host box is the same symbol as
`host_queue_lock` on the GPU side; abbreviated to fit the diagram
width.)

Key properties:

- The host queue is gated by a single binary lock that is **shared
  between the host and GPU WGs**. The host acquires it to push; a WG
  acquires it to pop. Whichever side wins the CAS owns the queue for
  one operation; the lock-holder identity rotates dynamically. Pops
  release the lock immediately so multiple WGs can run
  `distribute_batch` concurrently after popping (§2.3).
- Each tile is gated by **two independent binary locks** (§2.2.2):
  `queue_lock` (brief; held while mutating the tile's primitive
  buffer) and `render_lock` (long; held across the renderer's full
  drain-rasterize phase). Because they are independent, a distributor
  appending more primitives can run concurrently with a renderer
  rasterizing the prims drained earlier.
- The **global `ready` ring** holds tile IDs that may have pending
  primitives. Distributor's empty→non-empty transition pushes the
  tile ID; renderer pops one in O(1). The per-tile `in_ready` flag
  prevents duplicate enqueues and is cleared by the renderer inside
  `queue_lock` when its drain finds the tile empty (§2.5).
- WG roles are dynamic per iteration, not fixed by `wg_id`.

### 2.2 The Lock Types

Three locks plus one global ready ring. All three locks are plain
binary locks — a `uint32_t` taking values 0 (UNLOCKED) or 1 (LOCKED),
acquired by `atomicCAS(0, 1)` and released by `atomicExch(_, 0)` after
a release fence. The same `try_claim` / `release` helpers are used
everywhere:

```c
__device__ /* or host */
bool try_claim(_Atomic uint32_t *lock) {
  return atomicCAS(lock, 0, 1) == 0;
}

__device__ /* or host */
void release(_Atomic uint32_t *lock) {
  atomic_thread_fence(memory_order_release);
  atomicExch(lock, 0);
}
```

Per-tile state lives in **VRAM** (GPU-only); the host queue lock is in
**fine-grained SVM** so both sides can CAS it.

```c
// fine-grained SVM (host + GPU access)
volatile _Atomic uint32_t host_queue_lock;

// VRAM (GPU-only)
struct Tile {
  _Atomic uint32_t queue_lock;     // brief: protects buf/head mutations
  _Atomic uint32_t render_lock;    // long:  protects rasterization
  _Atomic uint32_t in_ready;       // 1 = queued in `ready` OR a renderer is draining
  uint32_t         head;           // current count in buf[]
  TileEntry        buf[CAP];       // primitive ids + frame_id, packed (see §5.1)
};
__device__ struct Tile tile[B];

__device__ MPMCRing<uint32_t> ready;    // global ring of tile IDs
```

The buffer is treated as a **stack**, not a ring: distributors append
at `head`, the renderer drains the entire `[0, head)` slice into LDS
in one shot and resets `head = 0`. This drops the FIFO `tail` pointer
that the previous design carried, simplifying the queue mutations to a
single counter.

#### 2.2.1 Host queue lock (binary, host+GPU shared)

This lock is in fine-grained SVM — both the host and GPU WGs target
it with `atomicCAS`/`atomicExch`. On the host side these compile to
appropriate `__atomic_compare_exchange` / `__atomic_exchange` builtins
with PCIe-coherent semantics. Acquire-side memory ordering is on the
CAS itself; the matching release fence ensures payload writes are
visible to the next acquirer regardless of which side it's on.

Whichever side holds the flag has exclusive access to the queue's
`front` and `back` pointers and to the queue slots that it touches
during one push or one pop. Neither side spins while holding the lock
— if the work cannot be completed (host: queue full; GPU: queue
empty), the lock is released first and the wait happens unlocked
(§2.7).

#### 2.2.2 Per-tile `queue_lock` (binary, brief)

`queue_lock` is held only across mutations of `buf[]` and `head`:

- A distributor holds it for the duration of one bucket append:
  `was_empty = (head == 0); memcpy(buf + head, bucket, n); head += n`.
- A renderer holds it for the duration of one drain:
  `count = head; memcpy(lds, buf, count); head = 0` (and, on the
  exit path, `atomic_store(&in_ready, 0)`).

Both are bounded by `BUCKET_DEPTH` and `CAP` respectively — a few
hundred ns. Contention is between distributors targeting the same
hot tile and the renderer of that same tile; either way the holder
finishes quickly.

#### 2.2.3 Per-tile `render_lock` (binary, long)

`render_lock` is held by a renderer for the entire drain-rasterize
loop on one tile. Hold time is dominated by rasterization (~tens of
µs). It exists only to enforce **single-renderer-per-tile**: two
renderers writing the same coarse bin's pixels would race. It is
explicitly *not* held during `queue_lock` mutations, so a distributor
can append while a renderer is rasterizing earlier prims.

#### 2.2.4 The `in_ready` flag and the `ready` ring

`in_ready` is a per-tile `uint32_t` with two values:

| Value | Meaning                                                   |
|-------|-----------------------------------------------------------|
| 0     | Tile has no pending work and is not in the `ready` ring.  |
| 1     | Tile is in the `ready` ring **or** a renderer is currently draining/rasterizing it. |

Transitions:

- Distributor's empty→non-empty push CASes `in_ready` 0→1; on
  success it pushes the tile ID into the global `ready` ring.
- Renderer pops a tile ID from `ready` (which conceptually consumes
  the "in ring" half of the disjunction; `in_ready` stays 1 because
  the renderer now owns the work).
- Renderer clears `in_ready` to 0 only when its drain finds
  `head == 0` — i.e., no new pushes since the previous drain — and
  it is about to release `render_lock`. The clear happens **inside
  the same `queue_lock` critical section** as the empty-head observation
  (§2.5). This is the linearization point that prevents lost wakeups:
  any distributor that subsequently acquires `queue_lock` will see
  both `head == 0` and `in_ready == 0`, hence treat its own push as
  the new empty→non-empty transition and re-publish the tile.

The ready ring is a multi-producer multi-consumer ring of `uint32_t`
tile IDs. `ready.push(T)` appends; `ready.pop(&T)` removes one. Both
are O(1). Sizing: at most one entry per distinct tile in flight — so
ring capacity ≥ B suffices (rounded up to the next power of two for
the slot mask). See §5.5 for memory cost.

**Lock ordering for deadlock freedom**:

- Distributor holds `host_queue_lock` first (during pop), releases
  it, then takes per-tile `queue_lock`s one at a time (never two
  simultaneously). Never holds `render_lock`.
- Renderer holds `render_lock` for the full pass, plus `queue_lock`
  briefly inside each drain iteration. The two locks always go in
  the order `render_lock` → `queue_lock` → release `queue_lock` →
  (rasterize) → repeat → release `render_lock`. Never the reverse.

The lock-acquisition graph has no cycle. Trivially deadlock-free.

**Memory ordering**: a release fence precedes every `release(...)`
call, so any writes a holder did to `buf[]`, `head`, framebuffer
pixels, or per-frame ack counters are visible to the next acquirer.
This is the same plain binary-lock pattern used everywhere else in
this design.

### 2.3 Workgroup Loop With Mode Dispatch

The lock-hold window for `host_queue_lock` is kept as small as
possible — only the actual pop. Bucketing and tile pushes happen
*after* releasing it, so the host's competing pushes are not blocked
by the WG's distribution work. Render-mode work discovery is a
single `ready.pop()` — no scan.

The pseudocode below is the **greedy** mode policy: try distribute
first every iteration, fall back to render. It is correct and
readable; the steady-state distributor count self-caps because
`host_queue_lock` is serializing (only one WG can pop per cycle).
The recommended production-grade policy biases the per-iteration
choice on observed queue depths to cut wasted CAS traffic — see
§3.3. §3 covers all alternatives.

```c
uint32_t attempts = 0;                    // §2.8 backoff counter, per-WG SGPR
for (;;) {
  bool terminating = atomic_load(&terminate);

  // Distribute mode: pop one batch under host_queue_lock, then
  // distribute outside the lock. §3.3 is the recommended mode policy.
  StreamBatch batch;
  bool got_batch = false;
  if (try_claim(&host_queue_lock)) {
    got_batch = pop_host_queue(&batch);
    release(&host_queue_lock);            // release ASAP — host may be waiting
  }
  if (got_batch) {
    distribute_batch(&batch, &attempts);  // §2.4; no host_queue_lock held
    attempts = 0;                         // productive: reset backoff
    continue;
  }

  // Render mode: pop one tile ID from the global ready ring. O(1).
  uint32_t T;
  if (ready.pop(&T)) {
    render(T);                            // §2.5; handles render_lock contention
    attempts = 0;
    continue;
  }

  // No work this iteration in either mode. The exit condition (§7.1)
  // is exactly this: terminate is set AND no work was found, so this
  // WG has nothing left to drain. Otherwise back off and retry.
  if (terminating) break;
  wg_backoff(&attempts);                  // §2.8
}
```

`pop_host_queue` returns false if the queue is empty; that's the
fast-out path for an idle frame. When the host has work pending, the
typical WG iteration is: CAS-acquire (~50 ns) + pop (~few hundred ns,
one cache line) + release (~50 ns) ≈ 1 μs of lock hold. That's also
the maximum window the host has to wait per pop.

`ready.pop` is a single MPMC-ring dequeue — typically one CAS plus a
relaxed load of the slot. ~50 ns on the success path, comparable on
the empty path (returns false without spinning). This replaces the
O(B) `pick_tile()` scan from the previous design (§1.1).

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
pushes each bucket under that tile's `queue_lock`.

```c
__device__ void distribute_batch(StreamBatch *batch, uint32_t *attempts) {
  // Phase 1: bucket-by-tile in LDS (parallel across waves, no atomics
  //          on tile queues yet).
  __shared__ uint32_t  lds_bucket_count[B_SUBSET];
  __shared__ TileEntry lds_bucket_prim [B_SUBSET][BUCKET_DEPTH];
  bucket_by_tile(batch, lds_bucket_count, lds_bucket_prim);

  // Phase 2: push each non-empty bucket under its tile's queue_lock.
  uint32_t pending = num_active_buckets(lds_bucket_count);
  while (pending > 0) {
    for (uint32_t b = 0; b < B_SUBSET; ++b) {
      if (lds_bucket_count[b] == 0) continue;
      uint32_t T = bucket_to_tile(b);
      if (try_push_to_tile(T,
                           lds_bucket_prim[b],
                           lds_bucket_count[b])) {
        lds_bucket_count[b] = 0;
        --pending;
      }
    }
    if (pending > 0) wg_backoff(attempts);   // §2.8; retry contended buckets
  }
}

// Append `n` entries into tile T's primitive buffer; promote T into
// the ready ring if this is the empty→non-empty transition.
__device__ bool try_push_to_tile(uint32_t T, const TileEntry *bucket,
                                 uint32_t n) {
  if (!try_claim(&tile[T].queue_lock)) return false;

  // §5.4 overflow policy: if head + n > CAP, drop or spill here.
  bool was_empty = (tile[T].head == 0);
  memcpy(tile[T].buf + tile[T].head, bucket, n * sizeof(TileEntry));
  tile[T].head += n;
  release(&tile[T].queue_lock);

  // Empty→non-empty: promote tile T into the ready ring. CAS prevents
  // duplicate enqueues when another distributor or a renderer in
  // its drain-rasterize loop already considers T "in flight".
  if (was_empty) {
    if (atomicCAS(&tile[T].in_ready, 0, 1) == 0) {
      while (!ready.push(T)) wg_backoff(/*ring full*/);
    }
  }
  return true;
}
```

Phases 1 and 2 do not touch `host_queue_lock`. Multiple WGs can be in
`distribute_batch()` concurrently — each works on its own popped batch
and pushes to tile queues using per-tile `queue_lock`s. The only
serialized point is the host-queue pop itself.

The distributor never waits on rasterization. `queue_lock` is held
only during the brief append (one `memcpy` + counter update), and
`render_lock` is independent — a renderer rasterizing the prims it
already drained does not block our push.

Phase 1 parallelizes across the WG's `W` waves: each wave handles a
disjoint subset of primitives, AABB-tests each against all bins,
writes hits into LDS buckets indexed by tile. Bucket-counter updates
use LDS atomics (~20 cycles), much cheaper than global.

Phase 2 lock acquisitions per batch: one for the host queue plus one
per distinct target tile, plus at most one `in_ready` CAS and one
`ready.push` per first-touched tile. For a 256-primitive batch
hitting ~16 tiles on average, that's 17 `queue_lock` acquires plus
≤ 16 ring pushes — bounded and predictable.

If a tile's `queue_lock` is held (another distributor pushing or the
renderer of that tile in its brief drain), `try_push_to_tile` returns
false and the distributor moves on to other buckets in this round,
then comes back. Bounded by the longest `queue_lock` hold time —
hundreds of ns.

`B_SUBSET` is the number of tiles the distributor handles per batch.
If the batch's primitives can hit any of B tiles, bucketing naively
requires B LDS slots, which scales poorly. Two practical mitigations:

- **LDS hash buckets**: bucket into `B_SUBSET = 64` LDS slots indexed
  by `tile_id % 64`. Each slot holds a small list of (tile, prims)
  pairs. Bounded LDS, slight collision handling.
- **Per-batch tile set**: precompute the set of tiles the batch's
  primitives can hit (upper bound by primitive AABBs). For typical
  demo batches (rotating teapot, ~6 K triangles split into batches
  of 256) this is 8–32 distinct tiles per batch.

The second is preferable for moderate batches.

### 2.5 Render Mode

A render pass on tile `T` runs a drain-rasterize loop under
`render_lock`. Each iteration drains whatever has accumulated since
the previous one — so prims that distributors push *during*
rasterization are picked up the next time around without releasing
`render_lock`. The loop exits exactly when a drain finds the buffer
empty.

The first thing the renderer does after popping `T` from the ready
ring is try to acquire `render_lock`. If another renderer holds it
(see §6.2 for when this happens), the work isn't dropped — the tile
ID is simply re-pushed back into the ready ring and this WG returns
to the main loop to look for other work.

```c
__device__ void render(uint32_t T) {
  // Caller popped T from the ready ring. Try to take render_lock;
  // re-queue on contention rather than spinning.
  if (!try_claim(&tile[T].render_lock)) {
    while (!ready.push(T)) wg_backoff(/*ring full*/);
    return;
  }

  __shared__ uint32_t  lds_count;
  __shared__ TileEntry lds_prim[CAP];          // packs prim_id + frame_id
  __shared__ uint16_t  lds_prim_frame_id[CAP]; // see §6.9

  for (;;) {
    // Drain phase: brief queue_lock hold.
    while (!try_claim(&tile[T].queue_lock)) wg_backoff(/*queue contended*/);

    uint32_t count = tile[T].head;
    if (count > 0) {
      drain_to_lds(tile[T].buf, count,
                   lds_prim, lds_prim_frame_id);
      tile[T].head = 0;
    } else {
      // Empty drain: nothing arrived since the previous rasterize.
      // Open the ready-gate inside queue_lock so the next distributor
      // push observes (head==0, in_ready==0) and re-queues T.
      atomic_store(&tile[T].in_ready, 0);
    }
    release(&tile[T].queue_lock);

    if (count == 0) break;

    // Rasterize from LDS. queue_lock is free — distributors may push
    // more prims for T concurrently; we'll see them next iteration.
    rasterize_tile(T, lds_prim, count);

    // §6.9 frame-ack flush; full protocol in
    // `frame-completion-detection.md` §5.5.
    ack_drain(lds_prim_frame_id, count);
  }

  release(&tile[T].render_lock);
}
```

**Linearization of the empty-drain exit.** This is the only subtle
part. The `atomic_store(&in_ready, 0)` is done *inside* the
`queue_lock` critical section that observed `head == 0`. Any
distributor that subsequently acquires `queue_lock` therefore sees
both `head == 0` (`was_empty == true`) and `in_ready == 0`, so its
own append is the new empty→non-empty transition: it CASes
`in_ready` 0→1 and re-publishes T into the ready ring. No wakeup is
lost.

Conversely, a distributor whose push *raced* the renderer's last
drain — i.e., it acquired `queue_lock` before the renderer's
last-iteration acquire — leaves `head > 0`. The renderer's drain
then sees `count > 0`, takes the prims, rasterizes, and loops again.
No prim is missed.

The `try_claim`/`re-queue` path on `render_lock` is reachable only
in a narrow window: between a previous renderer clearing
`in_ready = 0` (still inside its `queue_lock` critical section) and
its `release(render_lock)`. A distributor that pushes inside that
window CASes `in_ready` 0→1 and pushes T to ready; another renderer
can pop T before the previous one releases `render_lock`. The
re-push keeps the work alive without spinning.

The blocking timing is asymmetric, which is the point of having
two independent locks rather than one:

- **Distributor blocked on this tile**: only during another WG's
  drain or another distributor's push (a few hundred ns of
  `queue_lock` hold).
- **Other renderer blocked on this tile**: through this renderer's
  entire pass (`render_lock` held for tens of µs). The other
  renderer doesn't actually wait — it re-pushes the tile to ready
  and returns.
- **This renderer blocked on its own tile**: only during another
  distributor's concurrent push (a few hundred ns of `queue_lock`),
  and only at the start of each drain iteration. The rasterize
  phase holds no `queue_lock`.

New primitives pushed during a render pass are picked up by **this
renderer's next loop iteration** (not the next claim of T) — no
release/re-acquire of `render_lock` between drains. For opaque
"first closer wins" depth this is correct (the depth test handles
ordering). For blended draws or alpha-test, ordering across drain
iterations needs to match the host's submit order; the per-tile
buffer is FIFO by append order, and `drain_to_lds` preserves that.

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
claimed sub-tile exclusively. A static one-wave-per-fine-tile
assignment has the same property; dynamic claiming preserves it
while improving load balance when wave runtimes diverge.

#### 2.6.1 Per-Pixel Work: Demo Shader

Inside `rasterize_subtile`, each lane owns one pixel for one pass and
performs the standard inner loop: edge tests against the triangle's
half-spaces, depth interpolation, depth test, then the pixel shader.

The demo's shader is a **procedural checkerboard** sampled against
interpolated `(u, v)` — no texture memory, no sampler state, no cache
pressure. Closed-form:

```c
__device__ uint32_t shade_checker(float u, float v) {
  // 1.0 of UV maps to 8 checker cells. Adjust scale per object.
  uint32_t iu = (uint32_t)floorf(u * 8.0f);
  uint32_t iv = (uint32_t)floorf(v * 8.0f);
  return ((iu ^ iv) & 1u) ? 0xffe0e0e0u  // light cell
                          : 0xff202020u; // dark cell
}
```

UV is supplied per-vertex by the host (planar projection onto the
teapot is fine for the demo) and stored alongside screen-space
position in the primitive store (§5.5). The rasterizer interpolates
UV barycentrically across the triangle. **Affine** interpolation is
the starting point; perspective-correct (`u/w`, `v/w`, `1/w`) is a
small extension that doesn't change the pipeline shape.

This split — procedural shader on top of a real binning/rasterization
pipeline — exercises the UV interpolation and per-pixel hook points
that any future texture-sampler implementation will plug into,
without coupling the demo to a texture cache, sampler descriptors,
or LOD selection. The demo replaces `shade_checker` with a real
sampled lookup later; nothing else in the pipeline changes.

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
| sleep | unbounded | progressive 1 μs → 100 μs; the application's own host-side hang watchdog (e.g. §7.3 GPU heartbeat going stale) is what eventually breaks the loop, not a fixed cap |

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
unavailable: phase-2 retry on a contended `queue_lock` in
`distribute_batch` (§2.4), `ready.pop()` returning false in §2.3,
the renderer's drain-time `queue_lock` retry in §2.5, the host-queue
push retry on a full `ready` ring in §2.4, and the case where neither
mode finds work in one iteration of the main loop. The right
primitive on AMDGPU is
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
// Simplified shape; the canonical loop is in §2.3.
uint32_t attempts = 0;
for (;;) {
  bool terminating = atomic_load(&terminate);
  if (try_pop_or_render()) { attempts = 0; continue; }
  if (terminating) break;
  wg_backoff(&attempts);
}
```

The cap matters for **responsiveness**: a 2 μs sleep means a WG sees
a new host push within ~2 μs of the host releasing
`host_queue_lock`. It does *not* matter for watchdog avoidance —
`s_sleep` keeps the wave resident and issuing the `s_sleep`
instruction itself, so the KFD GPU watchdog (§7.4, 1–10 s) sees CP
progress regardless of sleep depth. The host heartbeat watchdog
(§7.3) is a separate concern: §2.8 backoff iterations do not bump
the heartbeat (only productive iterations do), so a fully idle
pipeline trips §7.3 unless the host suspends the watchdog when it
stops pushing.

#### Where to call wg_backoff

The §2.3 main loop calls `wg_backoff(&attempts)` at exactly one
site — the bottom-of-iteration "no work" path — and resets
`attempts = 0` whenever a productive distribute or render
completes. `distribute_batch` (§2.4) calls it inside its phase-2
retry loop while there are still buckets pending against contended
tile locks.

| Site | Backoff? | Rationale |
|---|---|---|
| §2.3 main loop, no work in either mode this iteration | yes | nothing to do; sleep before retry |
| §2.3 productive distribute or render | no | reset `attempts = 0` |
| §2.4 phase-2 retry loop with `pending > 0` | yes, between scans | per-tile lock contention |

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

Each WG decides per iteration whether to attempt distribute or render.
The §2.3 pattern is the simplest correct version — try the host
queue, fall back to render. The interesting question is *how the
distributor:renderer ratio settles*, given that there is no fixed
WG partition.

The answer is **back-pressure self-balancing**: both queues carry
depth signals that automatically push the system toward the right
ratio. There is no `N_dist` / `N_rend` configuration parameter, no
dependence on the WG count `N`, and no cost-model retuning per
resolution or framerate.

- Too few distributors → host queue fills (`h` rises) → the host's
  own backoff (§2.7) kicks in *and* WGs that observe `h > 0` shift
  toward distribute.
- Too many distributors → ready ring fills (`r` rises) → distributors
  produce slower (push to a back-pressured ring) *and* WGs that
  observe `r > 0` shift toward render.

The §3.1 / §3.2 variants are primitives that operate on this signal
implicitly (CAS serialization on `host_queue_lock` already caps the
rate at which distribute mode runs). §3.3 makes the bias explicit
and is the canonical baseline. §3.5 / §3.6 are escape hatches that
override the dynamic balance with a fixed split — useful as
diagnostic tools, not as a normal operating mode.

The `host_queue_lock` is held only across the pop (~1 μs); multiple
WGs can be in `distribute_batch` concurrently after popping, each
operating on a different batch. Mode rotation happens naturally as
WGs and the host alternate as lock holders.

### 3.1 Greedy Pop-Then-Distribute (Baseline)

```c
StreamBatch batch; bool got = false;
if (try_claim(&host_queue_lock)) {
  got = pop_host_queue(&batch);
  release(&host_queue_lock);
}
if (got) {
  distribute_batch(&batch, &attempts);   // §2.4
} else {
  uint32_t T;
  if (ready.pop(&T)) render(T);          // §2.5
}
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
if (got) {
  distribute_batch(&batch, &attempts);   // §2.4
} else {
  uint32_t T;
  if (ready.pop(&T)) render(T);          // §2.5
}
```

Particularly useful when the host produces in bursts with idle gaps
between them.

### 3.3 Queue-Depth-Biased Mode Selection (Recommended)

§3.1 has every free WG attempt the `host_queue_lock` CAS each
iteration even though only one can win. With `N ≈ 80` WGs hammering
a fine-grained-SVM lock that costs ~500 ns – 1 μs per CAS round-trip,
the losing WGs spend most of their time generating cache traffic
instead of doing work. §3.2 cuts this when the host is fully idle
(`h == 0`); when `h > 0` the storm returns.

The fix is to bias each WG's choice of mode using **observed queue
depths** read with relaxed atomics: the host queue depth `h` and the
ready ring depth `r`. The bias adapts itself; the only state per WG
is one SGPR for an xorshift seed.

```c
uint32_t h = host_queue_back - host_queue_front;   // relaxed
uint32_t r = ready.depth_relaxed();                // relaxed

// Probability of attempting distribute first this iteration:
//   p_distribute = h / (h + r + 1)
uint32_t denom = h + r + 1;
bool try_distribute_first = (rng_next(&wg_seed) % denom) < h;

if (try_distribute_first) {
  StreamBatch batch; bool got = false;
  if (try_claim(&host_queue_lock)) {
    got = pop_host_queue(&batch);
    release(&host_queue_lock);
  }
  if (got) {
    distribute_batch(&batch, &attempts);
    attempts = 0; continue;          // §2.3 productive-work reset
  }

  // CAS-fail or empty queue. Render this iteration so we don't spin.
  uint32_t T;
  if (ready.pop(&T)) { render(T); attempts = 0; continue; }
} else {
  uint32_t T;
  if (ready.pop(&T)) { render(T); attempts = 0; continue; }

  // Ready ring empty. Try distribute as fallback so the WG doesn't idle.
  StreamBatch batch; bool got = false;
  if (try_claim(&host_queue_lock)) {
    got = pop_host_queue(&batch);
    release(&host_queue_lock);
  }
  if (got) {
    distribute_batch(&batch, &attempts);
    attempts = 0; continue;
  }
}
// Nothing on either side. Fall through to §2.3's terminate / backoff.
```

Behaviour at the corners:

- **`h = 0`** (host idle): `p_distribute = 0`, every WG renders. No
  CAS pressure on the SVM lock — important because that lock is the
  most expensive atomic in the system.
- **`r = 0`** (no rasterizable tiles): `p_distribute = 1`, every WG
  attempts distribute. Bootstraps the system from an empty state
  (e.g. start-of-frame).
- **Steady state**: `h` and `r` track the production/consumption
  imbalance. If renderers fall behind, `r` rises → fewer
  distributors → the imbalance corrects. If distributors fall behind,
  `h` rises → more distributors → ditto. The fixed point sits where
  ready entries are produced and consumed at matching rates.

Per-iteration cost on top of §2.3 is two relaxed loads (~50 ns total)
plus one xorshift32 step plus a 32-bit mod. The xorshift state is
seeded once at WG launch (any non-zero value; xorshift32 has a fixed
point at 0):

```c
__device__ uint32_t rng_next(uint32_t *s) {
  uint32_t x = *s;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  *s = x;
  return x;
}

// At WG launch, lane 0 of wave 0 initialises the per-WG seed:
uint32_t wg_seed = (wg_id * 0x9E3779B9u) | 1u;   // golden ratio, force non-zero
```

The relaxed depth reads can lag the true value by tens of nanoseconds
and may be off by O(N) WGs in flight; that's fine. This is a
heuristic, and the per-branch fallbacks cover any wrong guess at zero
correctness cost. `ready.depth_relaxed()` is the unsigned difference
of the ring's modular head/tail counters (saturated to 0 if torn).

### 3.4 α-Bias Variant

If profiling shows the natural ratio under §3.3 is wrong (e.g.
distribute-side is the bottleneck or, conversely, the ready ring
drowns everything), tune with a single integer multiplier `α`:

```c
//   p_distribute = (W_H · h) / (W_H · h + W_R · r + 1)
uint32_t num = W_H * h;
uint32_t den = W_R * r + 1;
bool try_distribute_first = (rng_next(&wg_seed) % (num + den)) < num;
```

- `W_H == W_R == 1`: §3.3 default.
- `W_H > W_R`: prefer distribute — use when host queue tends to back
  up under load (large batches, slow `distribute_batch`).
- `W_R > W_H`: prefer render — use when distribute is cheap relative
  to render and the ready ring fills quickly (small batches, fat
  primitives).

`W_H` / `W_R` are global constants, independent of `N`, `B`,
resolution, or framerate. Treat them as profiling knobs, not as
configuration parameters; the §3.3 default should work for the
demo workload.

### 3.5 Rotating Pop

After a WG finishes a `distribute_batch()`, it skips the host queue
claim on the next iteration (sets a per-WG flag). Rotates the
distributor identity across WGs to spread the LDS bucketing cost.
Useful only if distribute mode imposes meaningful per-WG overhead
beyond the work itself; usually unnecessary, and §3.3 already
spreads the role across WGs probabilistically.

### 3.6 Reserved-Distributor Pool

Dedicate a subset of WGs (e.g., `wg_id % K == 0`) to distribute only;
others to render only. Re-introduces a soft role split as a tuning
hint, not a structural constraint. Distributor pool can be small (1–4
WGs) since distribution is light per batch.

Defeats the point of dynamic role selection, but useful as a
diagnostic tool: if a fixed split outperforms the dynamic one,
profiling will quickly show which side (distribute or render) is
bottlenecking. Once identified, prefer raising `W_H` / `W_R` (§3.4)
to keep the back-pressure feedback loop intact.

### 3.7 Recommendation

§3.3 is the canonical baseline. §3.1 and §3.2 are the primitives it
falls back to inside each branch — keep them in the kernel for the
fallback paths but do not use them as the top-level mode pick.

If profiling shows persistent imbalance under §3.3, try §3.4
(`α`-bias) before reaching for §3.5 (rotating pop) or §3.6
(reserved-distributor pool); the latter two override the back-pressure
self-tuning property and should be a last resort.

## 4. Tile Selection Strategy

Renderers don't pick a tile — they consume a tile ID from the global
`ready` ring. The full strategy is a single `ready.pop(&T)` on the
hot path, plus a re-push on `render_lock` contention.

### 4.1 Ready-Ring Pop (Baseline)

```c
uint32_t T;
if (ready.pop(&T)) render(T);   // §2.5
```

- **Cost**: one MPMC-ring pop per pick, ~50 ns. No tile-by-tile
  probing.
- **Selection policy**: FIFO by enqueue time (= empty→non-empty
  transition). The host's stream order maps directly to render
  order, modulo the inevitable interleavings from concurrent
  distributors.
- **Wakeup latency**: bounded by the time between distributor's
  `ready.push(T)` and any free renderer's `ready.pop`. With B/N ≥ 16
  every renderer tends to find work on the first pop.

The `in_ready` flag in the per-tile state coalesces enqueues so the
ring never carries duplicate T entries, even under heavy pushing
(§2.2.4). The renderer's re-push on `render_lock` contention is the
only way an entry transiently re-enters the ring; that path is
self-limiting since the holding renderer eventually exits.

### 4.2 Variants (Optional, Profile First)

The ring is FIFO and ignores spatial layout. Two refinements are
plausible if profiling shows headroom:

- **Two-stage ring with locality bias**. Use one MPMC ring per
  Morton-octant of the screen; renderers prefer their home octant's
  ring before stealing from neighbours. Trades some imbalance for
  texture/lightmap cache reuse. Adds B-bit-counted per-octant
  state.
- **Workload-weighted ordering**. Have distributors push the tile
  ID *plus* a backlog hint; renderers preferentially pop the
  highest-backlog entry. Requires a priority-queue-style ring,
  which is more expensive than the plain MPMC ring on the hot
  path.

Neither is needed for the demo workload (§1) — stick with the
single FIFO ring as the baseline and add a variant only if a
profiled bottleneck calls for it.

### 4.3 Why Not A Per-Tile Scan?

The earlier revision of this document proposed a stride-offset linear
scan over `tile_lock[]` (`pick_tile()`). We tried it and found
(§1.1):

- The scan walks all B tiles per pick on sparse frames (most tiles
  empty), eating ~100 ns per probed tile. At 1080p / 32×32 with
  B ≈ 2 040 the scan dominated renderer wakeup latency.
- Stride-offset hashing helps only on the *first* probe; subsequent
  probes still walk through empty tiles.
- The two relaxed loads per probe (lock state and pending count)
  invalidate cache lines as distributors push, so even an empty
  scan generates write-back traffic.

The ready ring removes the scan entirely and the per-tile
`tile_pending` counter becomes redundant — the empty-vs-non-empty
question is now answered by `head` inside `queue_lock`, and the
"is it queued for rendering" question is answered by `in_ready` /
the ring contents.

## 5. Memory Footprint

The pipeline's global memory cost is dominated by **per-tile queue state**.
Everything else is fixed-overhead or scales with primitive count rather
than tile count. Numbers below assume the per-tile header laid out in
§5.1 (three 4-byte locks/flags + `head`) and a 4-byte `TileEntry` per
queue slot; see §5.3 for sizing the queue capacity `C`.

**Units.** All sizes in this section use **MiB = 2²⁰ B** unless
explicitly stated otherwise (e.g. `50 K × 96 B = 4.8 MB` in §5.5
remains in decimal MB because that's the natural arithmetic; it
converts to ≈ 4.6 MiB).

### 5.1 Per-Tile State

Each coarse tile carries a fixed-size `Tile` record (§2.2):

| Field | Bytes | Where it lives |
|---|---|---|
| `tile[i].queue_lock` (binary, §2.2.2) | 4 | VRAM |
| `tile[i].render_lock` (binary, §2.2.3) | 4 | VRAM |
| `tile[i].in_ready` (1 bit, padded, §2.2.4) | 4 | VRAM |
| `tile[i].head` (current count) | 4 | VRAM |
| `tile[i].buf[C]` (packed prim_id + frame_id) | 4·C | VRAM |
| **Total** | **16 + 4·C** | VRAM |

Each `TileEntry` is a `uint32_t` packing the 16-bit primitive index
and 16-bit frame id (used by `ack_drain` in §6.9 for frame-completion
detection). Actual per-primitive screen-space data lives once in the
streaming primitive store (§5.5), not duplicated per tile.

The buffer is treated as a stack: distributors append at `head`, the
renderer drains the entire prefix `[0, head)` in one shot. This drops
the FIFO `tail` pointer that the previous design used.

The tile array lives in **device-local VRAM**, not fine-grained SVM:
only GPU WGs touch it, the host never reads or writes. PCIe-coherent
SVM is reserved for the host queue, the `terminate` flag, and the
heartbeat counter (§5.5). The total per-tile cost is unchanged from
the previous (3-state-lock) revision — the four header words just
play different roles now.

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
| 1080p | 32 | 2 040 | 1.03 MiB | 2.02 MiB | 4.02 MiB | 8.00 MiB |
| 1080p | 64 | 510   | 0.26 MiB | 0.51 MiB | 1.00 MiB | 2.00 MiB |
| 1080p | 128 | 135  | 0.07 MiB | 0.13 MiB | 0.27 MiB | 0.53 MiB |
| 1440p | 32 | 3 600 | 1.81 MiB | 3.57 MiB | 7.09 MiB | 14.1 MiB |
| 1440p | 64 | 920   | 0.46 MiB | 0.91 MiB | 1.81 MiB | 3.61 MiB |
| 4K    | 32 | 8 160 | 4.11 MiB | 8.09 MiB | 16.1 MiB | 32.0 MiB |
| 4K    | 64 | 2 040 | 1.03 MiB | 2.02 MiB | 4.02 MiB | 8.00 MiB |
| 4K    | 128 | 510  | 0.26 MiB | 0.51 MiB | 1.00 MiB | 2.00 MiB |

All comfortably below 1% of an 8 GiB GPU at any sane configuration.
The 4K 32×32 with C=1024 (32 MiB) is the only entry approaching
"noticeable", and still negligible against framebuffer + textures.

### 5.3 Sizing C

Queue capacity needs to absorb the largest plausible burst between
drains. Three regimes to consider:

- **Per-push burst.** Distribute mode pushes one LDS bucket per
  `queue_lock` acquire. Bucket cap is `BUCKET_DEPTH` (typical 32–64
  prims). One push can never exceed this.
- **Multi-push between drains.** While a tile's `render_lock` is
  held by a renderer mid-rasterize (~50 μs), distributors can push
  concurrently — `queue_lock` is independent of `render_lock`. With
  one active distributor and ~10 μs per push, that's ≤5 pushes
  ≤320 prims per tile during a single render pass. The renderer's
  next drain iteration absorbs them.
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
   ~4 MiB) holds primitives that didn't fit; a fallback rasterization
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
| Host queue (batch ring) | 256 batches × ~256 B header ≈ 64 KiB | fine SVM |
| Primitive store | P_max × ~96 B; 50K × 96 B = 4.8 MB ≈ 4.6 MiB | fine SVM |
| `host_queue_lock` | 4 B | fine SVM |
| `terminate` flag | 4 B | fine SVM |
| Heartbeat counter | 4 B | fine SVM |
| `frame_ack[NUM_INFLIGHT_FRAMES]` (§6.9) | 4 × 16 B = 64 B | fine SVM |
| `ready` ring (slots + head/tail + lock) | next_pow2(B) × 4 B + ~64 B | VRAM |
| Coarse depth (HiZ, optional) | B × 8 B | VRAM |
| Framebuffer (color + depth) | W·H · 8 B | VRAM |
| LDS scratch (per WG) | ~16 KiB / WG | LDS (on-chip) |

The primitive entry packs three vertex records of `(x, y, z, u, v)`
in screen space (5 × 4 B = 20 B per vertex; 60 B per triangle), plus
a small header (material id, flags, padding) — call it 96 B per
triangle to leave headroom. Adding perspective-correct interpolation
later costs one extra `1/w` per vertex (4 B), still well under
the 128-B alignment.

Fine-grained SVM total ≈ **4.7 MiB** at the 50 K-prim cap (host
queue 0.06 MiB + primitive store 4.58 MiB + locks/flags/frame_ack
< 1 KiB; round up for page alignment). Round up to **5 MiB** for
budgeting. This matters because fine-grained SVM atomics cross
PCIe and are significantly more expensive than VRAM atomics.

The procedural checker (§2.6.1) is closed-form — **no texture
allocation in fine SVM or VRAM**. A real sampled-texture extension
would add the texture data in VRAM (typical demo texture: 256×256
× 4 B = **256 KiB**) and is read-only, so it does not interact
with the locking protocol.

If a hierarchical-Z extension is added (per-tile `(min, max)` depth
in VRAM, used to skip primitives that fail the tile-Z test before
binning), the per-tile `B × 8 B` is negligible (~64 KiB at 4K 32×32).

The **`ready` ring** sizing follows from the in_ready coalescing
invariant: at most one entry per tile can be in the ring at any
time, so `ring_capacity ≥ B` suffices. Round up to the next power
of two for the slot mask:

| Resolution | Tile | B | Ring capacity | Memory |
|---|---|---|---|---|
| 1080p | 32 | 2 040 | 2 048 | 8 KiB |
| 1080p | 64 | 510   | 512   | 2 KiB |
| 1440p | 32 | 3 600 | 4 096 | 16 KiB |
| 4K    | 32 | 8 160 | 8 192 | 32 KiB |
| 4K    | 64 | 2 040 | 2 048 | 8 KiB |

Plus a small fixed header (head, tail, lock, padding) of ~64 B. The
ring stays below **32 KiB** even at 4K / 32×32, which is negligible
against the framebuffer's tens of MiB.

### 5.6 Total Pipeline Footprint

Sum at the recommended **C=256** default. Framebuffer column is
`W·H · 8 B` (color + depth) expressed in MiB. The ready ring
(≤ 32 KiB at 4K) is rolled into the tile-state column.

| Resolution | Tile | Tile state + ring | Other (fine SVM) | Other (VRAM, FB) | Total |
|---|---|---|---|---|---|
| 1080p | 32 | 2.03 MiB | ~5 MiB | 15.8 MiB | ~23 MiB |
| 1440p | 32 | 3.59 MiB | ~5 MiB | 28.1 MiB | ~37 MiB |
| 4K | 32 | 8.12 MiB | ~5 MiB | 63.3 MiB | ~76 MiB |
| 4K | 64 | 2.03 MiB | ~5 MiB | 63.3 MiB | ~70 MiB |

The framebuffer dominates at high resolutions; tile state is a small
contribution and the ready ring is a rounding error on top of it.
The `B/N ≥ 16` recommendation in §1 (favoring smaller tiles for
fewer renderer contention events, §6.2) costs only a few MiB of
extra VRAM — there is no memory pressure to push toward larger
tiles.

## 6. Open Issues

### 6.1 Distributor Push Contention On Hot Tiles (Severity: low)

`queue_lock` is independent of `render_lock` (§2.2), so distributors
are *not* blocked on rasterization. They only contend with another
distributor pushing to the same tile or with a renderer's brief drain
of that same tile — both windows of a few hundred ns. With `B >> N`
and per-batch bucket scattering, contention is rare; with hot-tile
workloads (many primitives target the same coarse bin), the
distributor's phase-2 retry loop may revisit the same bucket several
times in close succession.

Mitigations if profiling shows this matters:

- Sharded host queues (§6.7) — multiple parallel distributors push to
  different shards in parallel.
- Larger `B` — finer bins reduce per-bin push frequency.

### 6.2 Renderer Lock Contention (Severity: low)

A renderer that pops `T` from the ready ring fails to acquire
`render_lock` only if another renderer is mid-pass on `T`. The
window is narrow (§2.5 "Linearization of the empty-drain exit"):
between a previous renderer's `atomic_store(&in_ready, 0)` inside
`queue_lock` and its `release(render_lock)`, a distributor push can
re-publish `T` and another renderer can pop it before the first
finishes. The losing renderer simply re-pushes `T` into the ready
ring and goes back to the main loop — no spinning, no scan, no work
lost.

Compared to the previous design's `render_try_acquire` failures —
which rejected the tile *and required scanning for another* — this
costs only one ring push and one main-loop iteration before the WG
is productive again. Severity drops accordingly.

Mitigation: keep `B/N` large enough that the ring usually has
multiple distinct entries when a re-push lands. Profile re-push
rate (a debug counter on the `try_claim(&render_lock)` failure
path); if > 5% of `render()` invocations re-push, increase B.

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
  by skipping the CAS when `back == front`; the queue-depth-biased
  policy (§3.3) extends this to skip the CAS *probabilistically*
  whenever the ready ring is non-trivially full, which addresses the
  expensive-CAS storm on fine-grained-SVM atomics.

Crossing into tier 3 (sleep) regularly is the symptom that even with
parallel distribute the GPU consumer cannot keep up — typically
because individual `distribute_batch` calls are slow (large batches,
hot tiles). Mitigations: sharded host queues (§6.7), larger batch
size (fewer push round-trips per primitive), or — as a diagnostic —
fall back to a fixed-role distributor pool (§3.6) to confirm the
bottleneck is consistent.

### 6.4 Deadlock Avoidance (Severity: low)

Lock ordering rule (§2.2): distributor holds `host_queue_lock` then
at most one `queue_lock`; renderer holds `render_lock` then at most
one `queue_lock`; no cycles. The ready ring is lock-free w.r.t. the
tile locks. Trivially deadlock-free.

### 6.5 Choosing B (Severity: high)

`B` is the central tuning parameter:

- Too small: renderer re-push rate climbs as multiple WGs race for
  the same tile (§6.2); effective parallelism degrades.
- Too large: per-pixel overhead from finer bin granularity, tile-state
  cache pressure, more queue traffic, larger `tile[]` array (see §5.2
  for footprint at each B).

Initial guidance:

| Render res | Recommended bin size |
|---|---|
| 1080p | 32×32 |
| 1440p | 32×32 |
| 4K | 32×32 or 64×64 |

Profile lock contention and per-bin work distribution; tune. The
memory cost of "more bins" is negligible at any sane resolution
(§5.2), so the practical lower bound on T comes from §6.2 (renderer
contention) rather than memory pressure.

### 6.6 Fairness (Severity: low)

`atomicCAS` provides no fairness guarantee. In practice with
persistent WGs, `B >> N`, and the FIFO ready ring, all WGs make
forward progress and the host's submit order is preserved on the
visible ordering of empty→non-empty events. If a future workload
shows starvation, consider ticket locks on the host queue or a
priority-queued ready ring.

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
shape (V2: bulk per-frame counter with per-WG accumulator).
Integration points with this design:

- §2.4 (distributor): pack `batch.frame_id` into the high 16 bits
  of each `TileEntry`. The 4-byte `TileEntry` already accommodates
  this — `prim_id` (16) + `frame_id` (16) — so no per-tile size
  change.
- §2.5 (renderer): each loop iteration's `ack_drain` flushes a
  per-WG SGPR accumulator into the global
  `frame_ack[FRAME_SLOT(F)].rendered`, where
  `FRAME_SLOT(F) = F % NUM_INFLIGHT_FRAMES`
  (`frame-completion-detection.md` §5.1).
  `NUM_INFLIGHT_FRAMES` is the host's frame ring depth (2–4),
  unrelated to `N` (resident WG count).
- §2.7 (host push): host seals the frame by storing
  `frame_ack[FRAME_SLOT(F)].expected = N_F` after pushing all
  batches, where `N_F` is the primitive count of frame F.

The `ready` ring carries only tile IDs (no frame ID), so a tile that
has work for multiple frames is enqueued once and the renderer
processes them in append order — the per-entry `frame_id` makes
ack_drain safe across frames.

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

Workgroup exit conditions: the §2.3 main loop already encodes the
exit predicate. A WG breaks out of the loop on the iteration where
both modes returned "no work" *and* `terminate == 1`:

- `try_claim(&host_queue_lock)` either failed or the popped batch
  was empty;
- `ready.pop()` returned false (no tile is in the ready ring);
- `terminate == 1`.

This is the "drained" condition: with the host no longer pushing,
host queue empty, and the ready ring empty (every remaining tile
is either empty or already mid-render with another WG holding its
`render_lock`), there is nothing left for this WG to do. Other WGs
that still hold a `render_lock` finish their drain-rasterize loop
and reach the same predicate
on the next iteration. The dispatch fence signals once every WG
has exited.

The exit is wave-uniform because the §2.3 loop is wave-uniform —
lane 0 of wave 0 performs the atomic loads and the result is
broadcast via LDS, so all waves of the WG see the same "exit"
decision and reach the kernel return together.

The distributor does not need to broadcast `END_OF_STREAM` into
tile queues; the global flag is sufficient and avoids a fan-out
write across all bins.

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

### 7.7 Assumptions

The design depends on a handful of hardware/driver properties that
are believed-true on AMDGPU but should be **validated on every
target ASIC** before relying on them in production. None of these
are fundamental constants of the architecture; each could in
principle change with a kernel/driver/firmware update.

| Assumption | Where used | What breaks if false |
|---|---|---|
| Fine-grained coherent SVM with PCIe atomics on the same cache lines as host code (GFX9–GFX12). | Host queue, `host_queue_lock`, `terminate`, heartbeat, `frame_ack`. | Host↔GPU lock protocol; per-frame ack. Fall back to coarse-grained SVM + explicit flushes (slower). |
| The KFD GPU watchdog can be raised, disabled, or bypassed via a priority class for persistent compute queues. | §7.4. | The kernel hits the watchdog at ~1–10 s and is reset. No clean fallback; would need to re-architect as kernel-per-frame. |
| `s_sleep` is supported on the target gen and its `simm16` cycle counts approximate the documented values. | §2.8 backoff ladder. | `wg_backoff` becomes a busy-spin (worst case: cache-fabric saturation). Fallback: `s_nop` chain or LDS-only spin. |
| `RELEASE_MEM` (`INT_SEL=2`) and `WAIT_REG_MEM` PM4 packets have stable cross-gen semantics on a non-megakernel helper queue. | `frame-completion-detection.md` §5.8 interrupt-driven wake. | Interrupt-driven wake does not work; fall back to polling host wait (§5.3 of frame-completion). |
| No compute-queue preemption mid-kernel (lock-holder eviction). | §6.8. | A descheduled WG holds a lock indefinitely; lock-holder watchdog needed. |
| `s_sendmsg sendmsg(MSG_INTERRUPT)` is **not** a generic compute-kernel→KFD-event wake mechanism (only the trap handler routes it). | §5.8 of frame-completion explicitly documents this; design doesn't rely on it. | (No design impact; assumption listed for completeness.) |

Each row is a one-liner; the full discussion lives in the section
referenced. Bring-up should explicitly check each on the target
GPU before running anything beyond stage 0.

## 8. Concrete Refinement (If Implemented)

The minimal shippable instantiation, in priority order:

1. **WG count**: `N = 2 × hw_occupancy` symmetric WGs. No role
   distinction by `wg_id`.
2. **Waves per WG**: `W = 4`, wave64.
3. **Locks**: `host_queue_lock` (binary, single uint32_t) in
   fine-grained SVM (host+GPU shared, §2.2.1); per-tile
   `queue_lock` (binary, brief, §2.2.2) and `render_lock` (binary,
   long, §2.2.3) plus the `in_ready` flag (§2.2.4) in VRAM
   (GPU-only). CAS acquire, no spin on failure. Single MPMC `ready`
   ring of tile IDs in VRAM.
4. **Mode selection**: queue-depth-biased per-WG pick (§3.3) as the
   canonical baseline — `p_distribute = h / (h + r + 1)`, two relaxed
   reads + xorshift32 per iteration. §3.1 (greedy) and §3.2
   (demand-based) live as the fallbacks inside each branch. Tune with
   §3.4 `α`-weights only if profiling shows persistent imbalance.
5. **Distribute mode**: pop one batch under `host_queue_lock`,
   release immediately (§2.3); then bucket by tile in LDS (per-batch
   tile set) and push each bucket under that tile's `queue_lock`. On
   the empty→non-empty transition, CAS `in_ready` 0→1 and push the
   tile ID into the ready ring (§2.4). Phase-2 retries contended
   tiles.
6. **Render mode**: pop a tile ID from the ready ring; CAS
   `render_lock`; on contention, re-push the tile ID and return.
   Otherwise enter the drain-rasterize loop (§2.5) — each iteration
   takes `queue_lock` for a brief drain to LDS, releases, then
   rasterizes; loop exits when a drain finds the buffer empty
   (`in_ready` cleared inside `queue_lock`). Distributors may push
   to T concurrently with rasterization; their prims are picked up
   by the next loop iteration.
7. **Tile selection**: O(1) `ready.pop()`. No per-tile scan
   (§4.1).
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

Every stage runs **headless first** — `headless_runner` reads the
framebuffer back to host memory and validates against goldens or
invariants (`headless-testing.md`). The windowed dma-buf
swapchain mode is enabled at the end of stage 1 once frame
completion lands; until then there is nothing to flip on a vsync
boundary. Adopting this order means each stage has its own CI
test before any presentation code exists.

1. **Stage 0: Single-shot upload + fixed kernel-per-frame.** Host
   uploads the entire frame's primitives to VRAM, dispatches a
   non-persistent megakernel that drains the upload, exits. Same
   pipeline kernel as the final design but without the host↔GPU
   ring or persistent loop. Fastest to a pixel-on-screen
   (headless) and exercises the bin-ownership locks (`queue_lock`,
   `render_lock`, `in_ready`) and the ready-ring distribute/render
   protocol. Validated by pixel-equality + invariant tests
   (`headless-testing.md` §3.1, §3.2).
2. **Stage 1: Persistent megakernel with terminate flag.** Add the
   `terminate` flag (§7.1) and oversubscription (§7.2). The
   megakernel runs across multiple frames; per-frame completion
   is gated on a separate counter (`frame-completion-detection.md`
   §5). Validated by frame-completion + termination tests
   (`headless-testing.md` §3.4, §3.5). Windowed mode (dma-buf
   swapchain flip) lands here, replacing the readback path with
   a present.
3. **Stage 2: Host↔GPU streaming ring.** Replace the upload with
   a fine-grained-SVM ring that the host pushes to during frame N
   while the GPU is rendering frame N (§2.7 host push protocol).
   Earns the host-overlap-with-GPU benefit. Validated by
   concurrency stress tests (`headless-testing.md` §3.3) — the
   `host_queue_lock` contention path now has real load.
4. **Stage 3: Interrupt-driven host wait.** Replace the polling
   host wait with a helper PM4 queue (`frame-completion-detection.md`
   §5.8). Frees the host CPU for non-rendering work.

Stages can be implemented in order; each is incrementally testable.

## 10. Reading

- `cure-streaming-queues.md` — the queue primitive vocabulary,
  including the SPMC variant of `MultiIndexQueue`.
- `frame-completion-detection.md` — host-side frame-end detection
  and KFD-signal-driven wait, complementary to this proposal.
- `headless-testing.md` — test harness shape, test categories,
  and the windowless validation path used by every demo stage.
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
