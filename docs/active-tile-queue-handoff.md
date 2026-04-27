# Active Tile Queue and Cooperative Frame Handoff

Status: design note. Refines `bin-ownership-pipeline-proposal.md` by
replacing renderer-side whole-screen tile scanning with an explicit queue of
active tile queues, and sketches how that queue can simplify end-of-frame
handoff in a future cooperative-launch variant.

## 1. Motivation

The baseline bin-ownership proposal lets render-mode WGs find work by scanning
the tile set:

```c
tile = pick_tile();
if (tile != NO_TILE && render_try_acquire(tile))
  render(tile);
```

This is simple, but it has two costs:

1. Idle renderers wander over many empty tile queues, especially late in a
   frame or when the frame's primitive distribution is sparse.
2. End-of-frame detection must prove that no future tile queue can become
   renderable while renderers are independently scanning.

The proposed refinement adds an **active tile queue**: a global queue whose
entries are tile IDs. A distributor that pushes primitives into a per-tile queue
also pushes that tile ID into the active queue. Renderers pop tile IDs from the
active queue and then acquire/drain/render that tile. This turns render work
discovery into normal queue consumption instead of whole-frame probing.

The active queue does **not** replace per-tile queues or tile locks. It is only
a ready-list for tile queues that may contain work.

## 2. High-Level Shape

```text
Host batches / EOF
        |
        v
-----------------
| Host input ring |
-----------------
        |
        | distributor WGs pop batches
        v
------------------------------
| Per-tile primitive queues  |<------.
| + 3-state tile locks       |       |
------------------------------       |
              | distributor pushes   | renderer drains tile queue
              v tile_id              |
------------------------------       |
| Active tile queue          |-------'
| entries: tile_id           |
------------------------------
```

Distributor responsibilities:

- Pop host batches under the host queue protocol.
- Bucket primitives by tile.
- Push each non-empty bucket into the tile's primitive queue under that tile's
  queue-access lock.
- Push the tile ID into the active tile queue.

Renderer responsibilities:

- Pop a tile ID from the active tile queue.
- Try to acquire that tile for queue drain / render.
- Drain whatever is currently present into LDS.
- Rasterize the drained list while distributors may continue pushing later
  work for that tile according to the 3-state tile-lock protocol.

## 3. Active Queue Entries

The first implementation should allow **duplicate tile IDs**.

```c
struct ActiveTileEntry {
  uint32_t tile_id;
  uint32_t frame_id;   // or packed epoch bits, if multiple frames can overlap
};
```

Every distributor push of a non-empty bucket appends an entry. A renderer that
pops a duplicate may find that another renderer already drained the tile, or
that the tile is temporarily locked. This is acceptable: it wastes a small pop,
but it avoids lost-wakeup bugs while the protocol is being proven.

Later, if duplicate traffic becomes measurable, add a per-tile scheduling bit:

```c
queued_or_dirty[tile] = {
  QUEUED,          // active queue already has an entry for this tile
  DIRTY_RENDERING, // pushed while a renderer holds TILE_RENDER
};
```

The coalesced form is trickier because a push that races with a renderer must
not get lost. The simple duplicate form should be the correctness baseline.

## 4. Distributor Protocol

For each non-empty bucket:

```c
if (dist_try_push_lock(tile)) {
  push_bucket_to_tile_queue(tile, bucket);
  dist_release_push_lock(tile);
  active_tile_queue.push((ActiveTileEntry){ tile, frame_id });
}
```

Ordering requirement: the primitive queue push must be visible before the active
queue entry becomes visible. Otherwise a renderer could pop the tile ID and
drain an apparently-empty queue.

Recommended implementation choices:

- Use release ordering on the active-queue tail publication.
- Keep the tile queue push under the tile queue-lock bit.
- Treat active queue push failure as backpressure. The distributor should retain
  the bucket locally or retry before moving on; dropping the active entry can
  strand tile work.

## 5. Renderer Protocol

Renderer-mode WGs consume active tile entries instead of scanning all `B` tiles:

```c
bool render_one(void) {
  ActiveTileEntry e;
  if (!active_tile_queue.pop(&e))
    return false;

  if (e.frame_id != current_frame_id)
    return true; // stale entry; productive enough to retry loop

  if (!render_try_acquire(e.tile_id))
    return true; // duplicate or contended; retry later

  render(e.tile_id);
  return true;
}
```

For the duplicate-entry baseline, a failed `render_try_acquire` does not need to
requeue. Either another renderer is already draining/rasterizing that tile, or a
later distributor push will enqueue another active entry if more work arrives.

If profiling shows excessive duplicate pops, add the coalesced scheduling bit
only after the duplicate version is stable.

## 6. End-of-Frame Drain Predicate

The useful property of the active tile queue is that "no renderer-visible work"
becomes a queue-empty predicate rather than a screen-wide scan. However,
`active_tile_queue.empty()` alone is not enough. A safe frame-drained predicate
must also account for work currently in flight:

```text
sealed_epoch >= current_frame_id &&
active_tile_queue_empty &&
render_inflight == 0
```

Definitions:

- `sealed_epoch`: monotonic device-visible counter. `sealed_epoch >= F` means a
  distributor has popped `EOF(F)` and all batches before that EOF have been
  fully published into per-tile queues and active tile entries.
- `active_tile_queue_empty`: no queued tile IDs remain for renderers.
- `render_inflight`: renderers that popped a tile ID and have not finished
  drain/raster/release.

Render WGs do **not** need to check that the host queue itself is empty. Once a
distributor seals frame `F`, distributors stop popping host packets until the
frame handoff advances the GPU epoch. The host may continue pushing packets for
`F+1`; those packets can fill the host queue and eventually backpressure the
host, but they are not eligible for distribution into tile queues until frame
`F` completes.

This predicate prevents the key race:

1. A WG sees the active tile queue empty.
2. Another distributor is still holding a popped batch.
3. That distributor later pushes a tile and appends an active entry.

The race is prevented by the definition of `sealed_epoch`: EOF is not published
as sealed until the distributor has finished all tile and active-queue pushes
for batches before that EOF.

### EOF as a Monotonic Counter

EOF and completion should be monotonic counters, not boolean flags:

```c
_Atomic uint32_t current_epoch;    // frame WGs are allowed to distribute/render
_Atomic uint32_t sealed_epoch;     // EOF consumed and all prior work published
_Atomic uint32_t closing_epoch;    // a WG won the right to close this frame
_Atomic uint32_t completed_epoch;  // host-visible finished frame
```

Using counters avoids three common problems:

- Stale boolean state after frame-ring reuse.
- Ambiguity when the host queue already contains packets for future frames.
- Lost wakeups when a WG observes EOF state late; it can compare epochs instead
  of relying on an edge-triggered flag.

The EOF packet carries its frame ID:

```c
struct EndOfFrame {
  uint32_t frame_id;
};
```

A distributor that pops `EOF(F)` performs:

```c
// All earlier batches were popped and fully published by this point.
atomic_store_release(&sealed_epoch, F);
```

Then it stops distributing and switches to render/idle mode until the handoff
advances `current_epoch`.

## 7. Cooperative Frame Handoff

With a cooperative launch and a grid-wide barrier, the active tile queue gives a
clean frame boundary:

```c
for (;;) {
  // Normal streaming loop.
  bool did_work = try_distribute_one() || render_one();
  if (did_work) {
    reset_backoff();
    continue;
  }

  if (frame_drained_predicate())
    atomicCAS(&closing_epoch, current_frame_id - 1, current_frame_id);

  if (closing_epoch == current_frame_id) {
    grid_sync();

    if (wg_id == 0 && lane_id == 0)
      signal_host_frame_done(current_frame_id);

    grid_sync(); // no WG advances/reuses frame state before signal is ordered
    advance_to_next_frame(); // increments current_epoch
  } else {
    wg_backoff();
  }
}
```

Important rule: every WG in the cooperative launch must enter the same
`grid_sync()` sequence exactly once per frame. Therefore the EOF marker cannot
only be known to the distributor that popped it. EOF becomes the monotonic
`sealed_epoch`, and idle WGs keep polling the epoch counters while they fail to
find work.

### Host Interaction

The host stream for frame `F` ends with an EOF packet:

```c
host_push(batch_0(F));
host_push(batch_1(F));
...
host_push(EOF(F));
```

After `EOF(F)` is consumed, the distributor publishes:

```c
sealed_epoch = F;
```

After all active tile work and render-in-flight work drains, WG0 signals:

```c
completed_epoch = F;
```

The host may continue streaming packets for `F+1` before `completed_epoch == F`.
Those packets remain in the host queue. They are not distributed until the GPU
handoff advances `current_epoch` after the frame `F` barrier.

## 8. Non-Cooperative Fallback

The same active tile queue is useful without cooperative launch. In that mode,
the drained predicate is still:

```text
sealed_epoch >= current_frame_id &&
active_tile_queue_empty &&
render_inflight == 0
```

But instead of `grid_sync()`, one WG wins a CAS on a close gate:

```c
if (frame_drained_predicate() &&
    atomicCAS(&closing_epoch, frame_id - 1, frame_id) == frame_id - 1) {
  atomic_store_release(&completed_epoch, frame_id);
}
```

This preserves a portable path for devices or libkfd configurations without
cooperative launch / GWS support. It is more subtle than the cooperative form
because other WGs may still be between loop iterations, but the in-flight
counters make it safe.

## 9. Queue Capacity and Backpressure

The active tile queue capacity should be sized for bursty distributor output,
not for total primitives. With duplicate entries, the worst case is one active
entry per non-empty distributor bucket. Reasonable starting points:

- `active_capacity = 4 * B` for the demo workload.
- Larger capacity for scenes with many small batches that touch the same tiles.

If the active queue is full, distributors must not drop entries. Options:

1. Retry with backoff while holding the bucket in LDS.
2. Spill the bucket to a distributor-local retry list.
3. Fall back to setting a per-tile `queued_or_dirty` bit and let renderers scan
   a small overflow bitmap.

Start with retry/backoff. Overflow policies can be added after measuring.

## 10. Coalesced Active Entries (Future Optimization)

Duplicate tile IDs are simple but can be noisy. A coalesced design uses one
active entry per tile until a renderer drains it:

```c
if (atomicCAS(&tile_active[tile], 0, 1) == 0)
  active_tile_queue.push(tile);
```

The renderer clears `tile_active[tile]` after it drains the queue. The hard case
is a distributor push racing while the renderer is in `TILE_RENDER`:

- If the renderer clears too late, the distributor may skip enqueueing and the
  new primitives wait forever.
- If the renderer clears too early, duplicate entries reappear.

A robust coalesced design needs either:

- a `DIRTY_RENDERING` state that the renderer observes before clearing, or
- a per-tile generation counter where pushes advance the generation and the
  renderer requeues if generation changed while it rendered.

Do not start here. Prove the duplicate-entry queue first.

## 11. Relationship to Existing Docs

This note modifies two parts of `bin-ownership-pipeline-proposal.md`:

- Section 2.4 distributor mode gains an active tile queue push for every
  non-empty tile bucket.
- Section 4 tile selection changes from stride-offset scanning to active queue
  pop as the baseline render-mode strategy.

It also offers an alternative to parts of
`frame-completion-detection.md`: with cooperative launch, EOF + active queue
drain + in-flight counters + grid barriers can replace the current per-frame
bulk render-ack counter. Without cooperative launch, the same active queue still
simplifies the non-cooperative completion predicate.

## 12. Prototype Plan

1. Add an active tile queue with duplicate tile entries.
2. Keep existing tile locks and per-tile primitive queues unchanged.
3. Change renderer tile selection from whole-screen scan to active queue pop.
4. Add `distributor_inflight` and `render_inflight` debug counters.
5. Validate the non-cooperative drained predicate in headless stress tests.
6. Only then try cooperative launch / grid barrier for frame handoff.
7. After correctness, measure duplicate pop rate and decide whether coalescing
   is worth the complexity.

