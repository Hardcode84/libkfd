# cuRE Streaming Queues

cuRE (Kenzel, Kerbl, Schmalstieg, Steinberger, SIGGRAPH 2018) is the
primary academic reference for a production-quality
compute-rasterization pipeline. The streaming and bin-ownership
proposals in this directory borrow heavily from its primitive
vocabulary; this document gives those primitives a concrete reference
so the proposals can be read without ambient knowledge of the cuRE
codebase.

All code excerpts are from the published cuRE implementation at
`github.com/GPUPeople/cuRE`, MIT-licensed.

## 1. The Problem cuRE Solves

A naive software-rasterizer baseline follows the chunker-style shape
inherited from cudaraster (2011): each pipeline stage is a separate
compute dispatch, the host waits on the previous dispatch's fence
before dispatching the next, and stage outputs are sized for
worst-case inputs.

```text
host: dispatch tile_bin     -> implicit ordering on compute queue
host: dispatch world_raster -> gpu_fence_wait
```

Two structural costs follow:

- No overlap between stages. Every triangle in a frame must be fully binned
  before any pixel starts rasterizing. Stage utilization equals single-stage
  utilization.
- Worst-case intermediate buffers. Bin output (`tile_indices[tile][256]`,
  `tile_counts[tile]`, `tile_overflows[tile]`, `tile_depth_min/max[tile]`) is
  fully materialized before any consumer reads it. Sizing is set for the
  largest scene the renderer must handle.

cudaraster shipped this shape and was hit by the second cost on adversarial
inputs. cuRE's stated central claim is to replace it with concurrent streaming
stages connected by bounded queues, and to let stage output backpressure flow
through the queues instead of through worst-case sizing.

## 2. Execution Model: One Megakernel, Persistent Threads

cuRE compiles into a single CUDA kernel with the launch shape baked in as
compile-time constants:

```cpp
PipelineConfig<num_multiprocessors,
               num_blocks_per_multiprocessor,
               num_warps_per_block>
```

The README is explicit:

> The cuRE renderer is compiled not just for a specific target architecture,
> but for a specific launch configuration on the target hardware. This allows
> us to avoid unnecessary runtime-overhead in the scheduler. However, as a
> consequence, the parameters of the launch configuration must be provided as
> compile-time constants.

Concretely:

- The kernel is launched once per frame with as many blocks as fit. Threads
  stay resident for the whole frame and pull work from queues in a loop.
- There is no `__syncthreads()` across stages. Different blocks run different
  stages.
- Idle warps do not return; they poll a queue.
- Block IDs do not address work. Work is dynamic; warps grab queue items.

This is the Steinberger/Whippletree (TOG 2014) "task-based scheduling on the
GPU" model applied to a graphics pipeline. The Softshell paper (TOG 2012) is
the underlying scheduling-on-GPU foundation; Whippletree is the queue-based
specialization; cuRE is the graphics-pipeline application.

## 3. The Two Queue Primitives

cuRE uses two distinct queue types and the distinction is the crux of the
design. Confusing them or unifying them into one structure is the most common
way to get this kind of pipeline wrong.

### 3.1 `MultiIndexQueue` -- The Data Path

This is the queue that actually moves work. From
`source/cure/pipeline/index_queue.cuh`:

```cpp
template<typename T, int NUMQUEUES, int SIZE, class AccessControl,
         T UNUSED = static_cast<T>(-1), bool TRACK_FILL_LEVEL = false>
class MultiIndexQueue
{
  static constexpr int InternalCounterSize =
      (NUMQUEUES + 1023U) / 1024U * 1024U;

  int          count[InternalCounterSize];
  unsigned int front[InternalCounterSize];
  unsigned int back[InternalCounterSize];
  int          max_fill_level[InternalCounterSize];
  T            indices[NUMQUEUES][SIZE];
  ...
};
```

Three properties matter, and none is obvious from the headline.

#### 3.1.1 It Is A Queue Of Indices, Not Data

`T` is a primitive ID, not a triangle. The actual triangle/vertex data lives
in a separate buffer that all queue consumers can read by ID. This is what
makes vertex reuse tractable: the same vertex-shaded output is referenced from
many primitive queues with no copy.

The streaming and bin-ownership pipelines proposed in this directory
do the same: instead of materializing per-primitive records into
per-tile lists, they materialize *indices into a host-shared
primitive buffer* into per-tile queues. The triangle data is written
once.

#### 3.1.2 The Empty Marker Is A Sentinel Value, Not A Bit

Each cell holds either a valid index or `UNUSED = -1`. Producers atomically
advance `back`, then spin until the cell they were assigned reads back
`UNUSED`, then write their index. Consumers atomically advance `front`, then
spin until the cell stops being `UNUSED`, then atomically swap it back to
`UNUSED`:

```cpp
struct AtomicCheckedAbortOnOverflow
{
  template <typename T, int SIZE>
  __device__ static int enqueue(const T& element, int& count,
                                unsigned int& back, T* indices, T UNUSED)
  {
    int fill = atomicAdd(&count, 1);
    if (fill < static_cast<int>(SIZE-1)) {
      unsigned int pos = atomicInc(&back, SIZE - 1U);
      while (atomicCAS(indices + pos, UNUSED, element) != UNUSED)
        __threadfence();
      return fill;
    } else {
      __trap();
    }
  }

  template <typename T>
  __device__ static void read(T& localElement, T* readElement, T UNUSED)
  {
    while ((localElement = atomicExch(readElement, UNUSED)) == UNUSED)
      __threadfence();
  }
};
```

The sentinel encodes both "free slot" and "memory ordering checkpoint" in one
word. There is no separate validity bit, no separate sequence number. This is
the SPSC ring-buffer trick generalized to MPMC by combining with
`count`/`front`/`back` atomics. The memory ordering between producer's
`atomicInc(&back)` and consumer's `atomicExch(slot, UNUSED)` is the cell
sentinel itself: the consumer cannot complete a read until the producer's
write makes the cell non-`UNUSED`.

#### 3.1.3 Four Access-Control Policies

`MultiIndexQueue` is templated on an `AccessControl` parameter with four
choices: `AtomicCheckedAbortOnOverflow`, `AtomicCheckedWaitOnOverflow`,
`NonAtomicCheckedAbortOnOverflow`, `NonAtomicCheckedWaitOnOverflow`. The
"non-atomic" variants use cache-bypassing global loads/stores
(`ldg_cg`/`stg_cg`) and read the slot until the sentinel goes away. The
"abort" variants `__trap()` on overflow; the "wait" variants spin until the
consumer drains the queue.

This is how cuRE achieves bounded memory, the property cudaraster lacked. The
queue size is fixed at compile time; if a stage outpaces its consumer, the
producer either traps (if you sized too small) or stalls (turning back-
pressure into real flow control). cudaraster sized intermediate storage for
worst-case scenes; cuRE sizes for steady-state and uses backpressure to
handle bursts.

#### 3.1.4 Three Dequeue Granularities

```cpp
__device__ int dequeue(int q, T& element);                     // 1 thread, 1 item
__device__ int dequeueWarp(int q, T* localElement, int num);   // 32 threads coop
__device__ int dequeueBlock(int q, T* localElement, int num);  // block coop
```

`dequeueBlock` reserves up to `num` items via a single `atomicSub(&count, num)`
in lane 0, broadcasts the offset through shared memory, and the entire block
reads in parallel. A 32-warp block pulling 256 items costs one atomic.
cudaraster did per-thread atomic appending; cuRE does block-coalesced
reservation. This is the major source of the constant-factor speedup over
cudaraster.

#### 3.1.5 What "Multi" Means

`MultiIndexQueue<T, NUMQUEUES, SIZE>` is a single management primitive sharing
counters/cells across `NUMQUEUES` independent queues. cuRE uses one queue per
spatial bin and one queue per tile. With ~thousands of tiles this is just
another dimension on the index array. There is one queue type for the whole
pipeline; the spatial subdivisions are queue indices.

### 3.2 `ProgressQueue` -- The Order Path

The data path described above is unordered. API draw order is enforced by a
separate primitive. From `source/cure/pipeline/progress_queue.cuh`:

```cpp
template<int SIZE>
class ProgressQueue
{
  static_assert(static_popcnt<SIZE>::value == 1,
                "ProgressQueue size must be a power of two");
private:
  static constexpr int QUEUESIZE = (SIZE + 31) / 32;

  unsigned int bitfields[QUEUESIZE];
  unsigned int already_done;
  ...
};
```

This is not a queue at all. It is a bitmap watermark. Each bit `id % SIZE`
represents "has primitive `id` finished the upstream stage?". Producers
`markDone(id)` to set their bit. Consumers find the first un-set bit and use
that as a drain barrier:

```cpp
__device__ void markDone(unsigned int id)
{
  unsigned int withinElementOffset = id % 32;
  unsigned int element = (id / 32) % QUEUESIZE;
  atomicOr(&bitfields[element], 0x1u << withinElementOffset);
}

template<int NUM_THREADS>
__device__ unsigned int checkProgressBlock(
    CheckProgressShared<NUM_THREADS>& sharedstorage)
{
  __shared__ unsigned int current_already_done, new_already_done;
  current_already_done = ldg_cg(&already_done);
  __syncthreads();

  unsigned int myMask =
      ldg_cg(&bitfields[(current_already_done / 32 + threadIdx.x) % QUEUESIZE]);

  unsigned int firstNotCompleted = __ffs(~myMask);
  firstNotCompleted = (current_already_done & 0xFFFFFFE0u) +
      ((firstNotCompleted == 0) ? (NUM_THREADS * 32)
                                : (threadIdx.x * 32 + firstNotCompleted - 1));

  unsigned int agg_result = CheckProgressReduce<NUM_THREADS>(sharedstorage)
      .Reduce(firstNotCompleted, cub::Min());

  if (threadIdx.x == 0)
    new_already_done = max(agg_result,
                           atomicMax(&already_done, agg_result));
  __syncthreads();
  ...
  return new_already_done;
}
```

The protocol per check:

- 32 threads each `ldg_cg` one `uint32_t` of the bitfield starting at the
  current watermark.
- Each thread runs `__ffs(~myMask)` (find first set bit in the negation =
  first 0) to locate the first incomplete primitive in their dword.
- A CUB block reduce takes the minimum across the block. That is the new
  watermark.
- `atomicMax(&already_done, agg_result)` advances it monotonically.

Constant-time per advance, parallel across threads, contention-free across
producers (`atomicOr` on disjoint bits commutes).

The downstream stage then dequeues from the data-path `MultiIndexQueue` only
indices where `id < watermark`. Out-of-order arrivals in the data path are
fine because the order-preservation invariant is maintained separately.

### 3.3 Why Two Queues

Splitting the data path from the order path is the key insight. A unified
ordered queue would force one of two undesirable behaviors: either consumers
spin waiting for in-order arrival (serializing the consumers), or producers
stall to write in-order (serializing the producers). The split lets producers
race to their queues, while consumers see ordered work via a separate cheap
watermark check.

| Concern | Data path (`MultiIndexQueue`) | Order path (`ProgressQueue`) |
|---|---|---|
| Storage | O(SIZE * NUMQUEUES * 4 bytes) | O(SIZE / 8 bytes) |
| Producer write | atomicAdd + sentinel CAS | atomicOr on a shared bit |
| Consumer read | dequeue (drains slot) | check watermark (peek, never drains) |
| Out-of-order tolerated | yes (sentinel handles races) | yes (bitmap is associative) |
| Backpressure | yes (queue fill) | no (fire-and-forget) |
| Order semantics | none | global FIFO via watermark |

## 4. Pipeline Stages

From the file layout in `source/cure/pipeline/`:

```text
                  vertex buffer (indexed)
                         |
                         v
        +---------------------------------+
        | PerWarpPatchGeometryStage       | <- vertex shader, dedup via cache
        +---------------------------------+
                         |
                         v   MultiIndexQueue<TriangleId, ...>
                         |   ProgressQueue tracks geometry-stage completion
                         v
        +---------------------------------+
        | clipping.cuh                    |
        +---------------------------------+
                         |
                         v   MultiIndexQueue<ClippedPrim, ...>
                         v
        +---------------------------------+
        | RasterizationStage              | <- triangle setup, edge equations
        +---------------------------------+
                         |
                         v   MultiIndexQueue<BinPrim, NUMBINS, ...>
                         |   ProgressQueue tracks raster completion
                         v
        +---------------------------------+
        | BinTileRasterizationStage       | <- coarse bin -> fine tile
        | (BinRasterizer + TileRasterizer)|
        +---------------------------------+
                         |
                         v   MultiIndexQueue<Quad, NUMTILES, ...>
                         v
        +---------------------------------+
        | StampShading + fragment_shader  | <- 2x2 quad-granular FS
        +---------------------------------+
                         |
                         v
                    framebuffer (ROP, depth, blend)
```

Notes worth keeping in mind:

- `PerWarpPatchGeometryStage` processes patches at warp granularity. A warp
  processes a patch, caches its post-shading vertices, and emits primitive
  records by index. Subsequent primitives in the same patch reuse cached
  vertex outputs. This is the vertex-reuse mechanism.
- Two raster paths exist: `RasterizationStage.cuh` (exclusive: locks the
  framebuffer region) and `RasterizationStageNonExclusive.cuh` (multiple
  blocks race-write a tile and resolve with atomics). For the demo
  workload exclusive is the right choice and is what the streaming and
  bin-ownership proposals in this directory adopt.
- `BinTileSpace.cuh` is 50 KB because it encodes the two-level binning
  geometry: which bins/tiles a triangle covers, edge-vs-bin tests, hierarchy
  walks. A demo can sidestep most of this with a conservative AABB-based
  triangle-vs-tile test, at the cost of binning some triangles into
  tiles they do not actually cover.
- `bitmask.cuh` provides per-tile sub-pixel coverage masks. This is closer to
  Masked Software Occlusion Culling's coverage-mask trick than to anything in
  cudaraster.
- 2x2 stamp shading is required for screen-space derivatives. It only works
  because tile queues hand out quads, not pixels: a quad's four pixels are
  guaranteed adjacent and shaded together so finite-difference derivatives are
  well-defined.

## 5. Specific Problems And How Each Piece Solves Them

| Problem | cuRE solution |
|---|---|
| Worst-case unbounded intermediate storage | Compile-time fixed `SIZE`, sentinel-based slot reuse, `*OnOverflow` policies |
| Stage barriers serializing the pipeline | Persistent megakernel; warps poll queues instead of returning |
| Producers stalling on consumers | Block-coalesced reservation in `dequeueBlock`; per-tile/per-bin queue parallelism |
| API draw order | Separate `ProgressQueue` watermark, decoupled from data movement |
| Vertex reuse | `MultiIndexQueue<VertexId>` instead of `<VertexData>`; cache stored once, referenced by index |
| Per-stage variable parallelism | Megakernel + dynamic work-pulling instead of fixed grid shapes per stage |
| Screen-space derivatives | 2x2 stamp shading requires quad-granular FS, which only works because tile queues hand out quads |
| Memory ordering across producer/consumer | `__threadfence()` plus sentinel-based slot reuse |

## 6. Porting Notes For AMDGPU + libkfd

What carries over to an AMDGPU/libkfd implementation as proposed in
`streaming-pipeline-proposal.md` and `bin-ownership-pipeline-proposal.md`:

- **Slot-array bounded ring with sentinel values.** Direct port. Use
  `atomic_compare_exchange_strong` on the slot to swap `UNUSED ↔
  index`. AMD `atomicCAS` is L2-coherent on RDNA and uncontended cost
  is competitive with NVIDIA.
- **Dual atomic counter pattern (`back_seq` + `back`).** Direct port.
- **Bitmap-based progress watermark** (`ProgressQueue`) — relevant
  only if API-order preservation matters. The demo skips it (one
  draw call per frame, ordering is implicit).

What needs adaptation:

- **Wave size.** cuRE was designed for NVIDIA warp32. RDNA wave32 is a
  clean port, but GFX9 wave64 changes `dequeueWarp` math: NVIDIA
  `__shfl_sync(~0U, ...)` becomes
  `__builtin_amdgcn_readfirstlane` / wave64-aware ops.
- **Memory ordering.** `__threadfence()` maps to GCR cache flush via
  `ACQUIRE_MEM` PM4. libkfd's `kfd::ComputeQueue` already wraps this
  for barrier-style operations; for in-kernel use the appropriate
  AMDGCN intrinsics (`__builtin_amdgcn_s_waitcnt`,
  `__builtin_amdgcn_buffer_wbinvl1`).
- **Persistent kernel + host-fed queues.** The cuRE megakernel runs
  for the duration of one draw call (kernel-per-frame). For a
  *persistent* megakernel that survives across frames, see
  `frame-completion-detection.md` for how the host detects
  per-frame completion without relying on kernel exit.
- **Host↔GPU shared queues.** cuRE has no host-streamed queue (it
  uploads the whole frame upfront). The streaming proposal adds one
  via fine-grained SVM; see `streaming-pipeline-proposal.md` §2.2.

Relevant AMDGPU/libkfd background:

- `atomicCAS` on global memory has different cost on AMD. The non-atomic
  sentinel-spin variants (`ldg_cg`/`stg_cg` equivalents) are likely better on
  AMD where coherent atomics go through L2.
- `kfd::Signal` and `kfd::ComputeQueue::wait_reg_mem` provide the
  PM4 plumbing for interrupt-driven host wait on GPU-written values
  (see `frame-completion-detection.md` §5.8).
- `PipelineConfig<>` baked at compile time multiplies the kernel matrix.
  cuRE's README explicitly warns that compile times grow linearly with
  active configs. Demo code should pin a single PipelineConfig and only
  vary it per-AMDGPU-arch via the libkfd cross-compile flow.

## 7. The Shorter Version

Two sentences:

1. cuRE moves work between pipeline stages through a slot-array bounded ring
   (`MultiIndexQueue`) using sentinel values to detect free/full slots
   without separate validity bits, and uses block-coalesced atomic
   reservation to keep the queue protocol cheap.
2. cuRE preserves API draw order by tracking completion as a bitmap watermark
   (`ProgressQueue`), a separate primitive whose only job is `markDone(id)`
   and "find first incomplete", decoupled from the data path so out-of-order
   producers do not pay an ordering cost.

Everything else -- megakernel, persistent threads, two-level binning, stamp
shading, vertex reuse -- is pipeline structure built on top of those two
primitives.

## 8. Recommended Reading

In priority order if any of this is to be ported:

- Markus Steinberger et al., *Softshell: Dynamic Scheduling on GPUs*,
  TOG 2012. The scheduling-on-GPU foundation.
- Markus Steinberger et al., *Whippletree: Task-based Scheduling of Dynamic
  Workloads on the GPU*, TOG 2014. The queue primitives that became
  `MultiIndexQueue`.
- Michael Kenzel, Bernhard Kerbl, Dieter Schmalstieg, Markus Steinberger,
  *A High-Performance Software Graphics Pipeline Architecture for the GPU*,
  ACM TOG / SIGGRAPH 2018. cuRE itself.
- `github.com/GPUPeople/cuRE`, MIT-licensed reference implementation. The
  files most worth reading are `index_queue.cuh`, `progress_queue.cuh`,
  `BinTileSpace.cuh`, and `BinTileRasterizationStage.cuh`.
- `streaming-pipeline-proposal.md` and
  `bin-ownership-pipeline-proposal.md` in this directory — the two
  candidate AMDGPU/libkfd ports of the cuRE shape.
- `frame-completion-detection.md` — how the host detects per-frame
  completion against a persistent megakernel, since cuRE's
  kernel-per-draw-call model does not generalize.
