# cuRE Streaming Queues

Companion to `quake-compute-rasterizer-prior-art.md`, which names cuRE
(Kenzel, Kerbl, Schmalstieg, Steinberger, SIGGRAPH 2018) as the primary
reference for production-quality compute-rasterization but does not describe
its mechanics. This document does. It exists so that a future migration of
`quake_raster` away from single-level barrier-per-stage binning has a concrete
reference for what "streaming pipeline with bounded queues" means in
implementation terms.

All code excerpts are from the published cuRE implementation at
`github.com/GPUPeople/cuRE`, MIT-licensed.

## 1. The Problem cuRE Solves

`quake_raster` today follows the chunker-style shape inherited from cudaraster
(2011): each pipeline stage is a separate compute dispatch, the host waits on
the previous dispatch's fence before dispatching the next, and stage outputs
are sized for worst-case inputs.

```text
host: dispatch tile_bin     -> implicit ordering on compute queue
host: dispatch world_raster -> kfd_gpu_fence_wait
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

For `quake_raster` the equivalent would be: instead of materializing
`qr_raster_triangle` records into per-tile lists, materialize *indices into
the global triangle buffer* into per-tile queues. The triangle data is
written once.

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
  blocks race-write a tile and resolve with atomics). For a Quake-shaped
  workload exclusive is the right choice and matches `quake_raster`'s current
  workgroup-per-tile assignment.
- `BinTileSpace.cuh` is 50 KB because it encodes the two-level binning
  geometry: which bins/tiles a triangle covers, edge-vs-bin tests, hierarchy
  walks. The complexity of this math is what `quake_raster`'s
  `qr_triangle_intersects_tile` AABB test sidesteps by being conservative.
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

## 6. Adoption Notes For `quake_raster`

`quake_raster` today is the cudaraster shape: separate dispatches per stage,
worst-case-sized intermediates, host-driven sequencing. The TODO doc lists
this under "Stage F: Hierarchical Binning" but does not yet specify whether
that stage adopts cuRE-style streaming or just adds a coarse level above
`16x16` tiles.

Migration deltas if `quake_raster` ever adopts cuRE's shape:

| Component | `quake_raster` today | cuRE-shaped |
|---|---|---|
| Kernel structure | 4 separate dispatches, host-driven | 1 megakernel, persistent waves |
| Bin storage | `tile_indices[tile][256]`, fixed worst-case | `MultiIndexQueue<TriangleId, num_tiles, SIZE>`, bounded |
| Order | implicit via host submission order + queue order | explicit `ProgressQueue` watermark |
| Hierarchy | `16x16` tiles only | coarse bin then fine tile |
| Work assignment | static: 1 thread per tile in bin, 1 thread per pixel in raster | dynamic: `dequeueBlock` pulls work |
| Vertex reuse | host duplicates vertices in fan triangulation | indexed vertex queue |

AMDGPU/KFD-specific notes if this direction is ever pursued:

- AMD wave size affects queue granularity. cuRE was designed for NVIDIA
  warp32. RDNA wave32 is a clean port, but GFX9 wave64 changes
  `dequeueWarp` math: `__shfl_sync(~0U, ...)` becomes
  `__builtin_amdgcn_readfirstlane`/wave64-aware ops.
- `atomicCAS` on global memory has different cost on AMD. The non-atomic
  sentinel-spin variants (`ldg_cg`/`stg_cg` equivalents) are likely better on
  AMD where coherent atomics go through L2. AMD GCR cache flush
  (`ACQUIRE_MEM`) is the `__threadfence()` equivalent and is already used by
  `kfd::ComputeQueue` barriers.
- `KFD_GPU_KERNEL` is a one-shot kernel. `libkfd` does not currently have a
  way to launch a long-lived kernel and feed work into it from the host. The
  KFD doorbell ring buffers carry PM4 packets, not in-flight task data. A
  cuRE-style port would need either a kernel that runs for the whole frame
  consuming an in-VRAM queue the host fills via SDMA, or a redesign of the
  dispatch primitive in `kfd::ComputeQueue`.
- `PipelineConfig<>` baked at compile time multiplies the kernel matrix.
  `quake_raster` already cross-compiles per AMDGPU arch (see
  `quake_raster_add_gpu_kernel` in `CMakeLists.txt`). Adding "per launch
  config" stacks on top of that. cuRE's README explicitly warns that
  compile times go up linearly with active configs.

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
- `quake-compute-rasterizer-prior-art.md` in this repository for the broader
  lineage and for why cuRE is the right reference for Quake-shaped
  workloads rather than Nanite or voxel-search renderers.
