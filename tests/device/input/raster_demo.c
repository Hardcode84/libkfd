#include <gpuintrin.h>

struct DemoVertex {
  float x;
  float y;
  float z;
  float u;
  float v;
};

struct DemoPrimitive {
  struct DemoVertex v0;
  struct DemoVertex v1;
  struct DemoVertex v2;
};

struct RasterArgs {
  const struct DemoPrimitive *prims;
  unsigned prim_count;
  unsigned *color;
  float *depth;
  unsigned width;
  unsigned height;
  unsigned pitch;
  unsigned tile_size;
  unsigned clear_color;
  float clear_depth;
};

struct PersistentControl {
  volatile unsigned terminate;
  volatile unsigned frame_id;
  volatile unsigned prim_count;
  volatile unsigned tiles_done;
  volatile unsigned rendered;
  volatile unsigned ready;
  volatile unsigned heartbeat;
  unsigned _pad;
};

struct PersistentRasterArgs {
  const struct DemoPrimitive *prims;
  unsigned *color;
  float *depth;
  struct PersistentControl *control;
  unsigned width;
  unsigned height;
  unsigned pitch;
  unsigned tile_size;
  unsigned clear_color;
  float clear_depth;
};

static constexpr unsigned TILE_PRIM_CAP = 4096;

[[clang::loader_uninitialized]]
static __gpu_local unsigned lds_prim_count;
[[clang::loader_uninitialized]]
static __gpu_local unsigned lds_overflow;
[[clang::loader_uninitialized]]
static __gpu_local unsigned lds_prim_indices[TILE_PRIM_CAP];

static float min2(float a, float b) { return a < b ? a : b; }
static float max2(float a, float b) { return a > b ? a : b; }
static float min3(float a, float b, float c) { return min2(min2(a, b), c); }
static float max3(float a, float b, float c) { return max2(max2(a, b), c); }

static float edge(float ax, float ay, float bx, float by, float px, float py) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

static bool overlaps_tile(struct DemoPrimitive tri, float x0, float y0,
                          float x1, float y1) {
  float min_x = min3(tri.v0.x, tri.v1.x, tri.v2.x);
  float max_x = max3(tri.v0.x, tri.v1.x, tri.v2.x);
  float min_y = min3(tri.v0.y, tri.v1.y, tri.v2.y);
  float max_y = max3(tri.v0.y, tri.v1.y, tri.v2.y);
  return max_x >= x0 && min_x < x1 && max_y >= y0 && min_y < y1;
}

static unsigned shade_checker(float u, float v) {
  unsigned iu = (unsigned)(u * 8.0f);
  unsigned iv = (unsigned)(v * 8.0f);
  return ((iu ^ iv) & 1u) ? 0xffe0e0e0u : 0xff202020u;
}

static void raster_tile(const struct DemoPrimitive *prims, unsigned prim_count,
                        unsigned *color, float *depth, unsigned width,
                        unsigned height, unsigned pitch, unsigned tile_size,
                        unsigned clear_color, float clear_depth,
                        unsigned tile_x, unsigned tile_y) {
  unsigned lx = __gpu_thread_id_x();
  unsigned ly = __gpu_thread_id_y();
  unsigned tx = __gpu_num_threads_x();
  unsigned ty = __gpu_num_threads_y();

  unsigned base_x = tile_x * tile_size;
  unsigned base_y = tile_y * tile_size;
  unsigned tid = lx + ly * tx;
  unsigned threads = tx * ty;
  unsigned tile_end_x = base_x + tile_size;
  unsigned tile_end_y = base_y + tile_size;
  if (tile_end_x > width)
    tile_end_x = width;
  if (tile_end_y > height)
    tile_end_y = height;

  if (tid == 0) {
    lds_prim_count = 0;
    lds_overflow = 0;
  }
  __gpu_sync_threads();

  for (unsigned i = tid; i < prim_count; i += threads) {
    struct DemoPrimitive tri = prims[i];
    if (!overlaps_tile(tri, (float)base_x, (float)base_y, (float)tile_end_x,
                       (float)tile_end_y))
      continue;
    unsigned slot = __atomic_fetch_add(&lds_prim_count, 1u, __ATOMIC_RELAXED);
    if (slot < TILE_PRIM_CAP)
      lds_prim_indices[slot] = i;
    else
      lds_overflow = 1;
  }
  __gpu_sync_threads();

  unsigned local_count = lds_prim_count;
  if (local_count > TILE_PRIM_CAP)
    local_count = TILE_PRIM_CAP;
  bool overflow = lds_overflow != 0;

  for (unsigned oy = ly; oy < tile_size; oy += ty) {
    unsigned y = base_y + oy;
    if (y >= height)
      continue;

    for (unsigned ox = lx; ox < tile_size; ox += tx) {
      unsigned x = base_x + ox;
      if (x >= width)
        continue;

      float px = (float)x + 0.5f;
      float py = (float)y + 0.5f;
      float best_depth = clear_depth;
      unsigned best_color = clear_color;

      unsigned count = overflow ? prim_count : local_count;
      for (unsigned j = 0; j < count; ++j) {
        unsigned i = overflow ? j : lds_prim_indices[j];
        struct DemoPrimitive tri = prims[i];
        float area =
            edge(tri.v0.x, tri.v0.y, tri.v1.x, tri.v1.y, tri.v2.x, tri.v2.y);
        if (area > -0.00001f && area < 0.00001f)
          continue;

        float w0 = edge(tri.v1.x, tri.v1.y, tri.v2.x, tri.v2.y, px, py) / area;
        float w1 = edge(tri.v2.x, tri.v2.y, tri.v0.x, tri.v0.y, px, py) / area;
        float w2 = edge(tri.v0.x, tri.v0.y, tri.v1.x, tri.v1.y, px, py) / area;
        if (w0 < -0.0001f || w1 < -0.0001f || w2 < -0.0001f)
          continue;

        float z = w0 * tri.v0.z + w1 * tri.v1.z + w2 * tri.v2.z;
        if (z < 0.0f || z > best_depth)
          continue;

        float u = w0 * tri.v0.u + w1 * tri.v1.u + w2 * tri.v2.u;
        float v = w0 * tri.v0.v + w1 * tri.v1.v + w2 * tri.v2.v;
        best_depth = z;
        best_color = shade_checker(u, v);
      }

      unsigned idx = y * pitch + x;
      depth[idx] = best_depth;
      color[idx] = best_color;
    }
  }
}

__gpu_kernel void headless_stage0_raster(struct RasterArgs args) {
  raster_tile(args.prims, args.prim_count, args.color, args.depth, args.width,
              args.height, args.pitch, args.tile_size, args.clear_color,
              args.clear_depth, __gpu_block_id_x(), __gpu_block_id_y());
}

__gpu_kernel void headless_stage1_persistent(struct PersistentRasterArgs args) {
  unsigned tile_x = __gpu_block_id_x();
  unsigned tile_y = __gpu_block_id_y();
  unsigned tile_id = tile_y * __gpu_num_blocks_x() + tile_x;
  unsigned tile_count = __gpu_num_blocks_x() * __gpu_num_blocks_y();
  unsigned seen_frame = 0xffffffffu;

  for (;;) {
    if (__atomic_load_n(&args.control->terminate, __ATOMIC_ACQUIRE) != 0)
      break;

    unsigned ready = __atomic_load_n(&args.control->ready, __ATOMIC_ACQUIRE);
    unsigned frame = __atomic_load_n(&args.control->frame_id, __ATOMIC_ACQUIRE);
    if (ready != 0 && frame != seen_frame) {
      unsigned prim_count =
          __atomic_load_n(&args.control->prim_count, __ATOMIC_ACQUIRE);
      if (tile_id < tile_count) {
        raster_tile(args.prims, prim_count, args.color, args.depth, args.width,
                    args.height, args.pitch, args.tile_size, args.clear_color,
                    args.clear_depth, tile_x, tile_y);
      }
      __gpu_sync_threads();

      if (__gpu_thread_id_x() == 0 && __gpu_thread_id_y() == 0 &&
          __gpu_thread_id_z() == 0) {
        __atomic_fetch_add(&args.control->heartbeat, 1u, __ATOMIC_RELAXED);
        unsigned done = __atomic_fetch_add(&args.control->tiles_done, 1u,
                                           __ATOMIC_ACQ_REL) +
                        1u;
        if (done == tile_count) {
          __atomic_store_n(&args.control->rendered, prim_count,
                           __ATOMIC_RELEASE);
          __atomic_store_n(&args.control->ready, 0u, __ATOMIC_RELEASE);
        }
      }
      seen_frame = frame;
      continue;
    }

    __builtin_amdgcn_s_sleep(4);
  }
}
