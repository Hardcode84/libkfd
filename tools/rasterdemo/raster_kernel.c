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

struct TileRange {
  unsigned offset;
  unsigned count;
};

struct RasterArgs {
  const struct DemoPrimitive *prims;
  const unsigned *tile_indices;
  const struct TileRange *tile_ranges;
  unsigned *color;
  float *depth;
  unsigned width;
  unsigned height;
  unsigned pitch;
  unsigned tile_size;
  unsigned tiles_x;
  unsigned clear_color;
  float clear_depth;
};

struct PersistentFrame {
  const struct DemoPrimitive *prims;
  const unsigned *tile_indices;
  const struct TileRange *tile_ranges;
  volatile unsigned *tile_claims;
  unsigned *color;
  float *depth;
  unsigned width;
  unsigned height;
  unsigned pitch;
  unsigned tile_size;
  unsigned tiles_x;
  unsigned tiles_y;
  unsigned clear_color;
  float clear_depth;
  unsigned clear_only;
};

struct ClaimFrameArgs {
  const struct DemoPrimitive *prims;
  const unsigned *tile_indices;
  const struct TileRange *tile_ranges;
  volatile unsigned *tile_claims;
  unsigned *color;
  float *depth;
  unsigned width;
  unsigned height;
  unsigned pitch;
  unsigned tile_size;
  unsigned tiles_x;
  unsigned tiles_y;
  unsigned clear_color;
  float clear_depth;
  unsigned clear_only;
  unsigned tile_count;
  unsigned frame_epoch;
};

struct PersistentControl {
  volatile unsigned terminate;
  volatile unsigned ready;
  volatile unsigned frame_id;
  volatile unsigned active_slot;
  volatile unsigned init_cursor;
  volatile unsigned init_tiles_done;
  volatile unsigned tiles_done;
  volatile unsigned frame_done;
  volatile unsigned tile_count;
};

struct PersistentArgs {
  struct PersistentControl *control;
  const struct DemoPrimitive *prims;
  const unsigned *tile_indices;
  const struct TileRange *tile_ranges;
  volatile unsigned *tile_claims0;
  volatile unsigned *tile_claims1;
  volatile unsigned *tile_claims2;
  unsigned *color0;
  unsigned *color1;
  unsigned *color2;
  float *depth;
  unsigned width;
  unsigned height;
  unsigned pitch;
  unsigned tile_size;
  unsigned tiles_x;
  unsigned tiles_y;
  unsigned clear_color;
  float clear_depth;
  unsigned clear_only;
};

struct PersistentLaunchArgs {
  const struct PersistentArgs *args;
};

struct ProbeArgs {
  unsigned *out;
};

[[clang::loader_uninitialized]]
static __gpu_local volatile unsigned lds_claimed_tile;
[[clang::loader_uninitialized]]
static __gpu_local volatile unsigned lds_tile_range_offset;
[[clang::loader_uninitialized]]
static __gpu_local volatile unsigned lds_tile_range_count;

static float edge(float ax, float ay, float bx, float by, float px, float py) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

static unsigned shade_checker(float u, float v) {
  unsigned iu = (unsigned)(u * 8.0f);
  unsigned iv = (unsigned)(v * 8.0f);
  return ((iu ^ iv) & 1u) ? 0xffe0e0e0u : 0xff202020u;
}

static unsigned pick_tile(volatile unsigned *tile_claims, unsigned tile_count,
                          unsigned frame_epoch) {
  unsigned wg = __gpu_block_id_x();
  unsigned start = (wg * 2654435761u) % tile_count;
  unsigned claimed = frame_epoch | 0x80000000u;
  for (unsigned off = 0; off < tile_count; ++off) {
    unsigned tile = start + off;
    if (tile >= tile_count)
      tile -= tile_count;
    unsigned expected = frame_epoch;
    if (__atomic_compare_exchange_n(&tile_claims[tile], &expected, claimed,
                                    false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_ACQUIRE))
      return tile;
  }
  return tile_count;
}

__gpu_kernel void rasterdemo_probe(struct ProbeArgs args) {
  if (__gpu_block_id_x() == 0 && __gpu_thread_id_x() == 0)
    *args.out = 0xcafebabeu;
}

static void clear_tile(unsigned *color, float *depth, unsigned width,
                       unsigned height, unsigned pitch, unsigned tile_size,
                       unsigned clear_color, float clear_depth,
                       unsigned tile_x, unsigned tile_y) {
  unsigned lx = __gpu_thread_id_x();
  unsigned ly = __gpu_thread_id_y();
  unsigned tx = __gpu_num_threads_x();
  unsigned ty = __gpu_num_threads_y();

  unsigned base_x = tile_x * tile_size;
  unsigned base_y = tile_y * tile_size;
  for (unsigned oy = ly; oy < tile_size; oy += ty) {
    unsigned y = base_y + oy;
    if (y >= height)
      continue;
    for (unsigned ox = lx; ox < tile_size; ox += tx) {
      unsigned x = base_x + ox;
      if (x >= width)
        continue;
      unsigned idx = y * pitch + x;
      depth[idx] = clear_depth;
      color[idx] = clear_color;
    }
  }
}

static void raster_tile(const struct DemoPrimitive *prims,
                        const unsigned *tile_indices,
                        struct TileRange range, unsigned *color, float *depth,
                        unsigned width, unsigned height, unsigned pitch,
                        unsigned tile_size,
                        unsigned clear_color, float clear_depth,
                        unsigned tile_x, unsigned tile_y) {
  unsigned lx = __gpu_thread_id_x();
  unsigned ly = __gpu_thread_id_y();
  unsigned tx = __gpu_num_threads_x();
  unsigned ty = __gpu_num_threads_y();

  unsigned base_x = tile_x * tile_size;
  unsigned base_y = tile_y * tile_size;

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

      for (unsigned j = 0; j < range.count; ++j) {
        unsigned i = tile_indices[range.offset + j];
        struct DemoPrimitive tri = prims[i];
        float area = edge(tri.v0.x, tri.v0.y, tri.v1.x, tri.v1.y, tri.v2.x,
                          tri.v2.y);
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

__gpu_kernel void rasterdemo_clear_frame(struct RasterArgs args) {
  clear_tile(args.color, args.depth, args.width, args.height, args.pitch,
             args.tile_size, args.clear_color, args.clear_depth,
             __gpu_block_id_x(), __gpu_block_id_y());
}

__gpu_kernel void rasterdemo_frame(struct RasterArgs args) {
  unsigned tile_x = __gpu_block_id_x();
  unsigned tile_y = __gpu_block_id_y();
  struct TileRange range = args.tile_ranges[tile_y * args.tiles_x + tile_x];
  raster_tile(args.prims, args.tile_indices, range, args.color, args.depth,
              args.width, args.height, args.pitch, args.tile_size,
              args.clear_color, args.clear_depth, tile_x, tile_y);
}

__gpu_kernel void rasterdemo_claim_frame(struct ClaimFrameArgs args) {
  unsigned tid = __gpu_thread_id_x() +
                 __gpu_thread_id_y() * __gpu_num_threads_x() +
                 __gpu_thread_id_z() * __gpu_num_threads_x() *
                     __gpu_num_threads_y();

  for (;;) {
    if (tid == 0)
      lds_claimed_tile =
          pick_tile(args.tile_claims, args.tile_count, args.frame_epoch);
    __gpu_sync_threads();
    unsigned tile = lds_claimed_tile;
    if (tile >= args.tile_count)
      break;

    if (tid == 0) {
      struct TileRange range = args.tile_ranges[tile];
      lds_tile_range_offset = range.offset;
      lds_tile_range_count = range.count;
    }
    __gpu_sync_threads();

    unsigned tile_x = tile % args.tiles_x;
    unsigned tile_y = tile / args.tiles_x;
    if (args.clear_only) {
      clear_tile(args.color, args.depth, args.width, args.height, args.pitch,
                 args.tile_size, args.clear_color, args.clear_depth, tile_x,
                 tile_y);
    } else {
      struct TileRange range = {
          .offset = lds_tile_range_offset,
          .count = lds_tile_range_count,
      };
      raster_tile(args.prims, args.tile_indices, range, args.color, args.depth,
                  args.width, args.height, args.pitch, args.tile_size,
                  args.clear_color, args.clear_depth, tile_x, tile_y);
    }
    __gpu_sync_threads();
  }
}

__gpu_kernel void rasterdemo_persistent(struct PersistentLaunchArgs launch) {
  struct PersistentArgs args = *launch.args;
  unsigned seen_frame = 0xffffffffu;
  unsigned tid = __gpu_thread_id_x() +
                 __gpu_thread_id_y() * __gpu_num_threads_x() +
                 __gpu_thread_id_z() * __gpu_num_threads_x() *
                     __gpu_num_threads_y();

  for (;;) {
    if (__atomic_load_n(&args.control->terminate, __ATOMIC_ACQUIRE) != 0)
      break;

    unsigned ready = __atomic_load_n(&args.control->ready, __ATOMIC_ACQUIRE);
    unsigned frame = __atomic_load_n(&args.control->frame_id, __ATOMIC_ACQUIRE);
    if (ready == 0 || frame == seen_frame) {
      __builtin_amdgcn_s_sleep(4);
      continue;
    }

    unsigned slot = __atomic_load_n(&args.control->active_slot,
                                    __ATOMIC_ACQUIRE);
    struct PersistentFrame f;
    f.prims = args.prims;
    f.tile_indices = args.tile_indices;
    f.tile_ranges = args.tile_ranges;
    f.tile_claims = slot == 0 ? args.tile_claims0
                    : slot == 1 ? args.tile_claims1
                                : args.tile_claims2;
    f.color = slot == 0 ? args.color0 : slot == 1 ? args.color1 : args.color2;
    f.depth = args.depth;
    f.width = args.width;
    f.height = args.height;
    f.pitch = args.pitch;
    f.tile_size = args.tile_size;
    f.tiles_x = args.tiles_x;
    f.tiles_y = args.tiles_y;
    f.clear_color = args.clear_color;
    f.clear_depth = args.clear_depth;
    f.clear_only = args.clear_only;
    unsigned tile_count = __atomic_load_n(&args.control->tile_count,
                                          __ATOMIC_ACQUIRE);
    unsigned frame_epoch = frame & 0x7fffffffu;

    if (tid == 0) {
      for (;;) {
        unsigned tile =
            __atomic_fetch_add(&args.control->init_cursor, 1u,
                               __ATOMIC_ACQ_REL);
        if (tile >= tile_count)
          break;
        __atomic_exchange_n(&f.tile_claims[tile], frame_epoch,
                            __ATOMIC_ACQ_REL);
        __atomic_fetch_add(&args.control->init_tiles_done, 1u,
                           __ATOMIC_ACQ_REL);
      }
    }
    while (__atomic_load_n(&args.control->init_tiles_done, __ATOMIC_ACQUIRE) <
           tile_count) {
      __builtin_amdgcn_s_sleep(4);
    }
    for (;;) {
      if (tid == 0)
        lds_claimed_tile =
            pick_tile(f.tile_claims, tile_count, frame_epoch);
      __gpu_sync_threads();
      unsigned tile = lds_claimed_tile;
      if (tile >= tile_count)
        break;
      if (tid == 0) {
        struct TileRange range = f.tile_ranges[tile];
        lds_tile_range_offset = range.offset;
        lds_tile_range_count = range.count;
      }
      __gpu_sync_threads();
      unsigned tile_x = tile % f.tiles_x;
      unsigned tile_y = tile / f.tiles_x;
      if (f.clear_only) {
        clear_tile(f.color, f.depth, f.width, f.height, f.pitch, f.tile_size,
                   f.clear_color, f.clear_depth, tile_x, tile_y);
      } else {
        struct TileRange range = {
            .offset = lds_tile_range_offset,
            .count = lds_tile_range_count,
        };
        raster_tile(f.prims, f.tile_indices, range, f.color, f.depth, f.width,
                    f.height, f.pitch, f.tile_size, f.clear_color,
                    f.clear_depth, tile_x, tile_y);
      }
      __gpu_sync_threads();

      if (__gpu_thread_id_x() == 0 && __gpu_thread_id_y() == 0 &&
          __gpu_thread_id_z() == 0) {
        unsigned done =
            __atomic_fetch_add(&args.control->tiles_done, 1u,
                               __ATOMIC_ACQ_REL) +
            1u;
        if (done == tile_count) {
          __atomic_store_n(&args.control->frame_done, frame, __ATOMIC_RELEASE);
          __atomic_store_n(&args.control->ready, 0u, __ATOMIC_RELEASE);
        }
      }
      __gpu_sync_threads();
    }

    seen_frame = frame;
  }
}
