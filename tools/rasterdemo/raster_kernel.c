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
  const volatile unsigned *active_tiles;
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
  const volatile unsigned *active_tiles;
  volatile unsigned *active_cursor;
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
  unsigned active_count;
};

struct PersistentControl {
  volatile unsigned terminate;
  volatile unsigned current_epoch;
  volatile unsigned sealed_epoch;
  volatile unsigned closing_epoch;
  volatile unsigned completed_epoch;
  volatile unsigned active_slot;
  volatile unsigned active_cursor;
  volatile unsigned active_count;
  volatile unsigned render_done;
};

struct PersistentArgs {
  struct PersistentControl *control;
  const struct DemoPrimitive *prims;
  const unsigned *tile_indices;
  const struct TileRange *tile_ranges;
  const volatile unsigned *active_tiles0;
  const volatile unsigned *active_tiles1;
  const volatile unsigned *active_tiles2;
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
[[clang::loader_uninitialized]]
static __gpu_local volatile unsigned lds_frame_closed;

static float edge(float ax, float ay, float bx, float by, float px, float py) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

static unsigned shade_checker(float u, float v) {
  unsigned iu = (unsigned)(u * 8.0f);
  unsigned iv = (unsigned)(v * 8.0f);
  return ((iu ^ iv) & 1u) ? 0xffe0e0e0u : 0xff202020u;
}

static unsigned pop_active_tile(volatile unsigned *cursor,
                                const volatile unsigned *active_tiles,
                                unsigned active_count) {
  unsigned index = __atomic_fetch_add(cursor, 1u, __ATOMIC_ACQ_REL);
  if (index >= active_count)
    return 0xffffffffu;
  return __atomic_load_n(&active_tiles[index], __ATOMIC_ACQUIRE);
}

static unsigned try_complete_epoch(struct PersistentControl *control,
                                   unsigned epoch, unsigned active_count) {
  unsigned sealed = __atomic_load_n(&control->sealed_epoch, __ATOMIC_ACQUIRE);
  unsigned cursor = __atomic_load_n(&control->active_cursor, __ATOMIC_ACQUIRE);
  unsigned done = __atomic_load_n(&control->render_done, __ATOMIC_ACQUIRE);
  if (sealed < epoch || cursor < active_count || done < active_count)
    return __atomic_load_n(&control->completed_epoch, __ATOMIC_ACQUIRE) >= epoch;

  unsigned expected = epoch - 1u;
  if (__atomic_compare_exchange_n(&control->closing_epoch, &expected, epoch,
                                  false, __ATOMIC_ACQ_REL,
                                  __ATOMIC_ACQUIRE)) {
    __atomic_store_n(&control->completed_epoch, epoch, __ATOMIC_RELEASE);
    return 1u;
  }

  return __atomic_load_n(&control->completed_epoch, __ATOMIC_ACQUIRE) >= epoch;
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
          pop_active_tile(args.active_cursor, args.active_tiles,
                          args.active_count);
    __gpu_sync_threads();
    unsigned tile = lds_claimed_tile;
    if (tile == 0xffffffffu)
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
  unsigned seen_epoch = 0;
  unsigned tid = __gpu_thread_id_x() +
                 __gpu_thread_id_y() * __gpu_num_threads_x() +
                 __gpu_thread_id_z() * __gpu_num_threads_x() *
                     __gpu_num_threads_y();

  for (;;) {
    if (__atomic_load_n(&args.control->terminate, __ATOMIC_ACQUIRE) != 0)
      break;

    unsigned epoch =
        __atomic_load_n(&args.control->current_epoch, __ATOMIC_ACQUIRE);
    if (epoch == 0 || epoch == seen_epoch) {
      __builtin_amdgcn_s_sleep(4);
      continue;
    }

    unsigned slot = __atomic_load_n(&args.control->active_slot,
                                    __ATOMIC_ACQUIRE);
    struct PersistentFrame f;
    f.prims = args.prims;
    f.tile_indices = args.tile_indices;
    f.tile_ranges = args.tile_ranges;
    f.active_tiles = slot == 0 ? args.active_tiles0
                     : slot == 1 ? args.active_tiles1
                                 : args.active_tiles2;
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
    unsigned active_count = __atomic_load_n(&args.control->active_count,
                                            __ATOMIC_ACQUIRE);

    for (;;) {
      if (__atomic_load_n(&args.control->terminate, __ATOMIC_ACQUIRE) != 0)
        return;

      if (tid == 0)
        lds_claimed_tile =
            pop_active_tile(&args.control->active_cursor, f.active_tiles,
                            active_count);
      __gpu_sync_threads();
      unsigned tile = lds_claimed_tile;
      if (tile == 0xffffffffu) {
        for (;;) {
          if (__atomic_load_n(&args.control->terminate, __ATOMIC_ACQUIRE) != 0)
            return;
          if (tid == 0)
            lds_frame_closed =
                try_complete_epoch(args.control, epoch, active_count);
          __gpu_sync_threads();
          if (lds_frame_closed)
            break;
          __builtin_amdgcn_s_sleep(4);
        }
        break;
      }
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
        __atomic_fetch_add(&args.control->render_done, 1u, __ATOMIC_ACQ_REL);
        lds_frame_closed = try_complete_epoch(args.control, epoch, active_count);
      }
      __gpu_sync_threads();
      if (lds_frame_closed)
        break;
    }

    seen_epoch = epoch;
  }
}
