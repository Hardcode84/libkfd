#include <gpuintrin.h>

#include "raster_common.h"

struct PersistentFrame {
  const struct DemoPrimitive *prims;
  const unsigned *tile_indices;
  const struct TileRange *tile_ranges;
  const unsigned *active_tiles;
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

[[clang::loader_uninitialized]]
static __gpu_local unsigned lds_claimed_tile;
[[clang::loader_uninitialized]]
static __gpu_local unsigned lds_tile_range_offset;
[[clang::loader_uninitialized]]
static __gpu_local unsigned lds_tile_range_count;
[[clang::loader_uninitialized]]
static __gpu_local unsigned lds_current_epoch;
[[clang::loader_uninitialized]]
static __gpu_local unsigned lds_terminate;

static float edge(float ax, float ay, float bx, float by, float px, float py) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

static unsigned shade_checker(float u, float v) {
  unsigned iu = (unsigned)(u * 8.0f);
  unsigned iv = (unsigned)(v * 8.0f);
  return ((iu ^ iv) & 1u) ? 0xffe0e0e0u : 0xff202020u;
}

static unsigned pop_active_tile(unsigned *cursor, const unsigned *active_tiles,
                                unsigned active_count) {
  unsigned index = __atomic_fetch_add(cursor, 1u, __ATOMIC_ACQ_REL);
  if (index >= active_count)
    return 0xffffffffu;
  return __atomic_load_n(&active_tiles[index], __ATOMIC_ACQUIRE);
}

__gpu_kernel void rasterdemo_probe(struct ProbeArgs args) {
  if (__gpu_block_id_x() == 0 && __gpu_thread_id_x() == 0)
    *args.out = 0xcafebabeu;
}

static void clear_tile(unsigned *color, float *depth, unsigned width,
                       unsigned height, unsigned pitch, unsigned tile_size,
                       unsigned clear_color, float clear_depth, unsigned tile_x,
                       unsigned tile_y) {
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
                        const unsigned *tile_indices, struct TileRange range,
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

static void clear_subtile(unsigned *color, float *depth, unsigned width,
                          unsigned height, unsigned pitch, unsigned tile_size,
                          unsigned clear_color, float clear_depth,
                          unsigned tile_x, unsigned tile_y, unsigned subtile,
                          unsigned lane) {
  unsigned sub_x = (subtile % SUBTILES_X) * SUBTILE_WIDTH;
  unsigned sub_y = (subtile / SUBTILES_X) * SUBTILE_HEIGHT;
  unsigned base_x = tile_x * tile_size + sub_x;
  unsigned base_y = tile_y * tile_size + sub_y;

  for (unsigned p = lane; p < SUBTILE_PIXELS; p += 32u) {
    unsigned x = base_x + (p % SUBTILE_WIDTH);
    unsigned y = base_y + (p / SUBTILE_WIDTH);
    if (x >= width || y >= height)
      continue;
    unsigned idx = y * pitch + x;
    depth[idx] = clear_depth;
    color[idx] = clear_color;
  }
}

static void raster_subtile(const struct DemoPrimitive *prims,
                           const unsigned *tile_indices, struct TileRange range,
                           unsigned *color, float *depth, unsigned width,
                           unsigned height, unsigned pitch, unsigned tile_size,
                           unsigned clear_color, float clear_depth,
                           unsigned tile_x, unsigned tile_y, unsigned subtile,
                           unsigned lane) {
  unsigned sub_x = (subtile % SUBTILES_X) * SUBTILE_WIDTH;
  unsigned sub_y = (subtile / SUBTILES_X) * SUBTILE_HEIGHT;
  unsigned base_x = tile_x * tile_size + sub_x;
  unsigned base_y = tile_y * tile_size + sub_y;

  unsigned p0 = lane;
  unsigned x0 = base_x + (p0 % SUBTILE_WIDTH);
  unsigned y0 = base_y + (p0 / SUBTILE_WIDTH);
  unsigned valid0 = x0 < width && y0 < height;
  float px0 = (float)x0 + 0.5f;
  float py0 = (float)y0 + 0.5f;
  float best_depth0 = clear_depth;
  unsigned best_color0 = clear_color;

  unsigned p1 = lane + 32u;
  unsigned x1 = base_x + (p1 % SUBTILE_WIDTH);
  unsigned y1 = base_y + (p1 / SUBTILE_WIDTH);
  unsigned valid1 = x1 < width && y1 < height;
  float px1 = (float)x1 + 0.5f;
  float py1 = (float)y1 + 0.5f;
  float best_depth1 = clear_depth;
  unsigned best_color1 = clear_color;

  for (unsigned j = 0; j < range.count; ++j) {
    unsigned i = tile_indices[range.offset + j];
    struct DemoPrimitive tri = prims[i];
    float area =
        edge(tri.v0.x, tri.v0.y, tri.v1.x, tri.v1.y, tri.v2.x, tri.v2.y);
    if (area > -0.00001f && area < 0.00001f)
      continue;

    float inv_area = 1.0f / area;
    if (valid0) {
      float w0 =
          edge(tri.v1.x, tri.v1.y, tri.v2.x, tri.v2.y, px0, py0) * inv_area;
      float w1 =
          edge(tri.v2.x, tri.v2.y, tri.v0.x, tri.v0.y, px0, py0) * inv_area;
      float w2 =
          edge(tri.v0.x, tri.v0.y, tri.v1.x, tri.v1.y, px0, py0) * inv_area;
      if (w0 >= -0.0001f && w1 >= -0.0001f && w2 >= -0.0001f) {
        float z = w0 * tri.v0.z + w1 * tri.v1.z + w2 * tri.v2.z;
        if (z >= 0.0f && z <= best_depth0) {
          float u = w0 * tri.v0.u + w1 * tri.v1.u + w2 * tri.v2.u;
          float v = w0 * tri.v0.v + w1 * tri.v1.v + w2 * tri.v2.v;
          best_depth0 = z;
          best_color0 = shade_checker(u, v);
        }
      }
    }
    if (valid1) {
      float w0 =
          edge(tri.v1.x, tri.v1.y, tri.v2.x, tri.v2.y, px1, py1) * inv_area;
      float w1 =
          edge(tri.v2.x, tri.v2.y, tri.v0.x, tri.v0.y, px1, py1) * inv_area;
      float w2 =
          edge(tri.v0.x, tri.v0.y, tri.v1.x, tri.v1.y, px1, py1) * inv_area;
      if (w0 >= -0.0001f && w1 >= -0.0001f && w2 >= -0.0001f) {
        float z = w0 * tri.v0.z + w1 * tri.v1.z + w2 * tri.v2.z;
        if (z >= 0.0f && z <= best_depth1) {
          float u = w0 * tri.v0.u + w1 * tri.v1.u + w2 * tri.v2.u;
          float v = w0 * tri.v0.v + w1 * tri.v1.v + w2 * tri.v2.v;
          best_depth1 = z;
          best_color1 = shade_checker(u, v);
        }
      }
    }
  }

  if (valid0) {
    unsigned idx = y0 * pitch + x0;
    depth[idx] = best_depth0;
    color[idx] = best_color0;
  }
  if (valid1) {
    unsigned idx = y1 * pitch + x1;
    depth[idx] = best_depth1;
    color[idx] = best_color1;
  }
}

static void process_tile_subtiled(const struct DemoPrimitive *prims,
                                  const unsigned *tile_indices,
                                  struct TileRange range, unsigned *color,
                                  float *depth, unsigned width, unsigned height,
                                  unsigned pitch, unsigned tile_size,
                                  unsigned clear_color, float clear_depth,
                                  unsigned tile_x, unsigned tile_y,
                                  unsigned clear_only, unsigned tid) {
  const unsigned lane = tid & 31u;
  const unsigned wave = tid >> 5u;

  if (wave < PERSISTENT_WAVE_COUNT) {
    for (unsigned pass = 0; pass < SUBTILE_PASSES_PER_WAVE; ++pass) {
      unsigned subtile = pass * PERSISTENT_WAVE_COUNT + wave;
      if (subtile >= NUM_SUBTILES)
        continue;

      if (clear_only) {
        clear_subtile(color, depth, width, height, pitch, tile_size,
                      clear_color, clear_depth, tile_x, tile_y, subtile, lane);
      } else {
        raster_subtile(prims, tile_indices, range, color, depth, width, height,
                       pitch, tile_size, clear_color, clear_depth, tile_x,
                       tile_y, subtile, lane);
      }
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

static unsigned linear_id() {
  return __gpu_thread_id_x() + __gpu_thread_id_y() * __gpu_num_threads_x() +
         __gpu_thread_id_z() * __gpu_num_threads_x() * __gpu_num_threads_y();
}

__gpu_kernel void rasterdemo_claim_frame(struct ClaimFrameArgs args) {
  unsigned tid = linear_id();

  for (;;) {
    if (tid == 0)
      lds_claimed_tile = pop_active_tile(args.active_cursor, args.active_tiles,
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
  struct PersistentControl *control = launch.args->control;
  unsigned seen_epoch = 0;
  unsigned tid = linear_id();

  for (;;) {
    if (tid == 0) {
      lds_terminate = __atomic_load_n(&control->terminate, __ATOMIC_ACQUIRE);
      lds_current_epoch =
          __atomic_load_n(&control->current_epoch, __ATOMIC_ACQUIRE);
    }
    __gpu_sync_threads();

    if (lds_terminate != 0)
      break;

    unsigned epoch = lds_current_epoch;
    if (epoch == 0 || epoch == seen_epoch) {
      __builtin_amdgcn_s_sleep(4);
      continue;
    }

    struct PersistentArgs args = *launch.args;
    struct PersistentFrame f;
    f.prims = args.prims;
    f.tile_indices = args.tile_indices;
    f.tile_ranges = args.tile_ranges;
    f.active_tiles = args.active_tiles;
    f.color = args.color;
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
    unsigned active_count = f.tiles_x * f.tiles_y;

    for (;;) {
      struct TileRange range;
      if (tid == 0) {
        unsigned tile = pop_active_tile(&control->active_cursor, f.active_tiles,
                                        active_count);

        if (tile != 0xffffffffu)
          range = f.tile_ranges[tile];

        lds_claimed_tile = tile;
      }
      __gpu_sync_threads();
      unsigned tile = lds_claimed_tile;
      if (tile == 0xffffffffu)
        break;

      if (tid == 0) {
        lds_tile_range_offset = range.offset;
        lds_tile_range_count = range.count;
      }
      unsigned tile_x = tile % f.tiles_x;
      unsigned tile_y = tile / f.tiles_x;
      __gpu_sync_threads();
      range = (struct TileRange){
          .offset = lds_tile_range_offset,
          .count = lds_tile_range_count,
      };
      process_tile_subtiled(f.prims, f.tile_indices, range, f.color, f.depth,
                            f.width, f.height, f.pitch, f.tile_size,
                            f.clear_color, f.clear_depth, tile_x, tile_y,
                            f.clear_only, tid);
    }

    if (tid == 0)
      __atomic_fetch_add(&control->completed_wgs, 1u, __ATOMIC_ACQ_REL);
    seen_epoch = epoch;
  }
}
