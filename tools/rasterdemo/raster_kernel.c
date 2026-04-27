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

struct ProbeArgs {
  unsigned *out;
};

static float edge(float ax, float ay, float bx, float by, float px, float py) {
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

static unsigned shade_checker(float u, float v) {
  unsigned iu = (unsigned)(u * 8.0f);
  unsigned iv = (unsigned)(v * 8.0f);
  return ((iu ^ iv) & 1u) ? 0xffe0e0e0u : 0xff202020u;
}

__gpu_kernel void rasterdemo_probe(struct ProbeArgs args) {
  if (__gpu_block_id_x() == 0 && __gpu_thread_id_x() == 0)
    *args.out = 0xcafebabeu;
}

__gpu_kernel void rasterdemo_clear_frame(struct RasterArgs args) {
  unsigned tile_x = __gpu_block_id_x();
  unsigned tile_y = __gpu_block_id_y();
  unsigned lx = __gpu_thread_id_x();
  unsigned ly = __gpu_thread_id_y();
  unsigned tx = __gpu_num_threads_x();
  unsigned ty = __gpu_num_threads_y();

  unsigned base_x = tile_x * args.tile_size;
  unsigned base_y = tile_y * args.tile_size;
  for (unsigned oy = ly; oy < args.tile_size; oy += ty) {
    unsigned y = base_y + oy;
    if (y >= args.height)
      continue;
    for (unsigned ox = lx; ox < args.tile_size; ox += tx) {
      unsigned x = base_x + ox;
      if (x >= args.width)
        continue;
      unsigned idx = y * args.pitch + x;
      args.depth[idx] = args.clear_depth;
      args.color[idx] = args.clear_color;
    }
  }
}

__gpu_kernel void rasterdemo_frame(struct RasterArgs args) {
  unsigned tile_x = __gpu_block_id_x();
  unsigned tile_y = __gpu_block_id_y();
  unsigned lx = __gpu_thread_id_x();
  unsigned ly = __gpu_thread_id_y();
  unsigned tx = __gpu_num_threads_x();
  unsigned ty = __gpu_num_threads_y();

  unsigned base_x = tile_x * args.tile_size;
  unsigned base_y = tile_y * args.tile_size;
  struct TileRange range = args.tile_ranges[tile_y * args.tiles_x + tile_x];

  for (unsigned oy = ly; oy < args.tile_size; oy += ty) {
    unsigned y = base_y + oy;
    if (y >= args.height)
      continue;

    for (unsigned ox = lx; ox < args.tile_size; ox += tx) {
      unsigned x = base_x + ox;
      if (x >= args.width)
        continue;

      float px = (float)x + 0.5f;
      float py = (float)y + 0.5f;
      float best_depth = args.clear_depth;
      unsigned best_color = args.clear_color;

      for (unsigned j = 0; j < range.count; ++j) {
        unsigned i = args.tile_indices[range.offset + j];
        struct DemoPrimitive tri = args.prims[i];
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

      unsigned idx = y * args.pitch + x;
      args.depth[idx] = best_depth;
      args.color[idx] = best_color;
    }
  }
}
