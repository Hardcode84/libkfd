#include "libkfd/gpu/kernel.h"

#define QR_TILE_SIZE 16U

struct QrRasterVertex {
  float x;
  float y;
  float z;
  float u;
  float v;
  float light_u;
  float light_v;
};

struct QrRasterTriangle {
  struct QrRasterVertex v0;
  struct QrRasterVertex v1;
  struct QrRasterVertex v2;
  unsigned surface;
};

struct QrTileBinArgs {
  struct QrRasterTriangle *triangles;
  unsigned *tile_indices;
  unsigned *tile_counts;
  unsigned *tile_overflows;
  unsigned width;
  unsigned height;
  unsigned triangle_count;
  unsigned tile_cols;
  unsigned tile_rows;
  unsigned tile_triangle_capacity;
};

static float qr_min3(float a, float b, float c)
{
  float out = a < b ? a : b;
  return out < c ? out : c;
}

static float qr_max3(float a, float b, float c)
{
  float out = a > b ? a : b;
  return out > c ? out : c;
}

static int qr_triangle_intersects_tile(struct QrRasterTriangle *triangle,
                                       unsigned tile_x, unsigned tile_y,
                                       unsigned width, unsigned height)
{
  float min_x = qr_min3(triangle->v0.x, triangle->v1.x, triangle->v2.x);
  float max_x = qr_max3(triangle->v0.x, triangle->v1.x, triangle->v2.x);
  float min_y = qr_min3(triangle->v0.y, triangle->v1.y, triangle->v2.y);
  float max_y = qr_max3(triangle->v0.y, triangle->v1.y, triangle->v2.y);
  unsigned tile_min_x = tile_x * QR_TILE_SIZE;
  unsigned tile_min_y = tile_y * QR_TILE_SIZE;
  unsigned tile_max_x = tile_min_x + QR_TILE_SIZE;
  unsigned tile_max_y = tile_min_y + QR_TILE_SIZE;

  if (tile_max_x > width) {
    tile_max_x = width;
  }
  if (tile_max_y > height) {
    tile_max_y = height;
  }
  if (max_x < (float)tile_min_x || min_x >= (float)tile_max_x ||
      max_y < (float)tile_min_y || min_y >= (float)tile_max_y) {
    return 0;
  }
  return 1;
}

KFD_GPU_KERNEL void qr_bin_tiles(struct QrTileBinArgs *args)
{
  unsigned tile = kfd_global_id_x();
  unsigned tile_count = args->tile_cols * args->tile_rows;
  unsigned tile_x;
  unsigned tile_y;
  unsigned count = 0U;
  unsigned overflow = 0U;
  unsigned i;

  if (tile >= tile_count) {
    return;
  }

  tile_x = tile % args->tile_cols;
  tile_y = tile / args->tile_cols;
  for (i = 0U; i < args->triangle_count; ++i) {
    if (!qr_triangle_intersects_tile(&args->triangles[i], tile_x, tile_y,
                                     args->width, args->height)) {
      continue;
    }
    if (count < args->tile_triangle_capacity) {
      args->tile_indices[tile * args->tile_triangle_capacity + count] = i;
      ++count;
    } else {
      ++overflow;
    }
  }
  args->tile_counts[tile] = count;
  args->tile_overflows[tile] = overflow;
}
