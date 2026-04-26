#include "libkfd/gpu/kernel.h"

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

struct QrTextureRecord {
  unsigned mip_offset[4];
  unsigned width[4];
  unsigned height[4];
  unsigned mip_count;
  unsigned flags;
};

struct QrLightmapRecord {
  unsigned offset;
  unsigned width;
  unsigned height;
};

struct QrSurfaceRecord {
  unsigned texture;
  unsigned lightmap;
  unsigned flags;
  float plane[4];
  float tex_s[4];
  float tex_t[4];
  float light_s[4];
  float light_t[4];
};

struct QrWorldRasterArgs {
  unsigned char *dst;
  float *depth;
  struct QrRasterTriangle *triangles;
  struct QrSurfaceRecord *surfaces;
  struct QrTextureRecord *textures;
  struct QrLightmapRecord *lightmaps;
  unsigned char *texture_atlas;
  unsigned char *lightmap_atlas;
  unsigned char *colormap;
  unsigned width;
  unsigned height;
  unsigned triangle_count;
  unsigned debug_mode;
  unsigned *tile_indices;
  unsigned *tile_counts;
  unsigned *tile_overflows;
  unsigned tile_cols;
  unsigned tile_rows;
  unsigned tile_triangle_capacity;
};

static float qr_edge(float ax, float ay, float bx, float by, float px, float py)
{
  return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}

static unsigned qr_clamp_coord(float value, unsigned limit)
{
  int coord;

  if (limit == 0U || value <= 0.0f) {
    return 0U;
  }
  coord = (int)value;
  if ((unsigned)coord >= limit) {
    return limit - 1U;
  }
  return (unsigned)coord;
}

static unsigned qr_wrap_coord(float value, unsigned limit)
{
  int coord;
  int signed_limit;

  if (limit == 0U) {
    return 0U;
  }
  coord = (int)value;
  signed_limit = (int)limit;
  coord %= signed_limit;
  if (coord < 0) {
    coord += signed_limit;
  }
  return (unsigned)coord;
}

static unsigned char qr_sample_texture(struct QrTextureRecord *texture,
                                       unsigned char *atlas, float u, float v)
{
  unsigned width = texture->width[0];
  unsigned height = texture->height[0];
  unsigned x = qr_wrap_coord(u, width);
  unsigned y = qr_wrap_coord(v, height);

  return atlas[texture->mip_offset[0] + y * width + x];
}

static unsigned char qr_sample_lightmap(struct QrLightmapRecord *lightmap,
                                        unsigned char *atlas, float u, float v)
{
  unsigned width = lightmap->width;
  unsigned height = lightmap->height;
  unsigned x = qr_clamp_coord(u, width);
  unsigned y = qr_clamp_coord(v, height);

  return atlas[lightmap->offset + y * width + x];
}

static void qr_consider_triangle(struct QrWorldRasterArgs *args,
                                 struct QrRasterTriangle *triangle, float px,
                                 float py, float *best_depth,
                                 unsigned char *best_color)
{
  float area = qr_edge(triangle->v0.x, triangle->v0.y, triangle->v1.x,
                       triangle->v1.y, triangle->v2.x, triangle->v2.y);
  float w0;
  float w1;
  float w2;
  float depth;
  struct QrSurfaceRecord *surface;
  struct QrTextureRecord *texture;
  struct QrLightmapRecord *lightmap;
  unsigned char texel;
  unsigned char light;
  unsigned char color;

  if (area > -0.00001f && area < 0.00001f) {
    return;
  }

  w0 = qr_edge(triangle->v1.x, triangle->v1.y, triangle->v2.x,
               triangle->v2.y, px, py) / area;
  w1 = qr_edge(triangle->v2.x, triangle->v2.y, triangle->v0.x,
               triangle->v0.y, px, py) / area;
  w2 = 1.0f - w0 - w1;
  if (w0 < -0.0001f || w1 < -0.0001f || w2 < -0.0001f) {
    return;
  }

  depth = w0 * triangle->v0.z + w1 * triangle->v1.z + w2 * triangle->v2.z;
  if (depth >= *best_depth) {
    return;
  }

  surface = &args->surfaces[triangle->surface];
  texture = &args->textures[surface->texture];
  lightmap = &args->lightmaps[surface->lightmap];
  texel = qr_sample_texture(texture, args->texture_atlas,
                            w0 * triangle->v0.u + w1 * triangle->v1.u +
                                w2 * triangle->v2.u,
                            w0 * triangle->v0.v + w1 * triangle->v1.v +
                                w2 * triangle->v2.v);
  light = qr_sample_lightmap(lightmap, args->lightmap_atlas,
                             w0 * triangle->v0.light_u +
                                 w1 * triangle->v1.light_u +
                                 w2 * triangle->v2.light_u,
                             w0 * triangle->v0.light_v +
                                 w1 * triangle->v1.light_v +
                                 w2 * triangle->v2.light_v);

  if (args->debug_mode == 1U) {
    color = (unsigned char)((triangle->surface + 1U) & 0xffU);
  } else if (args->debug_mode == 2U) {
    float scaled = depth * 32.0f;
    if (scaled < 0.0f) {
      scaled = 0.0f;
    }
    if (scaled > 255.0f) {
      scaled = 255.0f;
    }
    color = (unsigned char)(255U - (unsigned)scaled);
  } else if (args->debug_mode == 3U) {
    color = texel;
  } else if (args->debug_mode == 4U) {
    color = light;
  } else {
    color = args->colormap[((unsigned)light << 8U) | (unsigned)texel];
  }

  *best_depth = depth;
  *best_color = color;
}

KFD_GPU_KERNEL void qr_world_raster(struct QrWorldRasterArgs *args)
{
  unsigned x = kfd_global_id_x();
  unsigned y = kfd_global_id_y();
  unsigned pixel;
  unsigned tile;
  unsigned count;
  float px;
  float py;
  float best_depth;
  unsigned char best_color;
  unsigned i;

  if (x >= args->width || y >= args->height) {
    return;
  }

  pixel = y * args->width + x;
  tile = (y / 16U) * args->tile_cols + (x / 16U);
  count = args->tile_counts[tile];
  px = (float)x + 0.5f;
  py = (float)y + 0.5f;
  best_depth = 3.402823466e38f;
  best_color = args->dst[pixel];

  if (args->tile_overflows[tile] != 0U) {
    for (i = 0U; i < args->triangle_count; ++i) {
      qr_consider_triangle(args, &args->triangles[i], px, py, &best_depth,
                           &best_color);
    }
  } else {
    unsigned base = tile * args->tile_triangle_capacity;

    for (i = 0U; i < count; ++i) {
      unsigned triangle_index = args->tile_indices[base + i];
      qr_consider_triangle(args, &args->triangles[triangle_index], px, py,
                           &best_depth, &best_color);
    }
  }

  args->depth[pixel] = best_depth;
  args->dst[pixel] = best_color;
}
