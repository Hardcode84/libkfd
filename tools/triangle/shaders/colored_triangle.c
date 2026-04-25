#include "libkfd/gpu/kernel.h"

typedef float __attribute__((ext_vector_type(2))) float2;
typedef float __attribute__((ext_vector_type(3))) float3;

struct Vertex {
  float x;
  float y;
  float r;
  float g;
  float b;
};

struct TriangleArgs {
  unsigned *framebuffer;
  unsigned width;
  unsigned height;
  float time;
  struct Vertex v0;
  struct Vertex v1;
  struct Vertex v2;
};

static float edge(float2 a, float2 b, float2 p) {
  return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
}

static float2 vertex_pos(struct Vertex v) { return (float2){v.x, v.y}; }

static float3 vertex_color(struct Vertex v) { return (float3){v.r, v.g, v.b}; }

static float3 clamp3(float3 x, float lo, float hi) {
  float3 lo3 = {lo, lo, lo};
  float3 hi3 = {hi, hi, hi};
  return __builtin_elementwise_minnum(__builtin_elementwise_maxnum(x, lo3),
                                      hi3);
}

static unsigned pack_xrgb(float3 c) {
  c = clamp3(c, 0.0f, 1.0f);
  return 0xff000000u | ((unsigned)(c.r * 255.0f) << 16u) |
         ((unsigned)(c.g * 255.0f) << 8u) | (unsigned)(c.b * 255.0f);
}

KFD_GPU_KERNEL void colored_triangle(struct TriangleArgs *args) {
  unsigned x = kfd_global_id_x();
  unsigned y = kfd_global_id_y();
  if (x >= args->width || y >= args->height)
    return;

  float2 p = {(float)x + 0.5f, (float)y + 0.5f};
  float2 p0 = vertex_pos(args->v0);
  float2 p1 = vertex_pos(args->v1);
  float2 p2 = vertex_pos(args->v2);
  float area = edge(p0, p1, p2);

  float3 bg = {0.015f, 0.018f, 0.03f};
  float3 color = bg;
  if (area != 0.0f) {
    float w0 = edge(p1, p2, p) / area;
    float w1 = edge(p2, p0, p) / area;
    float w2 = edge(p0, p1, p) / area;
    if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
      color = w0 * vertex_color(args->v0) + w1 * vertex_color(args->v1) +
              w2 * vertex_color(args->v2);
    }
  }

  args->framebuffer[y * args->width + x] = pack_xrgb(color);
}
