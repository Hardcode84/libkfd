#define _POSIX_C_SOURCE 200809L

#include "libkfd/quake_raster.h"

#include "libkfd/gpu.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QR_CLEAR_BLOCK_X 16U
#define QR_CLEAR_BLOCK_Y 16U
#define QR_DEFAULT_MAX_TEXTURES 1024U
#define QR_DEFAULT_MAX_LIGHTMAPS 1024U
#define QR_DEFAULT_MAX_SURFACES 65536U
#define QR_DEFAULT_MAX_WORLDS 256U
#define QR_DEFAULT_MAX_FRAME_TRIANGLES 65536U
#define QR_DEFAULT_MAX_ALIAS_MODELS 1024U
#define QR_DEFAULT_MAX_ALIAS_TRIANGLES 65536U
#define QR_DEFAULT_TEXTURE_ATLAS_BYTES (16U * 1024U * 1024U)
#define QR_DEFAULT_LIGHTMAP_ATLAS_BYTES (4U * 1024U * 1024U)
#define QR_RASTER_BLOCK_X 8U
#define QR_RASTER_BLOCK_Y 8U
#define QR_TILE_BIN_BLOCK_X 64U
#define QR_MAX_KERNEL_IMAGE_BYTES (64U * 1024U * 1024U)

typedef struct qr_kernel_candidate {
  const char *path;
  const char *arch;
} qr_kernel_candidate;

typedef struct qr_clear_indexed_args {
  uint8_t *dst;
  uint32_t width;
  uint32_t height;
  uint32_t color;
} qr_clear_indexed_args;

typedef struct qr_resolve_xrgb_args {
  uint8_t *src;
  uint32_t *dst;
  uint32_t *palette;
  uint32_t width;
  uint32_t height;
} qr_resolve_xrgb_args;

typedef struct qr_raster_vertex {
  float x;
  float y;
  float z;
  float u;
  float v;
  float light_u;
  float light_v;
} qr_raster_vertex;

typedef struct qr_raster_triangle {
  qr_raster_vertex v0;
  qr_raster_vertex v1;
  qr_raster_vertex v2;
  uint32_t surface;
} qr_raster_triangle;

typedef struct qr_texture_record {
  uint32_t mip_offset[QR_TEXTURE_MIP_COUNT];
  uint32_t width[QR_TEXTURE_MIP_COUNT];
  uint32_t height[QR_TEXTURE_MIP_COUNT];
  uint32_t mip_count;
  uint32_t flags;
} qr_texture_record;

typedef struct qr_lightmap_record {
  uint32_t offset;
  uint32_t width;
  uint32_t height;
} qr_lightmap_record;

typedef struct qr_surface_record {
  uint32_t texture;
  uint32_t lightmap;
  uint32_t flags;
  float plane[4];
  float tex_s[4];
  float tex_t[4];
  float light_s[4];
  float light_t[4];
} qr_surface_record;

typedef struct qr_world_record {
  uint32_t first_surface;
  uint32_t surface_count;
} qr_world_record;

typedef struct qr_alias_triangle_record {
  qr_raster_vertex v0;
  qr_raster_vertex v1;
  qr_raster_vertex v2;
  uint32_t surface;
} qr_alias_triangle_record;

typedef struct qr_alias_model_record {
  uint32_t world;
  uint32_t first_triangle;
  uint32_t triangle_count;
} qr_alias_model_record;

typedef struct qr_world_raster_args {
  uint8_t *dst;
  float *depth;
  qr_raster_triangle *triangles;
  qr_surface_record *surfaces;
  qr_texture_record *textures;
  qr_lightmap_record *lightmaps;
  uint8_t *texture_atlas;
  uint8_t *lightmap_atlas;
  uint8_t *colormap;
  uint32_t width;
  uint32_t height;
  uint32_t triangle_count;
  uint32_t debug_mode;
  float time_seconds;
  uint32_t *tile_indices;
  uint32_t *tile_counts;
  uint32_t *tile_overflows;
  uint32_t *tile_depth_min;
  uint32_t *tile_depth_max;
  uint32_t tile_cols;
  uint32_t tile_rows;
  uint32_t tile_triangle_capacity;
} qr_world_raster_args;

typedef struct qr_tile_bin_args {
  qr_raster_triangle *triangles;
  uint32_t *tile_indices;
  uint32_t *tile_counts;
  uint32_t *tile_overflows;
  uint32_t *tile_depth_min;
  uint32_t *tile_depth_max;
  uint32_t width;
  uint32_t height;
  uint32_t triangle_count;
  uint32_t tile_cols;
  uint32_t tile_rows;
  uint32_t tile_triangle_capacity;
} qr_tile_bin_args;

/* Keep these in sync with the device kernel structs; see docs. */
typedef char qr_assert_unsigned_abi_width[(sizeof(unsigned) == sizeof(uint32_t))
                                              ? 1
                                              : -1];
typedef char qr_assert_raster_vertex_abi_size
    [(sizeof(qr_raster_vertex) == 7U * sizeof(float)) ? 1 : -1];
typedef char qr_assert_raster_triangle_abi_size
    [(sizeof(qr_raster_triangle) ==
      (3U * sizeof(qr_raster_vertex)) + sizeof(uint32_t))
         ? 1
         : -1];
typedef char qr_assert_texture_record_abi_size
    [(sizeof(qr_texture_record) ==
      (3U * QR_TEXTURE_MIP_COUNT + 2U) * sizeof(uint32_t))
         ? 1
         : -1];

static const qr_kernel_candidate qr_clear_indexed_kernels[] = {
#include "quake_raster_clear_indexed_kernels.inc"
    {NULL, NULL}
};

static const qr_kernel_candidate qr_resolve_xrgb_kernels[] = {
#include "quake_raster_resolve_xrgb_kernels.inc"
    {NULL, NULL}
};

static const qr_kernel_candidate qr_world_raster_kernels[] = {
#include "quake_raster_world_raster_kernels.inc"
    {NULL, NULL}
};

static const qr_kernel_candidate qr_tile_bin_kernels[] = {
#include "quake_raster_tile_bin_kernels.inc"
    {NULL, NULL}
};

struct qr_frame {
  struct qr_context *ctx;
};

struct qr_context {
  kfd_gpu_context *gpu;
  kfd_gpu_buffer *indexed;
  kfd_gpu_buffer *xrgb;
  kfd_gpu_buffer *resolve_palette;
  kfd_gpu_buffer *depth;
  kfd_gpu_buffer *texture_atlas;
  kfd_gpu_buffer *lightmap_atlas;
  kfd_gpu_buffer *texture_metadata;
  kfd_gpu_buffer *lightmap_metadata;
  kfd_gpu_buffer *surface_metadata;
  kfd_gpu_buffer *raster_colormap;
  qr_texture_record *textures;
  qr_lightmap_record *lightmaps;
  qr_world_record *worlds;
  qr_alias_model_record *alias_models;
  qr_alias_triangle_record *alias_triangles;
  kfd_gpu_module *clear_module;
  kfd_gpu_kernel *clear_kernel;
  kfd_gpu_buffer *clear_root;
  kfd_gpu_buffer *clear_kernarg;
  kfd_gpu_fence *clear_fence;
  kfd_gpu_dispatch_config clear_dispatch;
  kfd_gpu_module *resolve_module;
  kfd_gpu_kernel *resolve_kernel;
  kfd_gpu_buffer *resolve_root;
  kfd_gpu_buffer *resolve_kernarg;
  kfd_gpu_fence *resolve_fence;
  kfd_gpu_dispatch_config resolve_dispatch;
  kfd_gpu_module *raster_module;
  kfd_gpu_kernel *raster_kernel;
  kfd_gpu_buffer *raster_root;
  kfd_gpu_buffer *raster_kernarg;
  kfd_gpu_fence *raster_fence;
  kfd_gpu_dispatch_config raster_dispatch;
  kfd_gpu_module *tile_bin_module;
  kfd_gpu_kernel *tile_bin_kernel;
  kfd_gpu_buffer *tile_bin_root;
  kfd_gpu_buffer *tile_bin_kernarg;
  kfd_gpu_fence *tile_bin_fence;
  kfd_gpu_dispatch_config tile_bin_dispatch;
  kfd_gpu_buffer *tile_indices;
  kfd_gpu_buffer *tile_counts;
  kfd_gpu_buffer *tile_overflows;
  kfd_gpu_buffer *tile_depth_min;
  kfd_gpu_buffer *tile_depth_max;
  kfd_gpu_buffer *triangle_buffer;
  qr_raster_triangle *triangles;
  uint32_t *triangulation_indices;
  uint32_t triangle_capacity;
  uint32_t tile_cols;
  uint32_t tile_rows;
  uint32_t tile_count;
  qr_raster_stats last_stats;
  uint32_t width;
  uint32_t height;
  size_t indexed_bytes;
  size_t xrgb_bytes;
  size_t texture_atlas_bytes;
  size_t texture_atlas_used;
  size_t lightmap_atlas_bytes;
  size_t lightmap_atlas_used;
  uint32_t texture_capacity;
  uint32_t texture_count;
  uint32_t lightmap_capacity;
  uint32_t lightmap_count;
  uint32_t surface_capacity;
  uint32_t surface_count;
  uint32_t world_capacity;
  uint32_t world_count;
  uint32_t alias_model_capacity;
  uint32_t alias_model_count;
  uint32_t alias_triangle_capacity;
  uint32_t alias_triangle_count;
  qr_output_mode output_mode;
  qr_present_callback present;
  void *present_userdata;
  uint32_t present_palette_xrgb[256];
  qr_perf_counters perf;
  struct qr_frame frame;
  int frame_active;
};

static qr_result qr_result_from_code(int code, qr_result fallback)
{
  switch (code) {
  case 0:
    return QR_SUCCESS;
  case EINVAL:
    return QR_ERROR_INVALID_ARGUMENT;
  case ENOTSUP:
    return QR_ERROR_UNSUPPORTED;
  case EOVERFLOW:
    return QR_ERROR_OVERFLOW;
  case ENOMEM:
    return QR_ERROR_OUT_OF_MEMORY;
  case ENOSPC:
    return QR_ERROR_NO_SPACE;
  case EBUSY:
    return QR_ERROR_BUSY;
  case ENOENT:
    return QR_ERROR_NOT_FOUND;
  case EIO:
    return QR_ERROR_IO;
  default:
    return fallback;
  }
}

static qr_result qr_result_from_errno(int code)
{
  return qr_result_from_code(code, QR_ERROR_SYSTEM);
}

static qr_result qr_result_from_gpu_error(int code)
{
  return qr_result_from_code(code, QR_ERROR_GPU);
}

static void *qr_alloc_bytes(size_t size)
{
  return malloc(size);
}

static void *qr_calloc_bytes(size_t count, size_t size)
{
  if (count != 0U && size > SIZE_MAX / count) {
    return NULL;
  }
  return calloc(count, size);
}

static void qr_free_bytes(void *ptr)
{
  free(ptr);
}

static uint32_t qr_default_u32(uint32_t value, uint32_t fallback)
{
  return value != 0U ? value : fallback;
}

static size_t qr_default_size(size_t value, size_t fallback)
{
  return value != 0U ? value : fallback;
}

static uint64_t qr_now_ns(void)
{
  struct timespec ts;

  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return 0U;
  }
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int qr_rect_tight_size(uint32_t width, uint32_t height, size_t *out);
static int qr_create_upload_buffer(qr_context *ctx, size_t size,
                                   kfd_gpu_buffer **out);

static int qr_mul_size(size_t lhs, size_t rhs, size_t *out)
{
  if (out == NULL) {
    return EINVAL;
  }
  if (lhs != 0U && rhs > SIZE_MAX / lhs) {
    return EOVERFLOW;
  }
  *out = lhs * rhs;
  return 0;
}

static int qr_copy_indexed_rect(uint8_t *dst, size_t dst_offset,
                                size_t dst_size, const void *src,
                                uint32_t width, uint32_t height,
                                size_t stride)
{
  const uint8_t *src_row;
  uint8_t *dst_row;
  size_t bytes;
  uint32_t y;
  int err;

  if (dst == NULL || src == NULL || width == 0U || height == 0U) {
    return EINVAL;
  }
  if (stride == 0U) {
    stride = (size_t)width;
  }
  err = qr_rect_tight_size(width, height, &bytes);
  if (err != 0) {
    return err;
  }
  if (stride < (size_t)width || dst_offset > dst_size ||
      bytes > dst_size - dst_offset) {
    return EINVAL;
  }

  src_row = (const uint8_t *)src;
  dst_row = dst + dst_offset;
  for (y = 0U; y < height; ++y) {
    memcpy(dst_row, src_row, (size_t)width);
    src_row += stride;
    dst_row += width;
  }
  return 0;
}

static int qr_rect_tight_size(uint32_t width, uint32_t height, size_t *out)
{
  if (out == NULL || width == 0U || height == 0U) {
    return EINVAL;
  }
  if ((size_t)width > SIZE_MAX / (size_t)height) {
    return EOVERFLOW;
  }
  *out = (size_t)width * (size_t)height;
  return 0;
}

static int qr_validate_desc(const qr_desc *desc)
{
  if (desc == NULL) {
    return EINVAL;
  }
  if (desc->width == 0U || desc->height == 0U) {
    return EINVAL;
  }
  if (desc->output_mode != QR_OUTPUT_NOOUTPUT &&
      desc->output_mode != QR_OUTPUT_PRESENT) {
    return ENOTSUP;
  }
  if (desc->output_mode == QR_OUTPUT_PRESENT &&
      (desc->present == NULL || desc->present_palette_xrgb == NULL)) {
    return EINVAL;
  }
  if (desc->framebuffer_format != QR_FRAMEBUFFER_INDEXED8) {
    return ENOTSUP;
  }
  if (desc->width > UINT32_MAX - (QR_CLEAR_BLOCK_X - 1U) ||
      desc->height > UINT32_MAX - (QR_CLEAR_BLOCK_Y - 1U)) {
    return EOVERFLOW;
  }
  if ((size_t)desc->width > SIZE_MAX / (size_t)desc->height) {
    return EOVERFLOW;
  }
  if ((size_t)desc->width * (size_t)desc->height > (size_t)UINT32_MAX) {
    return EOVERFLOW;
  }
  if ((size_t)desc->width * (size_t)desc->height >
      SIZE_MAX / sizeof(uint32_t)) {
    return EOVERFLOW;
  }
  return 0;
}

static int qr_indexed_read_size(uint32_t width, uint32_t height,
                                size_t stride, size_t *out)
{
  size_t rows;

  if (out == NULL || width == 0U || height == 0U) {
    return EINVAL;
  }
  if (stride == 0U) {
    stride = (size_t)width;
  }
  if (stride < (size_t)width) {
    return EINVAL;
  }
  rows = (size_t)height - 1U;
  if (rows > (SIZE_MAX - (size_t)width) / stride) {
    return EOVERFLOW;
  }
  *out = rows * stride + (size_t)width;
  return 0;
}

static int qr_xrgb_read_size(uint32_t width, uint32_t height,
                             size_t stride_pixels, size_t *out)
{
  size_t pixels;
  size_t rows;

  if (out == NULL || width == 0U || height == 0U) {
    return EINVAL;
  }
  if (stride_pixels == 0U) {
    stride_pixels = (size_t)width;
  }
  if (stride_pixels < (size_t)width) {
    return EINVAL;
  }
  rows = (size_t)height - 1U;
  if (rows > (SIZE_MAX - (size_t)width) / stride_pixels) {
    return EOVERFLOW;
  }
  pixels = rows * stride_pixels + (size_t)width;
  if (pixels > SIZE_MAX / sizeof(uint32_t)) {
    return EOVERFLOW;
  }
  *out = pixels * sizeof(uint32_t);
  return 0;
}

static int qr_write_all(FILE *file, const void *data, size_t size)
{
  if (size == 0U) {
    return 0;
  }
  if (fwrite(data, 1U, size, file) != size) {
    return EIO;
  }
  return 0;
}

static int qr_read_file(const char *path, uint8_t **out_data, size_t *out_size)
{
  FILE *file;
  uint8_t *data;
  long size;
  int err;

  if (path == NULL || out_data == NULL || out_size == NULL) {
    return EINVAL;
  }
  *out_data = NULL;
  *out_size = 0U;

  file = fopen(path, "rb");
  if (file == NULL) {
    return errno != 0 ? errno : EIO;
  }
  if (fseek(file, 0L, SEEK_END) != 0) {
    err = errno != 0 ? errno : EIO;
    (void)fclose(file);
    return err;
  }
  size = ftell(file);
  if (size <= 0L) {
    (void)fclose(file);
    return EIO;
  }
  if ((unsigned long)size > QR_MAX_KERNEL_IMAGE_BYTES) {
    (void)fclose(file);
    return EFBIG;
  }
  if (fseek(file, 0L, SEEK_SET) != 0) {
    err = errno != 0 ? errno : EIO;
    (void)fclose(file);
    return err;
  }
  data = (uint8_t *)qr_alloc_bytes((size_t)size);
  if (data == NULL) {
    (void)fclose(file);
    return ENOMEM;
  }
  if (fread(data, 1U, (size_t)size, file) != (size_t)size) {
    qr_free_bytes(data);
    (void)fclose(file);
    return EIO;
  }
  if (fclose(file) != 0) {
    err = errno != 0 ? errno : EIO;
    qr_free_bytes(data);
    return err;
  }

  *out_data = data;
  *out_size = (size_t)size;
  return 0;
}

static void qr_destroy_clear_kernel(qr_context *ctx)
{
  if (ctx == NULL) {
    return;
  }
  kfd_gpu_fence_destroy(ctx->clear_fence);
  kfd_gpu_buffer_destroy(ctx->clear_kernarg);
  kfd_gpu_buffer_destroy(ctx->clear_root);
  kfd_gpu_kernel_destroy(ctx->clear_kernel);
  kfd_gpu_module_destroy(ctx->clear_module);
  ctx->clear_fence = NULL;
  ctx->clear_kernarg = NULL;
  ctx->clear_root = NULL;
  ctx->clear_kernel = NULL;
  ctx->clear_module = NULL;
}

static void qr_destroy_resolve_kernel(qr_context *ctx)
{
  if (ctx == NULL) {
    return;
  }
  kfd_gpu_fence_destroy(ctx->resolve_fence);
  kfd_gpu_buffer_destroy(ctx->resolve_kernarg);
  kfd_gpu_buffer_destroy(ctx->resolve_root);
  kfd_gpu_kernel_destroy(ctx->resolve_kernel);
  kfd_gpu_module_destroy(ctx->resolve_module);
  kfd_gpu_buffer_destroy(ctx->resolve_palette);
  kfd_gpu_buffer_destroy(ctx->xrgb);
  ctx->resolve_fence = NULL;
  ctx->resolve_kernarg = NULL;
  ctx->resolve_root = NULL;
  ctx->resolve_kernel = NULL;
  ctx->resolve_module = NULL;
  ctx->resolve_palette = NULL;
  ctx->xrgb = NULL;
}

static void qr_destroy_raster_kernel(qr_context *ctx)
{
  if (ctx == NULL) {
    return;
  }
  kfd_gpu_fence_destroy(ctx->tile_bin_fence);
  kfd_gpu_buffer_destroy(ctx->tile_bin_kernarg);
  kfd_gpu_buffer_destroy(ctx->tile_bin_root);
  kfd_gpu_kernel_destroy(ctx->tile_bin_kernel);
  kfd_gpu_module_destroy(ctx->tile_bin_module);
  kfd_gpu_buffer_destroy(ctx->tile_overflows);
  kfd_gpu_buffer_destroy(ctx->tile_depth_max);
  kfd_gpu_buffer_destroy(ctx->tile_depth_min);
  kfd_gpu_buffer_destroy(ctx->tile_counts);
  kfd_gpu_buffer_destroy(ctx->tile_indices);
  kfd_gpu_buffer_destroy(ctx->triangle_buffer);
  kfd_gpu_fence_destroy(ctx->raster_fence);
  kfd_gpu_buffer_destroy(ctx->raster_kernarg);
  kfd_gpu_buffer_destroy(ctx->raster_root);
  kfd_gpu_kernel_destroy(ctx->raster_kernel);
  kfd_gpu_module_destroy(ctx->raster_module);
  kfd_gpu_buffer_destroy(ctx->raster_colormap);
  kfd_gpu_buffer_destroy(ctx->depth);
  ctx->raster_fence = NULL;
  ctx->raster_kernarg = NULL;
  ctx->raster_root = NULL;
  ctx->raster_kernel = NULL;
  ctx->raster_module = NULL;
  ctx->raster_colormap = NULL;
  ctx->depth = NULL;
  ctx->tile_bin_fence = NULL;
  ctx->tile_bin_kernarg = NULL;
  ctx->tile_bin_root = NULL;
  ctx->tile_bin_kernel = NULL;
  ctx->tile_bin_module = NULL;
  ctx->tile_overflows = NULL;
  ctx->tile_depth_max = NULL;
  ctx->tile_depth_min = NULL;
  ctx->tile_counts = NULL;
  ctx->tile_indices = NULL;
  ctx->triangle_buffer = NULL;
  ctx->triangles = NULL;
}

static int qr_load_clear_kernel(qr_context *ctx)
{
  const qr_kernel_candidate *candidate;
  qr_clear_indexed_args *args;
  int last_err;

  if (ctx == NULL) {
    return EINVAL;
  }
  last_err = ENOENT;
  for (candidate = qr_clear_indexed_kernels; candidate->path != NULL;
       ++candidate) {
    uint8_t *image;
    size_t image_size;
    int err;

    image = NULL;
    image_size = 0U;
    err = qr_read_file(candidate->path, &image, &image_size);
    if (err != 0) {
      last_err = err;
      continue;
    }
    err = kfd_gpu_module_load(ctx->gpu, image, image_size, &ctx->clear_module);
    qr_free_bytes(image);
    if (err != 0) {
      last_err = err;
      continue;
    }
    err = kfd_gpu_module_kernel(ctx->clear_module, "qr_clear_indexed.kd",
                                &ctx->clear_kernel);
    if (err != 0) {
      qr_destroy_clear_kernel(ctx);
      return err;
    }
    err = kfd_gpu_buffer_create(ctx->gpu, sizeof(qr_clear_indexed_args),
                                KFD_GPU_MEMORY_UPLOAD,
                                KFD_GPU_MEMORY_WRITABLE |
                                    KFD_GPU_MEMORY_COHERENT |
                                    KFD_GPU_MEMORY_UNCACHED,
                                &ctx->clear_root);
    if (err != 0) {
      qr_destroy_clear_kernel(ctx);
      return err;
    }
    err = kfd_gpu_kernel_alloc_args(ctx->clear_kernel, &ctx->clear_kernarg);
    if (err != 0) {
      qr_destroy_clear_kernel(ctx);
      return err;
    }
    err = kfd_gpu_fence_create(ctx->gpu, 1U, &ctx->clear_fence);
    if (err != 0) {
      qr_destroy_clear_kernel(ctx);
      return err;
    }

    ctx->clear_dispatch.grid.x =
        (ctx->width + QR_CLEAR_BLOCK_X - 1U) / QR_CLEAR_BLOCK_X;
    ctx->clear_dispatch.grid.y =
        (ctx->height + QR_CLEAR_BLOCK_Y - 1U) / QR_CLEAR_BLOCK_Y;
    ctx->clear_dispatch.grid.z = 1U;
    ctx->clear_dispatch.block.x = QR_CLEAR_BLOCK_X;
    ctx->clear_dispatch.block.y = QR_CLEAR_BLOCK_Y;
    ctx->clear_dispatch.block.z = 1U;
    ctx->clear_dispatch.dynamic_lds = 0U;
    ctx->clear_dispatch.private_segment_size = 0U;

    args = (qr_clear_indexed_args *)kfd_gpu_buffer_cpu(ctx->clear_root);
    if (args == NULL) {
      qr_destroy_clear_kernel(ctx);
      return EIO;
    }
    args->dst = (uint8_t *)kfd_gpu_buffer_gpu(ctx->indexed);
    args->width = ctx->width;
    args->height = ctx->height;
    args->color = 0U;
    kfd_gpu_kernel_set_root(ctx->clear_kernel, ctx->clear_kernarg,
                            kfd_gpu_buffer_gpu(ctx->clear_root),
                            &ctx->clear_dispatch);
    return 0;
  }
  return last_err;
}

static int qr_load_resolve_kernel(qr_context *ctx)
{
  const qr_kernel_candidate *candidate;
  qr_resolve_xrgb_args *args;
  int last_err;
  int err;

  if (ctx == NULL) {
    return EINVAL;
  }
  err = kfd_gpu_buffer_create(ctx->gpu, ctx->xrgb_bytes,
                              KFD_GPU_MEMORY_UPLOAD,
                              KFD_GPU_MEMORY_WRITABLE |
                                  KFD_GPU_MEMORY_COHERENT |
                                  KFD_GPU_MEMORY_UNCACHED,
                              &ctx->xrgb);
  if (err != 0) {
    return err;
  }
  err = kfd_gpu_buffer_create(ctx->gpu, 256U * sizeof(uint32_t),
                              KFD_GPU_MEMORY_UPLOAD,
                              KFD_GPU_MEMORY_WRITABLE |
                                  KFD_GPU_MEMORY_COHERENT |
                                  KFD_GPU_MEMORY_UNCACHED,
                              &ctx->resolve_palette);
  if (err != 0) {
    qr_destroy_resolve_kernel(ctx);
    return err;
  }

  last_err = ENOENT;
  for (candidate = qr_resolve_xrgb_kernels; candidate->path != NULL;
       ++candidate) {
    uint8_t *image;
    size_t image_size;

    image = NULL;
    image_size = 0U;
    err = qr_read_file(candidate->path, &image, &image_size);
    if (err != 0) {
      last_err = err;
      continue;
    }
    err = kfd_gpu_module_load(ctx->gpu, image, image_size,
                              &ctx->resolve_module);
    qr_free_bytes(image);
    if (err != 0) {
      last_err = err;
      continue;
    }
    err = kfd_gpu_module_kernel(ctx->resolve_module, "qr_resolve_xrgb.kd",
                                &ctx->resolve_kernel);
    if (err != 0) {
      qr_destroy_resolve_kernel(ctx);
      return err;
    }
    err = kfd_gpu_buffer_create(ctx->gpu, sizeof(qr_resolve_xrgb_args),
                                KFD_GPU_MEMORY_UPLOAD,
                                KFD_GPU_MEMORY_WRITABLE |
                                    KFD_GPU_MEMORY_COHERENT |
                                    KFD_GPU_MEMORY_UNCACHED,
                                &ctx->resolve_root);
    if (err != 0) {
      qr_destroy_resolve_kernel(ctx);
      return err;
    }
    err = kfd_gpu_kernel_alloc_args(ctx->resolve_kernel,
                                    &ctx->resolve_kernarg);
    if (err != 0) {
      qr_destroy_resolve_kernel(ctx);
      return err;
    }
    err = kfd_gpu_fence_create(ctx->gpu, 1U, &ctx->resolve_fence);
    if (err != 0) {
      qr_destroy_resolve_kernel(ctx);
      return err;
    }

    ctx->resolve_dispatch.grid.x =
        (ctx->width + QR_CLEAR_BLOCK_X - 1U) / QR_CLEAR_BLOCK_X;
    ctx->resolve_dispatch.grid.y =
        (ctx->height + QR_CLEAR_BLOCK_Y - 1U) / QR_CLEAR_BLOCK_Y;
    ctx->resolve_dispatch.grid.z = 1U;
    ctx->resolve_dispatch.block.x = QR_CLEAR_BLOCK_X;
    ctx->resolve_dispatch.block.y = QR_CLEAR_BLOCK_Y;
    ctx->resolve_dispatch.block.z = 1U;
    ctx->resolve_dispatch.dynamic_lds = 0U;
    ctx->resolve_dispatch.private_segment_size = 0U;

    args = (qr_resolve_xrgb_args *)kfd_gpu_buffer_cpu(ctx->resolve_root);
    if (args == NULL) {
      qr_destroy_resolve_kernel(ctx);
      return EIO;
    }
    args->src = (uint8_t *)kfd_gpu_buffer_gpu(ctx->indexed);
    args->dst = (uint32_t *)kfd_gpu_buffer_gpu(ctx->xrgb);
    args->palette = (uint32_t *)kfd_gpu_buffer_gpu(ctx->resolve_palette);
    args->width = ctx->width;
    args->height = ctx->height;
    kfd_gpu_kernel_set_root(ctx->resolve_kernel, ctx->resolve_kernarg,
                            kfd_gpu_buffer_gpu(ctx->resolve_root),
                            &ctx->resolve_dispatch);
    return 0;
  }

  qr_destroy_resolve_kernel(ctx);
  return last_err;
}

static int qr_load_tile_bin_kernel(qr_context *ctx)
{
  const qr_kernel_candidate *candidate;
  qr_tile_bin_args *args;
  int last_err;
  int err;

  if (ctx == NULL) {
    return EINVAL;
  }

  last_err = ENOENT;
  for (candidate = qr_tile_bin_kernels; candidate->path != NULL; ++candidate) {
    uint8_t *image;
    size_t image_size;

    image = NULL;
    image_size = 0U;
    err = qr_read_file(candidate->path, &image, &image_size);
    if (err != 0) {
      last_err = err;
      continue;
    }
    err = kfd_gpu_module_load(ctx->gpu, image, image_size,
                              &ctx->tile_bin_module);
    qr_free_bytes(image);
    if (err != 0) {
      last_err = err;
      continue;
    }
    err = kfd_gpu_module_kernel(ctx->tile_bin_module, "qr_bin_tiles.kd",
                                &ctx->tile_bin_kernel);
    if (err != 0) {
      qr_destroy_raster_kernel(ctx);
      return err;
    }
    err = kfd_gpu_buffer_create(ctx->gpu, sizeof(qr_tile_bin_args),
                                KFD_GPU_MEMORY_UPLOAD,
                                KFD_GPU_MEMORY_WRITABLE |
                                    KFD_GPU_MEMORY_COHERENT |
                                    KFD_GPU_MEMORY_UNCACHED,
                                &ctx->tile_bin_root);
    if (err != 0) {
      qr_destroy_raster_kernel(ctx);
      return err;
    }
    err = kfd_gpu_kernel_alloc_args(ctx->tile_bin_kernel,
                                    &ctx->tile_bin_kernarg);
    if (err != 0) {
      qr_destroy_raster_kernel(ctx);
      return err;
    }
    err = kfd_gpu_fence_create(ctx->gpu, 1U, &ctx->tile_bin_fence);
    if (err != 0) {
      qr_destroy_raster_kernel(ctx);
      return err;
    }

    ctx->tile_bin_dispatch.grid.x =
        (ctx->tile_count + QR_TILE_BIN_BLOCK_X - 1U) / QR_TILE_BIN_BLOCK_X;
    ctx->tile_bin_dispatch.grid.y = 1U;
    ctx->tile_bin_dispatch.grid.z = 1U;
    ctx->tile_bin_dispatch.block.x = QR_TILE_BIN_BLOCK_X;
    ctx->tile_bin_dispatch.block.y = 1U;
    ctx->tile_bin_dispatch.block.z = 1U;
    ctx->tile_bin_dispatch.dynamic_lds = 0U;
    ctx->tile_bin_dispatch.private_segment_size = 0U;

    args = (qr_tile_bin_args *)kfd_gpu_buffer_cpu(ctx->tile_bin_root);
    if (args == NULL) {
      qr_destroy_raster_kernel(ctx);
      return EIO;
    }
    memset(args, 0, sizeof(*args));
    args->tile_indices = (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_indices);
    args->tile_counts = (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_counts);
    args->tile_overflows = (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_overflows);
    args->tile_depth_min = (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_depth_min);
    args->tile_depth_max = (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_depth_max);
    args->width = ctx->width;
    args->height = ctx->height;
    args->tile_cols = ctx->tile_cols;
    args->tile_rows = ctx->tile_rows;
    args->tile_triangle_capacity = QR_TILE_TRIANGLE_CAPACITY;
    kfd_gpu_kernel_set_root(ctx->tile_bin_kernel, ctx->tile_bin_kernarg,
                            kfd_gpu_buffer_gpu(ctx->tile_bin_root),
                            &ctx->tile_bin_dispatch);
    return 0;
  }

  return last_err;
}

static int qr_load_raster_kernel(qr_context *ctx)
{
  const qr_kernel_candidate *candidate;
  qr_world_raster_args *args;
  size_t tile_refs;
  size_t tile_index_bytes;
  size_t tile_counter_bytes;
  size_t triangle_bytes;
  int last_err;
  int err;

  if (ctx == NULL) {
    return EINVAL;
  }
  ctx->tile_cols = (ctx->width + QR_TILE_SIZE - 1U) / QR_TILE_SIZE;
  ctx->tile_rows = (ctx->height + QR_TILE_SIZE - 1U) / QR_TILE_SIZE;
  if (ctx->tile_cols == 0U || ctx->tile_rows == 0U ||
      ctx->tile_cols > UINT32_MAX / ctx->tile_rows) {
    return EOVERFLOW;
  }
  ctx->tile_count = ctx->tile_cols * ctx->tile_rows;
  if (qr_mul_size((size_t)ctx->tile_count, QR_TILE_TRIANGLE_CAPACITY,
                  &tile_refs) != 0 ||
      qr_mul_size(tile_refs, sizeof(uint32_t), &tile_index_bytes) != 0 ||
      qr_mul_size((size_t)ctx->tile_count, sizeof(uint32_t),
                  &tile_counter_bytes) != 0 ||
      qr_mul_size((size_t)ctx->triangle_capacity, sizeof(qr_raster_triangle),
                  &triangle_bytes) != 0) {
    return EOVERFLOW;
  }
  err = kfd_gpu_buffer_create(ctx->gpu, ctx->indexed_bytes * sizeof(float),
                              KFD_GPU_MEMORY_UPLOAD,
                              KFD_GPU_MEMORY_WRITABLE |
                                  KFD_GPU_MEMORY_COHERENT |
                                  KFD_GPU_MEMORY_UNCACHED,
                              &ctx->depth);
  if (err != 0) {
    return err;
  }
  err = kfd_gpu_buffer_create(ctx->gpu, QR_COLORMAP_SIZE,
                              KFD_GPU_MEMORY_UPLOAD,
                              KFD_GPU_MEMORY_WRITABLE |
                                  KFD_GPU_MEMORY_COHERENT |
                                  KFD_GPU_MEMORY_UNCACHED,
                              &ctx->raster_colormap);
  if (err != 0) {
    qr_destroy_raster_kernel(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, tile_index_bytes, &ctx->tile_indices);
  if (err != 0) {
    qr_destroy_raster_kernel(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, tile_counter_bytes, &ctx->tile_counts);
  if (err != 0) {
    qr_destroy_raster_kernel(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, tile_counter_bytes, &ctx->tile_overflows);
  if (err != 0) {
    qr_destroy_raster_kernel(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, tile_counter_bytes, &ctx->tile_depth_min);
  if (err != 0) {
    qr_destroy_raster_kernel(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, tile_counter_bytes, &ctx->tile_depth_max);
  if (err != 0) {
    qr_destroy_raster_kernel(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, triangle_bytes, &ctx->triangle_buffer);
  if (err != 0) {
    qr_destroy_raster_kernel(ctx);
    return err;
  }
  ctx->triangles = (qr_raster_triangle *)kfd_gpu_buffer_cpu(ctx->triangle_buffer);
  if (ctx->triangles == NULL) {
    qr_destroy_raster_kernel(ctx);
    return EIO;
  }
  err = qr_load_tile_bin_kernel(ctx);
  if (err != 0) {
    qr_destroy_raster_kernel(ctx);
    return err;
  }

  last_err = ENOENT;
  for (candidate = qr_world_raster_kernels; candidate->path != NULL;
       ++candidate) {
    uint8_t *image;
    size_t image_size;

    image = NULL;
    image_size = 0U;
    err = qr_read_file(candidate->path, &image, &image_size);
    if (err != 0) {
      last_err = err;
      continue;
    }
    err = kfd_gpu_module_load(ctx->gpu, image, image_size, &ctx->raster_module);
    qr_free_bytes(image);
    if (err != 0) {
      last_err = err;
      continue;
    }
    err = kfd_gpu_module_kernel(ctx->raster_module, "qr_world_raster.kd",
                                &ctx->raster_kernel);
    if (err != 0) {
      qr_destroy_raster_kernel(ctx);
      return err;
    }
    err = kfd_gpu_buffer_create(ctx->gpu, sizeof(qr_world_raster_args),
                                KFD_GPU_MEMORY_UPLOAD,
                                KFD_GPU_MEMORY_WRITABLE |
                                    KFD_GPU_MEMORY_COHERENT |
                                    KFD_GPU_MEMORY_UNCACHED,
                                &ctx->raster_root);
    if (err != 0) {
      qr_destroy_raster_kernel(ctx);
      return err;
    }
    err = kfd_gpu_kernel_alloc_args(ctx->raster_kernel, &ctx->raster_kernarg);
    if (err != 0) {
      qr_destroy_raster_kernel(ctx);
      return err;
    }
    err = kfd_gpu_fence_create(ctx->gpu, 1U, &ctx->raster_fence);
    if (err != 0) {
      qr_destroy_raster_kernel(ctx);
      return err;
    }

    ctx->raster_dispatch.grid.x = ctx->tile_cols;
    ctx->raster_dispatch.grid.y = ctx->tile_rows;
    ctx->raster_dispatch.grid.z = 1U;
    ctx->raster_dispatch.block.x = QR_TILE_SIZE;
    ctx->raster_dispatch.block.y = QR_TILE_SIZE;
    ctx->raster_dispatch.block.z = 1U;
    ctx->raster_dispatch.dynamic_lds = 0U;
    ctx->raster_dispatch.private_segment_size = 0U;

    args = (qr_world_raster_args *)kfd_gpu_buffer_cpu(ctx->raster_root);
    if (args == NULL) {
      qr_destroy_raster_kernel(ctx);
      return EIO;
    }
    memset(args, 0, sizeof(*args));
    args->dst = (uint8_t *)kfd_gpu_buffer_gpu(ctx->indexed);
    args->depth = (float *)kfd_gpu_buffer_gpu(ctx->depth);
    args->surfaces = (qr_surface_record *)kfd_gpu_buffer_gpu(ctx->surface_metadata);
    args->textures = (qr_texture_record *)kfd_gpu_buffer_gpu(ctx->texture_metadata);
    args->lightmaps =
        (qr_lightmap_record *)kfd_gpu_buffer_gpu(ctx->lightmap_metadata);
    args->texture_atlas = (uint8_t *)kfd_gpu_buffer_gpu(ctx->texture_atlas);
    args->lightmap_atlas = (uint8_t *)kfd_gpu_buffer_gpu(ctx->lightmap_atlas);
    args->colormap = (uint8_t *)kfd_gpu_buffer_gpu(ctx->raster_colormap);
    args->width = ctx->width;
    args->height = ctx->height;
    args->tile_indices = (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_indices);
    args->tile_counts = (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_counts);
    args->tile_overflows =
        (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_overflows);
    args->tile_depth_min = (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_depth_min);
    args->tile_depth_max = (uint32_t *)kfd_gpu_buffer_gpu(ctx->tile_depth_max);
    args->tile_cols = ctx->tile_cols;
    args->tile_rows = ctx->tile_rows;
    args->tile_triangle_capacity = QR_TILE_TRIANGLE_CAPACITY;
    kfd_gpu_kernel_set_root(ctx->raster_kernel, ctx->raster_kernarg,
                            kfd_gpu_buffer_gpu(ctx->raster_root),
                            &ctx->raster_dispatch);
    return 0;
  }

  qr_destroy_raster_kernel(ctx);
  return last_err;
}

static int qr_dispatch_resolve_xrgb(qr_context *ctx,
                                    const uint32_t *palette_xrgb)
{
  void *palette_dst;
  uint64_t start_ns;
  int err;

  if (ctx == NULL || palette_xrgb == NULL || ctx->resolve_palette == NULL) {
    return EINVAL;
  }
  palette_dst = kfd_gpu_buffer_cpu(ctx->resolve_palette);
  if (palette_dst == NULL) {
    return EIO;
  }
  memcpy(palette_dst, palette_xrgb, 256U * sizeof(uint32_t));
  start_ns = qr_now_ns();
  err = kfd_gpu_dispatch(ctx->gpu, ctx->resolve_kernel,
                         &ctx->resolve_dispatch, ctx->resolve_kernarg,
                         ctx->resolve_fence);
  if (err != 0) {
    return err;
  }
  err = kfd_gpu_fence_wait(ctx->resolve_fence, 0U, UINT64_MAX);
  if (err == 0) {
    uint64_t end_ns = qr_now_ns();

    ++ctx->perf.resolve_count;
    if (end_ns >= start_ns) {
      ctx->perf.resolve_time_ns += end_ns - start_ns;
    }
  }
  return err;
}

static void qr_destroy_resources(qr_context *ctx)
{
  if (ctx == NULL) {
    return;
  }
  qr_free_bytes(ctx->alias_triangles);
  qr_free_bytes(ctx->alias_models);
  qr_free_bytes(ctx->triangulation_indices);
  qr_free_bytes(ctx->worlds);
  kfd_gpu_buffer_destroy(ctx->surface_metadata);
  kfd_gpu_buffer_destroy(ctx->lightmap_metadata);
  kfd_gpu_buffer_destroy(ctx->texture_metadata);
  kfd_gpu_buffer_destroy(ctx->lightmap_atlas);
  kfd_gpu_buffer_destroy(ctx->texture_atlas);
  ctx->worlds = NULL;
  ctx->alias_models = NULL;
  ctx->alias_triangles = NULL;
  ctx->triangulation_indices = NULL;
  ctx->lightmaps = NULL;
  ctx->textures = NULL;
  ctx->surface_metadata = NULL;
  ctx->lightmap_metadata = NULL;
  ctx->texture_metadata = NULL;
  ctx->lightmap_atlas = NULL;
  ctx->texture_atlas = NULL;
}

static int qr_create_upload_buffer(qr_context *ctx, size_t size,
                                   kfd_gpu_buffer **out)
{
  return kfd_gpu_buffer_create(ctx->gpu, size, KFD_GPU_MEMORY_UPLOAD,
                               KFD_GPU_MEMORY_WRITABLE |
                                   KFD_GPU_MEMORY_COHERENT |
                                   KFD_GPU_MEMORY_UNCACHED,
                               out);
}

static int qr_init_resources(qr_context *ctx, const qr_desc *desc)
{
  size_t texture_slots;
  size_t lightmap_slots;
  size_t world_slots;
  size_t alias_model_slots;
  size_t texture_metadata_bytes;
  size_t lightmap_metadata_bytes;
  size_t surface_bytes;
  int err;

  if (ctx == NULL || desc == NULL) {
    return EINVAL;
  }
  ctx->texture_capacity =
      qr_default_u32(desc->max_textures, QR_DEFAULT_MAX_TEXTURES);
  ctx->lightmap_capacity =
      qr_default_u32(desc->max_lightmaps, QR_DEFAULT_MAX_LIGHTMAPS);
  ctx->surface_capacity =
      qr_default_u32(desc->max_surfaces, QR_DEFAULT_MAX_SURFACES);
  ctx->world_capacity = qr_default_u32(desc->max_worlds, QR_DEFAULT_MAX_WORLDS);
  ctx->triangle_capacity =
      qr_default_u32(desc->max_frame_triangles, QR_DEFAULT_MAX_FRAME_TRIANGLES);
  ctx->alias_model_capacity =
      qr_default_u32(desc->max_alias_models, QR_DEFAULT_MAX_ALIAS_MODELS);
  ctx->alias_triangle_capacity =
      qr_default_u32(desc->max_alias_triangles, QR_DEFAULT_MAX_ALIAS_TRIANGLES);
  ctx->texture_atlas_bytes =
      qr_default_size(desc->texture_atlas_bytes, QR_DEFAULT_TEXTURE_ATLAS_BYTES);
  ctx->lightmap_atlas_bytes =
      qr_default_size(desc->lightmap_atlas_bytes, QR_DEFAULT_LIGHTMAP_ATLAS_BYTES);

  if (ctx->texture_capacity == UINT32_MAX ||
      ctx->lightmap_capacity == UINT32_MAX ||
      ctx->world_capacity == UINT32_MAX ||
      ctx->triangle_capacity == UINT32_MAX ||
      ctx->alias_model_capacity == UINT32_MAX ||
      ctx->alias_triangle_capacity == UINT32_MAX) {
    return EOVERFLOW;
  }
  texture_slots = (size_t)ctx->texture_capacity + 1U;
  lightmap_slots = (size_t)ctx->lightmap_capacity + 1U;
  world_slots = (size_t)ctx->world_capacity + 1U;
  alias_model_slots = (size_t)ctx->alias_model_capacity + 1U;
  if (qr_mul_size(texture_slots, sizeof(*ctx->textures),
                  &texture_metadata_bytes) != 0 ||
      qr_mul_size(lightmap_slots, sizeof(*ctx->lightmaps),
                  &lightmap_metadata_bytes) != 0 ||
      world_slots > SIZE_MAX / sizeof(*ctx->worlds) ||
      alias_model_slots > SIZE_MAX / sizeof(*ctx->alias_models) ||
      (size_t)ctx->alias_triangle_capacity >
          SIZE_MAX / sizeof(*ctx->alias_triangles) ||
      (size_t)ctx->surface_capacity > SIZE_MAX / sizeof(qr_surface_record)) {
    return EOVERFLOW;
  }
  surface_bytes = (size_t)ctx->surface_capacity * sizeof(qr_surface_record);

  ctx->worlds =
      (qr_world_record *)qr_calloc_bytes(world_slots, sizeof(*ctx->worlds));
  if (ctx->worlds == NULL) {
    qr_destroy_resources(ctx);
    return ENOMEM;
  }
  ctx->alias_models = (qr_alias_model_record *)qr_calloc_bytes(
      alias_model_slots, sizeof(*ctx->alias_models));
  ctx->alias_triangles = (qr_alias_triangle_record *)qr_calloc_bytes(
      (size_t)ctx->alias_triangle_capacity, sizeof(*ctx->alias_triangles));
  ctx->triangulation_indices = (uint32_t *)qr_calloc_bytes(
      (size_t)ctx->triangle_capacity + 2U, sizeof(*ctx->triangulation_indices));
  if (ctx->alias_models == NULL || ctx->alias_triangles == NULL ||
      ctx->triangulation_indices == NULL) {
    qr_destroy_resources(ctx);
    return ENOMEM;
  }

  err = qr_create_upload_buffer(ctx, ctx->texture_atlas_bytes,
                                &ctx->texture_atlas);
  if (err != 0) {
    qr_destroy_resources(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, ctx->lightmap_atlas_bytes,
                                &ctx->lightmap_atlas);
  if (err != 0) {
    qr_destroy_resources(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, texture_metadata_bytes,
                                &ctx->texture_metadata);
  if (err != 0) {
    qr_destroy_resources(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, lightmap_metadata_bytes,
                                &ctx->lightmap_metadata);
  if (err != 0) {
    qr_destroy_resources(ctx);
    return err;
  }
  err = qr_create_upload_buffer(ctx, surface_bytes, &ctx->surface_metadata);
  if (err != 0) {
    qr_destroy_resources(ctx);
    return err;
  }
  ctx->textures = (qr_texture_record *)kfd_gpu_buffer_cpu(ctx->texture_metadata);
  ctx->lightmaps =
      (qr_lightmap_record *)kfd_gpu_buffer_cpu(ctx->lightmap_metadata);
  if (ctx->textures == NULL || ctx->lightmaps == NULL) {
    qr_destroy_resources(ctx);
    return EIO;
  }
  memset(ctx->textures, 0, texture_metadata_bytes);
  memset(ctx->lightmaps, 0, lightmap_metadata_bytes);
  return 0;
}

uint32_t qr_api_version(void)
{
  return (QR_API_VERSION_MAJOR << 16U) | (QR_API_VERSION_MINOR << 8U) |
         QR_API_VERSION_PATCH;
}

const char *qr_strerror(qr_result code)
{
  switch (code) {
  case QR_SUCCESS:
    return "success";
  case QR_ERROR_INVALID_ARGUMENT:
    return "invalid argument";
  case QR_ERROR_UNSUPPORTED:
    return "unsupported operation";
  case QR_ERROR_OVERFLOW:
    return "numeric overflow";
  case QR_ERROR_OUT_OF_MEMORY:
    return "out of memory";
  case QR_ERROR_BUFFER_TOO_SMALL:
    return "destination buffer too small";
  case QR_ERROR_NO_SPACE:
    return "no space available";
  case QR_ERROR_BUSY:
    return "resource busy";
  case QR_ERROR_IO:
    return "I/O error";
  case QR_ERROR_NOT_FOUND:
    return "not found";
  case QR_ERROR_SYSTEM:
    return "system error";
  case QR_ERROR_GPU:
    return "GPU operation failed";
  }
  return "unknown error";
}

uint64_t qr_pack_depth_payload(uint32_t depth_key, uint32_t payload)
{
  return ((uint64_t)depth_key << 32U) | (uint64_t)payload;
}

qr_result qr_create(const qr_desc *desc, qr_context **out)
{
  qr_context *ctx;
  kfd_gpu_context *gpu;
  kfd_gpu_buffer *indexed;
  size_t indexed_bytes;
  size_t xrgb_bytes;
  int err;

  if (out == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  *out = NULL;
  err = qr_validate_desc(desc);
  if (err != 0) {
    return qr_result_from_errno(err);
  }

  indexed_bytes = (size_t)desc->width * (size_t)desc->height;
  xrgb_bytes = indexed_bytes * sizeof(uint32_t);
  err = kfd_gpu_context_create(desc->device_index, &gpu);
  if (err != 0) {
    return qr_result_from_gpu_error(err);
  }
  err = kfd_gpu_buffer_create(gpu, indexed_bytes, KFD_GPU_MEMORY_UPLOAD,
                              KFD_GPU_MEMORY_WRITABLE |
                                  KFD_GPU_MEMORY_COHERENT |
                                  KFD_GPU_MEMORY_UNCACHED,
                              &indexed);
  if (err != 0) {
    kfd_gpu_context_destroy(gpu);
    return qr_result_from_gpu_error(err);
  }

  ctx = (qr_context *)qr_calloc_bytes(1U, sizeof(*ctx));
  if (ctx == NULL) {
    kfd_gpu_buffer_destroy(indexed);
    kfd_gpu_context_destroy(gpu);
    return QR_ERROR_OUT_OF_MEMORY;
  }

  ctx->gpu = gpu;
  ctx->indexed = indexed;
  ctx->width = desc->width;
  ctx->height = desc->height;
  ctx->indexed_bytes = indexed_bytes;
  ctx->xrgb_bytes = xrgb_bytes;
  ctx->output_mode = desc->output_mode;
  ctx->present = desc->present;
  ctx->present_userdata = desc->present_userdata;
  if (desc->present_palette_xrgb != NULL) {
    memcpy(ctx->present_palette_xrgb, desc->present_palette_xrgb,
           sizeof(ctx->present_palette_xrgb));
  }
  ctx->frame.ctx = ctx;
  err = qr_init_resources(ctx, desc);
  if (err != 0) {
    qr_destroy_resources(ctx);
    kfd_gpu_buffer_destroy(indexed);
    kfd_gpu_context_destroy(gpu);
    qr_free_bytes(ctx);
    return qr_result_from_errno(err);
  }
  err = qr_load_clear_kernel(ctx);
  if (err != 0) {
    qr_destroy_clear_kernel(ctx);
    qr_destroy_resources(ctx);
    kfd_gpu_buffer_destroy(indexed);
    kfd_gpu_context_destroy(gpu);
    qr_free_bytes(ctx);
    return qr_result_from_errno(err);
  }
  err = qr_load_resolve_kernel(ctx);
  if (err != 0) {
    qr_destroy_resolve_kernel(ctx);
    qr_destroy_clear_kernel(ctx);
    qr_destroy_resources(ctx);
    kfd_gpu_buffer_destroy(indexed);
    kfd_gpu_context_destroy(gpu);
    qr_free_bytes(ctx);
    return qr_result_from_errno(err);
  }
  err = qr_load_raster_kernel(ctx);
  if (err != 0) {
    qr_destroy_raster_kernel(ctx);
    qr_destroy_resolve_kernel(ctx);
    qr_destroy_clear_kernel(ctx);
    qr_destroy_resources(ctx);
    kfd_gpu_buffer_destroy(indexed);
    kfd_gpu_context_destroy(gpu);
    qr_free_bytes(ctx);
    return qr_result_from_errno(err);
  }
  *out = ctx;
  return QR_SUCCESS;
}

void qr_destroy(qr_context *ctx)
{
  if (ctx == NULL) {
    return;
  }
  qr_destroy_raster_kernel(ctx);
  qr_destroy_resolve_kernel(ctx);
  qr_destroy_clear_kernel(ctx);
  qr_destroy_resources(ctx);
  kfd_gpu_buffer_destroy(ctx->indexed);
  kfd_gpu_context_destroy(ctx->gpu);
  qr_free_bytes(ctx);
}

uint32_t qr_width(const qr_context *ctx)
{
  return ctx != NULL ? ctx->width : 0U;
}

uint32_t qr_height(const qr_context *ctx)
{
  return ctx != NULL ? ctx->height : 0U;
}

qr_output_mode qr_output(const qr_context *ctx)
{
  return ctx != NULL ? ctx->output_mode : QR_OUTPUT_NOOUTPUT;
}

qr_result qr_begin_frame(qr_context *ctx, const qr_frame_desc *desc,
                         qr_frame **out)
{
  (void)desc;
  if (ctx == NULL || out == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (ctx->frame_active != 0) {
    return QR_ERROR_BUSY;
  }
  ctx->frame_active = 1;
  *out = &ctx->frame;
  return QR_SUCCESS;
}

qr_result qr_frame_clear_indexed(qr_frame *frame, uint8_t color)
{
  qr_clear_indexed_args *args;
  uint64_t start_ns;
  int err;

  if (frame == NULL || frame->ctx == NULL || frame->ctx->clear_root == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (frame->ctx->frame_active == 0) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  args = (qr_clear_indexed_args *)kfd_gpu_buffer_cpu(frame->ctx->clear_root);
  if (args == NULL) {
    return QR_ERROR_IO;
  }
  args->color = (uint32_t)color;
  start_ns = qr_now_ns();
  err = kfd_gpu_dispatch(frame->ctx->gpu, frame->ctx->clear_kernel,
                         &frame->ctx->clear_dispatch,
                         frame->ctx->clear_kernarg, frame->ctx->clear_fence);
  if (err != 0) {
    return qr_result_from_gpu_error(err);
  }
  err = kfd_gpu_fence_wait(frame->ctx->clear_fence, 0U, UINT64_MAX);
  if (err == 0) {
    uint64_t end_ns = qr_now_ns();

    ++frame->ctx->perf.clear_count;
    if (end_ns >= start_ns) {
      frame->ctx->perf.clear_time_ns += end_ns - start_ns;
    }
  }
  return qr_result_from_gpu_error(err);
}

static qr_raster_vertex qr_make_raster_vertex(const qr_world_vertex *vertex)
{
  qr_raster_vertex out;

  out.x = vertex->x;
  out.y = vertex->y;
  out.z = vertex->z;
  out.u = vertex->u;
  out.v = vertex->v;
  out.light_u = vertex->light_u;
  out.light_v = vertex->light_v;
  return out;
}

static int qr_valid_world_vertex(const qr_world_vertex *vertex)
{
  return vertex != NULL && isfinite(vertex->x) && isfinite(vertex->y) &&
         isfinite(vertex->z) && isfinite(vertex->u) && isfinite(vertex->v) &&
         isfinite(vertex->light_u) && isfinite(vertex->light_v);
}

static int qr_valid_alias_triangle_desc(const qr_alias_triangle_desc *triangle)
{
  return triangle != NULL && qr_valid_world_vertex(&triangle->v0) &&
         qr_valid_world_vertex(&triangle->v1) &&
         qr_valid_world_vertex(&triangle->v2);
}

static int qr_valid_sprite_desc(const qr_sprite_draw_desc *desc)
{
  return desc != NULL && isfinite(desc->x0) && isfinite(desc->y0) &&
         isfinite(desc->x1) && isfinite(desc->y1) && isfinite(desc->z) &&
         isfinite(desc->u0) && isfinite(desc->v0) && isfinite(desc->u1) &&
         isfinite(desc->v1) && isfinite(desc->time_seconds);
}

static int qr_valid_particle_desc(const qr_particle_desc *particle)
{
  return particle != NULL && isfinite(particle->x) && isfinite(particle->y) &&
         isfinite(particle->z) && isfinite(particle->size) &&
         isfinite(particle->u) && isfinite(particle->v) &&
         particle->size > 0.0f;
}

static int qr_valid_float4(const float values[4])
{
  return values != NULL && isfinite(values[0]) && isfinite(values[1]) &&
         isfinite(values[2]) && isfinite(values[3]);
}

static int qr_valid_surface_desc(const qr_world_surface_desc *surface)
{
  return surface != NULL && qr_valid_float4(surface->plane) &&
         qr_valid_float4(surface->tex_s) && qr_valid_float4(surface->tex_t) &&
         qr_valid_float4(surface->light_s) && qr_valid_float4(surface->light_t);
}

static int qr_valid_debug_mode(qr_debug_mode mode)
{
  switch (mode) {
  case QR_DEBUG_SHADED:
  case QR_DEBUG_FLAT_SURFACE_ID:
  case QR_DEBUG_DEPTH:
  case QR_DEBUG_TEXTURE_ONLY:
  case QR_DEBUG_LIGHT_ONLY:
  case QR_DEBUG_DEPTH_ORDER:
    return 1;
  }
  return 0;
}

static float qr_triangle_area2(const qr_world_vertex *a, const qr_world_vertex *b,
                               const qr_world_vertex *c)
{
  return (b->x - a->x) * (c->y - a->y) -
         (b->y - a->y) * (c->x - a->x);
}

static float qr_polygon_area2(const qr_world_vertex *vertices, uint32_t count)
{
  float area = 0.0f;
  uint32_t i;

  for (i = 0U; i < count; ++i) {
    const qr_world_vertex *a = &vertices[i];
    const qr_world_vertex *b = &vertices[(i + 1U) % count];

    area += a->x * b->y - b->x * a->y;
  }
  return area;
}

static int qr_point_in_triangle_2d(const qr_world_vertex *p,
                                   const qr_world_vertex *a,
                                   const qr_world_vertex *b,
                                   const qr_world_vertex *c, float winding)
{
  const float ab = qr_triangle_area2(a, b, p);
  const float bc = qr_triangle_area2(b, c, p);
  const float ca = qr_triangle_area2(c, a, p);

  if (winding >= 0.0f) {
    return ab >= -0.00001f && bc >= -0.00001f && ca >= -0.00001f;
  }
  return ab <= 0.00001f && bc <= 0.00001f && ca <= 0.00001f;
}

static void qr_emit_world_triangle(qr_raster_triangle *dst,
                                   const qr_world_polygon_desc *polygon,
                                   uint32_t first_surface, uint32_t a,
                                   uint32_t b, uint32_t c)
{
  dst->v0 = qr_make_raster_vertex(&polygon->vertices[a]);
  dst->v1 = qr_make_raster_vertex(&polygon->vertices[b]);
  dst->v2 = qr_make_raster_vertex(&polygon->vertices[c]);
  dst->surface = first_surface + polygon->surface;
}

static qr_result qr_triangulate_world_polygon(
    const qr_world_polygon_desc *polygon, uint32_t first_surface,
    qr_raster_triangle *triangles, size_t triangle_capacity, size_t *out_index,
    uint32_t *indices)
{
  uint32_t remaining;
  float winding;

  if (polygon == NULL || triangles == NULL || out_index == NULL ||
      indices == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (polygon->vertex_count < 3U) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (polygon->vertex_count == 3U) {
    if (*out_index >= triangle_capacity) {
      return QR_ERROR_NO_SPACE;
    }
    qr_emit_world_triangle(&triangles[(*out_index)++], polygon, first_surface,
                           0U, 1U, 2U);
    return QR_SUCCESS;
  }

  winding = qr_polygon_area2(polygon->vertices, polygon->vertex_count);
  if (winding > -0.00001f && winding < 0.00001f) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  for (uint32_t i = 0U; i < polygon->vertex_count; ++i) {
    indices[i] = i;
  }
  remaining = polygon->vertex_count;
  while (remaining > 3U) {
    int found_ear = 0;
    int removed_degenerate = 0;

    for (uint32_t i = 0U; i < remaining; ++i) {
      const uint32_t prev = indices[(i + remaining - 1U) % remaining];
      const uint32_t curr = indices[i];
      const uint32_t next = indices[(i + 1U) % remaining];
      const float area =
          qr_triangle_area2(&polygon->vertices[prev], &polygon->vertices[curr],
                            &polygon->vertices[next]);
      int contains_vertex = 0;

      if ((winding > 0.0f && area <= 0.00001f) ||
          (winding < 0.0f && area >= -0.00001f)) {
        continue;
      }
      for (uint32_t j = 0U; j < remaining; ++j) {
        const uint32_t candidate = indices[j];

        if (candidate == prev || candidate == curr || candidate == next) {
          continue;
        }
        if (qr_point_in_triangle_2d(&polygon->vertices[candidate],
                                    &polygon->vertices[prev],
                                    &polygon->vertices[curr],
                                    &polygon->vertices[next], winding) != 0) {
          contains_vertex = 1;
          break;
        }
      }
      if (contains_vertex != 0) {
        continue;
      }
      if (*out_index >= triangle_capacity) {
        return QR_ERROR_NO_SPACE;
      }
      qr_emit_world_triangle(&triangles[(*out_index)++], polygon, first_surface,
                             prev, curr, next);
      memmove(&indices[i], &indices[i + 1U],
              (size_t)(remaining - i - 1U) * sizeof(*indices));
      --remaining;
      found_ear = 1;
      break;
    }
    if (found_ear != 0) {
      continue;
    }

    for (uint32_t i = 0U; i < remaining; ++i) {
      const uint32_t prev = indices[(i + remaining - 1U) % remaining];
      const uint32_t curr = indices[i];
      const uint32_t next = indices[(i + 1U) % remaining];
      const float area =
          qr_triangle_area2(&polygon->vertices[prev], &polygon->vertices[curr],
                            &polygon->vertices[next]);

      if (area > -0.00001f && area < 0.00001f) {
        memmove(&indices[i], &indices[i + 1U],
                (size_t)(remaining - i - 1U) * sizeof(*indices));
        --remaining;
        removed_degenerate = 1;
        break;
      }
    }
    if (removed_degenerate == 0) {
      return QR_ERROR_INVALID_ARGUMENT;
    }
  }

  if (*out_index >= triangle_capacity) {
    return QR_ERROR_NO_SPACE;
  }
  qr_emit_world_triangle(&triangles[(*out_index)++], polygon, first_surface,
                         indices[0], indices[1], indices[2]);
  return QR_SUCCESS;
}

static void qr_reset_raster_stats(qr_context *ctx)
{
  if (ctx == NULL) {
    return;
  }
  memset(&ctx->last_stats, 0, sizeof(ctx->last_stats));
  ctx->last_stats.tile_size = QR_TILE_SIZE;
  ctx->last_stats.tile_cols = ctx->tile_cols;
  ctx->last_stats.tile_rows = ctx->tile_rows;
  ctx->last_stats.tile_count = ctx->tile_count;
  ctx->last_stats.tile_triangle_capacity = QR_TILE_TRIANGLE_CAPACITY;
  ctx->last_stats.depth_key_scale = QR_DEPTH_KEY_SCALE;
}

static qr_result qr_collect_tile_stats(qr_context *ctx, uint32_t triangle_count)
{
  const uint32_t *counts;
  const uint32_t *overflows;
  uint32_t i;

  if (ctx == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  counts = (const uint32_t *)kfd_gpu_buffer_cpu(ctx->tile_counts);
  overflows = (const uint32_t *)kfd_gpu_buffer_cpu(ctx->tile_overflows);
  if (counts == NULL || overflows == NULL) {
    return QR_ERROR_IO;
  }
  qr_reset_raster_stats(ctx);
  ctx->last_stats.triangle_count = triangle_count;
  for (i = 0U; i < ctx->tile_count; ++i) {
    if (counts[i] != 0U) {
      ++ctx->last_stats.occupied_tile_count;
      ++ctx->last_stats.depth_bound_tile_count;
    }
    if (counts[i] > ctx->last_stats.max_tile_triangle_count) {
      ctx->last_stats.max_tile_triangle_count = counts[i];
    }
    if (overflows[i] != 0U) {
      ++ctx->last_stats.overflow_tile_count;
      if (UINT32_MAX - ctx->last_stats.hiz_overflow_fallback_count <
          overflows[i]) {
        ctx->last_stats.hiz_overflow_fallback_count = UINT32_MAX;
      } else {
        ctx->last_stats.hiz_overflow_fallback_count += overflows[i];
      }
      if (UINT32_MAX - ctx->last_stats.overflow_reference_count <
          overflows[i]) {
        ctx->last_stats.overflow_reference_count = UINT32_MAX;
      } else {
        ctx->last_stats.overflow_reference_count += overflows[i];
      }
    }
    if (UINT32_MAX - ctx->last_stats.hiz_candidate_reference_count <
        counts[i]) {
      ctx->last_stats.hiz_candidate_reference_count = UINT32_MAX;
    } else {
      ctx->last_stats.hiz_candidate_reference_count += counts[i];
    }
  }
  return QR_SUCCESS;
}

static qr_result qr_dispatch_prepared_triangles(qr_context *ctx,
                                                size_t triangle_count,
                                                const uint8_t *colormap,
                                                size_t colormap_size,
                                                qr_debug_mode debug_mode,
                                                float time_seconds)
{
  qr_tile_bin_args *bin_args;
  qr_world_raster_args *args;
  uint64_t start_ns;
  uint64_t end_ns;
  int err;

  if (ctx == NULL || qr_valid_debug_mode(debug_mode) == 0 ||
      !isfinite(time_seconds)) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (debug_mode == QR_DEBUG_SHADED &&
      (colormap == NULL || colormap_size < QR_COLORMAP_SIZE)) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (triangle_count == 0U) {
    qr_reset_raster_stats(ctx);
    return QR_SUCCESS;
  }
  if (triangle_count > UINT32_MAX) {
    return QR_ERROR_OVERFLOW;
  }
  if (triangle_count > ctx->triangle_capacity) {
    return QR_ERROR_NO_SPACE;
  }
  if (ctx->triangles == NULL || ctx->triangle_buffer == NULL) {
    return QR_ERROR_IO;
  }

  if (debug_mode == QR_DEBUG_SHADED) {
    void *dst_colormap = kfd_gpu_buffer_cpu(ctx->raster_colormap);

    if (dst_colormap == NULL) {
      return QR_ERROR_IO;
    }
    memcpy(dst_colormap, colormap, QR_COLORMAP_SIZE);
  }

  args = (qr_world_raster_args *)kfd_gpu_buffer_cpu(ctx->raster_root);
  if (args == NULL) {
    return QR_ERROR_IO;
  }
  args->triangles = (qr_raster_triangle *)kfd_gpu_buffer_gpu(ctx->triangle_buffer);
  args->triangle_count = (uint32_t)triangle_count;
  args->debug_mode = (uint32_t)debug_mode;
  args->time_seconds = time_seconds;

  bin_args = (qr_tile_bin_args *)kfd_gpu_buffer_cpu(ctx->tile_bin_root);
  if (bin_args == NULL) {
    return QR_ERROR_IO;
  }
  bin_args->triangles =
      (qr_raster_triangle *)kfd_gpu_buffer_gpu(ctx->triangle_buffer);
  bin_args->triangle_count = (uint32_t)triangle_count;

  start_ns = qr_now_ns();
  err = kfd_gpu_dispatch(ctx->gpu, ctx->tile_bin_kernel,
                         &ctx->tile_bin_dispatch, ctx->tile_bin_kernarg,
                         ctx->tile_bin_fence);
  if (err == 0) {
    err = kfd_gpu_fence_wait(ctx->tile_bin_fence, 0U, UINT64_MAX);
  }
  if (err != 0) {
    return qr_result_from_gpu_error(err);
  }
  end_ns = qr_now_ns();
  if (end_ns >= start_ns) {
    ctx->perf.tile_bin_time_ns += end_ns - start_ns;
  }
  {
    qr_result stats_result =
        qr_collect_tile_stats(ctx, (uint32_t)triangle_count);
    if (stats_result != QR_SUCCESS) {
      return stats_result;
    }
  }

  start_ns = qr_now_ns();
  err = kfd_gpu_dispatch(ctx->gpu, ctx->raster_kernel, &ctx->raster_dispatch,
                         ctx->raster_kernarg, ctx->raster_fence);
  if (err == 0) {
    err = kfd_gpu_fence_wait(ctx->raster_fence, 0U, UINT64_MAX);
  }
  if (err == 0) {
    end_ns = qr_now_ns();
    ++ctx->perf.draw_count;
    ctx->perf.primitive_count += triangle_count;
    ctx->perf.tile_count += ctx->last_stats.occupied_tile_count;
    ctx->perf.tile_overflow_count += ctx->last_stats.overflow_tile_count;
    ctx->perf.tile_overflow_reference_count +=
        ctx->last_stats.overflow_reference_count;
    ctx->perf.hiz_candidate_reference_count +=
        ctx->last_stats.hiz_candidate_reference_count;
    ctx->perf.hiz_overflow_fallback_count +=
        ctx->last_stats.hiz_overflow_fallback_count;
    if (end_ns >= start_ns) {
      ctx->perf.raster_time_ns += end_ns - start_ns;
    }
  }
  return qr_result_from_gpu_error(err);
}

qr_result qr_frame_draw_world(qr_frame *frame, const qr_world_draw_desc *desc)
{
  qr_context *ctx;
  qr_world_record *world;
  qr_raster_triangle *triangles;
  size_t triangle_count;
  size_t i;
  size_t out_index;

  if (frame == NULL || frame->ctx == NULL || desc == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  ctx = frame->ctx;
  if (ctx->frame_active == 0 || desc->world == QR_INVALID_HANDLE ||
      desc->world > ctx->world_count ||
      qr_valid_debug_mode(desc->debug_mode) == 0 ||
      !isfinite(desc->time_seconds)) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (desc->debug_mode == QR_DEBUG_SHADED &&
      (desc->colormap == NULL || desc->colormap_size < QR_COLORMAP_SIZE)) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (desc->polygon_count == 0U) {
    qr_reset_raster_stats(ctx);
    return QR_SUCCESS;
  }
  if (desc->polygons == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }

  world = &ctx->worlds[desc->world];
  triangle_count = 0U;
  for (i = 0U; i < desc->polygon_count; ++i) {
    const qr_world_polygon_desc *polygon = &desc->polygons[i];

    if (polygon->vertices == NULL || polygon->vertex_count < 3U ||
        polygon->surface >= world->surface_count) {
      return QR_ERROR_INVALID_ARGUMENT;
    }
    for (uint32_t j = 0U; j < polygon->vertex_count; ++j) {
      if (qr_valid_world_vertex(&polygon->vertices[j]) == 0) {
        return QR_ERROR_INVALID_ARGUMENT;
      }
    }
    if (triangle_count > SIZE_MAX - ((size_t)polygon->vertex_count - 2U)) {
      return QR_ERROR_OVERFLOW;
    }
    triangle_count += (size_t)polygon->vertex_count - 2U;
  }
  if (triangle_count > UINT32_MAX) {
    return QR_ERROR_OVERFLOW;
  }
  if (triangle_count > ctx->triangle_capacity) {
    return QR_ERROR_NO_SPACE;
  }
  triangles = ctx->triangles;
  if (triangles == NULL || ctx->triangulation_indices == NULL) {
    return QR_ERROR_IO;
  }

  out_index = 0U;
  for (i = 0U; i < desc->polygon_count; ++i) {
    const qr_world_polygon_desc *polygon = &desc->polygons[i];
    qr_result result;

    result = qr_triangulate_world_polygon(
        polygon, world->first_surface, triangles, ctx->triangle_capacity,
        &out_index, ctx->triangulation_indices);
    if (result != QR_SUCCESS) {
      return result;
    }
  }
  return qr_dispatch_prepared_triangles(ctx, out_index, desc->colormap,
                                        desc->colormap_size, desc->debug_mode,
                                        desc->time_seconds);
}

qr_result qr_frame_draw_alias_model(qr_frame *frame,
                                    const qr_alias_draw_desc *desc)
{
  qr_context *ctx;
  qr_alias_model_record *model;
  qr_world_record *world;
  size_t i;

  if (frame == NULL || frame->ctx == NULL || desc == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  ctx = frame->ctx;
  if (ctx->frame_active == 0 || desc->model == QR_INVALID_HANDLE ||
      desc->model > ctx->alias_model_count ||
      qr_valid_debug_mode(desc->debug_mode) == 0) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  model = &ctx->alias_models[desc->model];
  world = &ctx->worlds[model->world];
  if (model->triangle_count > ctx->triangle_capacity || ctx->triangles == NULL) {
    return QR_ERROR_NO_SPACE;
  }
  for (i = 0U; i < model->triangle_count; ++i) {
    const qr_alias_triangle_record *src =
        &ctx->alias_triangles[(size_t)model->first_triangle + i];
    qr_raster_triangle *dst = &ctx->triangles[i];

    dst->v0 = src->v0;
    dst->v1 = src->v1;
    dst->v2 = src->v2;
    dst->surface = world->first_surface + src->surface;
  }
  return qr_dispatch_prepared_triangles(ctx, model->triangle_count,
                                        desc->colormap, desc->colormap_size,
                                        desc->debug_mode, desc->time_seconds);
}

qr_result qr_frame_draw_sprite(qr_frame *frame,
                               const qr_sprite_draw_desc *desc)
{
  qr_context *ctx;
  qr_world_record *world;

  if (frame == NULL || frame->ctx == NULL || desc == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  ctx = frame->ctx;
  if (ctx->frame_active == 0 || desc->world == QR_INVALID_HANDLE ||
      desc->world > ctx->world_count ||
      qr_valid_debug_mode(desc->debug_mode) == 0 ||
      qr_valid_sprite_desc(desc) == 0) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  world = &ctx->worlds[desc->world];
  if (desc->surface >= world->surface_count || ctx->triangle_capacity < 2U ||
      ctx->triangles == NULL) {
    return desc->surface >= world->surface_count ? QR_ERROR_INVALID_ARGUMENT
                                                 : QR_ERROR_NO_SPACE;
  }

  ctx->triangles[0].v0 = (qr_raster_vertex){desc->x0, desc->y0, desc->z,
                                             desc->u0, desc->v0, 0.0f, 0.0f};
  ctx->triangles[0].v1 = (qr_raster_vertex){desc->x1, desc->y0, desc->z,
                                             desc->u1, desc->v0, 0.0f, 0.0f};
  ctx->triangles[0].v2 = (qr_raster_vertex){desc->x1, desc->y1, desc->z,
                                             desc->u1, desc->v1, 0.0f, 0.0f};
  ctx->triangles[0].surface = world->first_surface + desc->surface;
  ctx->triangles[1].v0 = ctx->triangles[0].v0;
  ctx->triangles[1].v1 = ctx->triangles[0].v2;
  ctx->triangles[1].v2 = (qr_raster_vertex){desc->x0, desc->y1, desc->z,
                                             desc->u0, desc->v1, 0.0f, 0.0f};
  ctx->triangles[1].surface = world->first_surface + desc->surface;
  return qr_dispatch_prepared_triangles(ctx, 2U, desc->colormap,
                                        desc->colormap_size, desc->debug_mode,
                                        desc->time_seconds);
}

qr_result qr_frame_draw_particles(qr_frame *frame,
                                  const qr_particles_draw_desc *desc)
{
  qr_context *ctx;
  size_t triangle_count;
  size_t out_index;
  size_t i;

  if (frame == NULL || frame->ctx == NULL || desc == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  ctx = frame->ctx;
  if (ctx->frame_active == 0 || qr_valid_debug_mode(desc->debug_mode) == 0 ||
      !isfinite(desc->time_seconds)) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (desc->particle_count == 0U) {
    qr_reset_raster_stats(ctx);
    return QR_SUCCESS;
  }
  if (desc->particles == NULL || desc->particle_count > SIZE_MAX / 2U) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  triangle_count = desc->particle_count * 2U;
  if (triangle_count > ctx->triangle_capacity || ctx->triangles == NULL) {
    return QR_ERROR_NO_SPACE;
  }

  out_index = 0U;
  for (i = 0U; i < desc->particle_count; ++i) {
    const qr_particle_desc *particle = &desc->particles[i];
    qr_world_record *world;
    float half_size;
    float x0;
    float y0;
    float x1;
    float y1;
    uint32_t surface;

    if (qr_valid_particle_desc(particle) == 0 ||
        particle->world == QR_INVALID_HANDLE ||
        particle->world > ctx->world_count) {
      return QR_ERROR_INVALID_ARGUMENT;
    }
    world = &ctx->worlds[particle->world];
    if (particle->surface >= world->surface_count) {
      return QR_ERROR_INVALID_ARGUMENT;
    }
    half_size = particle->size * 0.5f;
    x0 = particle->x - half_size;
    y0 = particle->y - half_size;
    x1 = particle->x + half_size;
    y1 = particle->y + half_size;
    surface = world->first_surface + particle->surface;

    ctx->triangles[out_index].v0 =
        (qr_raster_vertex){x0, y0, particle->z, particle->u, particle->v,
                           0.0f, 0.0f};
    ctx->triangles[out_index].v1 =
        (qr_raster_vertex){x1, y0, particle->z, particle->u + 1.0f,
                           particle->v, 0.0f, 0.0f};
    ctx->triangles[out_index].v2 =
        (qr_raster_vertex){x1, y1, particle->z, particle->u + 1.0f,
                           particle->v + 1.0f, 0.0f, 0.0f};
    ctx->triangles[out_index].surface = surface;
    ++out_index;
    ctx->triangles[out_index].v0 = ctx->triangles[out_index - 1U].v0;
    ctx->triangles[out_index].v1 = ctx->triangles[out_index - 1U].v2;
    ctx->triangles[out_index].v2 =
        (qr_raster_vertex){x0, y1, particle->z, particle->u,
                           particle->v + 1.0f, 0.0f, 0.0f};
    ctx->triangles[out_index].surface = surface;
    ++out_index;
  }

  return qr_dispatch_prepared_triangles(ctx, triangle_count, desc->colormap,
                                        desc->colormap_size, desc->debug_mode,
                                        desc->time_seconds);
}

qr_result qr_end_frame(qr_frame *frame)
{
  qr_context *ctx;
  qr_result result;
  int err;

  if (frame == NULL || frame->ctx == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  ctx = frame->ctx;
  if (ctx->frame_active == 0) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (ctx->output_mode == QR_OUTPUT_PRESENT) {
    const uint32_t *xrgb;

    err = qr_dispatch_resolve_xrgb(ctx, ctx->present_palette_xrgb);
    if (err != 0) {
      ctx->frame_active = 0;
      return qr_result_from_gpu_error(err);
    }
    xrgb = (const uint32_t *)kfd_gpu_buffer_cpu(ctx->xrgb);
    if (xrgb == NULL) {
      ctx->frame_active = 0;
      return QR_ERROR_IO;
    }
    result = ctx->present(ctx->present_userdata, xrgb, ctx->width, ctx->height,
                          ctx->width);
    if (result != QR_SUCCESS) {
      ctx->frame_active = 0;
      return result;
    }
  }
  ++ctx->perf.frame_count;
  ctx->frame_active = 0;
  return QR_SUCCESS;
}

qr_result qr_read_indexed(qr_context *ctx, void *dst, size_t dst_size,
                          size_t dst_stride)
{
  const uint8_t *src_row;
  uint8_t *dst_row;
  size_t required;
  size_t stride;
  uint32_t y;
  int err;

  if (ctx == NULL || dst == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  stride = dst_stride != 0U ? dst_stride : (size_t)ctx->width;
  err = qr_indexed_read_size(ctx->width, ctx->height, stride, &required);
  if (err != 0) {
    return qr_result_from_errno(err);
  }
  if (dst_size < required) {
    return QR_ERROR_BUFFER_TOO_SMALL;
  }
  src_row = (const uint8_t *)kfd_gpu_buffer_cpu(ctx->indexed);
  if (src_row == NULL) {
    return QR_ERROR_IO;
  }
  dst_row = (uint8_t *)dst;
  for (y = 0U; y < ctx->height; ++y) {
    memcpy(dst_row, src_row, (size_t)ctx->width);
    src_row += ctx->width;
    dst_row += stride;
  }
  return QR_SUCCESS;
}

qr_result qr_read_xrgb(qr_context *ctx, const uint32_t *palette_xrgb, void *dst,
                       size_t dst_size, size_t dst_stride_pixels)
{
  const uint32_t *src_row;
  uint32_t *dst_row;
  size_t required;
  size_t stride_pixels;
  uint32_t y;
  int err;

  if (ctx == NULL || palette_xrgb == NULL || dst == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  stride_pixels =
      dst_stride_pixels != 0U ? dst_stride_pixels : (size_t)ctx->width;
  err = qr_xrgb_read_size(ctx->width, ctx->height, stride_pixels, &required);
  if (err != 0) {
    return qr_result_from_errno(err);
  }
  if (dst_size < required) {
    return QR_ERROR_BUFFER_TOO_SMALL;
  }
  err = qr_dispatch_resolve_xrgb(ctx, palette_xrgb);
  if (err != 0) {
    return qr_result_from_gpu_error(err);
  }
  src_row = (const uint32_t *)kfd_gpu_buffer_cpu(ctx->xrgb);
  if (src_row == NULL) {
    return QR_ERROR_IO;
  }
  dst_row = (uint32_t *)dst;
  for (y = 0U; y < ctx->height; ++y) {
    memcpy(dst_row, src_row, (size_t)ctx->width * sizeof(uint32_t));
    src_row += ctx->width;
    dst_row += stride_pixels;
  }
  return QR_SUCCESS;
}

qr_result qr_dump_indexed(qr_context *ctx, const char *path)
{
  const uint8_t *src_row;
  FILE *file;
  uint32_t y;
  int err;

  if (ctx == NULL || path == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  src_row = (const uint8_t *)kfd_gpu_buffer_cpu(ctx->indexed);
  if (src_row == NULL) {
    return QR_ERROR_IO;
  }
  file = fopen(path, "wb");
  if (file == NULL) {
    return qr_result_from_errno(errno != 0 ? errno : EIO);
  }
  for (y = 0U; y < ctx->height; ++y) {
    err = qr_write_all(file, src_row, (size_t)ctx->width);
    if (err != 0) {
      (void)fclose(file);
      return qr_result_from_errno(err);
    }
    src_row += ctx->width;
  }
  if (fclose(file) != 0) {
    return qr_result_from_errno(errno != 0 ? errno : EIO);
  }
  return QR_SUCCESS;
}

qr_result qr_dump_xrgb(qr_context *ctx, const uint32_t *palette_xrgb,
                       const char *path)
{
  const uint32_t *src_row;
  char header[64];
  FILE *file;
  int header_size;
  uint32_t y;
  int err;

  if (ctx == NULL || palette_xrgb == NULL || path == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  err = qr_dispatch_resolve_xrgb(ctx, palette_xrgb);
  if (err != 0) {
    return qr_result_from_gpu_error(err);
  }
  src_row = (const uint32_t *)kfd_gpu_buffer_cpu(ctx->xrgb);
  if (src_row == NULL) {
    return QR_ERROR_IO;
  }
  header_size = snprintf(header, sizeof(header), "P6\n%u %u\n255\n",
                         ctx->width, ctx->height);
  if (header_size < 0 || (size_t)header_size >= sizeof(header)) {
    return QR_ERROR_OVERFLOW;
  }
  file = fopen(path, "wb");
  if (file == NULL) {
    return qr_result_from_errno(errno != 0 ? errno : EIO);
  }
  err = qr_write_all(file, header, (size_t)header_size);
  if (err != 0) {
    (void)fclose(file);
    return qr_result_from_errno(err);
  }
  for (y = 0U; y < ctx->height; ++y) {
    uint32_t x;

    for (x = 0U; x < ctx->width; ++x) {
      uint32_t color = src_row[x];
      uint8_t rgb[3] = {
          (uint8_t)((color >> 16U) & 0xffU),
          (uint8_t)((color >> 8U) & 0xffU),
          (uint8_t)(color & 0xffU),
      };

      err = qr_write_all(file, rgb, sizeof(rgb));
      if (err != 0) {
        (void)fclose(file);
        return qr_result_from_errno(err);
      }
    }
    src_row += ctx->width;
  }
  if (fclose(file) != 0) {
    return qr_result_from_errno(errno != 0 ? errno : EIO);
  }
  return QR_SUCCESS;
}

qr_result qr_upload_texture(qr_context *ctx, const qr_texture_desc *desc,
                            qr_texture *out)
{
  qr_texture_record record;
  uint8_t *atlas;
  size_t mip_bytes[QR_TEXTURE_MIP_COUNT];
  size_t total_bytes;
  uint32_t i;
  int err;

  if (ctx == NULL || desc == NULL || out == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  *out = QR_INVALID_HANDLE;
  if (desc->mip_count == 0U || desc->mip_count > QR_TEXTURE_MIP_COUNT) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (ctx->texture_count >= ctx->texture_capacity) {
    return QR_ERROR_NO_SPACE;
  }
  atlas = (uint8_t *)kfd_gpu_buffer_cpu(ctx->texture_atlas);
  if (atlas == NULL) {
    return QR_ERROR_IO;
  }

  memset(&record, 0, sizeof(record));
  total_bytes = 0U;
  for (i = 0U; i < desc->mip_count; ++i) {
    const qr_texture_mip_desc *mip = &desc->mips[i];
    size_t stride = mip->stride != 0U ? mip->stride : (size_t)mip->width;

    if (mip->pixels == NULL || mip->width == 0U || mip->height == 0U ||
        stride < (size_t)mip->width) {
      return QR_ERROR_INVALID_ARGUMENT;
    }
    err = qr_rect_tight_size(mip->width, mip->height, &mip_bytes[i]);
    if (err != 0) {
      return qr_result_from_errno(err);
    }
    if (total_bytes > SIZE_MAX - mip_bytes[i] ||
        ctx->texture_atlas_used > ctx->texture_atlas_bytes ||
        total_bytes > ctx->texture_atlas_bytes - ctx->texture_atlas_used ||
        mip_bytes[i] >
            ctx->texture_atlas_bytes - ctx->texture_atlas_used - total_bytes ||
        ctx->texture_atlas_used + total_bytes > UINT32_MAX) {
      return QR_ERROR_NO_SPACE;
    }
    record.mip_offset[i] =
        (uint32_t)(ctx->texture_atlas_used + total_bytes);
    record.width[i] = mip->width;
    record.height[i] = mip->height;
    total_bytes += mip_bytes[i];
  }

  for (i = 0U; i < desc->mip_count; ++i) {
    const qr_texture_mip_desc *mip = &desc->mips[i];
    err = qr_copy_indexed_rect(atlas, record.mip_offset[i],
                               ctx->texture_atlas_bytes, mip->pixels,
                               mip->width, mip->height, mip->stride);
    if (err != 0) {
      return qr_result_from_errno(err);
    }
  }

  record.mip_count = desc->mip_count;
  record.flags = desc->flags;
  ++ctx->texture_count;
  ctx->textures[ctx->texture_count] = record;
  ctx->texture_atlas_used += total_bytes;
  ctx->perf.upload_bytes += total_bytes;
  *out = ctx->texture_count;
  return QR_SUCCESS;
}

qr_result qr_upload_lightmap(qr_context *ctx, const qr_lightmap_desc *desc,
                             qr_lightmap *out)
{
  qr_lightmap_record record;
  uint8_t *atlas;
  size_t bytes;
  int err;

  if (ctx == NULL || desc == NULL || out == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  *out = QR_INVALID_HANDLE;
  if (desc->pixels == NULL || desc->width == 0U || desc->height == 0U) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (ctx->lightmap_count >= ctx->lightmap_capacity) {
    return QR_ERROR_NO_SPACE;
  }
  err = qr_rect_tight_size(desc->width, desc->height, &bytes);
  if (err != 0) {
    return qr_result_from_errno(err);
  }
  if (ctx->lightmap_atlas_used > ctx->lightmap_atlas_bytes ||
      bytes > ctx->lightmap_atlas_bytes - ctx->lightmap_atlas_used ||
      ctx->lightmap_atlas_used > UINT32_MAX ||
      bytes > (size_t)UINT32_MAX - ctx->lightmap_atlas_used) {
    return QR_ERROR_NO_SPACE;
  }
  atlas = (uint8_t *)kfd_gpu_buffer_cpu(ctx->lightmap_atlas);
  if (atlas == NULL) {
    return QR_ERROR_IO;
  }
  err = qr_copy_indexed_rect(atlas, ctx->lightmap_atlas_used,
                             ctx->lightmap_atlas_bytes, desc->pixels,
                             desc->width, desc->height, desc->stride);
  if (err != 0) {
    return qr_result_from_errno(err);
  }

  record.offset = (uint32_t)ctx->lightmap_atlas_used;
  record.width = desc->width;
  record.height = desc->height;
  ++ctx->lightmap_count;
  ctx->lightmaps[ctx->lightmap_count] = record;
  ctx->lightmap_atlas_used += bytes;
  ctx->perf.upload_bytes += bytes;
  *out = ctx->lightmap_count;
  return QR_SUCCESS;
}

qr_result qr_update_lightmap(qr_context *ctx, qr_lightmap lightmap,
                             const qr_lightmap_desc *desc)
{
  qr_lightmap_record *record;
  uint8_t *atlas;
  int err;

  if (ctx == NULL || desc == NULL || lightmap == QR_INVALID_HANDLE ||
      lightmap > ctx->lightmap_count || desc->pixels == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  record = &ctx->lightmaps[lightmap];
  if (desc->width != record->width || desc->height != record->height) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  atlas = (uint8_t *)kfd_gpu_buffer_cpu(ctx->lightmap_atlas);
  if (atlas == NULL) {
    return QR_ERROR_IO;
  }
  err = qr_copy_indexed_rect(atlas, record->offset, ctx->lightmap_atlas_bytes,
                             desc->pixels, desc->width, desc->height,
                             desc->stride);
  if (err != 0) {
    return qr_result_from_errno(err);
  }
  ctx->perf.upload_bytes += (size_t)record->width * (size_t)record->height;
  return QR_SUCCESS;
}

qr_result qr_create_world(qr_context *ctx, const qr_world_surface_desc *surfaces,
                          size_t surface_count, qr_world *out)
{
  qr_surface_record *dst;
  uint32_t first_surface;
  uint32_t remaining_surfaces;
  size_t i;

  if (ctx == NULL || surfaces == NULL || out == NULL || surface_count == 0U) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  *out = QR_INVALID_HANDLE;
  if (ctx->surface_count > ctx->surface_capacity) {
    return QR_ERROR_SYSTEM;
  }
  remaining_surfaces = ctx->surface_capacity - ctx->surface_count;
  if (surface_count > UINT32_MAX ||
      surface_count > (size_t)remaining_surfaces) {
    return QR_ERROR_NO_SPACE;
  }
  if (ctx->world_count >= ctx->world_capacity) {
    return QR_ERROR_NO_SPACE;
  }
  dst = (qr_surface_record *)kfd_gpu_buffer_cpu(ctx->surface_metadata);
  if (dst == NULL) {
    return QR_ERROR_IO;
  }
  for (i = 0U; i < surface_count; ++i) {
    const qr_world_surface_desc *src = &surfaces[i];

    if (src->texture == QR_INVALID_HANDLE ||
        src->texture > ctx->texture_count ||
        src->lightmap == QR_INVALID_HANDLE ||
        src->lightmap > ctx->lightmap_count ||
        qr_valid_surface_desc(src) == 0) {
      return QR_ERROR_INVALID_ARGUMENT;
    }
  }

  first_surface = ctx->surface_count;
  for (i = 0U; i < surface_count; ++i) {
    const qr_world_surface_desc *src = &surfaces[i];
    qr_surface_record *record = &dst[(size_t)first_surface + i];

    record->texture = src->texture;
    record->lightmap = src->lightmap;
    record->flags = src->flags;
    memcpy(record->plane, src->plane, sizeof(record->plane));
    memcpy(record->tex_s, src->tex_s, sizeof(record->tex_s));
    memcpy(record->tex_t, src->tex_t, sizeof(record->tex_t));
    memcpy(record->light_s, src->light_s, sizeof(record->light_s));
    memcpy(record->light_t, src->light_t, sizeof(record->light_t));
  }

  ++ctx->world_count;
  ctx->worlds[ctx->world_count].first_surface = first_surface;
  ctx->worlds[ctx->world_count].surface_count = (uint32_t)surface_count;
  ctx->surface_count += (uint32_t)surface_count;
  *out = ctx->world_count;
  return QR_SUCCESS;
}

qr_result qr_upload_alias_model(qr_context *ctx,
                                const qr_alias_model_desc *desc,
                                qr_alias_model *out)
{
  qr_world_record *world;
  uint32_t first_triangle;
  uint32_t remaining_triangles;
  size_t i;

  if (ctx == NULL || desc == NULL || out == NULL ||
      desc->world == QR_INVALID_HANDLE || desc->world > ctx->world_count ||
      desc->triangles == NULL || desc->triangle_count == 0U) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  *out = QR_INVALID_HANDLE;
  if (ctx->alias_model_count >= ctx->alias_model_capacity ||
      desc->triangle_count > UINT32_MAX) {
    return QR_ERROR_NO_SPACE;
  }
  if (ctx->alias_triangle_count > ctx->alias_triangle_capacity) {
    return QR_ERROR_SYSTEM;
  }
  remaining_triangles =
      ctx->alias_triangle_capacity - ctx->alias_triangle_count;
  if (desc->triangle_count > (size_t)remaining_triangles) {
    return QR_ERROR_NO_SPACE;
  }

  world = &ctx->worlds[desc->world];
  for (i = 0U; i < desc->triangle_count; ++i) {
    if (desc->triangles[i].surface >= world->surface_count ||
        qr_valid_alias_triangle_desc(&desc->triangles[i]) == 0) {
      return QR_ERROR_INVALID_ARGUMENT;
    }
  }

  first_triangle = ctx->alias_triangle_count;
  for (i = 0U; i < desc->triangle_count; ++i) {
    const qr_alias_triangle_desc *src = &desc->triangles[i];
    qr_alias_triangle_record *dst =
        &ctx->alias_triangles[(size_t)first_triangle + i];

    dst->v0 = qr_make_raster_vertex(&src->v0);
    dst->v1 = qr_make_raster_vertex(&src->v1);
    dst->v2 = qr_make_raster_vertex(&src->v2);
    dst->surface = src->surface;
  }

  ++ctx->alias_model_count;
  ctx->alias_models[ctx->alias_model_count].world = desc->world;
  ctx->alias_models[ctx->alias_model_count].first_triangle = first_triangle;
  ctx->alias_models[ctx->alias_model_count].triangle_count =
      (uint32_t)desc->triangle_count;
  ctx->alias_triangle_count += (uint32_t)desc->triangle_count;
  *out = ctx->alias_model_count;
  return QR_SUCCESS;
}

qr_result qr_get_capacity_info(qr_context *ctx, qr_capacity_info *out)
{
  if (ctx == NULL || out == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  out->texture_count = ctx->texture_count;
  out->texture_capacity = ctx->texture_capacity;
  out->texture_atlas_used = ctx->texture_atlas_used;
  out->texture_atlas_capacity = ctx->texture_atlas_bytes;
  out->lightmap_count = ctx->lightmap_count;
  out->lightmap_capacity = ctx->lightmap_capacity;
  out->lightmap_atlas_used = ctx->lightmap_atlas_used;
  out->lightmap_atlas_capacity = ctx->lightmap_atlas_bytes;
  out->surface_count = ctx->surface_count;
  out->surface_capacity = ctx->surface_capacity;
  out->world_count = ctx->world_count;
  out->world_capacity = ctx->world_capacity;
  out->frame_triangle_capacity = ctx->triangle_capacity;
  out->alias_model_count = ctx->alias_model_count;
  out->alias_model_capacity = ctx->alias_model_capacity;
  out->alias_triangle_count = ctx->alias_triangle_count;
  out->alias_triangle_capacity = ctx->alias_triangle_capacity;
  return QR_SUCCESS;
}

qr_result qr_get_perf_counters(qr_context *ctx, qr_perf_counters *out)
{
  if (ctx == NULL || out == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  *out = ctx->perf;
  return QR_SUCCESS;
}

qr_result qr_get_raster_stats(qr_context *ctx, qr_raster_stats *out)
{
  if (ctx == NULL || out == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  *out = ctx->last_stats;
  return QR_SUCCESS;
}
