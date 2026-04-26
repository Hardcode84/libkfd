#include "libkfd/quake_raster.h"

#include "libkfd/gpu.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QR_CLEAR_BLOCK_X 16U
#define QR_CLEAR_BLOCK_Y 16U
#define QR_DEFAULT_MAX_TEXTURES 1024U
#define QR_DEFAULT_MAX_LIGHTMAPS 1024U
#define QR_DEFAULT_MAX_SURFACES 65536U
#define QR_DEFAULT_MAX_WORLDS 256U
#define QR_DEFAULT_TEXTURE_ATLAS_BYTES (16U * 1024U * 1024U)
#define QR_DEFAULT_LIGHTMAP_ATLAS_BYTES (4U * 1024U * 1024U)

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

static const qr_kernel_candidate qr_clear_indexed_kernels[] = {
#include "quake_raster_clear_indexed_kernels.inc"
    {NULL, NULL}
};

static const qr_kernel_candidate qr_resolve_xrgb_kernels[] = {
#include "quake_raster_resolve_xrgb_kernels.inc"
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
  kfd_gpu_buffer *texture_atlas;
  kfd_gpu_buffer *lightmap_atlas;
  kfd_gpu_buffer *surface_metadata;
  qr_texture_record *textures;
  qr_lightmap_record *lightmaps;
  qr_world_record *worlds;
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
  qr_output_mode output_mode;
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

static int qr_rect_tight_size(uint32_t width, uint32_t height, size_t *out);

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
  if (desc->output_mode != QR_OUTPUT_NOOUTPUT) {
    return ENOTSUP;
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

static int qr_dispatch_resolve_xrgb(qr_context *ctx,
                                    const uint32_t *palette_xrgb)
{
  void *palette_dst;
  int err;

  if (ctx == NULL || palette_xrgb == NULL || ctx->resolve_palette == NULL) {
    return EINVAL;
  }
  palette_dst = kfd_gpu_buffer_cpu(ctx->resolve_palette);
  if (palette_dst == NULL) {
    return EIO;
  }
  memcpy(palette_dst, palette_xrgb, 256U * sizeof(uint32_t));
  err = kfd_gpu_dispatch(ctx->gpu, ctx->resolve_kernel,
                         &ctx->resolve_dispatch, ctx->resolve_kernarg,
                         ctx->resolve_fence);
  if (err != 0) {
    return err;
  }
  return kfd_gpu_fence_wait(ctx->resolve_fence, 0U, UINT64_MAX);
}

static void qr_destroy_resources(qr_context *ctx)
{
  if (ctx == NULL) {
    return;
  }
  qr_free_bytes(ctx->worlds);
  qr_free_bytes(ctx->lightmaps);
  qr_free_bytes(ctx->textures);
  kfd_gpu_buffer_destroy(ctx->surface_metadata);
  kfd_gpu_buffer_destroy(ctx->lightmap_atlas);
  kfd_gpu_buffer_destroy(ctx->texture_atlas);
  ctx->worlds = NULL;
  ctx->lightmaps = NULL;
  ctx->textures = NULL;
  ctx->surface_metadata = NULL;
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
  ctx->texture_atlas_bytes =
      qr_default_size(desc->texture_atlas_bytes, QR_DEFAULT_TEXTURE_ATLAS_BYTES);
  ctx->lightmap_atlas_bytes =
      qr_default_size(desc->lightmap_atlas_bytes, QR_DEFAULT_LIGHTMAP_ATLAS_BYTES);

  if (ctx->texture_capacity == UINT32_MAX ||
      ctx->lightmap_capacity == UINT32_MAX ||
      ctx->world_capacity == UINT32_MAX) {
    return EOVERFLOW;
  }
  texture_slots = (size_t)ctx->texture_capacity + 1U;
  lightmap_slots = (size_t)ctx->lightmap_capacity + 1U;
  world_slots = (size_t)ctx->world_capacity + 1U;
  if (texture_slots > SIZE_MAX / sizeof(*ctx->textures) ||
      lightmap_slots > SIZE_MAX / sizeof(*ctx->lightmaps) ||
      world_slots > SIZE_MAX / sizeof(*ctx->worlds) ||
      (size_t)ctx->surface_capacity > SIZE_MAX / sizeof(qr_surface_record)) {
    return EOVERFLOW;
  }
  surface_bytes = (size_t)ctx->surface_capacity * sizeof(qr_surface_record);

  ctx->textures =
      (qr_texture_record *)qr_calloc_bytes(texture_slots, sizeof(*ctx->textures));
  ctx->lightmaps = (qr_lightmap_record *)qr_calloc_bytes(
      lightmap_slots, sizeof(*ctx->lightmaps));
  ctx->worlds =
      (qr_world_record *)qr_calloc_bytes(world_slots, sizeof(*ctx->worlds));
  if (ctx->textures == NULL || ctx->lightmaps == NULL || ctx->worlds == NULL) {
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
  err = qr_create_upload_buffer(ctx, surface_bytes, &ctx->surface_metadata);
  if (err != 0) {
    qr_destroy_resources(ctx);
    return err;
  }
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
  *out = ctx;
  return QR_SUCCESS;
}

void qr_destroy(qr_context *ctx)
{
  if (ctx == NULL) {
    return;
  }
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
  err = kfd_gpu_dispatch(frame->ctx->gpu, frame->ctx->clear_kernel,
                         &frame->ctx->clear_dispatch,
                         frame->ctx->clear_kernarg, frame->ctx->clear_fence);
  if (err != 0) {
    return qr_result_from_gpu_error(err);
  }
  return qr_result_from_gpu_error(
      kfd_gpu_fence_wait(frame->ctx->clear_fence, 0U, UINT64_MAX));
}

qr_result qr_end_frame(qr_frame *frame)
{
  if (frame == NULL || frame->ctx == NULL) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  if (frame->ctx->frame_active == 0) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  frame->ctx->frame_active = 0;
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
  *out = ctx->lightmap_count;
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
        src->lightmap > ctx->lightmap_count) {
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
  return QR_SUCCESS;
}
