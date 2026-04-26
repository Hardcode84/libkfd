#include "libkfd/quake_raster.h"

#include "libkfd/gpu.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct qr_frame {
  struct qr_context *ctx;
};

struct qr_context {
  kfd_gpu_context *gpu;
  kfd_gpu_buffer *indexed;
  uint32_t width;
  uint32_t height;
  size_t indexed_bytes;
  qr_output_mode output_mode;
  struct qr_frame frame;
  int frame_active;
};

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
  if ((size_t)desc->width > SIZE_MAX / (size_t)desc->height) {
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

const char *qr_strerror(int code)
{
  if (code == 0) {
    return "success";
  }
  return kfd_gpu_strerror(code);
}

int qr_create(const qr_desc *desc, qr_context **out)
{
  qr_context *ctx;
  kfd_gpu_context *gpu;
  kfd_gpu_buffer *indexed;
  size_t indexed_bytes;
  int err;

  if (out == NULL) {
    return EINVAL;
  }
  *out = NULL;
  err = qr_validate_desc(desc);
  if (err != 0) {
    return err;
  }

  indexed_bytes = (size_t)desc->width * (size_t)desc->height;
  err = kfd_gpu_context_create(desc->device_index, &gpu);
  if (err != 0) {
    return err;
  }
  err = kfd_gpu_buffer_create(gpu, indexed_bytes, KFD_GPU_MEMORY_UPLOAD,
                              KFD_GPU_MEMORY_WRITABLE |
                                  KFD_GPU_MEMORY_COHERENT |
                                  KFD_GPU_MEMORY_UNCACHED,
                              &indexed);
  if (err != 0) {
    kfd_gpu_context_destroy(gpu);
    return err;
  }

  ctx = (qr_context *)calloc(1U, sizeof(*ctx));
  if (ctx == NULL) {
    kfd_gpu_buffer_destroy(indexed);
    kfd_gpu_context_destroy(gpu);
    return ENOMEM;
  }

  ctx->gpu = gpu;
  ctx->indexed = indexed;
  ctx->width = desc->width;
  ctx->height = desc->height;
  ctx->indexed_bytes = indexed_bytes;
  ctx->output_mode = desc->output_mode;
  ctx->frame.ctx = ctx;
  *out = ctx;
  return 0;
}

void qr_destroy(qr_context *ctx)
{
  if (ctx == NULL) {
    return;
  }
  kfd_gpu_buffer_destroy(ctx->indexed);
  kfd_gpu_context_destroy(ctx->gpu);
  free(ctx);
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

int qr_begin_frame(qr_context *ctx, const qr_frame_desc *desc, qr_frame **out)
{
  (void)desc;
  if (ctx == NULL || out == NULL) {
    return EINVAL;
  }
  if (ctx->frame_active != 0) {
    return EBUSY;
  }
  ctx->frame_active = 1;
  *out = &ctx->frame;
  return 0;
}

int qr_frame_clear_indexed(qr_frame *frame, uint8_t color)
{
  void *dst;

  if (frame == NULL || frame->ctx == NULL || frame->ctx->indexed == NULL) {
    return EINVAL;
  }
  if (frame->ctx->frame_active == 0) {
    return EINVAL;
  }
  dst = kfd_gpu_buffer_cpu(frame->ctx->indexed);
  if (dst == NULL) {
    return EIO;
  }
  memset(dst, (int)color, frame->ctx->indexed_bytes);
  return 0;
}

int qr_end_frame(qr_frame *frame)
{
  if (frame == NULL || frame->ctx == NULL) {
    return EINVAL;
  }
  if (frame->ctx->frame_active == 0) {
    return EINVAL;
  }
  frame->ctx->frame_active = 0;
  return 0;
}

int qr_read_indexed(qr_context *ctx, void *dst, size_t dst_size,
                    size_t dst_stride)
{
  const uint8_t *src_row;
  uint8_t *dst_row;
  size_t required;
  size_t stride;
  uint32_t y;
  int err;

  if (ctx == NULL || dst == NULL) {
    return EINVAL;
  }
  stride = dst_stride != 0U ? dst_stride : (size_t)ctx->width;
  err = qr_indexed_read_size(ctx->width, ctx->height, stride, &required);
  if (err != 0) {
    return err;
  }
  if (dst_size < required) {
    return ENOSPC;
  }
  src_row = (const uint8_t *)kfd_gpu_buffer_cpu(ctx->indexed);
  if (src_row == NULL) {
    return EIO;
  }
  dst_row = (uint8_t *)dst;
  for (y = 0U; y < ctx->height; ++y) {
    memcpy(dst_row, src_row, (size_t)ctx->width);
    src_row += ctx->width;
    dst_row += stride;
  }
  return 0;
}
