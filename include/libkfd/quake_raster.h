/*===-- libkfd/quake_raster.h - Quake compute rasterizer API ------*- C -*-===*\
 *
 * Experimental C99 renderer boundary for a compute software rasterizer that can
 * run with or without an active presentation target.
 *
\*===----------------------------------------------------------------------===*/

#ifndef LIBKFD_QUAKE_RASTER_H
#define LIBKFD_QUAKE_RASTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qr_context qr_context;
typedef struct qr_frame qr_frame;

#define QR_API_VERSION_MAJOR 0U
#define QR_API_VERSION_MINOR 1U
#define QR_API_VERSION_PATCH 0U

typedef enum qr_result {
  QR_SUCCESS = 0,
  QR_ERROR_INVALID_ARGUMENT,
  QR_ERROR_UNSUPPORTED,
  QR_ERROR_OVERFLOW,
  QR_ERROR_OUT_OF_MEMORY,
  QR_ERROR_BUFFER_TOO_SMALL,
  QR_ERROR_NO_SPACE,
  QR_ERROR_BUSY,
  QR_ERROR_IO,
  QR_ERROR_NOT_FOUND,
  QR_ERROR_SYSTEM,
  QR_ERROR_GPU
} qr_result;

typedef enum qr_output_mode {
  QR_OUTPUT_NOOUTPUT = 0,
  QR_OUTPUT_PRESENT = 1
} qr_output_mode;

typedef enum qr_framebuffer_format {
  QR_FRAMEBUFFER_INDEXED8 = 0
} qr_framebuffer_format;

typedef struct qr_desc {
  uint32_t width;
  uint32_t height;
  size_t device_index;
  qr_output_mode output_mode;
  qr_framebuffer_format framebuffer_format;
} qr_desc;

typedef struct qr_frame_desc {
  uint32_t reserved;
} qr_frame_desc;

/* Returns QR_API_VERSION_* packed in 8-bit fields as 0x00MMmmPP. */
uint32_t qr_api_version(void);
const char *qr_strerror(qr_result code);

/*
 * Creates a renderer context. The context owns all GPU buffers, loaded kernels,
 * and synchronization objects it allocates. The caller owns only the returned
 * handle and must destroy it with qr_destroy().
 *
 * Contexts are not thread-safe. Calls that operate on the same qr_context or
 * qr_frame must be externally serialized by the caller.
 */
qr_result qr_create(const qr_desc *desc, qr_context **out);
void qr_destroy(qr_context *ctx);

uint32_t qr_width(const qr_context *ctx);
uint32_t qr_height(const qr_context *ctx);
qr_output_mode qr_output(const qr_context *ctx);

/*
 * Frame handles are borrowed from their parent context and stay valid until
 * qr_end_frame(). The current implementation executes clear/resolve work
 * synchronously before returning from the API that submits it.
 */
qr_result qr_begin_frame(qr_context *ctx, const qr_frame_desc *desc,
                         qr_frame **out);
qr_result qr_frame_clear_indexed(qr_frame *frame, uint8_t color);
qr_result qr_end_frame(qr_frame *frame);

/*
 * Readback and dump helpers are intended for headless diagnostics. A zero stride
 * means tightly packed rows. qr_dump_indexed() writes raw indexed bytes;
 * qr_dump_xrgb() writes binary PPM (P6) RGB data after palette resolve.
 */
qr_result qr_read_indexed(qr_context *ctx, void *dst, size_t dst_size,
                          size_t dst_stride);
qr_result qr_read_xrgb(qr_context *ctx, const uint32_t *palette_xrgb, void *dst,
                       size_t dst_size, size_t dst_stride_pixels);

qr_result qr_dump_indexed(qr_context *ctx, const char *path);
qr_result qr_dump_xrgb(qr_context *ctx, const uint32_t *palette_xrgb,
                       const char *path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBKFD_QUAKE_RASTER_H */
