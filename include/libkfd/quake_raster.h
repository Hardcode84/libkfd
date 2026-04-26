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

const char *qr_strerror(int code);

int qr_create(const qr_desc *desc, qr_context **out);
void qr_destroy(qr_context *ctx);

uint32_t qr_width(const qr_context *ctx);
uint32_t qr_height(const qr_context *ctx);
qr_output_mode qr_output(const qr_context *ctx);

int qr_begin_frame(qr_context *ctx, const qr_frame_desc *desc, qr_frame **out);
int qr_frame_clear_indexed(qr_frame *frame, uint8_t color);
int qr_end_frame(qr_frame *frame);

int qr_read_indexed(qr_context *ctx, void *dst, size_t dst_size,
                    size_t dst_stride);
int qr_read_xrgb(qr_context *ctx, const uint32_t *palette_xrgb, void *dst,
                 size_t dst_size, size_t dst_stride_pixels);

int qr_dump_indexed(qr_context *ctx, const char *path);
int qr_dump_xrgb(qr_context *ctx, const uint32_t *palette_xrgb,
                 const char *path);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBKFD_QUAKE_RASTER_H */
