#include "libkfd/gpu/kernel.h"

struct QrResolveXrgbArgs {
  unsigned char *src;
  unsigned *dst;
  unsigned *palette;
  unsigned char *top;
  unsigned char *ui;
  unsigned char *sbar;
  unsigned *top_palette;
  unsigned *ui_palette;
  unsigned width;
  unsigned height;
  int ui_x;
  int ui_y;
  unsigned ui_w;
  unsigned ui_h;
  int sbar_x;
  int sbar_y;
  unsigned sbar_w;
  unsigned sbar_h;
  unsigned top_enabled;
  unsigned ui_enabled;
  unsigned sbar_enabled;
};

static unsigned sample_scaled(unsigned x, unsigned y, unsigned dst_w,
                              unsigned dst_h, unsigned src_w, unsigned src_h,
                              unsigned char *src)
{
  unsigned sx = (x * src_w) / dst_w;
  unsigned sy = (y * src_h) / dst_h;
  return src[sy * src_w + sx];
}

KFD_GPU_KERNEL void qr_resolve_xrgb(struct QrResolveXrgbArgs *args)
{
  unsigned x = kfd_global_id_x();
  unsigned y = kfd_global_id_y();

  if (x >= args->width || y >= args->height) {
    return;
  }
  unsigned index = args->src[y * args->width + x];
  unsigned color = args->palette[index];

  if (args->sbar_enabled != 0U && args->sbar != 0 &&
      args->ui_palette != 0) {
    int bx = (int)x - args->sbar_x;
    int by = (int)y - args->sbar_y;
    if (bx >= 0 && by >= 0 && (unsigned)bx < args->sbar_w &&
        (unsigned)by < args->sbar_h) {
      unsigned sbar_index =
          sample_scaled((unsigned)bx, (unsigned)by, args->sbar_w,
                        args->sbar_h, args->width, args->height, args->sbar);
      if (sbar_index != 255U) {
        color = args->ui_palette[sbar_index];
      }
    }
  }

  if (args->ui_enabled != 0U && args->ui != 0 && args->ui_palette != 0) {
    int bx = (int)x - args->ui_x;
    int by = (int)y - args->ui_y;
    if (bx >= 0 && by >= 0 && (unsigned)bx < args->ui_w &&
        (unsigned)by < args->ui_h) {
      unsigned ui_index =
          sample_scaled((unsigned)bx, (unsigned)by, args->ui_w, args->ui_h,
                        args->width, args->height, args->ui);
      if (ui_index != 255U) {
        color = args->ui_palette[ui_index];
      }
    }
  }

  if (args->top_enabled != 0U && args->top != 0 && args->top_palette != 0) {
    unsigned top_index = args->top[y * args->width + x];
    if (top_index != 255U) {
      color = args->top_palette[top_index];
    }
  }

  args->dst[y * args->width + x] = color;
}
