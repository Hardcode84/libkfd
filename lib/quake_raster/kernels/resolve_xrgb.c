#include "libkfd/gpu/kernel.h"

struct QrResolveXrgbArgs {
  unsigned char *src;
  unsigned *dst;
  unsigned *palette;
  unsigned width;
  unsigned height;
};

KFD_GPU_KERNEL void qr_resolve_xrgb(struct QrResolveXrgbArgs *args)
{
  unsigned x = kfd_global_id_x();
  unsigned y = kfd_global_id_y();

  if (x >= args->width || y >= args->height) {
    return;
  }
  unsigned index = args->src[y * args->width + x];
  args->dst[y * args->width + x] = args->palette[index];
}
