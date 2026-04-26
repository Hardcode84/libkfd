#include "libkfd/gpu/kernel.h"

struct QrClearIndexedArgs {
  unsigned char *dst;
  unsigned width;
  unsigned height;
  unsigned color;
};

KFD_GPU_KERNEL void qr_clear_indexed(struct QrClearIndexedArgs *args)
{
  unsigned x = kfd_global_id_x();
  unsigned y = kfd_global_id_y();

  if (x >= args->width || y >= args->height) {
    return;
  }
  args->dst[y * args->width + x] = (unsigned char)args->color;
}
