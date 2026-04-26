/*===-- libkfd/gpu.h - Minimal compute-first GPU API --------------*- C -*-===*\
 *
 * Pure C convenience layer over the lower-level C++ libkfd primitives. The
 * model is intentionally simple: allocate GPU memory, load Clang-produced
 * AMDGPU ELFs, pass one root GPU pointer to a kernel, dispatch, signal, and
 * present exported buffers through platform code.
 *
\*===----------------------------------------------------------------------===*/

#ifndef LIBKFD_GPU_H
#define LIBKFD_GPU_H

#include <limits.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kfd_gpu_context kfd_gpu_context;
typedef struct kfd_gpu_buffer kfd_gpu_buffer;
typedef struct kfd_gpu_surface kfd_gpu_surface;
typedef struct kfd_gpu_module kfd_gpu_module;
typedef struct kfd_gpu_kernel kfd_gpu_kernel;
typedef struct kfd_gpu_fence kfd_gpu_fence;

#if UINT_MAX == 0xffffffffU
typedef unsigned int kfd_gpu_u32;
#else
typedef unsigned long kfd_gpu_u32;
#endif

#if ULONG_MAX > 0xffffffffUL
typedef unsigned long kfd_gpu_u64;
#else
typedef unsigned long long kfd_gpu_u64;
#endif

typedef enum kfd_gpu_memory_type {
  KFD_GPU_MEMORY_UPLOAD = 0,
  KFD_GPU_MEMORY_DEVICE = 1,
  KFD_GPU_MEMORY_READBACK = 2
} kfd_gpu_memory_type;

enum kfd_gpu_memory_flags {
  KFD_GPU_MEMORY_NONE = 0,
  KFD_GPU_MEMORY_WRITABLE = 1u << 0,
  KFD_GPU_MEMORY_EXECUTABLE = 1u << 1,
  KFD_GPU_MEMORY_COHERENT = 1u << 2,
  KFD_GPU_MEMORY_UNCACHED = 1u << 3,
  KFD_GPU_MEMORY_EXPORTABLE = 1u << 4
};

typedef enum kfd_gpu_format { KFD_GPU_FORMAT_XRGB8 = 0 } kfd_gpu_format;

typedef struct kfd_gpu_dim3 {
  kfd_gpu_u32 x;
  kfd_gpu_u32 y;
  kfd_gpu_u32 z;
} kfd_gpu_dim3;

typedef struct kfd_gpu_dispatch_config {
  kfd_gpu_dim3 grid;
  kfd_gpu_dim3 block;
  kfd_gpu_u32 dynamic_lds;
  kfd_gpu_u32 private_segment_size;
} kfd_gpu_dispatch_config;

const char *kfd_gpu_last_error(void);
const char *kfd_gpu_strerror(int code);

int kfd_gpu_context_create(size_t device_index, kfd_gpu_context **out);
int kfd_gpu_context_create_compatible(const void *image, size_t image_size,
                                      kfd_gpu_context **out);
void kfd_gpu_context_destroy(kfd_gpu_context *ctx);

const char *kfd_gpu_context_device_name(kfd_gpu_context *ctx);
kfd_gpu_u32 kfd_gpu_context_gfx_version(kfd_gpu_context *ctx);

int kfd_gpu_buffer_create(kfd_gpu_context *ctx, size_t size,
                          kfd_gpu_memory_type type, kfd_gpu_u32 flags,
                          kfd_gpu_buffer **out);
void kfd_gpu_buffer_destroy(kfd_gpu_buffer *buffer);
size_t kfd_gpu_buffer_size(kfd_gpu_buffer *buffer);
void *kfd_gpu_buffer_cpu(kfd_gpu_buffer *buffer);
void *kfd_gpu_buffer_gpu(kfd_gpu_buffer *buffer);

int kfd_gpu_surface_create(kfd_gpu_context *ctx, kfd_gpu_u32 width,
                           kfd_gpu_u32 height, kfd_gpu_format format,
                           kfd_gpu_surface **out);
void kfd_gpu_surface_destroy(kfd_gpu_surface *surface);
kfd_gpu_buffer *kfd_gpu_surface_buffer(kfd_gpu_surface *surface);
kfd_gpu_u32 kfd_gpu_surface_width(kfd_gpu_surface *surface);
kfd_gpu_u32 kfd_gpu_surface_height(kfd_gpu_surface *surface);
kfd_gpu_u32 kfd_gpu_surface_stride(kfd_gpu_surface *surface);
size_t kfd_gpu_surface_size(kfd_gpu_surface *surface);
int kfd_gpu_surface_dmabuf_fd(kfd_gpu_surface *surface);

int kfd_gpu_module_load(kfd_gpu_context *ctx, const void *image,
                        size_t image_size, kfd_gpu_module **out);
void kfd_gpu_module_destroy(kfd_gpu_module *module);

int kfd_gpu_module_kernel(kfd_gpu_module *module, const char *name,
                          kfd_gpu_kernel **out);
void kfd_gpu_kernel_destroy(kfd_gpu_kernel *kernel);

int kfd_gpu_kernel_alloc_args(kfd_gpu_kernel *kernel, kfd_gpu_buffer **out);
void kfd_gpu_kernel_set_root(kfd_gpu_kernel *kernel, kfd_gpu_buffer *args,
                             void *root_gpu,
                             const kfd_gpu_dispatch_config *cfg);

int kfd_gpu_fence_create(kfd_gpu_context *ctx, kfd_gpu_u64 initial,
                         kfd_gpu_fence **out);
void kfd_gpu_fence_destroy(kfd_gpu_fence *fence);
int kfd_gpu_fence_reset(kfd_gpu_fence *fence, kfd_gpu_u64 value);
int kfd_gpu_fence_wait(kfd_gpu_fence *fence, kfd_gpu_u64 value,
                       kfd_gpu_u64 timeout_ns);

int kfd_gpu_copy(kfd_gpu_context *ctx, void *dst_gpu, const void *src_gpu,
                 size_t bytes);
int kfd_gpu_fill32(kfd_gpu_context *ctx, void *dst_gpu, kfd_gpu_u32 value,
                   kfd_gpu_u32 bytes);
int kfd_gpu_acquire_mem(kfd_gpu_context *ctx);
int kfd_gpu_dispatch(kfd_gpu_context *ctx, kfd_gpu_kernel *kernel,
                     const kfd_gpu_dispatch_config *cfg, kfd_gpu_buffer *args,
                     kfd_gpu_fence *completion);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBKFD_GPU_H */
