//===-- libkfd/gpu/kernel.h - Device-side MVP helpers ----------*- C/C++
//-*-===//
//
// Small header for freestanding AMDGPU C/C++ kernels used by the MVP API.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_GPU_KERNEL_H
#define LIBKFD_GPU_KERNEL_H

#include <gpuintrin.h>

#define KFD_GPU_KERNEL __attribute__((amdgpu_kernel))

static inline unsigned kfd_thread_id_x(void) { return __gpu_thread_id_x(); }
static inline unsigned kfd_thread_id_y(void) { return __gpu_thread_id_y(); }
static inline unsigned kfd_thread_id_z(void) { return __gpu_thread_id_z(); }

static inline unsigned kfd_block_id_x(void) { return __gpu_block_id_x(); }
static inline unsigned kfd_block_id_y(void) { return __gpu_block_id_y(); }
static inline unsigned kfd_block_id_z(void) { return __gpu_block_id_z(); }

static inline unsigned kfd_block_dim_x(void) { return __gpu_num_threads_x(); }
static inline unsigned kfd_block_dim_y(void) { return __gpu_num_threads_y(); }
static inline unsigned kfd_block_dim_z(void) { return __gpu_num_threads_z(); }

static inline unsigned kfd_global_id_x(void) {
  return kfd_thread_id_x() + kfd_block_id_x() * kfd_block_dim_x();
}

static inline unsigned kfd_global_id_y(void) {
  return kfd_thread_id_y() + kfd_block_id_y() * kfd_block_dim_y();
}

static inline unsigned kfd_global_id_z(void) {
  return kfd_thread_id_z() + kfd_block_id_z() * kfd_block_dim_z();
}

#endif // LIBKFD_GPU_KERNEL_H
