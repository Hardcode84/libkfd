//===-- lib/gpu.cpp - Minimal compute-first GPU API ----------------------===//
//
// Convenience wrappers over the lower-level libkfd primitives.
//
//===----------------------------------------------------------------------===//

#include "libkfd/gpu.h"

#include "libkfd/condition.h"
#include "libkfd/context.h"
#include "libkfd/detail/box.h"
#include "libkfd/device.h"
#include "libkfd/dispatch.h"
#include "libkfd/error.h"
#include "libkfd/loader.h"
#include "libkfd/memory.h"
#include "libkfd/queue.h"
#include "libkfd/signal.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <span>
#include <utility>

struct kfd_gpu_context {
  kfd_gpu_context(kfd::detail::Box<kfd::Context> &&ctx, size_t device_index,
                  kfd::ComputeQueue &&compute, kfd::SDMAQueue &&sdma)
      : ctx(std::move(ctx)), index(device_index), compute(std::move(compute)),
        sdma(std::move(sdma)) {
    refresh_device();
  }

  void refresh_device() { dev = &ctx->devices()[index]; }

  kfd::detail::Box<kfd::Context> ctx;
  size_t index = 0;
  kfd::Device *dev = nullptr;
  kfd::ComputeQueue compute;
  kfd::SDMAQueue sdma;
};

struct kfd_gpu_buffer {
  kfd_gpu_buffer(kfd::Buffer &&raw, bool host_visible)
      : raw(std::move(raw)), host_visible(host_visible) {}

  kfd::Buffer raw;
  bool host_visible = false;
};

struct kfd_gpu_surface {
  kfd_gpu_surface(kfd_gpu_buffer &&image, kfd::DMABuffer &&dmabuf,
                  uint32_t width, uint32_t height, uint32_t stride)
      : image(std::move(image)), dmabuf(std::move(dmabuf)), width(width),
        height(height), stride(stride) {}

  kfd_gpu_buffer image;
  kfd::DMABuffer dmabuf;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t stride = 0;
};

struct kfd_gpu_module {
  explicit kfd_gpu_module(kfd::Executable &&executable)
      : executable(std::move(executable)) {}

  kfd::Executable executable;
};

struct kfd_gpu_kernel {
  explicit kfd_gpu_kernel(kfd::Kernel raw) : raw(raw) {}

  kfd::Kernel raw;
};

struct kfd_gpu_fence {
  explicit kfd_gpu_fence(kfd::Signal &&signal) : signal(std::move(signal)) {}

  kfd::Signal signal;
};

namespace {

thread_local char last_error[256] = {};

int set_error(const kfd::Error &err) {
  std::snprintf(last_error, sizeof(last_error), "%s", kfd::strerror(err));
  return err.code;
}

int set_error(int code, const char *msg) {
  std::snprintf(last_error, sizeof(last_error), "%s: %s", msg,
                std::strerror(code));
  return code;
}

int clear_error() {
  last_error[0] = '\0';
  return 0;
}

template <typename T, typename... Args> T *make_handle(Args &&...args) {
  constexpr size_t size = (sizeof(T) + alignof(T) - 1) & ~(alignof(T) - 1);
  void *mem = std::aligned_alloc(alignof(T), size);
  if (!mem)
    return nullptr;
  return ::new (mem) T(std::forward<Args>(args)...);
}

template <typename T> void destroy_handle(T *handle) {
  if (!handle)
    return;
  handle->~T();
  std::free(handle);
}

kfd::MemFlags convert_flags(kfd_gpu_memory_type type, uint32_t flags) {
  kfd::MemFlags out = kfd::MemFlags::NONE;

  if (flags & KFD_GPU_MEMORY_WRITABLE)
    out = out | kfd::MemFlags::WRITABLE;
  if (flags & KFD_GPU_MEMORY_EXECUTABLE)
    out = out | kfd::MemFlags::EXECUTABLE;
  if (flags & KFD_GPU_MEMORY_COHERENT)
    out = out | kfd::MemFlags::COHERENT;
  if (flags & KFD_GPU_MEMORY_UNCACHED)
    out = out | kfd::MemFlags::UNCACHED;

  if (type != KFD_GPU_MEMORY_DEVICE)
    out = out | kfd::MemFlags::HOST_ACCESS;

  return out;
}

kfd::MemType convert_type(kfd_gpu_memory_type type) {
  switch (type) {
  case KFD_GPU_MEMORY_UPLOAD:
  case KFD_GPU_MEMORY_READBACK:
    return kfd::MemType::GTT;
  case KFD_GPU_MEMORY_DEVICE:
    return kfd::MemType::VRAM;
  }
  return kfd::MemType::GTT;
}

kfd::DispatchConfig convert_dispatch(const kfd_gpu_dispatch_config *cfg) {
  return kfd::DispatchConfig{
      .grid = {.x = cfg->grid.x ? cfg->grid.x : 1,
               .y = cfg->grid.y ? cfg->grid.y : 1,
               .z = cfg->grid.z ? cfg->grid.z : 1},
      .block = {.x = cfg->block.x ? cfg->block.x : 1,
                .y = cfg->block.y ? cfg->block.y : 1,
                .z = cfg->block.z ? cfg->block.z : 1},
      .dynamic_lds = cfg->dynamic_lds,
      .private_segment_size = cfg->private_segment_size,
  };
}

int make_buffer(kfd_gpu_context *ctx, size_t size, kfd_gpu_memory_type type,
                uint32_t flags, kfd_gpu_buffer **out) {
  if (!ctx || !out)
    return set_error(EINVAL, "invalid buffer_create argument");
  *out = nullptr;

  auto raw = kfd::Buffer::allocate(*ctx->dev, size, convert_type(type),
                                   convert_flags(type, flags));
  if (!raw)
    return set_error(raw.error());
  if (auto r = raw->map(*ctx->dev); !r)
    return set_error(r.error());

  bool host_visible = type != KFD_GPU_MEMORY_DEVICE;
  auto *buf = make_handle<kfd_gpu_buffer>(std::move(*raw), host_visible);
  if (!buf)
    return set_error(ENOMEM, "failed to allocate buffer handle");
  *out = buf;
  return clear_error();
}

} // namespace

extern "C" const char *kfd_gpu_last_error(void) {
  return last_error[0] ? last_error : "success";
}

extern "C" const char *kfd_gpu_strerror(int code) {
  return code == 0 ? "success" : std::strerror(code);
}

extern "C" int kfd_gpu_context_create(size_t device_index,
                                      kfd_gpu_context **out) {
  if (!out)
    return set_error(EINVAL, "invalid context_create argument");
  *out = nullptr;

  auto raw_ctx = kfd::Context::create();
  if (!raw_ctx)
    return set_error(raw_ctx.error());
  auto boxed_ctx = kfd::detail::Box<kfd::Context>::create(std::move(*raw_ctx));
  if (!boxed_ctx)
    return set_error(boxed_ctx.error());
  auto dev = (*boxed_ctx)->device(device_index);
  if (!dev)
    return set_error(dev.error());
  auto compute = kfd::ComputeQueue::create(**dev);
  if (!compute)
    return set_error(compute.error());
  auto sdma = kfd::SDMAQueue::create(**dev);
  if (!sdma)
    return set_error(sdma.error());

  auto *ctx =
      make_handle<kfd_gpu_context>(std::move(*boxed_ctx), device_index,
                                   std::move(*compute), std::move(*sdma));
  if (!ctx)
    return set_error(ENOMEM, "failed to allocate context handle");
  *out = ctx;
  return clear_error();
}

extern "C" int kfd_gpu_context_create_compatible(const void *image,
                                                 size_t image_size,
                                                 kfd_gpu_context **out) {
  if (!image || !out)
    return set_error(EINVAL, "invalid context_create_compatible argument");
  *out = nullptr;

  auto raw_ctx = kfd::Context::create();
  if (!raw_ctx)
    return set_error(raw_ctx.error());
  auto boxed_ctx = kfd::detail::Box<kfd::Context>::create(std::move(*raw_ctx));
  if (!boxed_ctx)
    return set_error(boxed_ctx.error());

  auto bytes =
      std::span(reinterpret_cast<const std::byte *>(image), image_size);
  for (size_t i = 0; i < (*boxed_ctx)->num_devices(); ++i) {
    auto dev = (*boxed_ctx)->device(i);
    if (!dev)
      return set_error(dev.error());
    if (!(**dev).loadable(bytes))
      continue;

    auto compute = kfd::ComputeQueue::create(**dev);
    if (!compute)
      return set_error(compute.error());
    auto sdma = kfd::SDMAQueue::create(**dev);
    if (!sdma)
      return set_error(sdma.error());

    auto *ctx = make_handle<kfd_gpu_context>(
        std::move(*boxed_ctx), i, std::move(*compute), std::move(*sdma));
    if (!ctx)
      return set_error(ENOMEM, "failed to allocate context handle");
    *out = ctx;
    return clear_error();
  }

  return set_error(ENOEXEC, "no device is compatible with module image");
}

extern "C" void kfd_gpu_context_destroy(kfd_gpu_context *ctx) {
  destroy_handle(ctx);
}

extern "C" const char *kfd_gpu_context_device_name(kfd_gpu_context *ctx) {
  if (!ctx)
    return "";
  auto name = ctx->dev->get_name();
  return name.data();
}

extern "C" uint32_t kfd_gpu_context_gfx_version(kfd_gpu_context *ctx) {
  return ctx ? ctx->dev->properties().gfx_target_version : 0;
}

extern "C" int kfd_gpu_buffer_create(kfd_gpu_context *ctx, size_t size,
                                     kfd_gpu_memory_type type, uint32_t flags,
                                     kfd_gpu_buffer **out) {
  return make_buffer(ctx, size, type, flags, out);
}

extern "C" void kfd_gpu_buffer_destroy(kfd_gpu_buffer *buffer) {
  destroy_handle(buffer);
}

extern "C" size_t kfd_gpu_buffer_size(kfd_gpu_buffer *buffer) {
  return buffer ? buffer->raw.size() : 0;
}

extern "C" void *kfd_gpu_buffer_cpu(kfd_gpu_buffer *buffer) {
  return buffer && buffer->host_visible ? buffer->raw.data() : nullptr;
}

extern "C" void *kfd_gpu_buffer_gpu(kfd_gpu_buffer *buffer) {
  return buffer ? buffer->raw.data() : nullptr;
}

extern "C" int kfd_gpu_surface_create(kfd_gpu_context *ctx, uint32_t width,
                                      uint32_t height, kfd_gpu_format format,
                                      kfd_gpu_surface **out) {
  if (!ctx || !out)
    return set_error(EINVAL, "invalid surface_create argument");
  *out = nullptr;
  if (format != KFD_GPU_FORMAT_XRGB8)
    return set_error(EINVAL, "only XRGB8 surfaces are supported");

  uint32_t stride = width * sizeof(uint32_t);
  size_t bytes = static_cast<size_t>(stride) * height;

  kfd_gpu_buffer *image_ptr = nullptr;
  int err = make_buffer(ctx, bytes, KFD_GPU_MEMORY_DEVICE,
                        KFD_GPU_MEMORY_WRITABLE | KFD_GPU_MEMORY_EXPORTABLE,
                        &image_ptr);
  if (err)
    return err;
  kfd_gpu_buffer image = std::move(*image_ptr);
  destroy_handle(image_ptr);

  auto dmabuf = kfd::DMABuffer::create(image.raw);
  if (!dmabuf)
    return set_error(dmabuf.error());

  auto *surface = make_handle<kfd_gpu_surface>(
      std::move(image), std::move(*dmabuf), width, height, stride);
  if (!surface)
    return set_error(ENOMEM, "failed to allocate surface handle");
  *out = surface;
  return clear_error();
}

extern "C" void kfd_gpu_surface_destroy(kfd_gpu_surface *surface) {
  destroy_handle(surface);
}

extern "C" kfd_gpu_buffer *kfd_gpu_surface_buffer(kfd_gpu_surface *surface) {
  return surface ? &surface->image : nullptr;
}

extern "C" uint32_t kfd_gpu_surface_width(kfd_gpu_surface *surface) {
  return surface ? surface->width : 0;
}

extern "C" uint32_t kfd_gpu_surface_height(kfd_gpu_surface *surface) {
  return surface ? surface->height : 0;
}

extern "C" uint32_t kfd_gpu_surface_stride(kfd_gpu_surface *surface) {
  return surface ? surface->stride : 0;
}

extern "C" size_t kfd_gpu_surface_size(kfd_gpu_surface *surface) {
  return surface ? surface->image.raw.size() : 0;
}

extern "C" int kfd_gpu_surface_dmabuf_fd(kfd_gpu_surface *surface) {
  return surface ? surface->dmabuf.fd() : -1;
}

extern "C" int kfd_gpu_module_load(kfd_gpu_context *ctx, const void *image,
                                   size_t image_size, kfd_gpu_module **out) {
  if (!ctx || !image || !out)
    return set_error(EINVAL, "invalid module_load argument");
  *out = nullptr;

  auto bytes =
      std::span(reinterpret_cast<const std::byte *>(image), image_size);
  auto exe = kfd::Executable::load(*ctx->dev, bytes, ctx->sdma, ctx->compute);
  if (!exe)
    return set_error(exe.error());

  auto *module = make_handle<kfd_gpu_module>(std::move(*exe));
  if (!module)
    return set_error(ENOMEM, "failed to allocate module handle");
  *out = module;
  return clear_error();
}

extern "C" void kfd_gpu_module_destroy(kfd_gpu_module *module) {
  destroy_handle(module);
}

extern "C" int kfd_gpu_module_kernel(kfd_gpu_module *module, const char *name,
                                     kfd_gpu_kernel **out) {
  if (!module || !name || !out)
    return set_error(EINVAL, "invalid module_kernel argument");
  *out = nullptr;

  auto kernel = module->executable.kernel(name);
  if (!kernel)
    return set_error(kernel.error());

  auto *handle = make_handle<kfd_gpu_kernel>(*kernel);
  if (!handle)
    return set_error(ENOMEM, "failed to allocate kernel handle");
  *out = handle;
  return clear_error();
}

extern "C" void kfd_gpu_kernel_destroy(kfd_gpu_kernel *kernel) {
  destroy_handle(kernel);
}

extern "C" int kfd_gpu_kernel_alloc_args(kfd_gpu_kernel *kernel,
                                         kfd_gpu_buffer **out) {
  if (!kernel || !out)
    return set_error(EINVAL, "invalid kernel_alloc_args argument");
  *out = nullptr;

  auto args = kernel->raw.alloc();
  if (!args)
    return set_error(args.error());

  auto *buffer = make_handle<kfd_gpu_buffer>(std::move(*args), true);
  if (!buffer)
    return set_error(ENOMEM, "failed to allocate buffer handle");
  *out = buffer;
  return clear_error();
}

extern "C" void kfd_gpu_kernel_set_root(kfd_gpu_kernel *kernel,
                                        kfd_gpu_buffer *args, void *root_gpu,
                                        const kfd_gpu_dispatch_config *cfg) {
  if (!kernel || !args || !cfg)
    return;

  struct RootArgs {
    void *root;
  };
  RootArgs root{root_gpu};
  auto dispatch = convert_dispatch(cfg);
  kernel->raw.fill(args->raw, root, dispatch);
}

extern "C" int kfd_gpu_fence_create(kfd_gpu_context *ctx, uint64_t initial,
                                    kfd_gpu_fence **out) {
  if (!ctx || !out)
    return set_error(EINVAL, "invalid fence_create argument");
  *out = nullptr;

  auto signal = kfd::Signal::create(*ctx->ctx, initial);
  if (!signal)
    return set_error(signal.error());

  auto *fence = make_handle<kfd_gpu_fence>(std::move(*signal));
  if (!fence)
    return set_error(ENOMEM, "failed to allocate fence handle");
  *out = fence;
  return clear_error();
}

extern "C" void kfd_gpu_fence_destroy(kfd_gpu_fence *fence) {
  destroy_handle(fence);
}

extern "C" int kfd_gpu_fence_reset(kfd_gpu_fence *fence, uint64_t value) {
  if (!fence)
    return set_error(EINVAL, "invalid fence_reset argument");
  auto r = fence->signal.reset(value);
  return r ? clear_error() : set_error(r.error());
}

extern "C" int kfd_gpu_fence_wait(kfd_gpu_fence *fence, uint64_t value,
                                  uint64_t timeout_ns) {
  if (!fence)
    return set_error(EINVAL, "invalid fence_wait argument");
  auto r = fence->signal.wait(kfd::Condition::EQ, value, timeout_ns);
  return r ? clear_error() : set_error(r.error());
}

extern "C" int kfd_gpu_copy(kfd_gpu_context *ctx, void *dst_gpu,
                            const void *src_gpu, size_t bytes) {
  if (!ctx)
    return set_error(EINVAL, "invalid copy argument");
  auto r = ctx->sdma.copy_linear(dst_gpu, src_gpu, bytes);
  return r ? clear_error() : set_error(r.error());
}

extern "C" int kfd_gpu_fill32(kfd_gpu_context *ctx, void *dst_gpu,
                              uint32_t value, uint32_t bytes) {
  if (!ctx)
    return set_error(EINVAL, "invalid fill32 argument");
  auto r = ctx->sdma.const_fill(dst_gpu, value, bytes);
  return r ? clear_error() : set_error(r.error());
}

extern "C" int kfd_gpu_dispatch(kfd_gpu_context *ctx, kfd_gpu_kernel *kernel,
                                const kfd_gpu_dispatch_config *cfg,
                                kfd_gpu_buffer *args,
                                kfd_gpu_fence *completion) {
  if (!ctx || !kernel || !cfg || !args)
    return set_error(EINVAL, "invalid dispatch argument");

  auto dispatch = convert_dispatch(cfg);
  if (completion) {
    if (auto r = completion->signal.reset(); !r)
      return set_error(r.error());
    auto r = ctx->compute.dispatch(kernel->raw, dispatch, args->raw,
                                   completion->signal);
    return r ? clear_error() : set_error(r.error());
  }

  auto r = ctx->compute.dispatch(kernel->raw, dispatch, args->raw);
  return r ? clear_error() : set_error(r.error());
}
