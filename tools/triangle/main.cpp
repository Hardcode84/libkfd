//===-- tools/triangle/main.cpp - MVP API colored triangle demo -*- C++ -*-===//
//
// Demonstrates the minimal compute-first GPU API by drawing a colored triangle
// in a compute kernel and presenting the result through DRI3/Present.
//
// $ triangle [WIDTHxHEIGHT]
//
//===----------------------------------------------------------------------===//

#include "window.h"

#include "libkfd/gpu.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t NUM_BUFFERS = 2;
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t BLOCK_Y = 16;

struct DemoKernel {
  const char *path;
  const char *arch;
};

const DemoKernel demo_kernels[] = {
#include "triangle_kernels.inc"
};

struct Vertex {
  float x;
  float y;
  float r;
  float g;
  float b;
};

struct TriangleArgs {
  unsigned *framebuffer;
  unsigned width;
  unsigned height;
  float time;
  Vertex v0;
  Vertex v1;
  Vertex v2;
};

struct Frame {
  kfd_gpu_surface *surface = nullptr;
  kfd_gpu_buffer *root = nullptr;
  kfd_gpu_buffer *kernarg = nullptr;
  kfd_gpu_fence *fence = nullptr;
};

struct LoadedDemo {
  kfd_gpu_context *ctx = nullptr;
  kfd_gpu_module *module = nullptr;
  kfd_gpu_kernel *kernel = nullptr;
};

void check(int err) {
  if (err) {
    std::fprintf(stderr, "error: %s\n", kfd_gpu_last_error());
    std::exit(1);
  }
}

std::vector<std::byte> read_file(const char *path) {
  std::FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "warning: cannot open '%s': %s\n", path,
                 std::strerror(errno));
    return {};
  }
  std::fseek(f, 0, SEEK_END);
  auto sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<std::byte> buf(static_cast<size_t>(sz));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    std::fclose(f);
    std::fprintf(stderr, "warning: short read on '%s'\n", path);
    return {};
  }
  std::fclose(f);
  return buf;
}

void parse_resolution(const char *str, uint32_t &w, uint32_t &h) {
  if (std::sscanf(str, "%ux%u", &w, &h) != 2) {
    std::fprintf(
        stderr, "error: invalid resolution '%s', expected WIDTHxHEIGHT\n", str);
    std::exit(1);
  }
}

bool load_demo(LoadedDemo &out) {
  for (const auto &candidate : demo_kernels) {
    auto image = read_file(candidate.path);
    if (image.empty())
      continue;

    kfd_gpu_context *ctx = nullptr;
    int err =
        kfd_gpu_context_create_compatible(image.data(), image.size(), &ctx);
    if (err)
      continue;

    kfd_gpu_module *module = nullptr;
    err = kfd_gpu_module_load(ctx, image.data(), image.size(), &module);
    if (err) {
      kfd_gpu_context_destroy(ctx);
      continue;
    }

    kfd_gpu_kernel *kernel = nullptr;
    err = kfd_gpu_module_kernel(module, "colored_triangle.kd", &kernel);
    if (err) {
      kfd_gpu_module_destroy(module);
      kfd_gpu_context_destroy(ctx);
      continue;
    }

    std::printf("Loaded %s kernel: %s\n", candidate.arch, candidate.path);
    out = LoadedDemo{ctx, module, kernel};
    return true;
  }
  return false;
}

Vertex make_vertex(float cx, float cy, float radius, float angle, float r,
                   float g, float b) {
  return Vertex{cx + radius * std::cos(angle), cy + radius * std::sin(angle), r,
                g, b};
}

void fill_triangle_args(TriangleArgs &args, void *framebuffer, uint32_t width,
                        uint32_t height, float time) {
  float cx = 0.5f * static_cast<float>(width);
  float cy = 0.5f * static_cast<float>(height);
  float radius = 0.38f * static_cast<float>(width < height ? width : height);
  float angle = time * 0.7f;

  args.framebuffer = static_cast<unsigned *>(framebuffer);
  args.width = width;
  args.height = height;
  args.time = time;
  args.v0 = make_vertex(cx, cy, radius, angle - 1.5707963f, 1.0f, 0.1f, 0.1f);
  args.v1 = make_vertex(cx, cy, radius, angle + 2.6179939f, 0.1f, 1.0f, 0.1f);
  args.v2 = make_vertex(cx, cy, radius, angle + 0.5235988f, 0.1f, 0.3f, 1.0f);
}

} // namespace

int main(int argc, char **argv) {
  uint32_t width = 1280;
  uint32_t height = 720;
  if (argc > 1)
    parse_resolution(argv[1], width, height);

  LoadedDemo demo;
  if (!load_demo(demo)) {
    std::fprintf(stderr,
                 "error: no compatible colored_triangle kernel found\n");
    return 1;
  }
  auto *ctx = demo.ctx;

  std::printf("GPU: %s (gfx%u)\n", kfd_gpu_context_device_name(ctx),
              kfd_gpu_context_gfx_version(ctx));

  kfd_gpu_dispatch_config cfg{
      .grid = {.x = (width + BLOCK_X - 1) / BLOCK_X,
               .y = (height + BLOCK_Y - 1) / BLOCK_Y,
               .z = 1},
      .block = {.x = BLOCK_X, .y = BLOCK_Y, .z = 1},
      .dynamic_lds = 0,
      .private_segment_size = 0,
  };

  std::array<Frame, NUM_BUFFERS> frames;
  for (auto &frame : frames) {
    check(kfd_gpu_surface_create(ctx, width, height, KFD_GPU_FORMAT_XRGB8,
                                 &frame.surface));
    check(kfd_gpu_buffer_create(
        ctx, sizeof(TriangleArgs), KFD_GPU_MEMORY_UPLOAD,
        KFD_GPU_MEMORY_WRITABLE | KFD_GPU_MEMORY_UNCACHED, &frame.root));
    check(kfd_gpu_kernel_alloc_args(demo.kernel, &frame.kernarg));
    kfd_gpu_kernel_set_root(demo.kernel, frame.kernarg,
                            kfd_gpu_buffer_gpu(frame.root), &cfg);
    check(kfd_gpu_fence_create(ctx, 1, &frame.fence));
  }

  auto win =
      KFD_EXPECT(Window::create(width, height, NUM_BUFFERS, "libkfd triangle"));
  for (uint32_t i = 0; i < NUM_BUFFERS; ++i) {
    KFD_EXPECT(win->import_buffer(i,
                                  kfd_gpu_surface_dmabuf_fd(frames[i].surface),
                                  kfd_gpu_surface_size(frames[i].surface),
                                  kfd_gpu_surface_stride(frames[i].surface)));
  }

  uint32_t current = 0;
  uint32_t frame_no = 0;
  auto start = std::chrono::high_resolution_clock::now();

  std::printf("Entering triangle loop at %ux%u. Press 'q' to quit.\n", width,
              height);

  while (win->poll()) {
    win->wait_idle(current);

    auto now = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(now - start).count();

    auto &frame = frames[current];
    auto *root = static_cast<TriangleArgs *>(kfd_gpu_buffer_cpu(frame.root));
    fill_triangle_args(
        *root, kfd_gpu_buffer_gpu(kfd_gpu_surface_buffer(frame.surface)), width,
        height, time);

    check(kfd_gpu_dispatch(ctx, demo.kernel, &cfg, frame.kernarg, frame.fence));
    check(kfd_gpu_fence_wait(frame.fence, 0, UINT64_MAX));

    win->present(current);
    current = (current + 1) % NUM_BUFFERS;
    ++frame_no;
  }

  std::printf("Exiting after %u frames.\n", frame_no);
  for (auto &frame : frames) {
    kfd_gpu_fence_destroy(frame.fence);
    kfd_gpu_buffer_destroy(frame.kernarg);
    kfd_gpu_buffer_destroy(frame.root);
    kfd_gpu_surface_destroy(frame.surface);
  }
  kfd_gpu_kernel_destroy(demo.kernel);
  kfd_gpu_module_destroy(demo.module);
  kfd_gpu_context_destroy(demo.ctx);
  return 0;
}
