#include "test_helpers.h"

#include "libkfd/abi.h"

#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

static const kfd::test::TestBinary raster_demo_kernels[] = {
#include "raster_demo_kernels.inc"
};

using kfd::test::make_device_fixture;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

namespace {

constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t BLOCK_Y = 16;
constexpr uint32_t CLEAR_COLOR = 0xff102030u;
constexpr float CLEAR_DEPTH = 1.0f;
constexpr uint32_t DARK_CHECKER = 0xff202020u;
constexpr uint32_t LIGHT_CHECKER = 0xffe0e0e0u;

struct DemoVertex {
  float x;
  float y;
  float z;
  float u;
  float v;
};

struct DemoPrimitive {
  DemoVertex v0;
  DemoVertex v1;
  DemoVertex v2;
};

struct RasterArgs {
  const DemoPrimitive *prims;
  uint32_t prim_count;
  uint32_t *color;
  float *depth;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;
  uint32_t tile_size;
  uint32_t clear_color;
  float clear_depth;
};

struct PersistentControl {
  uint32_t terminate;
  uint32_t frame_id;
  uint32_t prim_count;
  uint32_t tiles_done;
  uint32_t rendered;
  uint32_t ready;
  uint32_t heartbeat;
  uint32_t _pad;
};

struct PersistentRasterArgs {
  const DemoPrimitive *prims;
  uint32_t *color;
  float *depth;
  PersistentControl *control;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;
  uint32_t tile_size;
  uint32_t clear_color;
  float clear_depth;
};

static_assert(sizeof(DemoVertex) == 20);
static_assert(sizeof(DemoPrimitive) == 60);
static_assert(sizeof(PersistentControl) == 32);

struct RenderResult {
  std::vector<uint32_t> color;
  std::vector<float> depth;
};

DemoPrimitive tri(DemoVertex a, DemoVertex b, DemoVertex c) {
  return DemoPrimitive{.v0 = a, .v1 = b, .v2 = c};
}

void add_quad(std::vector<DemoPrimitive> &prims, float x0, float y0, float x1,
              float y1, float z, float u0, float v0, float u1, float v1) {
  DemoVertex a{.x = x0, .y = y0, .z = z, .u = u0, .v = v0};
  DemoVertex b{.x = x1, .y = y0, .z = z, .u = u1, .v = v0};
  DemoVertex c{.x = x1, .y = y1, .z = z, .u = u1, .v = v1};
  DemoVertex d{.x = x0, .y = y1, .z = z, .u = u0, .v = v1};
  prims.push_back(tri(a, b, d));
  prims.push_back(tri(b, c, d));
}

uint32_t expected_checker(float u, float v) {
  uint32_t iu = static_cast<uint32_t>(u * 8.0f);
  uint32_t iv = static_cast<uint32_t>(v * 8.0f);
  return ((iu ^ iv) & 1u) ? LIGHT_CHECKER : DARK_CHECKER;
}

float fsin(float x) { return __builtin_sinf(x); }
float fcos(float x) { return __builtin_cosf(x); }

std::vector<DemoPrimitive> make_demo_workload(uint32_t width, uint32_t height) {
  constexpr uint32_t SLICES = 64;
  constexpr uint32_t RINGS = 48;
  constexpr float PI = 3.14159265358979323846f;
  constexpr float TAU = 6.28318530717958647692f;
  constexpr float ANGLE = 0.65f;

  std::vector<DemoVertex> verts((RINGS + 1) * SLICES);
  auto at = [&](uint32_t ring, uint32_t slice) -> DemoVertex & {
    return verts[ring * SLICES + (slice % SLICES)];
  };

  for (uint32_t r = 0; r <= RINGS; ++r) {
    float t = static_cast<float>(r) / static_cast<float>(RINGS);
    float y = (t - 0.5f) * 2.0f;
    float radius = 0.42f + 0.10f * fcos((t * 3.0f + 0.2f) * PI) +
                   0.04f * fsin(t * 11.0f * PI);
    if (r == 0 || r == RINGS)
      radius = 0.05f;

    for (uint32_t s = 0; s < SLICES; ++s) {
      float u = static_cast<float>(s) / static_cast<float>(SLICES);
      float theta = u * TAU;
      float x = radius * fcos(theta);
      float z = radius * fsin(theta);

      float xr = x * fcos(ANGLE) + z * fsin(ANGLE);
      float zr = -x * fsin(ANGLE) + z * fcos(ANGLE);

      at(r, s) = DemoVertex{
          .x = static_cast<float>(width) * (0.5f + xr * 0.62f),
          .y = static_cast<float>(height) * (0.5f - y * 0.42f),
          .z = 0.48f + zr * 0.20f,
          .u = u,
          .v = t,
      };
    }
  }

  std::vector<DemoPrimitive> prims;
  prims.reserve(RINGS * SLICES * 2);
  for (uint32_t r = 0; r < RINGS; ++r) {
    for (uint32_t s = 0; s < SLICES; ++s) {
      DemoVertex a = at(r, s);
      DemoVertex b = at(r, s + 1);
      DemoVertex c = at(r + 1, s + 1);
      DemoVertex d = at(r + 1, s);
      prims.push_back(tri(a, b, d));
      prims.push_back(tri(b, c, d));
    }
  }
  return prims;
}

RenderResult render(kfd::test::DeviceFixture &fix,
                    std::span<const DemoPrimitive> prims, uint32_t width,
                    uint32_t height) {
  auto kernel = fix.exe.kernel("headless_stage0_raster.kd");
  REQUIRE_RESULT(kernel);

  uint32_t pitch = width;
  size_t pixel_count = static_cast<size_t>(pitch) * height;
  size_t color_bytes = pixel_count * sizeof(uint32_t);
  size_t depth_bytes = pixel_count * sizeof(float);
  size_t prim_bytes = prims.size_bytes();

  auto prim_buf = kfd::test::alloc_host_buffer(
      *fix.gpu,
      kfd::detail::align_up(prim_bytes ? prim_bytes : sizeof(DemoPrimitive),
                            kfd::detail::page_size()));
  std::memset(prim_buf.data(), 0, prim_buf.size());
  if (prim_bytes)
    std::memcpy(prim_buf.data(), prims.data(), prim_bytes);

  auto color = kfd::test::alloc_host_buffer(
      *fix.gpu, kfd::detail::align_up(color_bytes, kfd::detail::page_size()));
  auto depth = kfd::test::alloc_host_buffer(
      *fix.gpu, kfd::detail::align_up(depth_bytes, kfd::detail::page_size()));
  std::memset(color.data(), 0, color_bytes);
  std::memset(depth.data(), 0, depth_bytes);

  RasterArgs args{
      .prims = static_cast<const DemoPrimitive *>(prim_buf.data()),
      .prim_count = static_cast<uint32_t>(prims.size()),
      .color = static_cast<uint32_t *>(color.data()),
      .depth = static_cast<float *>(depth.data()),
      .width = width,
      .height = height,
      .pitch = pitch,
      .tile_size = TILE_SIZE,
      .clear_color = CLEAR_COLOR,
      .clear_depth = CLEAR_DEPTH,
  };

  kfd::DispatchConfig cfg{
      .grid = {.x = (width + TILE_SIZE - 1) / TILE_SIZE,
               .y = (height + TILE_SIZE - 1) / TILE_SIZE},
      .block = {.x = BLOCK_X, .y = BLOCK_Y},
  };
  auto kernarg = kernel->alloc();
  REQUIRE_RESULT(kernarg);
  kernel->fill(*kernarg, args, cfg);

  auto sig = kfd::Signal::create(fix.gpu->context());
  REQUIRE_RESULT(sig);
  REQUIRE_RESULT(fix.compute.dispatch(*kernel, cfg, *kernarg));
  REQUIRE_RESULT(fix.compute.signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

  RenderResult result{
      .color = std::vector<uint32_t>(pixel_count),
      .depth = std::vector<float>(pixel_count),
  };
  std::memcpy(result.color.data(), color.data(), color_bytes);
  std::memcpy(result.depth.data(), depth.data(), depth_bytes);
  return result;
}

bool wait_for_frame(PersistentControl *control, uint32_t expected_prims) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  for (;;) {
    uint32_t ready = __atomic_load_n(&control->ready, __ATOMIC_ACQUIRE);
    uint32_t rendered = __atomic_load_n(&control->rendered, __ATOMIC_ACQUIRE);
    if (ready == 0 && rendered == expected_prims)
      return true;
    if (std::chrono::steady_clock::now() > deadline)
      return false;
    kfd::detail::spin_hint();
  }
}

void for_each_fixture(auto &&body) {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, raster_demo_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);
      body(*fix);
    }
  }
}

} // namespace

TEST_CASE("Streaming raster demo - checker UV interpolation is pixel exact",
          "[device][raster-demo]") {
  for_each_fixture([](kfd::test::DeviceFixture &fix) {
    constexpr uint32_t WIDTH = 64;
    constexpr uint32_t HEIGHT = 64;

    std::vector<DemoPrimitive> prims;
    add_quad(prims, 0.0f, 0.0f, static_cast<float>(WIDTH),
             static_cast<float>(HEIGHT), 0.5f, 0.0f, 0.0f, 1.0f, 1.0f);

    auto result = render(fix, prims, WIDTH, HEIGHT);
    for (uint32_t y = 0; y < HEIGHT; ++y) {
      for (uint32_t x = 0; x < WIDTH; ++x) {
        float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(WIDTH);
        float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(HEIGHT);
        size_t idx = static_cast<size_t>(y) * WIDTH + x;
        INFO("pixel " << x << "," << y);
        CHECK(result.color[idx] == expected_checker(u, v));
        CHECK(result.depth[idx] > 0.49f);
        CHECK(result.depth[idx] < 0.51f);
      }
    }
  });
}

TEST_CASE("Streaming raster demo - closer primitives win the depth test",
          "[device][raster-demo]") {
  for_each_fixture([](kfd::test::DeviceFixture &fix) {
    constexpr uint32_t WIDTH = 48;
    constexpr uint32_t HEIGHT = 48;

    std::vector<DemoPrimitive> prims;
    add_quad(prims, 0.0f, 0.0f, static_cast<float>(WIDTH),
             static_cast<float>(HEIGHT), 0.8f, 0.0f, 0.0f, 1.0f, 1.0f);
    add_quad(prims, 12.0f, 12.0f, 36.0f, 36.0f, 0.2f, 0.125f, 0.0f, 1.125f,
             1.0f);

    auto result = render(fix, prims, WIDTH, HEIGHT);
    size_t center = static_cast<size_t>(24) * WIDTH + 24;
    CHECK(result.color[center] == LIGHT_CHECKER);
    CHECK(result.depth[center] > 0.19f);
    CHECK(result.depth[center] < 0.21f);

    size_t corner = static_cast<size_t>(4) * WIDTH + 4;
    CHECK(result.color[corner] != CLEAR_COLOR);
    CHECK(result.depth[corner] > 0.79f);
    CHECK(result.depth[corner] < 0.81f);
  });
}

TEST_CASE("Streaming raster demo - teapot-sized workload renders headlessly",
          "[device][raster-demo]") {
  for_each_fixture([](kfd::test::DeviceFixture &fix) {
    constexpr uint32_t WIDTH = 160;
    constexpr uint32_t HEIGHT = 120;
    auto prims = make_demo_workload(WIDTH, HEIGHT);
    REQUIRE(prims.size() == 6144);

    auto result = render(fix, prims, WIDTH, HEIGHT);
    uint32_t background = 0;
    uint32_t dark = 0;
    uint32_t light = 0;
    for (uint32_t color : result.color) {
      if (color == CLEAR_COLOR)
        ++background;
      else if (color == DARK_CHECKER)
        ++dark;
      else if (color == LIGHT_CHECKER)
        ++light;
    }

    CHECK(background > 1000);
    CHECK(dark > 1000);
    CHECK(light > 1000);
    CHECK(result.color.front() == CLEAR_COLOR);
    CHECK(result.color.back() == CLEAR_COLOR);
  });
}

TEST_CASE("Streaming raster demo - persistent mode completes frames and exits",
          "[device][raster-demo]") {
  for_each_fixture([](kfd::test::DeviceFixture &fix) {
    constexpr uint32_t WIDTH = 64;
    constexpr uint32_t HEIGHT = 64;
    constexpr size_t MAX_PRIMS = 8;
    constexpr size_t PIXELS = static_cast<size_t>(WIDTH) * HEIGHT;

    auto kernel = fix.exe.kernel("headless_stage1_persistent.kd");
    REQUIRE_RESULT(kernel);

    auto prim_buf = kfd::test::alloc_host_buffer(
        *fix.gpu, kfd::detail::align_up(MAX_PRIMS * sizeof(DemoPrimitive),
                                        kfd::detail::page_size()));
    auto color = kfd::test::alloc_host_buffer(
        *fix.gpu, kfd::detail::align_up(PIXELS * sizeof(uint32_t),
                                        kfd::detail::page_size()));
    auto depth = kfd::test::alloc_host_buffer(
        *fix.gpu, kfd::detail::align_up(PIXELS * sizeof(float),
                                        kfd::detail::page_size()));
    auto control_buf = kfd::test::alloc_host_buffer(
        *fix.gpu, kfd::detail::align_up(sizeof(PersistentControl),
                                        kfd::detail::page_size()));

    auto *control = static_cast<PersistentControl *>(control_buf.data());
    std::memset(prim_buf.data(), 0, prim_buf.size());
    std::memset(color.data(), 0, PIXELS * sizeof(uint32_t));
    std::memset(depth.data(), 0, PIXELS * sizeof(float));
    std::memset(control, 0, sizeof(PersistentControl));

    PersistentRasterArgs args{
        .prims = static_cast<const DemoPrimitive *>(prim_buf.data()),
        .color = static_cast<uint32_t *>(color.data()),
        .depth = static_cast<float *>(depth.data()),
        .control = control,
        .width = WIDTH,
        .height = HEIGHT,
        .pitch = WIDTH,
        .tile_size = TILE_SIZE,
        .clear_color = CLEAR_COLOR,
        .clear_depth = CLEAR_DEPTH,
    };
    kfd::DispatchConfig cfg{
        .grid = {.x = (WIDTH + TILE_SIZE - 1) / TILE_SIZE,
                 .y = (HEIGHT + TILE_SIZE - 1) / TILE_SIZE},
        .block = {.x = BLOCK_X, .y = BLOCK_Y},
    };
    auto kernarg = kernel->alloc();
    REQUIRE_RESULT(kernarg);
    kernel->fill(*kernarg, args, cfg);

    REQUIRE_RESULT(fix.compute.dispatch(*kernel, cfg, *kernarg));

    auto submit_frame = [&](uint32_t frame_id,
                            std::span<const DemoPrimitive> prims) {
      REQUIRE(prims.size() <= MAX_PRIMS);
      std::memcpy(prim_buf.data(), prims.data(),
                  prims.size() * sizeof(DemoPrimitive));
      __atomic_store_n(&control->rendered, 0u, __ATOMIC_RELEASE);
      __atomic_store_n(&control->tiles_done, 0u, __ATOMIC_RELEASE);
      __atomic_store_n(&control->prim_count,
                       static_cast<uint32_t>(prims.size()), __ATOMIC_RELEASE);
      __atomic_store_n(&control->frame_id, frame_id, __ATOMIC_RELEASE);
      kfd::detail::memory_barrier();
      __atomic_store_n(&control->ready, 1u, __ATOMIC_RELEASE);
      REQUIRE(wait_for_frame(control, static_cast<uint32_t>(prims.size())));
    };

    std::vector<DemoPrimitive> frame0;
    add_quad(frame0, 0.0f, 0.0f, static_cast<float>(WIDTH),
             static_cast<float>(HEIGHT), 0.5f, 0.0f, 0.0f, 1.0f, 1.0f);
    submit_frame(0, frame0);

    auto *pixels = static_cast<const uint32_t *>(color.data());
    CHECK(pixels[0] == expected_checker(0.5f / WIDTH, 0.5f / HEIGHT));
    CHECK(__atomic_load_n(&control->heartbeat, __ATOMIC_ACQUIRE) ==
          cfg.grid.x * cfg.grid.y);

    std::vector<DemoPrimitive> frame1;
    add_quad(frame1, 0.0f, 0.0f, static_cast<float>(WIDTH),
             static_cast<float>(HEIGHT), 0.8f, 0.0f, 0.0f, 1.0f, 1.0f);
    add_quad(frame1, 16.0f, 16.0f, 48.0f, 48.0f, 0.2f, 0.125f, 0.0f, 1.125f,
             1.0f);
    submit_frame(1, frame1);

    size_t center = static_cast<size_t>(32) * WIDTH + 32;
    CHECK(pixels[center] == LIGHT_CHECKER);
    CHECK(__atomic_load_n(&control->heartbeat, __ATOMIC_ACQUIRE) ==
          2 * cfg.grid.x * cfg.grid.y);

    std::vector<DemoPrimitive> empty_frame;
    submit_frame(2, empty_frame);
    CHECK(pixels[center] == CLEAR_COLOR);
    CHECK(__atomic_load_n(&control->heartbeat, __ATOMIC_ACQUIRE) ==
          3 * cfg.grid.x * cfg.grid.y);

    __atomic_store_n(&control->terminate, 1u, __ATOMIC_RELEASE);
    auto sig = kfd::Signal::create(fix.gpu->context());
    REQUIRE_RESULT(sig);
    REQUIRE_RESULT(fix.compute.signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  });
}
