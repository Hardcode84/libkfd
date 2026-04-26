#include "libkfd/gpu.h"
#include "libkfd/quake_raster.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

qr_context *create_context_or_skip(const qr_desc &desc) {
  qr_context *ctx = nullptr;
  qr_result err = qr_create(&desc, &ctx);
  if (err != QR_SUCCESS) {
    SKIP("KFD not available: " << qr_strerror(err) << " ("
                               << kfd_gpu_last_error() << ")");
  }
  REQUIRE(ctx != nullptr);
  return ctx;
}

std::filesystem::path temp_dump_path(const char *suffix) {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("libkfd_quake_raster_" + std::to_string(stamp) + suffix);
}

std::vector<std::uint8_t> read_binary_file(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  REQUIRE(file.good());
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

} // namespace

TEST_CASE("Quake raster - exposes API version and error strings",
          "[quake_raster]") {
  constexpr std::uint32_t VERSION =
      (QR_API_VERSION_MAJOR << 16U) | (QR_API_VERSION_MINOR << 8U) |
      QR_API_VERSION_PATCH;

  CHECK(qr_api_version() == VERSION);
  CHECK(qr_strerror(QR_SUCCESS) != nullptr);
  CHECK(qr_strerror(QR_ERROR_INVALID_ARGUMENT) != nullptr);
  CHECK(qr_strerror(QR_ERROR_BUFFER_TOO_SMALL) != nullptr);
  CHECK(qr_strerror(QR_ERROR_GPU) != nullptr);
}

TEST_CASE("Quake raster - rejects invalid descriptors", "[quake_raster]") {
  qr_context *ctx = nullptr;
  constexpr std::uint32_t U32_MAX =
      std::numeric_limits<std::uint32_t>::max();
  qr_desc desc{
      .width = 0,
      .height = 16,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
  };

  CHECK(qr_create(nullptr, &ctx) == QR_ERROR_INVALID_ARGUMENT);
  CHECK(qr_create(&desc, nullptr) == QR_ERROR_INVALID_ARGUMENT);
  CHECK(qr_create(&desc, &ctx) == QR_ERROR_INVALID_ARGUMENT);
  CHECK(ctx == nullptr);

  desc.width = 16;
  desc.output_mode = QR_OUTPUT_PRESENT;
  CHECK(qr_create(&desc, &ctx) == QR_ERROR_UNSUPPORTED);

  desc.output_mode = QR_OUTPUT_NOOUTPUT;
  desc.framebuffer_format = static_cast<qr_framebuffer_format>(99);
  CHECK(qr_create(&desc, &ctx) == QR_ERROR_UNSUPPORTED);

  desc.framebuffer_format = QR_FRAMEBUFFER_INDEXED8;
  desc.width = U32_MAX;
  desc.height = 1;
  CHECK(qr_create(&desc, &ctx) == QR_ERROR_OVERFLOW);

  desc.width = U32_MAX / 2U;
  desc.height = 3;
  CHECK(qr_create(&desc, &ctx) == QR_ERROR_OVERFLOW);
}

TEST_CASE("Quake raster - nooutput clear can be inspected",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 16;
  constexpr std::uint32_t HEIGHT = 8;
  constexpr std::uint8_t CLEAR_COLOR = 37;
  qr_context *ctx = nullptr;
  qr_frame *frame = nullptr;
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
  };
  qr_frame_desc frame_desc{};
  std::array<std::uint8_t, WIDTH * HEIGHT> pixels{};
  std::array<std::uint8_t, HEIGHT * (WIDTH + 3)> padded_pixels{};

  ctx = create_context_or_skip(desc);
  CHECK(qr_width(ctx) == WIDTH);
  CHECK(qr_height(ctx) == HEIGHT);
  CHECK(qr_output(ctx) == QR_OUTPUT_NOOUTPUT);

  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
  REQUIRE(frame != nullptr);
  REQUIRE(qr_frame_clear_indexed(frame, CLEAR_COLOR) == QR_SUCCESS);
  REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
  REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
          QR_SUCCESS);
  padded_pixels.fill(0xaaU);
  REQUIRE(qr_read_indexed(ctx, padded_pixels.data(), padded_pixels.size(),
                          WIDTH + 3) == QR_SUCCESS);
  CHECK(qr_read_indexed(ctx, pixels.data(), pixels.size() - 1U, WIDTH) ==
        QR_ERROR_BUFFER_TOO_SMALL);
  CHECK(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH - 1U) ==
        QR_ERROR_INVALID_ARGUMENT);

  for (std::uint8_t pixel : pixels) {
    CHECK(pixel == CLEAR_COLOR);
  }
  for (std::uint32_t y = 0; y < HEIGHT; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * (WIDTH + 3);
    for (std::uint32_t x = 0; x < WIDTH; ++x) {
      CHECK(padded_pixels[row + x] == CLEAR_COLOR);
    }
    CHECK(padded_pixels[row + WIDTH] == 0xaaU);
    CHECK(padded_pixels[row + WIDTH + 1U] == 0xaaU);
    CHECK(padded_pixels[row + WIDTH + 2U] == 0xaaU);
  }

  qr_destroy(ctx);
}

TEST_CASE("Quake raster - nooutput dumps indexed frames",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 7;
  constexpr std::uint32_t HEIGHT = 5;
  constexpr std::uint8_t CLEAR_COLOR = 91;
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
  };
  qr_frame_desc frame_desc{};
  qr_frame *frame = nullptr;
  qr_context *ctx = create_context_or_skip(desc);
  const auto path = temp_dump_path(".raw");

  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
  REQUIRE(qr_frame_clear_indexed(frame, CLEAR_COLOR) == QR_SUCCESS);
  REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
  REQUIRE(qr_dump_indexed(ctx, path.c_str()) == QR_SUCCESS);

  const auto bytes = read_binary_file(path);
  REQUIRE(bytes.size() == WIDTH * HEIGHT);
  for (std::uint8_t byte : bytes) {
    CHECK(byte == CLEAR_COLOR);
  }

  std::filesystem::remove(path);
  qr_destroy(ctx);
}

TEST_CASE("Quake raster - nooutput resolves XRGB for inspection",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 4;
  constexpr std::uint32_t HEIGHT = 2;
  constexpr std::uint8_t CLEAR_COLOR = 37;
  constexpr std::uint32_t RESOLVED = 0xff123456U;
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
  };
  qr_frame_desc frame_desc{};
  qr_frame *frame = nullptr;
  qr_context *ctx = create_context_or_skip(desc);
  std::array<std::uint32_t, 256> palette{};
  std::array<std::uint32_t, HEIGHT * (WIDTH + 1)> padded_xrgb{};
  const auto path = temp_dump_path(".ppm");

  palette[CLEAR_COLOR] = RESOLVED;
  padded_xrgb.fill(0xdeadbeefU);
  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
  REQUIRE(qr_frame_clear_indexed(frame, CLEAR_COLOR) == QR_SUCCESS);
  REQUIRE(qr_end_frame(frame) == QR_SUCCESS);

  REQUIRE(qr_read_xrgb(ctx, palette.data(), padded_xrgb.data(),
                       padded_xrgb.size() * sizeof(std::uint32_t),
                       WIDTH + 1) == QR_SUCCESS);
  CHECK(qr_read_xrgb(ctx, palette.data(), padded_xrgb.data(),
                     (WIDTH * HEIGHT * sizeof(std::uint32_t)) - 1U, WIDTH) ==
        QR_ERROR_BUFFER_TOO_SMALL);
  CHECK(qr_read_xrgb(ctx, palette.data(), padded_xrgb.data(),
                     padded_xrgb.size() * sizeof(std::uint32_t), WIDTH - 1U) ==
        QR_ERROR_INVALID_ARGUMENT);

  for (std::uint32_t y = 0; y < HEIGHT; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * (WIDTH + 1U);
    for (std::uint32_t x = 0; x < WIDTH; ++x) {
      CHECK(padded_xrgb[row + x] == RESOLVED);
    }
    CHECK(padded_xrgb[row + WIDTH] == 0xdeadbeefU);
  }

  REQUIRE(qr_dump_xrgb(ctx, palette.data(), path.c_str()) == QR_SUCCESS);
  const auto bytes = read_binary_file(path);
  const std::string header = "P6\n4 2\n255\n";
  REQUIRE(bytes.size() == header.size() + WIDTH * HEIGHT * 3U);
  CHECK(std::equal(header.begin(), header.end(), bytes.begin()));
  for (std::size_t i = header.size(); i < bytes.size(); i += 3U) {
    CHECK(bytes[i] == 0x12U);
    CHECK(bytes[i + 1U] == 0x34U);
    CHECK(bytes[i + 2U] == 0x56U);
  }

  std::filesystem::remove(path);
  qr_destroy(ctx);
}

TEST_CASE("Quake raster - uploads synthetic persistent resources",
          "[quake_raster][device]") {
  qr_context *ctx = nullptr;
  qr_desc desc{
      .width = 8,
      .height = 8,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 2,
      .max_lightmaps = 2,
      .max_surfaces = 4,
      .max_worlds = 1,
      .texture_atlas_bytes = 64,
      .lightmap_atlas_bytes = 32,
  };
  const std::array<std::uint8_t, 16> tex0{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const std::array<std::uint8_t, 4> tex1{16, 17, 18, 19};
  const std::array<std::uint8_t, 4> light{7, 8, 9, 10};
  qr_texture texture = QR_INVALID_HANDLE;
  qr_lightmap lightmap = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_texture_desc texture_desc{
      .mips =
          {
              {.pixels = tex0.data(), .width = 4, .height = 4, .stride = 4},
              {.pixels = tex1.data(), .width = 2, .height = 2, .stride = 2},
          },
      .mip_count = 2,
      .flags = 0,
  };
  qr_lightmap_desc lightmap_desc{
      .pixels = light.data(),
      .width = 2,
      .height = 2,
      .stride = 2,
  };
  std::array<qr_world_surface_desc, 2> surfaces{};
  qr_capacity_info capacity{};

  ctx = create_context_or_skip(desc);

  REQUIRE(qr_upload_texture(ctx, &texture_desc, &texture) == QR_SUCCESS);
  REQUIRE(texture != QR_INVALID_HANDLE);
  REQUIRE(qr_upload_lightmap(ctx, &lightmap_desc, &lightmap) == QR_SUCCESS);
  REQUIRE(lightmap != QR_INVALID_HANDLE);

  surfaces[0].texture = texture;
  surfaces[0].lightmap = lightmap;
  surfaces[0].plane[2] = 1.0F;
  surfaces[0].tex_s[0] = 1.0F;
  surfaces[0].tex_t[1] = 1.0F;
  surfaces[0].light_s[0] = 0.5F;
  surfaces[0].light_t[1] = 0.5F;
  surfaces[1] = surfaces[0];
  surfaces[1].flags = 1;

  surfaces[1].texture = 999U;
  CHECK(qr_create_world(ctx, surfaces.data(), surfaces.size(), &world) ==
        QR_ERROR_INVALID_ARGUMENT);
  surfaces[1].texture = texture;

  REQUIRE(qr_create_world(ctx, surfaces.data(), surfaces.size(), &world) ==
          QR_SUCCESS);
  REQUIRE(world != QR_INVALID_HANDLE);
  REQUIRE(qr_get_capacity_info(ctx, &capacity) == QR_SUCCESS);
  CHECK(capacity.texture_count == 1);
  CHECK(capacity.texture_capacity == 2);
  CHECK(capacity.texture_atlas_used == tex0.size() + tex1.size());
  CHECK(capacity.lightmap_count == 1);
  CHECK(capacity.lightmap_atlas_used == light.size());
  CHECK(capacity.surface_count == surfaces.size());
  CHECK(capacity.surface_capacity == 4);
  CHECK(capacity.world_count == 1);
  CHECK(capacity.world_capacity == 1);

  CHECK(qr_create_world(ctx, surfaces.data(), surfaces.size(), &world) ==
        QR_ERROR_NO_SPACE);

  REQUIRE(qr_upload_texture(ctx, &texture_desc, &texture) == QR_SUCCESS);
  CHECK(qr_upload_texture(ctx, &texture_desc, &texture) == QR_ERROR_NO_SPACE);
  REQUIRE(qr_upload_lightmap(ctx, &lightmap_desc, &lightmap) == QR_SUCCESS);
  CHECK(qr_upload_lightmap(ctx, &lightmap_desc, &lightmap) ==
        QR_ERROR_NO_SPACE);

  qr_destroy(ctx);
}

TEST_CASE("Quake raster - resource uploads reject undersized atlases",
          "[quake_raster][device]") {
  qr_desc desc{
      .width = 8,
      .height = 8,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 1,
      .max_lightmaps = 1,
      .max_surfaces = 1,
      .max_worlds = 1,
      .texture_atlas_bytes = 4,
      .lightmap_atlas_bytes = 2,
  };
  const std::array<std::uint8_t, 16> texture_pixels{};
  const std::array<std::uint8_t, 4> lightmap_pixels{};
  qr_texture texture = QR_INVALID_HANDLE;
  qr_lightmap lightmap = QR_INVALID_HANDLE;
  qr_texture_desc texture_desc{
      .mips =
          {
              {.pixels = texture_pixels.data(),
               .width = 4,
               .height = 4,
               .stride = 4},
          },
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc lightmap_desc{
      .pixels = lightmap_pixels.data(),
      .width = 2,
      .height = 2,
      .stride = 2,
  };
  qr_context *ctx = create_context_or_skip(desc);

  CHECK(qr_upload_texture(ctx, &texture_desc, &texture) == QR_ERROR_NO_SPACE);
  CHECK(texture == QR_INVALID_HANDLE);
  CHECK(qr_upload_lightmap(ctx, &lightmap_desc, &lightmap) ==
        QR_ERROR_NO_SPACE);
  CHECK(lightmap == QR_INVALID_HANDLE);

  qr_destroy(ctx);
}
