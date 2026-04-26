#include "libkfd/gpu.h"
#include "libkfd/quake_raster.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
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
  int err = qr_create(&desc, &ctx);
  if (err != 0) {
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

  CHECK(qr_create(nullptr, &ctx) != 0);
  CHECK(qr_create(&desc, nullptr) != 0);
  CHECK(qr_create(&desc, &ctx) != 0);
  CHECK(ctx == nullptr);

  desc.width = 16;
  desc.output_mode = QR_OUTPUT_PRESENT;
  CHECK(qr_create(&desc, &ctx) == ENOTSUP);

  desc.output_mode = QR_OUTPUT_NOOUTPUT;
  desc.framebuffer_format = static_cast<qr_framebuffer_format>(99);
  CHECK(qr_create(&desc, &ctx) == ENOTSUP);

  desc.framebuffer_format = QR_FRAMEBUFFER_INDEXED8;
  desc.width = U32_MAX;
  desc.height = 1;
  CHECK(qr_create(&desc, &ctx) == EOVERFLOW);

  desc.width = U32_MAX / 2U;
  desc.height = 3;
  CHECK(qr_create(&desc, &ctx) == EOVERFLOW);
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

  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == 0);
  REQUIRE(frame != nullptr);
  REQUIRE(qr_frame_clear_indexed(frame, CLEAR_COLOR) == 0);
  REQUIRE(qr_end_frame(frame) == 0);
  REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) == 0);
  padded_pixels.fill(0xaaU);
  REQUIRE(qr_read_indexed(ctx, padded_pixels.data(), padded_pixels.size(),
                          WIDTH + 3) == 0);
  CHECK(qr_read_indexed(ctx, pixels.data(), pixels.size() - 1U, WIDTH) ==
        ENOSPC);
  CHECK(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH - 1U) ==
        EINVAL);

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

  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == 0);
  REQUIRE(qr_frame_clear_indexed(frame, CLEAR_COLOR) == 0);
  REQUIRE(qr_end_frame(frame) == 0);
  REQUIRE(qr_dump_indexed(ctx, path.c_str()) == 0);

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
  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == 0);
  REQUIRE(qr_frame_clear_indexed(frame, CLEAR_COLOR) == 0);
  REQUIRE(qr_end_frame(frame) == 0);

  REQUIRE(qr_read_xrgb(ctx, palette.data(), padded_xrgb.data(),
                       padded_xrgb.size() * sizeof(std::uint32_t),
                       WIDTH + 1) == 0);
  CHECK(qr_read_xrgb(ctx, palette.data(), padded_xrgb.data(),
                     (WIDTH * HEIGHT * sizeof(std::uint32_t)) - 1U, WIDTH) ==
        ENOSPC);
  CHECK(qr_read_xrgb(ctx, palette.data(), padded_xrgb.data(),
                     padded_xrgb.size() * sizeof(std::uint32_t), WIDTH - 1U) ==
        EINVAL);

  for (std::uint32_t y = 0; y < HEIGHT; ++y) {
    const std::size_t row = static_cast<std::size_t>(y) * (WIDTH + 1U);
    for (std::uint32_t x = 0; x < WIDTH; ++x) {
      CHECK(padded_xrgb[row + x] == RESOLVED);
    }
    CHECK(padded_xrgb[row + WIDTH] == 0xdeadbeefU);
  }

  REQUIRE(qr_dump_xrgb(ctx, palette.data(), path.c_str()) == 0);
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
