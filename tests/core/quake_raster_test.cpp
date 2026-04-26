#include "libkfd/gpu.h"
#include "libkfd/quake_raster.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

TEST_CASE("Quake raster - rejects invalid descriptors", "[quake_raster]") {
  qr_context *ctx = nullptr;
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

  int err = qr_create(&desc, &ctx);
  if (err != 0) {
    SKIP("KFD not available: " << qr_strerror(err) << " ("
                               << kfd_gpu_last_error() << ")");
  }
  REQUIRE(ctx != nullptr);
  CHECK(qr_width(ctx) == WIDTH);
  CHECK(qr_height(ctx) == HEIGHT);
  CHECK(qr_output(ctx) == QR_OUTPUT_NOOUTPUT);

  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == 0);
  REQUIRE(frame != nullptr);
  REQUIRE(qr_frame_clear_indexed(frame, CLEAR_COLOR) == 0);
  REQUIRE(qr_end_frame(frame) == 0);
  REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) == 0);

  for (std::uint8_t pixel : pixels) {
    CHECK(pixel == CLEAR_COLOR);
  }

  qr_destroy(ctx);
}
