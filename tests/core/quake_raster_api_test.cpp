#include "libkfd/quake_raster.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

TEST_CASE("Quake raster API - exposes version and pure helpers",
          "[quake_raster][no_gpu]") {
  constexpr std::uint32_t VERSION =
      (QR_API_VERSION_MAJOR << 16U) | (QR_API_VERSION_MINOR << 8U) |
      QR_API_VERSION_PATCH;

  CHECK(qr_api_version() == VERSION);
  CHECK(qr_strerror(QR_SUCCESS) != nullptr);
  CHECK(qr_strerror(QR_ERROR_INVALID_ARGUMENT) != nullptr);
  CHECK(qr_strerror(QR_ERROR_GPU) != nullptr);
  CHECK(qr_pack_depth_payload(0x12345678U, 0x90abcdefU) ==
        0x1234567890abcdefULL);
}

TEST_CASE("Quake raster API - rejects descriptors before device setup",
          "[quake_raster][no_gpu]") {
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
  CHECK(qr_create(&desc, &ctx) == QR_ERROR_INVALID_ARGUMENT);

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
