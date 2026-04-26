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

struct PresentCapture {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::size_t stride = 0;
  std::vector<std::uint32_t> pixels;
  int calls = 0;
};

qr_result capture_present(void *userdata, const std::uint32_t *xrgb,
                          std::uint32_t width, std::uint32_t height,
                          std::size_t stride_pixels) {
  auto *capture = static_cast<PresentCapture *>(userdata);
  if (capture == nullptr || xrgb == nullptr || stride_pixels < width) {
    return QR_ERROR_INVALID_ARGUMENT;
  }
  capture->width = width;
  capture->height = height;
  capture->stride = stride_pixels;
  capture->pixels.assign(xrgb, xrgb + (stride_pixels * height));
  ++capture->calls;
  return QR_SUCCESS;
}

} // namespace

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

TEST_CASE("Quake raster - present mode resolves through callback",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 3;
  constexpr std::uint32_t HEIGHT = 2;
  constexpr std::uint8_t CLEAR_COLOR = 11;
  constexpr std::uint32_t RESOLVED = 0xff445566U;
  std::array<std::uint32_t, 256> palette{};
  PresentCapture capture{};
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_PRESENT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .present = capture_present,
      .present_userdata = &capture,
      .present_palette_xrgb = palette.data(),
  };
  qr_context *ctx = nullptr;
  qr_frame *frame = nullptr;
  qr_frame_desc frame_desc{};

  palette[CLEAR_COLOR] = RESOLVED;
  ctx = create_context_or_skip(desc);
  REQUIRE(qr_output(ctx) == QR_OUTPUT_PRESENT);
  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
  REQUIRE(qr_frame_clear_indexed(frame, CLEAR_COLOR) == QR_SUCCESS);
  REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
  CHECK(capture.calls == 1);
  CHECK(capture.width == WIDTH);
  CHECK(capture.height == HEIGHT);
  CHECK(capture.stride == WIDTH);
  REQUIRE(capture.pixels.size() == WIDTH * HEIGHT);
  CHECK(std::all_of(capture.pixels.begin(), capture.pixels.end(),
                    [](std::uint32_t pixel) { return pixel == RESOLVED; }));

  qr_destroy(ctx);
}

TEST_CASE("Quake raster - perspective-corrects world texture coordinates",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 4;
  constexpr std::uint32_t HEIGHT = 1;
  const std::array<std::uint8_t, 8> texture{0, 1, 2, 3, 4, 5, 6, 7};
  const std::array<std::uint8_t, 1> light{0};
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 1,
      .max_lightmaps = 1,
      .max_surfaces = 1,
      .max_worlds = 1,
      .texture_atlas_bytes = texture.size(),
      .lightmap_atlas_bytes = light.size(),
  };
  qr_context *ctx = create_context_or_skip(desc);
  qr_texture tex = QR_INVALID_HANDLE;
  qr_lightmap lm = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_texture_desc tex_desc{
      .mips = {{.pixels = texture.data(), .width = 8, .height = 1,
                .stride = 8}},
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc lm_desc{
      .pixels = light.data(),
      .width = 1,
      .height = 1,
      .stride = 1,
  };
  std::array<qr_world_surface_desc, 1> surfaces{};
  const std::array<qr_world_vertex, 4> quad{{
      {.x = 0.0F, .y = 0.0F, .z = 1.0F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 4.0F, .y = 0.0F, .z = 4.0F, .u = 8.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 4.0F, .y = 1.0F, .z = 4.0F, .u = 8.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 0.0F, .y = 1.0F, .z = 1.0F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
  }};
  qr_world_polygon_desc polygon{
      .surface = 0,
      .vertices = quad.data(),
      .vertex_count = static_cast<std::uint32_t>(quad.size()),
  };
  qr_world_draw_desc draw_desc{
      .world = QR_INVALID_HANDLE,
      .polygons = &polygon,
      .polygon_count = 1,
      .debug_mode = QR_DEBUG_TEXTURE_ONLY,
  };
  qr_frame *frame = nullptr;
  qr_frame_desc frame_desc{};
  std::array<std::uint8_t, WIDTH * HEIGHT> pixels{};

  REQUIRE(qr_upload_texture(ctx, &tex_desc, &tex) == QR_SUCCESS);
  REQUIRE(qr_upload_lightmap(ctx, &lm_desc, &lm) == QR_SUCCESS);
  surfaces[0].texture = tex;
  surfaces[0].lightmap = lm;
  REQUIRE(qr_create_world(ctx, surfaces.data(), surfaces.size(), &world) ==
          QR_SUCCESS);
  draw_desc.world = world;

  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
  REQUIRE(qr_frame_clear_indexed(frame, 99) == QR_SUCCESS);
  REQUIRE(qr_frame_draw_world(frame, &draw_desc) == QR_SUCCESS);
  REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
  REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
          QR_SUCCESS);

  CHECK(pixels[2] == 2);
  CHECK(pixels[3] == 5);
  qr_destroy(ctx);
}

TEST_CASE("Quake raster - handles classic surface flags and light updates",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 4;
  constexpr std::uint32_t HEIGHT = 4;
  const std::array<std::uint8_t, WIDTH * HEIGHT> cutout_texture{
      255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255};
  const std::array<std::uint8_t, WIDTH * HEIGHT> solid_texture{
      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  const std::array<std::uint8_t, WIDTH * HEIGHT> light_a{
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};
  const std::array<std::uint8_t, WIDTH * HEIGHT> light_b{
      77, 77, 77, 77, 77, 77, 77, 77,
      77, 77, 77, 77, 77, 77, 77, 77};
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 2,
      .max_lightmaps = 1,
      .max_surfaces = 3,
      .max_worlds = 1,
      .texture_atlas_bytes = cutout_texture.size() + solid_texture.size(),
      .lightmap_atlas_bytes = light_a.size(),
  };
  qr_context *ctx = create_context_or_skip(desc);
  qr_texture cutout_tex = QR_INVALID_HANDLE;
  qr_texture solid_tex = QR_INVALID_HANDLE;
  qr_lightmap light = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_texture_desc cutout_desc{
      .mips = {{.pixels = cutout_texture.data(), .width = WIDTH,
                .height = HEIGHT, .stride = WIDTH}},
      .mip_count = 1,
      .flags = 0,
  };
  qr_texture_desc solid_desc{
      .mips = {{.pixels = solid_texture.data(), .width = WIDTH,
                .height = HEIGHT, .stride = WIDTH}},
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc light_desc{
      .pixels = light_a.data(),
      .width = WIDTH,
      .height = HEIGHT,
      .stride = WIDTH,
  };
  qr_lightmap_desc light_update{
      .pixels = light_b.data(),
      .width = WIDTH,
      .height = HEIGHT,
      .stride = WIDTH,
  };
  std::array<qr_world_surface_desc, 3> surfaces{};
  const std::array<qr_world_vertex, 4> quad{{
      {.x = 0.0F, .y = 0.0F, .z = 0.25F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 4.0F, .y = 0.0F, .z = 0.25F, .u = 4.0F, .v = 0.0F,
       .light_u = 4.0F, .light_v = 0.0F},
      {.x = 4.0F, .y = 4.0F, .z = 0.25F, .u = 4.0F, .v = 4.0F,
       .light_u = 4.0F, .light_v = 4.0F},
      {.x = 0.0F, .y = 4.0F, .z = 0.25F, .u = 0.0F, .v = 4.0F,
       .light_u = 0.0F, .light_v = 4.0F},
  }};
  std::array<std::uint8_t, WIDTH * HEIGHT> pixels{};

  REQUIRE(qr_upload_texture(ctx, &cutout_desc, &cutout_tex) == QR_SUCCESS);
  REQUIRE(qr_upload_texture(ctx, &solid_desc, &solid_tex) == QR_SUCCESS);
  REQUIRE(qr_upload_lightmap(ctx, &light_desc, &light) == QR_SUCCESS);
  surfaces[0].texture = cutout_tex;
  surfaces[0].lightmap = light;
  surfaces[0].flags = QR_SURFACE_CUTOUT;
  surfaces[1].texture = solid_tex;
  surfaces[1].lightmap = light;
  surfaces[1].flags = QR_SURFACE_SKY;
  surfaces[2].texture = solid_tex;
  surfaces[2].lightmap = light;
  REQUIRE(qr_create_world(ctx, surfaces.data(), surfaces.size(), &world) ==
          QR_SUCCESS);

  auto draw_surface = [&](std::uint32_t surface, qr_debug_mode mode,
                          std::uint8_t clear) {
    qr_frame *frame = nullptr;
    qr_frame_desc frame_desc{};
    qr_world_polygon_desc polygon{.surface = surface,
                                  .vertices = quad.data(),
                                  .vertex_count = 4};
    qr_world_draw_desc draw_desc{
        .world = world,
        .polygons = &polygon,
        .polygon_count = 1,
        .debug_mode = mode,
    };

    REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
    REQUIRE(qr_frame_clear_indexed(frame, clear) == QR_SUCCESS);
    REQUIRE(qr_frame_draw_world(frame, &draw_desc) == QR_SUCCESS);
    REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
    REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
            QR_SUCCESS);
  };

  draw_surface(0, QR_DEBUG_TEXTURE_ONLY, 13);
  CHECK(std::all_of(pixels.begin(), pixels.end(),
                    [](std::uint8_t pixel) { return pixel == 13; }));

  draw_surface(1, QR_DEBUG_TEXTURE_ONLY, 0);
  CHECK(std::all_of(pixels.begin(), pixels.end(), [](std::uint8_t pixel) {
    return pixel == QR_SKY_COLOR_INDEX;
  }));

  draw_surface(2, QR_DEBUG_LIGHT_ONLY, 0);
  CHECK(pixels[0] == 5);
  REQUIRE(qr_update_lightmap(ctx, light, &light_update) == QR_SUCCESS);
  draw_surface(2, QR_DEBUG_LIGHT_ONLY, 0);
  CHECK(pixels[0] == 77);

  qr_destroy(ctx);
}

TEST_CASE("Quake raster - draws alias models, sprites, and particles",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 8;
  constexpr std::uint32_t HEIGHT = 8;
  const std::array<std::uint8_t, 4> texture{55, 55, 55, 55};
  const std::array<std::uint8_t, 4> light{1, 1, 1, 1};
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 1,
      .max_lightmaps = 1,
      .max_surfaces = 1,
      .max_worlds = 1,
      .max_frame_triangles = 8,
      .max_alias_models = 1,
      .max_alias_triangles = 1,
      .texture_atlas_bytes = texture.size(),
      .lightmap_atlas_bytes = light.size(),
  };
  qr_context *ctx = create_context_or_skip(desc);
  qr_texture tex = QR_INVALID_HANDLE;
  qr_lightmap lm = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_alias_model alias = QR_INVALID_HANDLE;
  qr_texture_desc tex_desc{
      .mips =
          {
              {.pixels = texture.data(), .width = 2, .height = 2, .stride = 2},
          },
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc lm_desc{
      .pixels = light.data(),
      .width = 2,
      .height = 2,
      .stride = 2,
  };
  qr_world_surface_desc surface{};
  qr_alias_triangle_desc tri{
      .surface = 0,
      .v0 = {.x = 0.0F, .y = 0.0F, .z = 0.4F, .u = 0.0F, .v = 0.0F,
             .light_u = 0.0F, .light_v = 0.0F},
      .v1 = {.x = 4.0F, .y = 0.0F, .z = 0.4F, .u = 2.0F, .v = 0.0F,
             .light_u = 1.0F, .light_v = 0.0F},
      .v2 = {.x = 0.0F, .y = 4.0F, .z = 0.4F, .u = 0.0F, .v = 2.0F,
             .light_u = 0.0F, .light_v = 1.0F},
  };
  qr_alias_model_desc alias_desc{
      .world = QR_INVALID_HANDLE,
      .triangles = &tri,
      .triangle_count = 1,
  };
  std::array<std::uint8_t, WIDTH * HEIGHT> pixels{};
  qr_frame_desc frame_desc{};

  REQUIRE(qr_upload_texture(ctx, &tex_desc, &tex) == QR_SUCCESS);
  REQUIRE(qr_upload_lightmap(ctx, &lm_desc, &lm) == QR_SUCCESS);
  surface.texture = tex;
  surface.lightmap = lm;
  REQUIRE(qr_create_world(ctx, &surface, 1, &world) == QR_SUCCESS);
  alias_desc.world = world;
  REQUIRE(qr_upload_alias_model(ctx, &alias_desc, &alias) == QR_SUCCESS);
  {
    qr_alias_model overflow_alias = QR_INVALID_HANDLE;
    CHECK(qr_upload_alias_model(ctx, &alias_desc, &overflow_alias) ==
          QR_ERROR_NO_SPACE);
    CHECK(overflow_alias == QR_INVALID_HANDLE);
  }

  {
    qr_frame *frame = nullptr;
    qr_alias_draw_desc draw{
        .model = alias,
        .debug_mode = QR_DEBUG_TEXTURE_ONLY,
    };

    REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
    REQUIRE(qr_frame_clear_indexed(frame, 0) == QR_SUCCESS);
    REQUIRE(qr_frame_draw_alias_model(frame, &draw) == QR_SUCCESS);
    REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
    REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
            QR_SUCCESS);
    CHECK(pixels[0] == 55);
  }

  {
    qr_frame *frame = nullptr;
    qr_sprite_draw_desc draw{
        .world = world,
        .surface = 0,
        .x0 = 4.0F,
        .y0 = 0.0F,
        .x1 = 8.0F,
        .y1 = 4.0F,
        .z = 0.3F,
        .u0 = 0.0F,
        .v0 = 0.0F,
        .u1 = 2.0F,
        .v1 = 2.0F,
        .debug_mode = QR_DEBUG_TEXTURE_ONLY,
    };

    REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
    REQUIRE(qr_frame_clear_indexed(frame, 0) == QR_SUCCESS);
    REQUIRE(qr_frame_draw_sprite(frame, &draw) == QR_SUCCESS);
    REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
    REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
            QR_SUCCESS);
    CHECK(pixels[4] == 55);
  }

  {
    qr_frame *frame = nullptr;
    std::array<qr_particle_desc, 1> particles{{
        {.world = world, .surface = 0, .x = 6.0F, .y = 6.0F, .z = 0.2F,
         .size = 2.0F, .u = 0.0F, .v = 0.0F},
    }};
    qr_particles_draw_desc draw{
        .particles = particles.data(),
        .particle_count = particles.size(),
        .debug_mode = QR_DEBUG_TEXTURE_ONLY,
    };

    REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
    REQUIRE(qr_frame_clear_indexed(frame, 0) == QR_SUCCESS);
    REQUIRE(qr_frame_draw_particles(frame, &draw) == QR_SUCCESS);
    REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
    REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
            QR_SUCCESS);
    CHECK(pixels[static_cast<std::size_t>(6) * WIDTH + 6U] == 55);
  }

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
      .max_frame_triangles = 8,
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
  CHECK(capacity.frame_triangle_capacity == 8);

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

TEST_CASE("Quake raster - world draw rejects transient triangle overflow",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 4;
  constexpr std::uint32_t HEIGHT = 4;
  const std::array<std::uint8_t, WIDTH * HEIGHT> texture{
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const std::array<std::uint8_t, WIDTH * HEIGHT> light{
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 1,
      .max_lightmaps = 1,
      .max_surfaces = 1,
      .max_worlds = 1,
      .max_frame_triangles = 1,
      .texture_atlas_bytes = texture.size(),
      .lightmap_atlas_bytes = light.size(),
  };
  qr_context *ctx = create_context_or_skip(desc);
  qr_texture tex = QR_INVALID_HANDLE;
  qr_lightmap lm = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_texture_desc tex_desc{
      .mips = {{.pixels = texture.data(), .width = WIDTH, .height = HEIGHT,
                .stride = WIDTH}},
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc lm_desc{
      .pixels = light.data(),
      .width = WIDTH,
      .height = HEIGHT,
      .stride = WIDTH,
  };
  qr_world_surface_desc surface{};
  const std::array<qr_world_vertex, 4> quad{{
      {.x = 0.0F, .y = 0.0F, .z = 0.25F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 4.0F, .y = 0.0F, .z = 0.25F, .u = 4.0F, .v = 0.0F,
       .light_u = 4.0F, .light_v = 0.0F},
      {.x = 4.0F, .y = 4.0F, .z = 0.25F, .u = 4.0F, .v = 4.0F,
       .light_u = 4.0F, .light_v = 4.0F},
      {.x = 0.0F, .y = 4.0F, .z = 0.25F, .u = 0.0F, .v = 4.0F,
       .light_u = 0.0F, .light_v = 4.0F},
  }};
  qr_world_polygon_desc polygon{
      .surface = 0,
      .vertices = quad.data(),
      .vertex_count = static_cast<std::uint32_t>(quad.size()),
  };
  qr_world_draw_desc draw_desc{
      .world = QR_INVALID_HANDLE,
      .polygons = &polygon,
      .polygon_count = 1,
      .debug_mode = QR_DEBUG_TEXTURE_ONLY,
  };
  qr_frame *frame = nullptr;
  qr_frame_desc frame_desc{};

  REQUIRE(qr_upload_texture(ctx, &tex_desc, &tex) == QR_SUCCESS);
  REQUIRE(qr_upload_lightmap(ctx, &lm_desc, &lm) == QR_SUCCESS);
  surface.texture = tex;
  surface.lightmap = lm;
  REQUIRE(qr_create_world(ctx, &surface, 1, &world) == QR_SUCCESS);
  draw_desc.world = world;

  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
  CHECK(qr_frame_draw_world(frame, &draw_desc) == QR_ERROR_NO_SPACE);
  REQUIRE(qr_end_frame(frame) == QR_SUCCESS);

  qr_destroy(ctx);
}

TEST_CASE("Quake raster - minimal world raster draws indexed polygons",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 4;
  constexpr std::uint32_t HEIGHT = 4;
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 2,
      .max_lightmaps = 2,
      .max_surfaces = 2,
      .max_worlds = 1,
      .texture_atlas_bytes = 32,
      .lightmap_atlas_bytes = 32,
  };
  const std::array<std::uint8_t, WIDTH * HEIGHT> far_texture{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const std::array<std::uint8_t, WIDTH * HEIGHT> near_texture{
      90, 90, 90, 90, 90, 90, 90, 90,
      90, 90, 90, 90, 90, 90, 90, 90};
  const std::array<std::uint8_t, WIDTH * HEIGHT> far_light{
      10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10};
  const std::array<std::uint8_t, WIDTH * HEIGHT> near_light{
      200, 200, 200, 200, 200, 200, 200, 200,
      200, 200, 200, 200, 200, 200, 200, 200};
  qr_context *ctx = create_context_or_skip(desc);
  qr_texture far_tex = QR_INVALID_HANDLE;
  qr_texture near_tex = QR_INVALID_HANDLE;
  qr_lightmap far_lm = QR_INVALID_HANDLE;
  qr_lightmap near_lm = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_texture_desc far_tex_desc{
      .mips =
          {
              {.pixels = far_texture.data(), .width = WIDTH, .height = HEIGHT,
               .stride = WIDTH},
          },
      .mip_count = 1,
      .flags = 0,
  };
  qr_texture_desc near_tex_desc{
      .mips =
          {
              {.pixels = near_texture.data(), .width = WIDTH, .height = HEIGHT,
               .stride = WIDTH},
          },
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc far_lm_desc{
      .pixels = far_light.data(),
      .width = WIDTH,
      .height = HEIGHT,
      .stride = WIDTH,
  };
  qr_lightmap_desc near_lm_desc{
      .pixels = near_light.data(),
      .width = WIDTH,
      .height = HEIGHT,
      .stride = WIDTH,
  };
  std::array<qr_world_surface_desc, 2> surfaces{};
  std::array<qr_world_vertex, 4> far_quad{{
      {.x = 0.0F, .y = 0.0F, .z = 0.5F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 4.0F, .y = 0.0F, .z = 0.5F, .u = 4.0F, .v = 0.0F,
       .light_u = 4.0F, .light_v = 0.0F},
      {.x = 4.0F, .y = 4.0F, .z = 0.5F, .u = 4.0F, .v = 4.0F,
       .light_u = 4.0F, .light_v = 4.0F},
      {.x = 0.0F, .y = 4.0F, .z = 0.5F, .u = 0.0F, .v = 4.0F,
       .light_u = 0.0F, .light_v = 4.0F},
  }};
  std::array<qr_world_vertex, 4> near_quad{{
      {.x = 1.0F, .y = 1.0F, .z = 0.1F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 3.0F, .y = 1.0F, .z = 0.1F, .u = 4.0F, .v = 0.0F,
       .light_u = 4.0F, .light_v = 0.0F},
      {.x = 3.0F, .y = 3.0F, .z = 0.1F, .u = 4.0F, .v = 4.0F,
       .light_u = 4.0F, .light_v = 4.0F},
      {.x = 1.0F, .y = 3.0F, .z = 0.1F, .u = 0.0F, .v = 4.0F,
       .light_u = 0.0F, .light_v = 4.0F},
  }};
  std::array<qr_world_polygon_desc, 2> polygons{{
      {.surface = 0, .vertices = far_quad.data(),
       .vertex_count = static_cast<std::uint32_t>(far_quad.size())},
      {.surface = 1, .vertices = near_quad.data(),
       .vertex_count = static_cast<std::uint32_t>(near_quad.size())},
  }};
  std::array<std::uint8_t, QR_COLORMAP_SIZE> colormap{};
  std::array<std::uint8_t, WIDTH * HEIGHT> pixels{};

  for (std::uint32_t light = 0; light < 256; ++light) {
    for (std::uint32_t texel = 0; texel < 256; ++texel) {
      colormap[(light << 8U) | texel] =
          static_cast<std::uint8_t>((light + texel) & 0xffU);
    }
  }

  REQUIRE(qr_upload_texture(ctx, &far_tex_desc, &far_tex) == QR_SUCCESS);
  REQUIRE(qr_upload_texture(ctx, &near_tex_desc, &near_tex) == QR_SUCCESS);
  REQUIRE(qr_upload_lightmap(ctx, &far_lm_desc, &far_lm) == QR_SUCCESS);
  REQUIRE(qr_upload_lightmap(ctx, &near_lm_desc, &near_lm) == QR_SUCCESS);
  surfaces[0].texture = far_tex;
  surfaces[0].lightmap = far_lm;
  surfaces[1].texture = near_tex;
  surfaces[1].lightmap = near_lm;
  REQUIRE(qr_create_world(ctx, surfaces.data(), surfaces.size(), &world) ==
          QR_SUCCESS);

  auto render_mode = [&](qr_debug_mode mode) {
    qr_frame *frame = nullptr;
    qr_frame_desc frame_desc{};
    qr_world_draw_desc draw_desc{
        .world = world,
        .polygons = polygons.data(),
        .polygon_count = polygons.size(),
        .colormap = colormap.data(),
        .colormap_size = colormap.size(),
        .debug_mode = mode,
    };

    REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
    REQUIRE(qr_frame_clear_indexed(frame, 0) == QR_SUCCESS);
    REQUIRE(qr_frame_draw_world(frame, &draw_desc) == QR_SUCCESS);
    REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
    REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
            QR_SUCCESS);
  };

  render_mode(QR_DEBUG_TEXTURE_ONLY);
  CHECK(pixels[0] == 0);
  CHECK(pixels[static_cast<std::size_t>(1) * WIDTH + 1U] == 90);
  {
    qr_raster_stats stats{};
    REQUIRE(qr_get_raster_stats(ctx, &stats) == QR_SUCCESS);
    CHECK(stats.tile_size == QR_TILE_SIZE);
    CHECK(stats.tile_cols == 1);
    CHECK(stats.tile_rows == 1);
    CHECK(stats.tile_count == 1);
    CHECK(stats.tile_triangle_capacity == QR_TILE_TRIANGLE_CAPACITY);
    CHECK(stats.triangle_count == 4);
    CHECK(stats.occupied_tile_count == 1);
    CHECK(stats.max_tile_triangle_count == 4);
    CHECK(stats.overflow_tile_count == 0);
    CHECK(stats.overflow_reference_count == 0);
    CHECK(stats.depth_key_scale == QR_DEPTH_KEY_SCALE);
    CHECK(stats.depth_bound_tile_count == 1);
    CHECK(stats.hiz_candidate_reference_count == 4);
    CHECK(stats.hiz_overflow_fallback_count == 0);
  }

  render_mode(QR_DEBUG_FLAT_SURFACE_ID);
  CHECK(pixels[0] == 1);
  CHECK(pixels[static_cast<std::size_t>(1) * WIDTH + 1U] == 2);

  render_mode(QR_DEBUG_LIGHT_ONLY);
  CHECK(pixels[0] == 10);
  CHECK(pixels[static_cast<std::size_t>(1) * WIDTH + 1U] == 200);

  render_mode(QR_DEBUG_SHADED);
  CHECK(pixels[0] == ((10 + 0) & 0xff));
  CHECK(pixels[static_cast<std::size_t>(1) * WIDTH + 1U] ==
        ((200 + 90) & 0xff));

  render_mode(QR_DEBUG_DEPTH);
  CHECK(pixels[0] == 239);
  CHECK(pixels[static_cast<std::size_t>(1) * WIDTH + 1U] == 252);
  {
    qr_perf_counters perf{};
    REQUIRE(qr_get_perf_counters(ctx, &perf) == QR_SUCCESS);
    CHECK(perf.frame_count >= 5);
    CHECK(perf.clear_count >= 5);
    CHECK(perf.draw_count >= 5);
    CHECK(perf.primitive_count >= 20);
    CHECK(perf.tile_count >= 5);
    CHECK(perf.upload_bytes >= far_texture.size() + near_texture.size() +
                                   far_light.size() + near_light.size());
    CHECK(perf.tile_bin_time_ns > 0);
    CHECK(perf.raster_time_ns > 0);
  }

  {
    qr_frame *frame = nullptr;
    qr_frame_desc frame_desc{};
    qr_world_draw_desc bad_draw{
        .world = world,
        .polygons = polygons.data(),
        .polygon_count = polygons.size(),
        .debug_mode = QR_DEBUG_SHADED,
    };
    std::array<qr_world_polygon_desc, 1> bad_polygons{{
        {.surface = 99, .vertices = far_quad.data(),
         .vertex_count = static_cast<std::uint32_t>(far_quad.size())},
    }};
    auto nonfinite_quad = far_quad;
    nonfinite_quad[0].z = std::numeric_limits<float>::quiet_NaN();

    REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
    CHECK(qr_frame_draw_world(frame, &bad_draw) == QR_ERROR_INVALID_ARGUMENT);
    bad_draw.colormap = colormap.data();
    bad_draw.colormap_size = colormap.size();
    bad_draw.polygons = bad_polygons.data();
    bad_draw.polygon_count = bad_polygons.size();
    CHECK(qr_frame_draw_world(frame, &bad_draw) == QR_ERROR_INVALID_ARGUMENT);
    bad_polygons[0].surface = 0;
    bad_polygons[0].vertices = nonfinite_quad.data();
    CHECK(qr_frame_draw_world(frame, &bad_draw) == QR_ERROR_INVALID_ARGUMENT);
    REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
  }

  qr_destroy(ctx);
}

TEST_CASE("Quake raster - depth order validation flags fixed-key ties",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 16;
  constexpr std::uint32_t HEIGHT = 16;
  const std::array<std::uint8_t, 4> texture{1, 1, 1, 1};
  const std::array<std::uint8_t, 4> lightmap{1, 1, 1, 1};
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 1,
      .max_lightmaps = 1,
      .max_surfaces = 1,
      .max_worlds = 1,
      .texture_atlas_bytes = texture.size(),
      .lightmap_atlas_bytes = lightmap.size(),
  };
  qr_context *ctx = create_context_or_skip(desc);
  qr_texture tex = QR_INVALID_HANDLE;
  qr_lightmap lm = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_texture_desc tex_desc{
      .mips =
          {
              {.pixels = texture.data(), .width = 2, .height = 2, .stride = 2},
          },
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc lm_desc{
      .pixels = lightmap.data(),
      .width = 2,
      .height = 2,
      .stride = 2,
  };
  qr_world_surface_desc surface{};
  const std::array<qr_world_vertex, 4> farther{{
      {.x = 0.0F, .y = 0.0F, .z = 0.10020F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 0.0F, .z = 0.10020F, .u = 2.0F, .v = 0.0F,
       .light_u = 2.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 16.0F, .z = 0.10020F, .u = 2.0F, .v = 2.0F,
       .light_u = 2.0F, .light_v = 2.0F},
      {.x = 0.0F, .y = 16.0F, .z = 0.10020F, .u = 0.0F, .v = 2.0F,
       .light_u = 0.0F, .light_v = 2.0F},
  }};
  const std::array<qr_world_vertex, 4> nearer{{
      {.x = 0.0F, .y = 0.0F, .z = 0.10010F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 0.0F, .z = 0.10010F, .u = 2.0F, .v = 0.0F,
       .light_u = 2.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 16.0F, .z = 0.10010F, .u = 2.0F, .v = 2.0F,
       .light_u = 2.0F, .light_v = 2.0F},
      {.x = 0.0F, .y = 16.0F, .z = 0.10010F, .u = 0.0F, .v = 2.0F,
       .light_u = 0.0F, .light_v = 2.0F},
  }};
  std::array<qr_world_polygon_desc, 2> polygons{{
      {.surface = 0, .vertices = farther.data(),
       .vertex_count = static_cast<std::uint32_t>(farther.size())},
      {.surface = 0, .vertices = nearer.data(),
       .vertex_count = static_cast<std::uint32_t>(nearer.size())},
  }};
  std::array<std::uint8_t, WIDTH * HEIGHT> pixels{};
  qr_frame *frame = nullptr;
  qr_frame_desc frame_desc{};
  qr_world_draw_desc draw_desc{
      .world = QR_INVALID_HANDLE,
      .polygons = polygons.data(),
      .polygon_count = polygons.size(),
      .debug_mode = QR_DEBUG_DEPTH_ORDER,
  };

  REQUIRE(qr_upload_texture(ctx, &tex_desc, &tex) == QR_SUCCESS);
  REQUIRE(qr_upload_lightmap(ctx, &lm_desc, &lm) == QR_SUCCESS);
  surface.texture = tex;
  surface.lightmap = lm;
  REQUIRE(qr_create_world(ctx, &surface, 1, &world) == QR_SUCCESS);
  draw_desc.world = world;

  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
  REQUIRE(qr_frame_clear_indexed(frame, 0) == QR_SUCCESS);
  REQUIRE(qr_frame_draw_world(frame, &draw_desc) == QR_SUCCESS);
  REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
  REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
          QR_SUCCESS);
  CHECK(std::any_of(pixels.begin(), pixels.end(),
                    [](std::uint8_t pixel) { return pixel == 255; }));

  qr_destroy(ctx);
}

TEST_CASE("Quake raster - fixed-key depth ties still choose nearest surface",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 16;
  constexpr std::uint32_t HEIGHT = 16;
  const std::array<std::uint8_t, 4> far_texture{11, 11, 11, 11};
  const std::array<std::uint8_t, 4> near_texture{22, 22, 22, 22};
  const std::array<std::uint8_t, 4> lightmap{1, 1, 1, 1};
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 2,
      .max_lightmaps = 1,
      .max_surfaces = 2,
      .max_worlds = 1,
      .texture_atlas_bytes = far_texture.size() + near_texture.size(),
      .lightmap_atlas_bytes = lightmap.size(),
  };
  qr_context *ctx = create_context_or_skip(desc);
  qr_texture far_tex = QR_INVALID_HANDLE;
  qr_texture near_tex = QR_INVALID_HANDLE;
  qr_lightmap lm = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_texture_desc far_tex_desc{
      .mips = {{.pixels = far_texture.data(), .width = 2, .height = 2,
                .stride = 2}},
      .mip_count = 1,
      .flags = 0,
  };
  qr_texture_desc near_tex_desc{
      .mips = {{.pixels = near_texture.data(), .width = 2, .height = 2,
                .stride = 2}},
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc lm_desc{
      .pixels = lightmap.data(),
      .width = 2,
      .height = 2,
      .stride = 2,
  };
  const std::array<qr_world_vertex, 4> farther{{
      {.x = 0.0F, .y = 0.0F, .z = 0.10020F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 0.0F, .z = 0.10020F, .u = 2.0F, .v = 0.0F,
       .light_u = 2.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 16.0F, .z = 0.10020F, .u = 2.0F, .v = 2.0F,
       .light_u = 2.0F, .light_v = 2.0F},
      {.x = 0.0F, .y = 16.0F, .z = 0.10020F, .u = 0.0F, .v = 2.0F,
       .light_u = 0.0F, .light_v = 2.0F},
  }};
  const std::array<qr_world_vertex, 4> nearer{{
      {.x = 0.0F, .y = 0.0F, .z = 0.10010F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 0.0F, .z = 0.10010F, .u = 2.0F, .v = 0.0F,
       .light_u = 2.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 16.0F, .z = 0.10010F, .u = 2.0F, .v = 2.0F,
       .light_u = 2.0F, .light_v = 2.0F},
      {.x = 0.0F, .y = 16.0F, .z = 0.10010F, .u = 0.0F, .v = 2.0F,
       .light_u = 0.0F, .light_v = 2.0F},
  }};
  std::array<qr_world_surface_desc, 2> surfaces{};
  std::array<qr_world_polygon_desc, 2> polygons{{
      {.surface = 0, .vertices = farther.data(),
       .vertex_count = static_cast<std::uint32_t>(farther.size())},
      {.surface = 1, .vertices = nearer.data(),
       .vertex_count = static_cast<std::uint32_t>(nearer.size())},
  }};
  qr_frame *frame = nullptr;
  qr_frame_desc frame_desc{};
  qr_world_draw_desc draw_desc{
      .world = QR_INVALID_HANDLE,
      .polygons = polygons.data(),
      .polygon_count = polygons.size(),
      .debug_mode = QR_DEBUG_TEXTURE_ONLY,
  };
  std::array<std::uint8_t, WIDTH * HEIGHT> pixels{};

  REQUIRE(qr_upload_texture(ctx, &far_tex_desc, &far_tex) == QR_SUCCESS);
  REQUIRE(qr_upload_texture(ctx, &near_tex_desc, &near_tex) == QR_SUCCESS);
  REQUIRE(qr_upload_lightmap(ctx, &lm_desc, &lm) == QR_SUCCESS);
  surfaces[0].texture = far_tex;
  surfaces[0].lightmap = lm;
  surfaces[1].texture = near_tex;
  surfaces[1].lightmap = lm;
  REQUIRE(qr_create_world(ctx, surfaces.data(), surfaces.size(), &world) ==
          QR_SUCCESS);
  draw_desc.world = world;

  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
  REQUIRE(qr_frame_clear_indexed(frame, 0) == QR_SUCCESS);
  REQUIRE(qr_frame_draw_world(frame, &draw_desc) == QR_SUCCESS);
  REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
  REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
          QR_SUCCESS);
  CHECK(pixels[static_cast<std::size_t>(8) * WIDTH + 8U] == 22);

  qr_destroy(ctx);
}

TEST_CASE("Quake raster - tiled raster reports bounded overflow",
          "[quake_raster][device]") {
  constexpr std::uint32_t WIDTH = 16;
  constexpr std::uint32_t HEIGHT = 16;
  constexpr std::size_t POLYGON_COUNT =
      (QR_TILE_TRIANGLE_CAPACITY / 2U) + 2U;
  const std::array<std::uint8_t, 4> texture{42, 42, 42, 42};
  const std::array<std::uint8_t, 4> lightmap{7, 7, 7, 7};
  qr_desc desc{
      .width = WIDTH,
      .height = HEIGHT,
      .device_index = 0,
      .output_mode = QR_OUTPUT_NOOUTPUT,
      .framebuffer_format = QR_FRAMEBUFFER_INDEXED8,
      .max_textures = 1,
      .max_lightmaps = 1,
      .max_surfaces = 1,
      .max_worlds = 1,
      .texture_atlas_bytes = texture.size(),
      .lightmap_atlas_bytes = lightmap.size(),
  };
  qr_context *ctx = create_context_or_skip(desc);
  qr_texture tex = QR_INVALID_HANDLE;
  qr_lightmap lm = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_texture_desc tex_desc{
      .mips =
          {
              {.pixels = texture.data(), .width = 2, .height = 2, .stride = 2},
          },
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc lm_desc{
      .pixels = lightmap.data(),
      .width = 2,
      .height = 2,
      .stride = 2,
  };
  qr_world_surface_desc surface{};
  const std::array<qr_world_vertex, 4> quad{{
      {.x = 0.0F, .y = 0.0F, .z = 0.25F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 0.0F, .z = 0.25F, .u = 2.0F, .v = 0.0F,
       .light_u = 2.0F, .light_v = 0.0F},
      {.x = 16.0F, .y = 16.0F, .z = 0.25F, .u = 2.0F, .v = 2.0F,
       .light_u = 2.0F, .light_v = 2.0F},
      {.x = 0.0F, .y = 16.0F, .z = 0.25F, .u = 0.0F, .v = 2.0F,
       .light_u = 0.0F, .light_v = 2.0F},
  }};
  std::vector<qr_world_polygon_desc> polygons(POLYGON_COUNT);
  std::array<std::uint8_t, WIDTH * HEIGHT> pixels{};
  qr_frame *frame = nullptr;
  qr_frame_desc frame_desc{};
  qr_world_draw_desc draw_desc{};
  qr_raster_stats stats{};

  REQUIRE(qr_get_raster_stats(nullptr, &stats) == QR_ERROR_INVALID_ARGUMENT);
  REQUIRE(qr_upload_texture(ctx, &tex_desc, &tex) == QR_SUCCESS);
  REQUIRE(qr_upload_lightmap(ctx, &lm_desc, &lm) == QR_SUCCESS);
  surface.texture = tex;
  surface.lightmap = lm;
  REQUIRE(qr_create_world(ctx, &surface, 1, &world) == QR_SUCCESS);

  for (qr_world_polygon_desc &polygon : polygons) {
    polygon.surface = 0;
    polygon.vertices = quad.data();
    polygon.vertex_count = static_cast<std::uint32_t>(quad.size());
  }

  draw_desc.world = world;
  draw_desc.polygons = polygons.data();
  draw_desc.polygon_count = polygons.size();
  draw_desc.debug_mode = QR_DEBUG_TEXTURE_ONLY;
  REQUIRE(qr_begin_frame(ctx, &frame_desc, &frame) == QR_SUCCESS);
  REQUIRE(qr_frame_clear_indexed(frame, 0) == QR_SUCCESS);
  REQUIRE(qr_frame_draw_world(frame, &draw_desc) == QR_SUCCESS);
  REQUIRE(qr_end_frame(frame) == QR_SUCCESS);
  REQUIRE(qr_get_raster_stats(ctx, &stats) == QR_SUCCESS);
  CHECK(stats.tile_count == 1);
  CHECK(stats.triangle_count == POLYGON_COUNT * 2U);
  CHECK(stats.max_tile_triangle_count == QR_TILE_TRIANGLE_CAPACITY);
  CHECK(stats.overflow_tile_count == 1);
  CHECK(stats.overflow_reference_count == (POLYGON_COUNT * 2U) -
                                              QR_TILE_TRIANGLE_CAPACITY);

  REQUIRE(qr_read_indexed(ctx, pixels.data(), pixels.size(), WIDTH) ==
          QR_SUCCESS);
  CHECK(std::all_of(pixels.begin(), pixels.end(),
                    [](std::uint8_t pixel) { return pixel == 42; }));

  qr_destroy(ctx);
}
