#include "libkfd/quake_raster.h"

#include <array>
#include <cstdint>
#include <iostream>

int main() {
  constexpr std::uint32_t WIDTH = 64;
  constexpr std::uint32_t HEIGHT = 64;
  std::array<std::uint8_t, 16> texture{};
  std::array<std::uint8_t, 16> lightmap{};
  std::array<std::uint8_t, QR_COLORMAP_SIZE> colormap{};
  qr_context *ctx = nullptr;
  qr_texture texture_handle = QR_INVALID_HANDLE;
  qr_lightmap lightmap_handle = QR_INVALID_HANDLE;
  qr_world world = QR_INVALID_HANDLE;
  qr_frame *frame = nullptr;
  qr_frame_desc frame_desc{};

  for (std::size_t i = 0; i < texture.size(); ++i) {
    texture[i] = static_cast<std::uint8_t>(i + 1U);
    lightmap[i] = 16U;
  }
  for (std::uint32_t light = 0; light < 256U; ++light) {
    for (std::uint32_t texel = 0; texel < 256U; ++texel) {
      colormap[(light << 8U) | texel] =
          static_cast<std::uint8_t>((light + texel) & 0xffU);
    }
  }

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
      .max_frame_triangles = 2,
      .texture_atlas_bytes = texture.size(),
      .lightmap_atlas_bytes = lightmap.size(),
  };
  qr_result result = qr_create(&desc, &ctx);
  if (result != QR_SUCCESS) {
    std::cerr << "qr_create: " << qr_strerror(result) << '\n';
    return 77;
  }

  qr_texture_desc texture_desc{
      .mips = {{.pixels = texture.data(), .width = 4, .height = 4, .stride = 4}},
      .mip_count = 1,
      .flags = 0,
  };
  qr_lightmap_desc lightmap_desc{
      .pixels = lightmap.data(),
      .width = 4,
      .height = 4,
      .stride = 4,
  };
  qr_world_surface_desc surface{};
  result = qr_upload_texture(ctx, &texture_desc, &texture_handle);
  if (result == QR_SUCCESS) {
    result = qr_upload_lightmap(ctx, &lightmap_desc, &lightmap_handle);
  }
  surface.texture = texture_handle;
  surface.lightmap = lightmap_handle;
  if (result == QR_SUCCESS) {
    result = qr_create_world(ctx, &surface, 1U, &world);
  }
  if (result != QR_SUCCESS) {
    std::cerr << "resource setup: " << qr_strerror(result) << '\n';
    qr_destroy(ctx);
    return 1;
  }

  std::array<qr_world_vertex, 4> quad{{
      {.x = 0.0F, .y = 0.0F, .z = 0.5F, .u = 0.0F, .v = 0.0F,
       .light_u = 0.0F, .light_v = 0.0F},
      {.x = 64.0F, .y = 0.0F, .z = 0.5F, .u = 4.0F, .v = 0.0F,
       .light_u = 4.0F, .light_v = 0.0F},
      {.x = 64.0F, .y = 64.0F, .z = 0.5F, .u = 4.0F, .v = 4.0F,
       .light_u = 4.0F, .light_v = 4.0F},
      {.x = 0.0F, .y = 64.0F, .z = 0.5F, .u = 0.0F, .v = 4.0F,
       .light_u = 0.0F, .light_v = 4.0F},
  }};
  qr_world_polygon_desc polygon{
      .surface = 0,
      .vertices = quad.data(),
      .vertex_count = static_cast<std::uint32_t>(quad.size()),
  };
  qr_world_draw_desc draw{
      .world = world,
      .polygons = &polygon,
      .polygon_count = 1,
      .colormap = colormap.data(),
      .colormap_size = colormap.size(),
      .debug_mode = QR_DEBUG_SHADED,
  };

  result = qr_begin_frame(ctx, &frame_desc, &frame);
  if (result == QR_SUCCESS) {
    result = qr_frame_clear_indexed(frame, 0U);
  }
  if (result == QR_SUCCESS) {
    result = qr_frame_draw_world(frame, &draw);
  }
  if (result == QR_SUCCESS) {
    result = qr_end_frame(frame);
  }
  qr_perf_counters perf{};
  if (result == QR_SUCCESS) {
    result = qr_get_perf_counters(ctx, &perf);
  }
  if (result != QR_SUCCESS) {
    std::cerr << "render: " << qr_strerror(result) << '\n';
    qr_destroy(ctx);
    return 1;
  }
  std::cout << "frames=" << perf.frame_count << " draws=" << perf.draw_count
            << " primitives=" << perf.primitive_count
            << " tiles=" << perf.tile_count
            << " raster_ns=" << perf.raster_time_ns << '\n';
  qr_destroy(ctx);
  return 0;
}
