# Quake Raster Host/Device ABI

The compute rasterizer uses plain C records in host memory and passes those
records directly to AMDGPU kernels. The following definitions are the ABI
contract and must stay in lockstep:

- Host records in `lib/quake_raster.c`: `qr_raster_vertex`,
  `qr_raster_triangle`, `qr_texture_record`, `qr_lightmap_record`,
  `qr_surface_record`, `qr_tile_bin_args`, and `qr_world_raster_args`.
- Device records in `lib/quake_raster/kernels/tile_bin.c` and
  `lib/quake_raster/kernels/world_raster_tiled.c`: the matching `Qr*`
  structs.
- Public constants in `include/libkfd/quake_raster.h`: `QR_TEXTURE_MIP_COUNT`,
  `QR_TILE_SIZE`, `QR_TILE_TRIANGLE_CAPACITY`, `QR_DEPTH_KEY_SCALE`,
  `QR_SURFACE_*`, and `QR_SKY_COLOR_INDEX`.

When changing any field order, type, pointer, or constant, update both the host
and device copies in the same commit. Keep integer fields 32-bit wide; the
kernel sources use `unsigned` because the AMDGPU C dialect maps it to a 32-bit
scalar. Host-side records use `uint32_t`.

Review checklist for ABI changes:

- Rebuild all `quake_raster_*` kernels for the configured architectures.
- Run `quake_raster_test` on a KFD-capable machine.
- Inspect `qr_raster_stats` on a known scene to confirm tile counts and overflow
  accounting still match expectations.
- Update this document if a new host/device record is added.
