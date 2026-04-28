#ifndef RASTERDEMO_RASTER_COMMON_H
#define RASTERDEMO_RASTER_COMMON_H

typedef unsigned RasterU32;

enum {
  NUM_BUFFERS = 3u,
  TILE_SIZE = 32u,
  BLOCK_X = 16u,
  BLOCK_Y = 16u,
  CLAIM_BLOCK_X = 32u,
  CLAIM_BLOCK_Y = 1u,
  PERSISTENT_BLOCK_X = 16u,
  PERSISTENT_BLOCK_Y = 8u,
  DEFAULT_PERSISTENT_WGS = 48u,
  STRIDE_ALIGN = 256u,
  CLEAR_COLOR = 0xff102030u,
  SLICES = 64u,
  RINGS = 48u,
  RASTER_TILE_SIZE = TILE_SIZE,
  SUBTILE_WIDTH = 16u,
  SUBTILE_HEIGHT = 4u,
  SUBTILES_X = RASTER_TILE_SIZE / SUBTILE_WIDTH,
  SUBTILES_Y = RASTER_TILE_SIZE / SUBTILE_HEIGHT,
  NUM_SUBTILES = SUBTILES_X * SUBTILES_Y,
  SUBTILE_PIXELS = SUBTILE_WIDTH * SUBTILE_HEIGHT,
  PERSISTENT_WAVE_COUNT = (PERSISTENT_BLOCK_X * PERSISTENT_BLOCK_Y) / 32u,
  SUBTILE_PASSES_PER_WAVE =
      (NUM_SUBTILES + PERSISTENT_WAVE_COUNT - 1u) / PERSISTENT_WAVE_COUNT,
};
#define CLEAR_DEPTH 1.0f
#define PI 3.14159265358979323846f
#define TAU 6.28318530717958647692f

typedef struct DemoVertex {
  float x;
  float y;
  float z;
  float u;
  float v;
} DemoVertex;

typedef struct DemoPrimitive {
  DemoVertex v0;
  DemoVertex v1;
  DemoVertex v2;
} DemoPrimitive;

typedef struct TileRange {
  RasterU32 offset;
  RasterU32 count;
} TileRange;

typedef struct RasterArgs {
  const DemoPrimitive *prims;
  const RasterU32 *tile_indices;
  const TileRange *tile_ranges;
  RasterU32 *color;
  float *depth;
  RasterU32 width;
  RasterU32 height;
  RasterU32 pitch;
  RasterU32 tile_size;
  RasterU32 tiles_x;
  RasterU32 clear_color;
  float clear_depth;
} RasterArgs;

typedef struct PersistentControl {
  volatile RasterU32 terminate;
  volatile RasterU32 current_epoch;
  volatile RasterU32 sealed_epoch;
  volatile RasterU32 closing_epoch;
  volatile RasterU32 completed_epoch;
  volatile RasterU32 active_cursor;
  volatile RasterU32 active_count;
  volatile RasterU32 render_done;
} PersistentControl;

typedef struct ClaimFrameArgs {
  const DemoPrimitive *prims;
  const RasterU32 *tile_indices;
  const TileRange *tile_ranges;
  const volatile RasterU32 *active_tiles;
  volatile RasterU32 *active_cursor;
  RasterU32 *color;
  float *depth;
  RasterU32 width;
  RasterU32 height;
  RasterU32 pitch;
  RasterU32 tile_size;
  RasterU32 tiles_x;
  RasterU32 tiles_y;
  RasterU32 clear_color;
  float clear_depth;
  RasterU32 clear_only;
  RasterU32 active_count;
} ClaimFrameArgs;

typedef struct PersistentArgs {
  PersistentControl *control;
  const DemoPrimitive *prims;
  const RasterU32 *tile_indices;
  const TileRange *tile_ranges;
  const volatile RasterU32 *active_tiles;
  RasterU32 *color;
  float *depth;
  RasterU32 width;
  RasterU32 height;
  RasterU32 pitch;
  RasterU32 tile_size;
  RasterU32 tiles_x;
  RasterU32 tiles_y;
  RasterU32 clear_color;
  float clear_depth;
  RasterU32 clear_only;
} PersistentArgs;

typedef struct PersistentLaunchArgs {
  const PersistentArgs *args;
} PersistentLaunchArgs;

typedef struct ProbeArgs {
  RasterU32 *out;
} ProbeArgs;

#endif // RASTERDEMO_RASTER_COMMON_H
