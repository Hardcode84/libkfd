/*===-- libkfd/quake_raster.h - Quake compute rasterizer API ------*- C -*-===*\
 *
 * Experimental C99 renderer boundary for a compute software rasterizer that can
 * run with or without an active presentation target.
 *
\*===----------------------------------------------------------------------===*/

#ifndef LIBKFD_QUAKE_RASTER_H
#define LIBKFD_QUAKE_RASTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qr_context qr_context;
typedef struct qr_frame qr_frame;

#define QR_API_VERSION_MAJOR 0U
#define QR_API_VERSION_MINOR 9U
#define QR_API_VERSION_PATCH 0U
#define QR_TEXTURE_MIP_COUNT 4U
#define QR_INVALID_HANDLE 0U
#define QR_COLORMAP_SIZE (256U * 256U)
#define QR_TILE_SIZE 16U
#define QR_TILE_TRIANGLE_CAPACITY 256U
#define QR_DEPTH_KEY_SCALE 4096U
#define QR_SURFACE_SKY 4U
#define QR_SURFACE_TURBULENT 0x10U
#define QR_SURFACE_CUTOUT 0x80U
#define QR_SKY_COLOR_INDEX 109U

typedef enum qr_result {
  QR_SUCCESS = 0,
  QR_ERROR_INVALID_ARGUMENT,
  QR_ERROR_UNSUPPORTED,
  QR_ERROR_OVERFLOW,
  QR_ERROR_OUT_OF_MEMORY,
  QR_ERROR_BUFFER_TOO_SMALL,
  QR_ERROR_NO_SPACE,
  QR_ERROR_BUSY,
  QR_ERROR_IO,
  QR_ERROR_NOT_FOUND,
  QR_ERROR_SYSTEM,
  QR_ERROR_GPU
} qr_result;

typedef enum qr_output_mode {
  QR_OUTPUT_NOOUTPUT = 0,
  QR_OUTPUT_PRESENT = 1
} qr_output_mode;

typedef enum qr_framebuffer_format {
  QR_FRAMEBUFFER_INDEXED8 = 0
} qr_framebuffer_format;

typedef enum qr_debug_mode {
  QR_DEBUG_SHADED = 0,
  /* Visualizes surface ids as flat indexed colors. */
  QR_DEBUG_FLAT_SURFACE_ID,
  /* Visualizes the fixed-point depth key used for ordering. */
  QR_DEBUG_DEPTH,
  /* Samples only texture/lightmap data, bypassing shaded colormap lookup. */
  QR_DEBUG_TEXTURE_ONLY,
  QR_DEBUG_LIGHT_ONLY,
  /* Marks pixels where fixed-key and float-depth ordering would disagree. */
  QR_DEBUG_DEPTH_ORDER
} qr_debug_mode;

typedef uint32_t qr_texture;
typedef uint32_t qr_lightmap;
typedef uint32_t qr_world;
typedef uint32_t qr_alias_model;

typedef qr_result (*qr_present_callback)(void *userdata, const uint32_t *xrgb,
                                         uint32_t width, uint32_t height,
                                         size_t stride_pixels);

typedef struct qr_desc {
  uint32_t width;
  uint32_t height;
  size_t device_index;
  qr_output_mode output_mode;
  qr_framebuffer_format framebuffer_format;
  qr_present_callback present;
  void *present_userdata;
  const uint32_t *present_palette_xrgb;
  /* Zero capacity/budget fields select renderer defaults. */
  uint32_t max_textures;
  uint32_t max_lightmaps;
  uint32_t max_surfaces;
  uint32_t max_worlds;
  uint32_t max_frame_triangles;
  uint32_t max_alias_models;
  uint32_t max_alias_triangles;
  size_t texture_atlas_bytes;
  size_t lightmap_atlas_bytes;
} qr_desc;

typedef struct qr_frame_desc {
  /* Reserved for future per-frame flags; callers must initialize it to zero. */
  uint32_t reserved;
} qr_frame_desc;

typedef struct qr_texture_mip_desc {
  const void *pixels;
  uint32_t width;
  uint32_t height;
  size_t stride;
} qr_texture_mip_desc;

typedef struct qr_texture_desc {
  qr_texture_mip_desc mips[QR_TEXTURE_MIP_COUNT];
  uint32_t mip_count;
  uint32_t flags;
} qr_texture_desc;

typedef struct qr_lightmap_desc {
  const void *pixels;
  uint32_t width;
  uint32_t height;
  size_t stride;
} qr_lightmap_desc;

typedef struct qr_world_surface_desc {
  qr_texture texture;
  qr_lightmap lightmap;
  uint32_t flags;
  float plane[4];
  float tex_s[4];
  float tex_t[4];
  float light_s[4];
  float light_t[4];
} qr_world_surface_desc;

typedef struct qr_world_vertex {
  float x;
  float y;
  float z;
  float u;
  float v;
  float light_u;
  float light_v;
} qr_world_vertex;

typedef struct qr_world_polygon_desc {
  uint32_t surface;
  const qr_world_vertex *vertices;
  uint32_t vertex_count;
} qr_world_polygon_desc;

typedef struct qr_world_draw_desc {
  qr_world world;
  const qr_world_polygon_desc *polygons;
  size_t polygon_count;
  const uint8_t *colormap;
  size_t colormap_size;
  qr_debug_mode debug_mode;
  float time_seconds;
} qr_world_draw_desc;

typedef struct qr_alias_triangle_desc {
  uint32_t surface;
  qr_world_vertex v0;
  qr_world_vertex v1;
  qr_world_vertex v2;
} qr_alias_triangle_desc;

typedef struct qr_alias_model_desc {
  qr_world world;
  const qr_alias_triangle_desc *triangles;
  size_t triangle_count;
} qr_alias_model_desc;

typedef struct qr_alias_draw_desc {
  qr_alias_model model;
  const uint8_t *colormap;
  size_t colormap_size;
  qr_debug_mode debug_mode;
  float time_seconds;
} qr_alias_draw_desc;

typedef struct qr_sprite_draw_desc {
  qr_world world;
  uint32_t surface;
  float x0;
  float y0;
  float x1;
  float y1;
  float z;
  float u0;
  float v0;
  float u1;
  float v1;
  const uint8_t *colormap;
  size_t colormap_size;
  qr_debug_mode debug_mode;
  float time_seconds;
} qr_sprite_draw_desc;

typedef struct qr_particle_desc {
  qr_world world;
  uint32_t surface;
  float x;
  float y;
  float z;
  float size;
  float u;
  float v;
} qr_particle_desc;

typedef struct qr_particles_draw_desc {
  const qr_particle_desc *particles;
  size_t particle_count;
  const uint8_t *colormap;
  size_t colormap_size;
  qr_debug_mode debug_mode;
  float time_seconds;
} qr_particles_draw_desc;

typedef struct qr_capacity_info {
  uint32_t texture_count;
  uint32_t texture_capacity;
  size_t texture_atlas_used;
  size_t texture_atlas_capacity;
  uint32_t lightmap_count;
  uint32_t lightmap_capacity;
  size_t lightmap_atlas_used;
  size_t lightmap_atlas_capacity;
  uint32_t surface_count;
  uint32_t surface_capacity;
  uint32_t world_count;
  uint32_t world_capacity;
  uint32_t frame_triangle_capacity;
  uint32_t alias_model_count;
  uint32_t alias_model_capacity;
  uint32_t alias_triangle_count;
  uint32_t alias_triangle_capacity;
} qr_capacity_info;

typedef struct qr_raster_stats {
  uint32_t tile_size;
  uint32_t tile_cols;
  uint32_t tile_rows;
  uint32_t tile_count;
  uint32_t tile_triangle_capacity;
  uint32_t triangle_count;
  uint32_t occupied_tile_count;
  uint32_t max_tile_triangle_count;
  uint32_t overflow_tile_count;
  uint32_t overflow_reference_count;
  uint32_t depth_key_scale;
  uint32_t depth_bound_tile_count;
  /* Triangle references visited by the most recent submitted draw batch. */
  uint32_t hiz_candidate_reference_count;
  /* Overflow references that forced full-list tile fallback in that batch. */
  uint32_t hiz_overflow_fallback_count;
} qr_raster_stats;

typedef struct qr_perf_counters {
  uint64_t frame_count;
  uint64_t clear_count;
  uint64_t draw_count;
  uint64_t resolve_count;
  uint64_t upload_bytes;
  uint64_t primitive_count;
  uint64_t tile_count;
  uint64_t tile_overflow_count;
  uint64_t tile_overflow_reference_count;
  uint64_t hiz_candidate_reference_count;
  uint64_t hiz_overflow_fallback_count;
  uint64_t clear_time_ns;
  uint64_t tile_bin_time_ns;
  uint64_t raster_time_ns;
  uint64_t resolve_time_ns;
} qr_perf_counters;

/* Returns QR_API_VERSION_* packed in 8-bit fields as 0x00MMmmPP. */
uint32_t qr_api_version(void);
const char *qr_strerror(qr_result code);
uint64_t qr_pack_depth_payload(uint32_t depth_key, uint32_t payload);

/*
 * Creates a renderer context. The context owns all GPU buffers, loaded kernels,
 * and synchronization objects it allocates. The caller owns only the returned
 * handle and must destroy it with qr_destroy().
 *
 * QR_OUTPUT_PRESENT uses the present callback in qr_desc. qr_create() copies
 * the supplied 256-entry XRGB palette, and qr_end_frame() resolves the indexed
 * framebuffer before invoking the callback with tightly packed XRGB rows.
 *
 * Contexts are not thread-safe. Calls that operate on the same qr_context or
 * qr_frame must be externally serialized by the caller.
 */
qr_result qr_create(const qr_desc *desc, qr_context **out);
void qr_destroy(qr_context *ctx);

uint32_t qr_width(const qr_context *ctx);
uint32_t qr_height(const qr_context *ctx);
qr_output_mode qr_output(const qr_context *ctx);

/*
 * Frame handles are borrowed from their parent context and stay valid until
 * qr_end_frame(). The current implementation executes clear/resolve work
 * synchronously before returning from the API that submits it.
 *
 * qr_frame_draw_world() takes finite screen-space polygon vertices. The CPU
 * setup path triangulates each polygon as a fan, writes the transient command
 * list into context-owned GPU memory, then a 16x16 tiled GPU raster pass draws
 * opaque surfaces with nearest texture/lightmap sampling. In shaded mode,
 * colormap must point at QR_COLORMAP_SIZE bytes using light*256 + texel
 * indexing. Submit the full visible opaque world batch in one call; each draw
 * call rebuilds its own depth state and tile lists.
 */
qr_result qr_begin_frame(qr_context *ctx, const qr_frame_desc *desc,
                         qr_frame **out);
qr_result qr_frame_clear_indexed(qr_frame *frame, uint8_t color);
qr_result qr_frame_draw_world(qr_frame *frame, const qr_world_draw_desc *desc);
qr_result qr_frame_draw_alias_model(qr_frame *frame,
                                    const qr_alias_draw_desc *desc);
qr_result qr_frame_draw_sprite(qr_frame *frame,
                               const qr_sprite_draw_desc *desc);
qr_result qr_frame_draw_particles(qr_frame *frame,
                                  const qr_particles_draw_desc *desc);
qr_result qr_end_frame(qr_frame *frame);

/*
 * Readback and dump helpers are intended for headless diagnostics. A zero stride
 * means tightly packed rows. qr_dump_indexed() writes raw indexed bytes;
 * qr_dump_xrgb() writes binary PPM (P6) RGB data after palette resolve.
 */
qr_result qr_read_indexed(qr_context *ctx, void *dst, size_t dst_size,
                          size_t dst_stride);
qr_result qr_read_xrgb(qr_context *ctx, const uint32_t *palette_xrgb, void *dst,
                       size_t dst_size, size_t dst_stride_pixels);

qr_result qr_dump_indexed(qr_context *ctx, const char *path);
qr_result qr_dump_xrgb(qr_context *ctx, const uint32_t *palette_xrgb,
                       const char *path);

/*
 * Persistent resources are append-only and owned by the context. Handles are
 * stable for the lifetime of the context and start at 1; QR_INVALID_HANDLE is
 * never returned.
 */
qr_result qr_upload_texture(qr_context *ctx, const qr_texture_desc *desc,
                            qr_texture *out);
qr_result qr_upload_lightmap(qr_context *ctx, const qr_lightmap_desc *desc,
                             qr_lightmap *out);
qr_result qr_update_lightmap(qr_context *ctx, qr_lightmap lightmap,
                             const qr_lightmap_desc *desc);
qr_result qr_create_world(qr_context *ctx, const qr_world_surface_desc *surfaces,
                          size_t surface_count, qr_world *out);
qr_result qr_upload_alias_model(qr_context *ctx,
                                const qr_alias_model_desc *desc,
                                qr_alias_model *out);
qr_result qr_get_capacity_info(qr_context *ctx, qr_capacity_info *out);
qr_result qr_get_perf_counters(qr_context *ctx, qr_perf_counters *out);

/*
 * Returns counters from the most recent triangle draw call, including world,
 * alias, sprite, or particle submissions. Overflow references are handled by a
 * deterministic full-list fallback for that tile.
 */
qr_result qr_get_raster_stats(qr_context *ctx, qr_raster_stats *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBKFD_QUAKE_RASTER_H */
