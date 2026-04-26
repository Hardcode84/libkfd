# Quake Compute Rasterizer TODO

This TODO turns `quake-compute-rasterizer-plan.md` into implementation
milestones. The intent is to keep the renderer library independent from
QrustyQuake while still letting QrustyQuake be the first frontend.

## Milestone 0: Renderer Library Skeleton

Status: started.

- [x] Add isolated `quake_raster` C99 library target.
- [x] Build with strict diagnostics: `-Wall -Wextra -Wpedantic -Werror`.
- [x] Add opaque `qr_context` and `qr_frame` API.
- [x] Add `QR_OUTPUT_NOOUTPUT`.
- [x] Add dummy indexed framebuffer clear/readback.
- [x] Add headless smoke test.
- [x] Replace ad hoc `calloc/free` with renderer-owned allocation helpers.
- [x] Add public API versioning.
- [x] Add structured error enum instead of raw errno-only returns.
- [x] Add API docs for ownership, threading, and frame lifetime.

Exit criteria:

- Library builds as C99 with strict warnings.
- No windowing dependency is needed for tests.
- No renderer globals.

## Milestone 1: Nooutput Diagnostics

- [x] Add `qr_dump_indexed()` for raw indexed framebuffer dumps.
- [x] Add `qr_dump_xrgb()` or equivalent resolved-output inspection path.
- [x] Define dump file format: raw, simple PPM, or both.
- [x] Add deterministic clear/fill pattern test.
- [x] Add stride/readback tests for padded destinations.
- [x] Add failure tests for undersized readback buffers.

Exit criteria:

- A headless CI run can produce inspectable frame output.
- Dump output does not require SDL, X11, or a present surface.

## Milestone 2: GPU Kernel Package

- [x] Add renderer-owned GPU kernel source directory.
- [x] Add CMake helper to build renderer kernels for supported AMDGPU targets.
- [x] Add kernel selection/load path inside `qr_context`.
- [x] Add GPU clear indexed kernel.
- [x] Add indexed-to-XRGB resolve kernel.
- [x] Add nooutput test that clears through GPU dispatch, not CPU `memset`.
- [x] Add readback path for indexed framebuffer after GPU work completion.

Exit criteria:

- `nooutput` executes at least one real GPU dispatch.
- CPU clear path remains only as a debug/fallback helper.

## Milestone 3: Resource Model

- [x] Define renderer resource handles: texture, lightmap, surface, world.
- [x] Add fixed-capacity or arena-backed resource tables.
- [x] Add linear texture atlas builder for 8-bit Quake mip levels.
- [x] Add lightmap atlas/page storage.
- [x] Add world surface metadata buffer.
- [x] Add upload/update APIs for static resources.
- [x] Add capacity telemetry for atlas usage and resource counts.

Exit criteria:

- A standalone test can upload synthetic textures/lightmaps/surfaces.
- No per-frame texture repacking is required.

## Milestone 4: QrustyQuake Adapter Layer

- [x] Add adapter code outside the renderer core.
- [x] Extract Quake textures and mip levels into renderer atlas inputs.
- [x] Extract world surface metadata into renderer structs.
- [x] Extract static lightmaps into renderer atlas inputs.
- [x] Map Quake surface IDs to renderer surface handles.
- [x] Record visible world surface command list from BSP/PVS traversal.
- [x] Keep existing KFD/palette backend as fallback during bring-up.

Exit criteria:

- The renderer core still does not include `quakedef.h`.
- QrustyQuake can initialize renderer resources for a loaded map.

## Milestone 5: Minimal World Raster

- [x] Add CPU or GPU setup path for visible world polygons.
- [x] Add first world raster kernel without tile binning if simpler.
- [x] Raster opaque world surfaces into indexed framebuffer.
- [x] Sample texture atlas with nearest filtering.
- [x] Sample static lightmap atlas.
- [x] Apply classic colormap rule.
- [x] Add depth buffer.
- [x] Add visual debug modes: flat surface ID, depth, texture only, light only.
- [x] Validate recognizable `e1m1` nooutput dump through the QrustyQuake
  adapter.

Exit criteria:

- `e1m1` opaque world renders recognizably in `nooutput`.
- Output can be dumped and inspected headlessly.

## Milestone 6: 16x16 Tiled Raster

- [x] Define `16x16` tile grid and tile metadata buffers.
- [x] Add tile list counter/offset buffers.
- [x] Add primitive-to-tile binning pass.
- [x] Add bounded overflow policy.
- [x] Add tile raster kernel.
- [x] Preserve deterministic per-tile primitive order for opaque world.
- [x] Compare performance against minimal non-tiled raster.

Exit criteria:

- Tiled world raster matches minimal world raster closely enough visually.
- Tile occupancy and overflow counters are available in perf output.

## Milestone 7: Depth, Color, And HiZ Policy

- [x] Choose MVP depth representation: fixed-point or ordered float bits.
- [x] Implement opaque-world ordering policy without byte color atomics.
- [x] Add validation mode for depth/order disagreements.
- [x] Add per-tile depth bounds.
- [x] Add early reject using tile depth bounds.
- [x] Prototype packed 64-bit atomic depth/payload path for later entities.

Exit criteria:

- Opaque world rendering is deterministic.
- HiZ counters show accepted/rejected primitive or tile work.

## Milestone 8: Present Mode

- [x] Add present output backend interface.
- [x] Reuse or factor existing DMA-BUF/X11 present code.
- [x] Keep `nooutput` as the default test path.
- [x] Add resolve-to-present-surface path.
- [x] Add windowed smoke test guarded by display availability.

Exit criteria:

- Same renderer frame can target either `nooutput` or `present`.
- Renderer creation does not require a window unless present mode is selected.

## Milestone 9: Classic Surface Types

- [x] Sky path.
- [x] Water/turbulence path.
- [x] Transparent index/cutout handling.
- [x] Animated light styles.
- [x] Dirty dynamic lightmap updates.

Exit criteria:

- Main world pass handles first-level visual features beyond opaque walls.

## Milestone 10: Entities

- [x] Alias model resource upload.
- [x] Alias model command buffer.
- [x] Alias triangle setup/raster path.
- [x] Sprite command/raster path.
- [x] Particle command/raster path.
- [x] Use packed atomic or separate ordered passes where world ordering no
  longer applies.

Exit criteria:

- First-level gameplay entities render without CPU software fallback.

## Milestone 11: Performance And Validation

- [x] Add renderer perf counters:
  - command counts
  - primitive counts
  - tile counts
  - tile overflow counts
  - HiZ reject counts
  - GPU pass timings
  - upload bytes
- [x] Add deterministic timedemo/headless benchmark mode.
- [x] Add image-diff tool against known-good dumps.
- [x] Add stress scenes for large polygons, dense entities, and overflow paths.
- [x] Add CI-friendly nooutput test subset.

Exit criteria:

- Performance regressions are visible without manual log inspection.
- Headless tests cover both API behavior and image output.

## First Implementation Order

1. Finish `nooutput` diagnostics and dump support.
2. Move framebuffer clear from CPU `memset` to a GPU kernel.
3. Add persistent renderer resource tables.
4. Build synthetic texture/lightmap/surface tests without Quake.
5. Add QrustyQuake adapter for resource extraction.
6. Render the first opaque textured/lightmapped world polygon in nooutput.

Do not start with full Quake integration. The renderer should become useful and
testable before it depends on a live game frame.
