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
- [ ] Replace ad hoc `calloc/free` with renderer-owned allocator or arena.
- [ ] Add public API versioning.
- [ ] Add structured error enum instead of raw errno-only returns.
- [ ] Add API docs for ownership, threading, and frame lifetime.

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

- [ ] Define renderer resource handles: texture, lightmap, surface, world.
- [ ] Add fixed-capacity or arena-backed resource tables.
- [ ] Add texture atlas builder for 8-bit Quake mip levels.
- [ ] Add lightmap atlas/page storage.
- [ ] Add world surface metadata buffer.
- [ ] Add upload/update APIs for static resources.
- [ ] Add capacity telemetry for atlas usage and resource counts.

Exit criteria:

- A standalone test can upload synthetic textures/lightmaps/surfaces.
- No per-frame texture repacking is required.

## Milestone 4: QrustyQuake Adapter Layer

- [ ] Add adapter code outside the renderer core.
- [ ] Extract Quake textures and mip levels into renderer atlas inputs.
- [ ] Extract world surface metadata into renderer structs.
- [ ] Extract static lightmaps into renderer atlas inputs.
- [ ] Map Quake surface IDs to renderer surface handles.
- [ ] Emit visible world surface command buffer from BSP/PVS traversal.
- [ ] Keep existing KFD/palette backend as fallback during bring-up.

Exit criteria:

- The renderer core still does not include `quakedef.h`.
- QrustyQuake can initialize renderer resources for a loaded map.

## Milestone 5: Minimal World Raster

- [ ] Add CPU or GPU setup path for visible world polygons.
- [ ] Add first world raster kernel without tile binning if simpler.
- [ ] Raster opaque world surfaces into indexed framebuffer.
- [ ] Sample texture atlas with nearest filtering.
- [ ] Sample static lightmap atlas.
- [ ] Apply classic colormap rule.
- [ ] Add depth buffer.
- [ ] Add visual debug modes: flat surface ID, depth, texture only, light only.

Exit criteria:

- `e1m1` opaque world renders recognizably in `nooutput`.
- Output can be dumped and inspected headlessly.

## Milestone 6: 16x16 Tiled Raster

- [ ] Define `16x16` tile grid and tile metadata buffers.
- [ ] Add tile list counter/offset buffers.
- [ ] Add primitive-to-tile binning pass.
- [ ] Add bounded overflow policy.
- [ ] Add tile raster kernel.
- [ ] Preserve deterministic per-tile primitive order for opaque world.
- [ ] Compare performance against minimal non-tiled raster.

Exit criteria:

- Tiled world raster matches minimal world raster closely enough visually.
- Tile occupancy and overflow counters are available in perf output.

## Milestone 7: Depth, Color, And HiZ Policy

- [ ] Choose MVP depth representation: fixed-point or ordered float bits.
- [ ] Implement opaque-world ordering policy without byte color atomics.
- [ ] Add validation mode for depth/order disagreements.
- [ ] Add per-tile depth bounds.
- [ ] Add early reject using tile depth bounds.
- [ ] Prototype packed 64-bit atomic depth/payload path for later entities.

Exit criteria:

- Opaque world rendering is deterministic.
- HiZ counters show accepted/rejected primitive or tile work.

## Milestone 8: Present Mode

- [ ] Add present output backend interface.
- [ ] Reuse or factor existing DMA-BUF/X11 present code.
- [ ] Keep `nooutput` as the default test path.
- [ ] Add resolve-to-present-surface path.
- [ ] Add windowed smoke test guarded by display availability.

Exit criteria:

- Same renderer frame can target either `nooutput` or `present`.
- Renderer creation does not require a window unless present mode is selected.

## Milestone 9: Classic Surface Types

- [ ] Sky path.
- [ ] Water/turbulence path.
- [ ] Transparent index/cutout handling.
- [ ] Animated light styles.
- [ ] Dirty dynamic lightmap updates.

Exit criteria:

- Main world pass handles first-level visual features beyond opaque walls.

## Milestone 10: Entities

- [ ] Alias model resource upload.
- [ ] Alias model command buffer.
- [ ] Alias triangle setup/raster path.
- [ ] Sprite command/raster path.
- [ ] Particle command/raster path.
- [ ] Use packed atomic or separate ordered passes where world ordering no
  longer applies.

Exit criteria:

- First-level gameplay entities render without CPU software fallback.

## Milestone 11: Performance And Validation

- [ ] Add renderer perf counters:
  - command counts
  - primitive counts
  - tile counts
  - tile overflow counts
  - HiZ reject counts
  - GPU pass timings
  - upload bytes
- [ ] Add deterministic timedemo/headless benchmark mode.
- [ ] Add image-diff tool against known-good dumps.
- [ ] Add stress scenes for large polygons, dense entities, and overflow paths.
- [ ] Add CI-friendly nooutput test subset.

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
