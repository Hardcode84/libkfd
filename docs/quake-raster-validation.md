# Quake Raster Validation

This rasterizer is developed for headless-first validation. The checks below are
the expected gate before changing milestone status or claiming adapter coverage.

## CI-Friendly Checks

Run these from `libkfd` after configuring with the Clang toolchain:

```sh
cmake --build build/clang --target quake_raster_api_test quake_raster_diff
ctest --test-dir build/clang/tests/core -R "Quake raster API" --output-on-failure
printf '\001\002\003\004' > /tmp/qr_expected.raw
printf '\001\002\003\004' > /tmp/qr_actual.raw
build/clang/tools/quake_raster_diff/quake_raster_diff /tmp/qr_expected.raw /tmp/qr_actual.raw
```

`quake_raster_api_test` is host-only and exercises descriptor rejection and pure
API helpers without requiring a KFD device. Device-backed tests remain labeled
`device` and should be run on AMDGPU CI or developer machines with KFD access:

```sh
ctest --test-dir build/clang/tests/core -L device --output-on-failure
```

The GitHub workflow always runs the host-only checks and diff-tool smoke. Device
coverage is an opt-in self-hosted job; set repository variable
`LIBKFD_RUN_DEVICE_TESTS=1` on an AMDGPU runner labeled `self-hosted`, `linux`,
and `amdgpu` to make it a merge gate.

See `quake-raster-abi.md` before changing host/device records or constants.

The current raster API bins and rasterizes once per triangle draw submission.
Keep opaque world surfaces batched in a single `qr_frame_draw_world()` call for
validation; future translucent/material batching should add an explicit
frame-scoped batch API instead of relying on many small draw calls.

## QrustyQuake Adapter Smoke

The sibling `qrustyquake` checkout includes `id1/qraster-e1m1.cfg` as a checked
in smoke script for the KFD backend:

```sh
make -C ../qrustyquake/src qrustyquake-kfd
../qrustyquake/qrustyquake-kfd -basedir ../qrustyquake +exec qraster-e1m1.cfg
```

The CMake path is equivalent when the `libkfd` build is not in the default
sibling location:

```sh
cmake -S ../qrustyquake -B ../qrustyquake/build/kfd \
  -DLIBKFD_SOURCE_DIR="$PWD" \
  -DLIBKFD_BUILD_DIR="$PWD/build/clang" \
  -DLIBKFD_LIBRARY="$PWD/build/clang/libkfd.a" \
  -DLIBKFD_QUAKE_RASTER_LIBRARY="$PWD/build/clang/libquake_raster.a"
cmake --build ../qrustyquake/build/kfd --target qrustyquake-kfd
```

The script loads `e1m1`, lets a few frames populate the visible-surface list,
writes `qraster-e1m1-indexed.raw` and `qraster-e1m1-xrgb.ppm`, then exits.
Compare raw outputs with:

```sh
build/clang/tools/quake_raster_diff/quake_raster_diff expected.raw actual.raw
```

Because Quake assets are not shipped in this repository, e1m1 validation is a
release smoke check rather than a default CI test. Record asset version, command
line, driver, GPU, and output hashes in release notes whenever this check is
used to support a milestone claim.
