# Quake Raster Validation

This rasterizer is developed for headless-first validation. The checks below are
the expected gate before changing milestone status or claiming adapter coverage.

## CI-Friendly Checks

Run these from `libkfd` after configuring with the Clang toolchain:

```sh
cmake --build build/clang --target quake_raster_api_test quake_raster_diff
ctest --test-dir build/clang/tests/core -R "Quake raster API" --output-on-failure
```

`quake_raster_api_test` is host-only and exercises descriptor rejection and pure
API helpers without requiring a KFD device. Device-backed tests remain labeled
`device` and should be run on AMDGPU CI or developer machines with KFD access:

```sh
ctest --test-dir build/clang/tests/core -L device --output-on-failure
```

## QrustyQuake Adapter Smoke

The sibling `qrustyquake` checkout includes `id1/qraster-e1m1.cfg` as a checked
in smoke script for the KFD backend:

```sh
make -C ../qrustyquake/src qrustyquake-kfd
../qrustyquake/qrustyquake-kfd -basedir ../qrustyquake +exec qraster-e1m1.cfg
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
