# libkfd

A minimal userspace library for dispatching compute work to AMDGPU hardware
directly through the Linux KFD (Kernel Fusion Driver) interface. The goal of
this project is to be a self-contained C++ library to explore and learn AMDGPU
internals.

> [!IMPORTANT]
> This is a personal project done for educational purposes only. It is not
> affiliated with nor a replacement for AMD's ROCm stack. The interface is
> largely untested and intentionally incomplete. For a supported compute
> runtime, use [ROCm](https://github.com/ROCm/ROCm).

## Overview

libkfd talks directly to `/dev/kfd` and the DRM render nodes to manage GPU
resources from user space. It is written in C++23 but the core library does not
depend on any C++ runtime features. The library handles:

- **Context** -- Opens the KFD device, enumerates the GPU topology, and
  initializes library state.
- **Device** -- Represents a single GPU node. Owns the DRM fd, GPUVM aperture,
  doorbell mapping, and trap handler.
- **Memory** -- RAII buffer type for allocating VRAM or GTT (system) memory,
  mapping it to device page tables, and pinning host memory.
- **Queues** -- User-space ring buffers for submitting PM4 compute packets
  (`ComputeQueue`) or SDMA memory-movement packets (`SDMAQueue`) via MMIO
  doorbells.
- **Loader** -- Loads AMDGPU ELF objects into GPU VRAM, performs relocations, and
  provides symbol lookup.
- **Signals** -- Combines KFD events (interrupt-driven wakeup) with GPU-writable
  fence slots for synchronization across queues and devices.
- **Trap handler** -- Per-architecture trap handler binaries are embedded at
  build time and installed on each device for exception delivery.
- **Topology** -- Parses the KFD sysfs topology to enumerate nodes, memory
  banks, caches, and IO links.

## Requirements

- **Clang >= 21** -- Required for C23 `#embed`, C++23 features, and the AMDGPU
  cross-compilation targets used for trap handlers and test kernels. A
  [nightly build](https://apt.llvm.org/) or a build from LLVM trunk will work.
- **CMake >= 3.28**
- **Ninja** (recommended) or Make
- **libdrm** with amdgpu support (`libdrm_amdgpu` via pkg-config)
- **xcb**, **xcb-dri3**, **xcb-present** (optional, for the `computetoy` tool)
- **Linux kernel >= 6.7 with KFD enabled** (typically `CONFIG_HSA_AMD=y`)
- An AMDGPU -- GFX9 (Vega), GFX10 (RDNA 1/2), GFX11 (RDNA 3), or GFX12
  (RDNA 4)

## Building

```sh
# presets: `debug`, `release`, `asan`, `tsan`.
cmake --preset release
cmake --build --preset release
```

To install:

```sh
cmake --preset release
cmake --build --preset release
cmake --install build/release --prefix /usr/local
```

After installing, run `kfdinfo` to verify that the system has usable GPU
devices:

```sh
/usr/local/bin/kfdinfo
```

## CMake Integration

After installing, use `find_package`:

```cmake
find_package(libkfd REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE libkfd::kfd)
```

Or pull the source directly with `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(libkfd
  GIT_REPOSITORY https://github.com/jhuber6/libkfd.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(libkfd)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE libkfd::kfd)
```

Both approaches require Clang >= 21 and `libdrm_amdgpu` to be available on the
system.

## Usage

The core workflow is: open a context, get a device, create queues, allocate
memory, load a kernel, dispatch, and synchronize. The signal interface is a
monotonically decreasing counter.

```cpp
#include <libkfd/libkfd.h>

// Open /dev/kfd and enumerate GPUs.
auto ctx  = KFD_EXPECT(kfd::Context::create());
auto &dev = ctx.devices().front();

// Create a compute queue.
auto compute = KFD_EXPECT(kfd::ComputeQueue::create(dev));

// Load a GPU ELF code object and look up a kernel.
auto exe    = KFD_EXPECT(kfd::Executable::load(dev, elf_bytes, compute));
auto kernel = KFD_EXPECT(exe.kernel("my_kernel.kd"));

// Allocate and map a GTT buffer for kernel arguments.
auto buf = KFD_EXPECT(kfd::Buffer::allocate(
    dev, size, kfd::MemType::GTT, kfd::MemFlags::WRITABLE));
KFD_EXPECT(buf.map(dev));

// Set up dispatch dimensions and build the kernarg buffer.
kfd::DispatchConfig cfg{.grid = {.x = num_blocks}, .block = {.x = 256}};
auto kernarg = KFD_EXPECT(kernel.alloc());
kernel.fill(kernarg, my_args, cfg);

// Dispatch and wait for completion.
auto sig = KFD_EXPECT(kfd::Signal::create(ctx));
KFD_EXPECT(compute.dispatch(kernel, cfg, kernarg, sig));
KFD_EXPECT(sig.wait(kfd::Condition::EQ, 0, UINT64_MAX));
```

Error handling uses `std::expected<T, kfd::Error>`, which wraps around standard
Linux `errno` values. The `KFD_EXPECT` macro unwraps a value or prints the error
and exits.

A complete working example is in [`tools/sandbox/`](tools/sandbox/), which runs
a SAXPY kernel on the GPU.

### Compiling GPU Kernels

GPU kernels are plain C compiled as freestanding AMDGPU executables. Use
`<gpuintrin.h>` for portable builtins, or the raw Clang attributes directly:

```c
// saxpy.c
#include <gpuintrin.h>

__gpu_kernel void saxpy(float *y, const float *x, float a, unsigned n) {
  unsigned i = __gpu_thread_id_x() + __gpu_block_id_x() * __gpu_num_threads_x();
  if (i < n)
    y[i] = a * x[i] + y[i];
}
```

Compile to an AMDGPU ELF for a specific GPU architecture. The resulting ELF can
be loaded at runtime via `kfd::Executable::load` like in the SAXPY example.

```sh
clang --target=amdgcn--amdhsa -mcpu=gfx1100 -nostdlibinc -O2 saxpy.c -o image
```

## Tests

The test suite covers all of the basic operations. It uses catch2 and can be
invoked through ctest.

```sh
ctest -j8
```

## Development

Install the pre-commit hooks before sending changes:

```sh
export PATH="$HOME/llvm/llvm-install/bin:$PATH"
pre-commit install
pre-commit run --all-files
```

## Tools

The project includes some command-line tools to serve as examples. The
`gpu-loader` utility requires the [LLVM libc GPU
headers](https://libc.llvm.org/gpu/building.html).

| Tool | Description |
|------|-------------|
| **kfdinfo** | Prints a detailed summary of the GPU topology - identity, compute layout, memory, caches, IO links, and firmware versions. |
| **sandbox** | Runs a SAXPY kernel end-to-end as a minimal libkfd demo. |
| **gpu-loader** | An `llvm-gpu-loader` equivalent that launches a `main()` function on the GPU as a hosted environment. |
| **computetoy** | Runs a compute kernel as a procedural image synthesizer and presents it to an X11 window via DRI3. Requires `xcb`, `xcb-dri3`, and `xcb-present`. |

## Project Structure

```
include/libkfd/          Public headers
  detail/                Internal utilities (ELF, allocators, mutex, etc.)
  packets/               PM4 and SDMA packet definitions
lib/                     Library implementation
  detail/                Internal utilities
  device/                Trap handler assembly
tests/
  detail/                Unit tests (no GPU needed)
  core/                  Core subsystem tests (GPU needed)
  device/                Full device integration tests (GPU needed)
tools/                   Command-line tools
cmake/                   CMake and pkg-config install templates
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
