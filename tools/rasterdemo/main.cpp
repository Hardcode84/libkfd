//===-- tools/rasterdemo/main.cpp - Windowed rasterizer demo --------------===//
//
// Presents the headless raster demo through the same dma-buf display path used
// by computetoy. The host generates a rotating checker-shaded demo mesh and the
// GPU raster kernel writes directly into presentable VRAM framebuffers.
//
//===----------------------------------------------------------------------===//

#include "window.h"

#include "raster_common.h"

#include "libkfd/detail/elf.h"
#include "libkfd/libkfd.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef HAVE_RASTERDEMO_XCB_SW
#include <unistd.h>
#include <xcb/xcb.h>
#endif

namespace {

constexpr kfd::MemFlags HOST_GTT_FLAGS =
    kfd::MemFlags::WRITABLE | kfd::MemFlags::EXECUTABLE |
    kfd::MemFlags::HOST_ACCESS | kfd::MemFlags::UNCACHED;

struct DemoBinary {
  const char *path;
  const char *arch;
};

static const DemoBinary rasterdemo_kernels[] = {
#include "rasterdemo_kernels.inc"
};

struct Framebuffer {
  kfd::Buffer color;
  std::unique_ptr<kfd::Signal> signal;
};

enum class KernelMode { Persistent, PerFrame, StaticGrid };

#ifdef HAVE_RASTERDEMO_XCB_SW
xcb_atom_t intern_atom(xcb_connection_t *conn, const char *name) {
  auto cookie =
      xcb_intern_atom(conn, 0, static_cast<uint16_t>(std::strlen(name)), name);
  auto *reply = xcb_intern_atom_reply(conn, cookie, nullptr);
  if (!reply)
    return XCB_ATOM_NONE;
  xcb_atom_t atom = reply->atom;
  std::free(reply);
  return atom;
}

class XcbSoftwareWindow {
public:
  ~XcbSoftwareWindow() {
    if (!conn)
      return;
    if (gc)
      xcb_free_gc(conn, gc);
    if (win)
      xcb_destroy_window(conn, win);
    xcb_disconnect(conn);
  }

  XcbSoftwareWindow(const XcbSoftwareWindow &) = delete;
  XcbSoftwareWindow &operator=(const XcbSoftwareWindow &) = delete;
  XcbSoftwareWindow(XcbSoftwareWindow &&o)
      : conn(std::exchange(o.conn, nullptr)), screen(o.screen),
        win(std::exchange(o.win, 0)), gc(std::exchange(o.gc, 0)),
        wm_delete(std::exchange(o.wm_delete, 0)), w(o.w), h(o.h),
        depth(o.depth) {}

  static std::expected<XcbSoftwareWindow, kfd::Error>
  create(uint32_t width, uint32_t height, const char *title) {
    if (!std::getenv("DISPLAY"))
      return kfd::unexpected(ENODEV, "No X11 DISPLAY available");

    int screen_num = 0;
    xcb_connection_t *conn = xcb_connect(nullptr, &screen_num);
    if (xcb_connection_has_error(conn)) {
      xcb_disconnect(conn);
      return kfd::unexpected(ECONNREFUSED, "Cannot connect to X server");
    }

    auto *setup = xcb_get_setup(conn);
    xcb_screen_iterator_t iter = xcb_setup_roots_iterator(setup);
    for (int i = 0; i < screen_num; ++i)
      xcb_screen_next(&iter);
    xcb_screen_t *screen = iter.data;

    xcb_window_t window = xcb_generate_id(conn);
    uint32_t event_mask = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS |
                          XCB_EVENT_MASK_STRUCTURE_NOTIFY;
    uint32_t values[] = {screen->black_pixel, event_mask};
    xcb_create_window(conn, screen->root_depth, window, screen->root, 0, 0,
                      static_cast<uint16_t>(width),
                      static_cast<uint16_t>(height), 0,
                      XCB_WINDOW_CLASS_INPUT_OUTPUT, screen->root_visual,
                      XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK, values);

    xcb_atom_t wm_protocols = intern_atom(conn, "WM_PROTOCOLS");
    xcb_atom_t wm_delete = intern_atom(conn, "WM_DELETE_WINDOW");
    xcb_change_property(conn, XCB_PROP_MODE_REPLACE, window, wm_protocols,
                        XCB_ATOM_ATOM, 32, 1, &wm_delete);

    size_t title_len = std::strlen(title);
    xcb_change_property(conn, XCB_PROP_MODE_REPLACE, window, XCB_ATOM_WM_NAME,
                        XCB_ATOM_STRING, 8, static_cast<uint32_t>(title_len),
                        title);

    xcb_gcontext_t gc = xcb_generate_id(conn);
    xcb_create_gc(conn, gc, window, 0, nullptr);
    xcb_map_window(conn, window);
    xcb_flush(conn);

    return XcbSoftwareWindow(conn, screen, window, gc, wm_delete, width, height,
                             screen->root_depth);
  }

  bool poll() {
    xcb_generic_event_t *event;
    while ((event = xcb_poll_for_event(conn))) {
      uint8_t type = event->response_type & 0x7f;
      if (type == 0) {
        auto *err = reinterpret_cast<xcb_generic_error_t *>(event);
        std::fprintf(stderr, "X11 error: code %u, sequence %u, resource %u\n",
                     err->error_code, err->sequence, err->resource_id);
        std::free(event);
        continue;
      }
      switch (type) {
      case XCB_KEY_PRESS: {
        auto *kp = reinterpret_cast<xcb_key_press_event_t *>(event);
        if (kp->detail == /*Esc=*/9 || kp->detail == /*q=*/24) {
          std::free(event);
          return false;
        }
        if (kp->detail == /*Space=*/65)
          pause_toggle = true;
        break;
      }
      case XCB_CLIENT_MESSAGE: {
        auto *cm = reinterpret_cast<xcb_client_message_event_t *>(event);
        if (cm->data.data32[0] == wm_delete) {
          std::free(event);
          return false;
        }
        break;
      }
      default:
        break;
      }
      std::free(event);
    }
    return !xcb_connection_has_error(conn);
  }

  bool take_pause_toggle() { return std::exchange(pause_toggle, false); }

  void present(const uint32_t *pixels, uint32_t pitch) {
    scratch.resize(static_cast<size_t>(w) * h);
    if (pitch == w) {
      std::memcpy(scratch.data(), pixels, scratch.size() * sizeof(uint32_t));
    } else {
      for (uint32_t y = 0; y < h; ++y) {
        std::memcpy(scratch.data() + static_cast<size_t>(y) * w,
                    pixels + static_cast<size_t>(y) * pitch,
                    static_cast<size_t>(w) * sizeof(uint32_t));
      }
    }

    uint32_t max_request_dwords = xcb_get_maximum_request_length(conn);
    uint32_t max_payload_bytes =
        max_request_dwords > 64 ? (max_request_dwords - 64) * 4 : 4096;
    uint32_t row_bytes = w * sizeof(uint32_t);
    uint32_t rows_per_chunk = max_payload_bytes / row_bytes;
    if (rows_per_chunk == 0)
      rows_per_chunk = 1;

    for (uint32_t y = 0; y < h; y += rows_per_chunk) {
      uint32_t rows = h - y;
      if (rows > rows_per_chunk)
        rows = rows_per_chunk;
      const uint32_t *src = scratch.data() + static_cast<size_t>(y) * w;
      xcb_put_image(conn, XCB_IMAGE_FORMAT_Z_PIXMAP, win, gc,
                    static_cast<uint16_t>(w), static_cast<uint16_t>(rows), 0,
                    static_cast<int16_t>(y), 0, depth, rows * row_bytes,
                    reinterpret_cast<const uint8_t *>(src));
    }
    xcb_flush(conn);
  }

  uint32_t width() const { return w; }
  uint32_t height() const { return h; }

private:
  XcbSoftwareWindow(xcb_connection_t *conn, xcb_screen_t *screen,
                    xcb_window_t win, xcb_gcontext_t gc, xcb_atom_t wm_delete,
                    uint32_t w, uint32_t h, uint8_t depth)
      : conn(conn), screen(screen), win(win), gc(gc), wm_delete(wm_delete),
        w(w), h(h), depth(depth) {}

  xcb_connection_t *conn = nullptr;
  xcb_screen_t *screen = nullptr;
  xcb_window_t win = 0;
  xcb_gcontext_t gc = 0;
  xcb_atom_t wm_delete = 0;
  uint32_t w = 0;
  uint32_t h = 0;
  uint8_t depth = 0;
  std::vector<uint32_t> scratch;
  bool pause_toggle = false;
};
#endif

std::vector<std::byte> read_file(const char *path) {
  std::FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "error: cannot open '%s'\n", path);
    std::exit(1);
  }
  std::fseek(f, 0, SEEK_END);
  auto sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<std::byte> buf(static_cast<size_t>(sz));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    std::fprintf(stderr, "error: short read on '%s'\n", path);
    std::exit(1);
  }
  std::fclose(f);
  return buf;
}

void parse_resolution(const char *str, uint32_t &w, uint32_t &h) {
  if (std::sscanf(str, "%ux%u", &w, &h) != 2) {
    std::fprintf(
        stderr, "error: invalid resolution '%s', expected WIDTHxHEIGHT\n", str);
    std::exit(1);
  }
}

WindowBackend parse_backend(std::string_view value) {
  if (value == "auto")
    return WindowBackend::Auto;
  if (value == "xcb")
    return WindowBackend::Xcb;
  if (value == "drm")
    return WindowBackend::Drm;
  if (value == "xcb-sw")
    return WindowBackend::Xcb;
  std::fprintf(stderr,
               "error: invalid backend '%.*s', expected auto|xcb|xcb-sw|drm\n",
               static_cast<int>(value.size()), value.data());
  std::exit(1);
}

uint32_t parse_u32(std::string_view value, const char *name) {
  char *end = nullptr;
  std::string text(value);
  unsigned long parsed = std::strtoul(text.c_str(), &end, 10);
  if (*end != '\0' || parsed > UINT32_MAX) {
    std::fprintf(stderr, "error: invalid %s '%s'\n", name, text.c_str());
    std::exit(1);
  }
  return static_cast<uint32_t>(parsed);
}

WindowBackend default_backend() {
  // Avoid taking over KMS from under an active desktop compositor. Raw DRM is
  // still available for console sessions via --backend=drm.
  if (std::getenv("DISPLAY"))
    return WindowBackend::Xcb;
  if (std::getenv("WAYLAND_DISPLAY"))
    return WindowBackend::Xcb;
  return WindowBackend::Auto;
}

const DemoBinary *find_compatible_binary(kfd::Device &dev) {
  namespace elf = kfd::detail::elf;
  for (const auto &bin : rasterdemo_kernels) {
    auto buf = read_file(bin.path);
    auto parsed = elf::ELF64LE::create(
        std::span<const std::byte>(buf.data(), buf.size()));
    if (!parsed)
      continue;
    bool sramecc = (dev.properties().capability &
                    kfd::NodeProperties::NODE_CAP_SRAM_EDCSUPPORTED) != 0;
    if (elf::is_compatible(parsed->header().e_flags, dev.gfx_version(),
                           dev.context().xnack_enabled(), sramecc))
      return &bin;
  }
  return nullptr;
}

std::expected<kfd::Device *, kfd::Error> find_device(kfd::Context &ctx) {
  for (size_t i = 0; i < ctx.num_devices(); ++i) {
    kfd::Device &dev = *KFD_EXPECT(ctx.device(i));
    if (find_compatible_binary(dev))
      return &dev;
  }
  return kfd::unexpected(ENOEXEC, "No compatible GPUs found");
}

DemoPrimitive tri(DemoVertex a, DemoVertex b, DemoVertex c) {
  return DemoPrimitive{.v0 = a, .v1 = b, .v2 = c};
}

void build_demo_workload(std::vector<DemoPrimitive> &prims, uint32_t width,
                         uint32_t height, float time) {
  std::vector<DemoVertex> verts((RINGS + 1) * SLICES);
  auto at = [&](uint32_t ring, uint32_t slice) -> DemoVertex & {
    return verts[ring * SLICES + (slice % SLICES)];
  };

  float angle = time * 0.75f;
  float ca = std::cos(angle);
  float sa = std::sin(angle);
  for (uint32_t r = 0; r <= RINGS; ++r) {
    float t = static_cast<float>(r) / static_cast<float>(RINGS);
    float y = (t - 0.5f) * 2.0f;
    float radius = 0.42f + 0.10f * std::cos((t * 3.0f + 0.2f) * PI) +
                   0.04f * std::sin(t * 11.0f * PI);
    if (r == 0 || r == RINGS)
      radius = 0.05f;

    for (uint32_t s = 0; s < SLICES; ++s) {
      float u = static_cast<float>(s) / static_cast<float>(SLICES);
      float theta = u * TAU;
      float x = radius * std::cos(theta);
      float z = radius * std::sin(theta);
      float xr = x * ca + z * sa;
      float zr = -x * sa + z * ca;
      float perspective = 1.0f / (1.45f + zr * 0.35f);

      at(r, s) = DemoVertex{
          .x = static_cast<float>(width) * (0.5f + xr * 0.78f * perspective),
          .y = static_cast<float>(height) * (0.5f - y * 0.54f * perspective),
          .z = 0.48f + zr * 0.20f,
          .u = u,
          .v = t,
      };
    }
  }

  prims.clear();
  prims.reserve(RINGS * SLICES * 2);
  for (uint32_t r = 0; r < RINGS; ++r) {
    for (uint32_t s = 0; s < SLICES; ++s) {
      DemoVertex a = at(r, s);
      DemoVertex b = at(r, s + 1);
      DemoVertex c = at(r + 1, s + 1);
      DemoVertex d = at(r + 1, s);
      prims.push_back(tri(a, b, d));
      prims.push_back(tri(b, c, d));
    }
  }
}

float min2(float a, float b) { return a < b ? a : b; }
float max2(float a, float b) { return a > b ? a : b; }
float min3(float a, float b, float c) { return min2(min2(a, b), c); }
float max3(float a, float b, float c) { return max2(max2(a, b), c); }

uint32_t clamp_tile(int value, uint32_t limit) {
  if (value < 0)
    return 0;
  uint32_t v = static_cast<uint32_t>(value);
  return v > limit ? limit : v;
}

bool build_tile_bins(std::span<const DemoPrimitive> prims, uint32_t width,
                     uint32_t height, uint32_t tiles_x, uint32_t tiles_y,
                     std::vector<TileRange> &ranges,
                     std::vector<uint32_t> &indices, size_t max_indices) {
  ranges.assign(static_cast<size_t>(tiles_x) * tiles_y, TileRange{});
  std::vector<uint32_t> counts(ranges.size());

  for (uint32_t i = 0; i < prims.size(); ++i) {
    const auto &tri = prims[i];
    float min_x = min3(tri.v0.x, tri.v1.x, tri.v2.x);
    float max_x = max3(tri.v0.x, tri.v1.x, tri.v2.x);
    float min_y = min3(tri.v0.y, tri.v1.y, tri.v2.y);
    float max_y = max3(tri.v0.y, tri.v1.y, tri.v2.y);
    if (max_x < 0.0f || max_y < 0.0f || min_x >= static_cast<float>(width) ||
        min_y >= static_cast<float>(height))
      continue;

    uint32_t x0 = clamp_tile(static_cast<int>(std::floor(min_x)) /
                                 static_cast<int>(TILE_SIZE),
                             tiles_x - 1);
    uint32_t x1 = clamp_tile(static_cast<int>(std::floor(max_x)) /
                                 static_cast<int>(TILE_SIZE),
                             tiles_x - 1);
    uint32_t y0 = clamp_tile(static_cast<int>(std::floor(min_y)) /
                                 static_cast<int>(TILE_SIZE),
                             tiles_y - 1);
    uint32_t y1 = clamp_tile(static_cast<int>(std::floor(max_y)) /
                                 static_cast<int>(TILE_SIZE),
                             tiles_y - 1);

    for (uint32_t y = y0; y <= y1; ++y)
      for (uint32_t x = x0; x <= x1; ++x)
        ++counts[static_cast<size_t>(y) * tiles_x + x];
  }

  uint32_t total = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    ranges[i].offset = total;
    ranges[i].count = counts[i];
    total += counts[i];
    counts[i] = 0;
  }
  if (total > max_indices)
    return false;

  indices.assign(total, 0);
  for (uint32_t i = 0; i < prims.size(); ++i) {
    const auto &tri = prims[i];
    float min_x = min3(tri.v0.x, tri.v1.x, tri.v2.x);
    float max_x = max3(tri.v0.x, tri.v1.x, tri.v2.x);
    float min_y = min3(tri.v0.y, tri.v1.y, tri.v2.y);
    float max_y = max3(tri.v0.y, tri.v1.y, tri.v2.y);
    if (max_x < 0.0f || max_y < 0.0f || min_x >= static_cast<float>(width) ||
        min_y >= static_cast<float>(height))
      continue;

    uint32_t x0 = clamp_tile(static_cast<int>(std::floor(min_x)) /
                                 static_cast<int>(TILE_SIZE),
                             tiles_x - 1);
    uint32_t x1 = clamp_tile(static_cast<int>(std::floor(max_x)) /
                                 static_cast<int>(TILE_SIZE),
                             tiles_x - 1);
    uint32_t y0 = clamp_tile(static_cast<int>(std::floor(min_y)) /
                                 static_cast<int>(TILE_SIZE),
                             tiles_y - 1);
    uint32_t y1 = clamp_tile(static_cast<int>(std::floor(max_y)) /
                                 static_cast<int>(TILE_SIZE),
                             tiles_y - 1);

    for (uint32_t y = y0; y <= y1; ++y) {
      for (uint32_t x = x0; x <= x1; ++x) {
        size_t tile = static_cast<size_t>(y) * tiles_x + x;
        indices[ranges[tile].offset + counts[tile]++] = i;
      }
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  uint32_t width = 1280;
  uint32_t height = 720;
  WindowBackend backend = default_backend();
  bool force_software_present = false;
  bool probe_only = false;
  bool clear_only = false;
  bool headless = false;
  KernelMode kernel_mode = KernelMode::Persistent;
  uint32_t max_frames = 0;
  uint32_t gpu_timeout_ms = 5000;
  uint32_t requested_persistent_wgs = 0;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg.starts_with("--backend=")) {
      std::string_view value = arg.substr(std::strlen("--backend="));
      force_software_present = value == "xcb-sw";
      backend = parse_backend(value);
      continue;
    }
    if (arg.starts_with("--frames=")) {
      max_frames = parse_u32(arg.substr(std::strlen("--frames=")), "frames");
      continue;
    }
    if (arg.starts_with("--gpu-timeout-ms=")) {
      gpu_timeout_ms = parse_u32(arg.substr(std::strlen("--gpu-timeout-ms=")),
                                 "gpu-timeout-ms");
      continue;
    }
    if (arg.starts_with("--persistent-wgs=")) {
      requested_persistent_wgs = parse_u32(
          arg.substr(std::strlen("--persistent-wgs=")), "persistent-wgs");
      continue;
    }
    if (arg == "--xcb-sw") {
      force_software_present = true;
      backend = WindowBackend::Xcb;
      continue;
    }
    if (arg == "--probe") {
      probe_only = true;
      continue;
    }
    if (arg == "--clear-only") {
      clear_only = true;
      continue;
    }
    if (arg == "--headless") {
      headless = true;
      continue;
    }
    if (arg == "--per-frame" || arg == "--single-shot") {
      kernel_mode = KernelMode::PerFrame;
      continue;
    }
    if (arg == "--static-grid") {
      kernel_mode = KernelMode::StaticGrid;
      continue;
    }
    if (arg == "--drm") {
      backend = WindowBackend::Drm;
      continue;
    }
    if (arg == "--xcb") {
      backend = WindowBackend::Xcb;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      std::printf("usage: %s [options] [WIDTHxHEIGHT]\n"
                  "options:\n"
                  "  --backend=xcb|xcb-sw|drm|auto\n"
                  "  --xcb, --xcb-sw, --drm\n"
                  "  --frames=N              exit after N frames\n"
                  "  --gpu-timeout-ms=N      per-frame GPU timeout "
                  "(default 5000)\n"
                  "  --persistent-wgs=N      persistent workgroups; default "
                  "is 48, capped by tile count\n"
                  "  --probe                 run a one-workgroup dispatch and "
                  "exit\n"
                  "  --clear-only            present clear frames without "
                  "triangle rasterization\n"
                  "  --headless              render frames without opening a "
                  "display\n"
                  "  --per-frame             one dynamic tile-claim dispatch "
                  "per frame\n"
                  "  --single-shot           alias for --per-frame\n"
                  "  --static-grid           one static grid dispatch per "
                  "frame for debugging\n"
                  "  xcb: X11 desktop mode; falls back to xcb_put_image when "
                  "DRI3 is unavailable\n"
                  "  xcb-sw: X11 software presentation; skips DRI3 entirely\n"
                  "  drm: raw KMS mode for a real TTY; can take over display\n",
                  argv[0]);
      return 0;
    }
    parse_resolution(argv[i], width, height);
  }

  auto ctx = KFD_EXPECT(kfd::Context::create());
  auto &dev = *KFD_EXPECT(find_device(ctx));
  auto *bin = find_compatible_binary(dev);
  if (!bin) {
    std::fprintf(stderr, "error: no compatible rasterdemo kernel found\n");
    return 1;
  }

  std::printf("GPU: %.*s (gfx%u), kernel arch %s\n",
              static_cast<int>(dev.get_name().size()), dev.get_name().data(),
              dev.properties().gfx_target_version, bin->arch);

  auto compute = KFD_EXPECT(kfd::ComputeQueue::create(dev));
  auto kernel_file = read_file(bin->path);
  auto exe = KFD_EXPECT(kfd::Executable::load(dev, kernel_file, compute));
  auto frame_kernel = KFD_EXPECT(exe.kernel(
      clear_only ? "rasterdemo_clear_frame.kd" : "rasterdemo_frame.kd"));
  auto claim_frame_kernel = KFD_EXPECT(exe.kernel("rasterdemo_claim_frame.kd"));
  auto persistent_kernel = KFD_EXPECT(exe.kernel("rasterdemo_persistent.kd"));

  if (probe_only) {
    auto probe = KFD_EXPECT(exe.kernel("rasterdemo_probe.kd"));
    auto out = KFD_EXPECT(kfd::Buffer::allocate(
        dev, kfd::detail::page_size(), kfd::MemType::GTT, HOST_GTT_FLAGS));
    KFD_EXPECT(out.map(dev));
    *static_cast<uint32_t *>(out.data()) = 0;

    ProbeArgs args{.out = static_cast<uint32_t *>(out.data())};
    kfd::DispatchConfig probe_cfg{.grid = {.x = 1}, .block = {.x = 64}};
    auto kernarg = KFD_EXPECT(probe.alloc());
    probe.fill(kernarg, args, probe_cfg);
    auto sig = KFD_EXPECT(kfd::Signal::create(ctx));
    KFD_EXPECT(compute.dispatch(probe, probe_cfg, kernarg, sig));
    auto waited = sig.wait(kfd::Condition::EQ, 0,
                           static_cast<uint64_t>(gpu_timeout_ms) * 1'000'000u);
    if (!waited) {
      std::fprintf(stderr, "error: probe GPU wait failed: %s\n",
                   kfd::strerror(waited));
      return 1;
    }
    uint32_t value = *static_cast<uint32_t *>(out.data());
    if (value != 0xcafebabeu) {
      std::fprintf(stderr, "error: probe wrote 0x%08x, expected 0xcafebabe\n",
                   value);
      return 1;
    }
    std::printf("Probe dispatch passed.\n");
    return 0;
  }

  std::unique_ptr<Window> win;
#ifdef HAVE_RASTERDEMO_XCB_SW
  std::unique_ptr<XcbSoftwareWindow> sw_win;
#endif
  bool software_present = false;

  if (headless && max_frames == 0)
    max_frames = 1;

  if (headless) {
    std::printf("Presentation: headless\n");
  } else
#ifdef HAVE_RASTERDEMO_XCB_SW
      if (force_software_present) {
    auto sw = XcbSoftwareWindow::create(width, height, "libkfd rasterdemo");
    if (!sw) {
      std::fprintf(stderr, "error: %s\n", kfd::strerror(sw));
      return 1;
    }
    sw_win = std::make_unique<XcbSoftwareWindow>(std::move(*sw));
    software_present = true;
    width = sw_win->width();
    height = sw_win->height();
    std::printf("Presentation: XCB software (xcb_put_image)\n");
  } else
#else
      if (force_software_present) {
    std::fprintf(stderr, "error: rasterdemo was built without XCB software "
                         "presentation support\n");
    return 1;
  } else
#endif
  {
    auto display =
        Window::create(width, height, NUM_BUFFERS, "libkfd rasterdemo",
                       dev.render_fd(), backend);
    if (display) {
      win = std::move(*display);
      width = win->width();
      height = win->height();
      std::printf("Presentation: dma-buf display backend\n");
    } else {
#ifdef HAVE_RASTERDEMO_XCB_SW
      if (backend != WindowBackend::Drm && std::getenv("DISPLAY")) {
        std::fprintf(stderr,
                     "warning: dma-buf presentation unavailable (%s); falling "
                     "back to XCB software presentation\n",
                     kfd::strerror(display));
        auto sw = XcbSoftwareWindow::create(width, height, "libkfd rasterdemo");
        if (!sw) {
          std::fprintf(stderr, "error: %s\n", kfd::strerror(sw));
          return 1;
        }
        sw_win = std::make_unique<XcbSoftwareWindow>(std::move(*sw));
        software_present = true;
        width = sw_win->width();
        height = sw_win->height();
      } else
#endif
      {
        std::fprintf(stderr, "error: %s\n", kfd::strerror(display));
        return 1;
      }
    }
  }

  uint32_t stride =
      software_present
          ? width * sizeof(uint32_t)
          : (width * sizeof(uint32_t) + STRIDE_ALIGN - 1) & ~(STRIDE_ALIGN - 1);
  uint32_t pitch = stride / sizeof(uint32_t);
  size_t color_bytes = static_cast<size_t>(stride) * height;
  size_t depth_bytes = static_cast<size_t>(pitch) * height * sizeof(float);
  uint32_t tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t tile_count = tiles_x * tiles_y;
  uint32_t default_persistent_wgs = DEFAULT_PERSISTENT_WGS;
  if (default_persistent_wgs > tile_count)
    default_persistent_wgs = tile_count;
  uint32_t persistent_wgs = requested_persistent_wgs ? requested_persistent_wgs
                                                     : default_persistent_wgs;
  if (persistent_wgs == 0) {
    std::fprintf(stderr,
                 "error: persistent workgroup count must be non-zero\n");
    return 1;
  }

  auto depth = KFD_EXPECT(kfd::Buffer::allocate(
      dev, depth_bytes, kfd::MemType::VRAM, kfd::MemFlags::WRITABLE));
  KFD_EXPECT(depth.map(dev));

  std::vector<DemoPrimitive> prims;
  prims.reserve(RINGS * SLICES * 2);
  size_t prim_bytes = RINGS * SLICES * 2 * sizeof(DemoPrimitive);
  size_t max_tile_indices = RINGS * SLICES * 2 * 256;
  size_t tile_indices_bytes = max_tile_indices * sizeof(uint32_t);
  size_t tile_ranges_bytes =
      static_cast<size_t>(tiles_x) * tiles_y * sizeof(TileRange);
  size_t active_tiles_bytes =
      static_cast<size_t>(tiles_x) * tiles_y * sizeof(uint32_t);
  auto prim_buf = KFD_EXPECT(kfd::Buffer::allocate(
      dev, kfd::detail::align_up(prim_bytes, kfd::detail::page_size()),
      kfd::MemType::GTT, HOST_GTT_FLAGS));
  KFD_EXPECT(prim_buf.map(dev));
  auto tile_indices_buf = KFD_EXPECT(kfd::Buffer::allocate(
      dev, kfd::detail::align_up(tile_indices_bytes, kfd::detail::page_size()),
      kfd::MemType::GTT, HOST_GTT_FLAGS));
  KFD_EXPECT(tile_indices_buf.map(dev));
  auto tile_ranges_buf = KFD_EXPECT(kfd::Buffer::allocate(
      dev, kfd::detail::align_up(tile_ranges_bytes, kfd::detail::page_size()),
      kfd::MemType::GTT, HOST_GTT_FLAGS));
  KFD_EXPECT(tile_ranges_buf.map(dev));
  auto active_tiles_buf = KFD_EXPECT(kfd::Buffer::allocate(
      dev, kfd::detail::align_up(active_tiles_bytes, kfd::detail::page_size()),
      kfd::MemType::GTT, HOST_GTT_FLAGS));
  KFD_EXPECT(active_tiles_buf.map(dev));
  auto *active_tiles = static_cast<uint32_t *>(active_tiles_buf.data());
  // The current demo clears as part of tile rendering, so every tile remains
  // active until clearing is split into its own pass.
  for (uint32_t tile = 0; tile < tile_count; ++tile)
    active_tiles[tile] = tile;
  std::vector<TileRange> tile_ranges;
  std::vector<uint32_t> tile_indices;

  Framebuffer fbs[NUM_BUFFERS];
  kfd::DispatchConfig cfg{
      .grid = {.x = tiles_x, .y = tiles_y},
      .block = {.x = BLOCK_X, .y = BLOCK_Y},
  };
  for (uint32_t i = 0; i < NUM_BUFFERS; ++i) {
    kfd::MemType color_type =
        (software_present || headless) ? kfd::MemType::GTT : kfd::MemType::VRAM;
    kfd::MemFlags color_flags = (software_present || headless)
                                    ? HOST_GTT_FLAGS
                                    : kfd::MemFlags::WRITABLE;
    fbs[i].color = KFD_EXPECT(
        kfd::Buffer::allocate(dev, color_bytes, color_type, color_flags));
    KFD_EXPECT(fbs[i].color.map(dev));
    fbs[i].signal =
        std::make_unique<kfd::Signal>(KFD_EXPECT(kfd::Signal::create(ctx)));
    if (!software_present && !headless) {
      auto dmabuf = KFD_EXPECT(kfd::DMABuffer::create(fbs[i].color));
      KFD_EXPECT(
          win->import_buffer(i, dmabuf.fd(), fbs[i].color.size(), stride));
    }
  }
  kfd::detail::memory_barrier();

  auto control_buf = KFD_EXPECT(
      kfd::Buffer::allocate(dev,
                            kfd::detail::align_up(sizeof(PersistentControl),
                                                  kfd::detail::page_size()),
                            kfd::MemType::GTT, HOST_GTT_FLAGS));
  KFD_EXPECT(control_buf.map(dev));

  auto *control = static_cast<PersistentControl *>(control_buf.data());
  std::memset(control, 0, sizeof(*control));

  kfd::DispatchConfig persistent_cfg{
      .grid = {.x = persistent_wgs},
      .block = {.x = PERSISTENT_BLOCK_X, .y = PERSISTENT_BLOCK_Y},
  };
  kfd::DispatchConfig claim_cfg{
      .grid = {.x = persistent_wgs},
      .block = {.x = CLAIM_BLOCK_X, .y = CLAIM_BLOCK_Y},
  };
  auto frame_kernarg = KFD_EXPECT(frame_kernel.alloc());
  auto claim_kernarg = KFD_EXPECT(claim_frame_kernel.alloc());
  auto persistent_kernarg = KFD_EXPECT(persistent_kernel.alloc());
  PersistentArgs persistent_args{
      .control = control,
      .prims = static_cast<const DemoPrimitive *>(prim_buf.data()),
      .tile_indices = static_cast<const uint32_t *>(tile_indices_buf.data()),
      .tile_ranges = static_cast<const TileRange *>(tile_ranges_buf.data()),
      .active_tiles = static_cast<const uint32_t *>(active_tiles_buf.data()),
      .color = static_cast<uint32_t *>(fbs[0].color.data()),
      .depth = static_cast<float *>(depth.data()),
      .width = width,
      .height = height,
      .pitch = pitch,
      .tile_size = TILE_SIZE,
      .tiles_x = tiles_x,
      .tiles_y = tiles_y,
      .clear_color = CLEAR_COLOR,
      .clear_depth = CLEAR_DEPTH,
      .clear_only = clear_only ? 1u : 0u,
  };
  auto persistent_args_buf = KFD_EXPECT(kfd::Buffer::allocate(
      dev,
      kfd::detail::align_up(sizeof(PersistentArgs), kfd::detail::page_size()),
      kfd::MemType::GTT, HOST_GTT_FLAGS));
  KFD_EXPECT(persistent_args_buf.map(dev));
  std::memcpy(persistent_args_buf.data(), &persistent_args,
              sizeof(persistent_args));
  auto *persistent_args_device =
      static_cast<PersistentArgs *>(persistent_args_buf.data());
  PersistentLaunchArgs persistent_launch{
      .args = persistent_args_device,
  };
  persistent_kernel.fill(persistent_kernarg, persistent_launch, persistent_cfg);
  auto shutdown_signal = KFD_EXPECT(kfd::Signal::create(ctx));

  if (kernel_mode == KernelMode::Persistent) {
    KFD_EXPECT(compute.dispatch(persistent_kernel, persistent_cfg,
                                persistent_kernarg));
    std::printf("Kernel mode: persistent (%u workgroups, %ux%u threads)\n",
                persistent_wgs, PERSISTENT_BLOCK_X, PERSISTENT_BLOCK_Y);
  } else if (kernel_mode == KernelMode::PerFrame) {
    std::printf(
        "Kernel mode: per-frame active tile queue (%u workgroups, %ux%u "
        "threads)\n",
        persistent_wgs, CLAIM_BLOCK_X, CLAIM_BLOCK_Y);
  } else {
    std::printf("Kernel mode: static grid dispatch per frame\n");
  }

  uint32_t current = 0;
  uint32_t frame = 0;
  auto start = std::chrono::steady_clock::now();
  auto fps_time = start;
  uint32_t fps_frames = 0;
  bool render_paused = false;

  std::printf(
      "Rendering %u triangles at %ux%u. Press Space to pause/resume, q or Esc "
      "to quit.\n",
      RINGS * SLICES * 2, width, height);
  uint64_t gpu_timeout_ns = static_cast<uint64_t>(gpu_timeout_ms) * 1'000'000u;

  for (;;) {
    bool running = headless ? true
                   : software_present
#ifdef HAVE_RASTERDEMO_XCB_SW
                       ? sw_win->poll()
#else
                       ? false
#endif
                       : win->poll();
    if (!running)
      break;

    bool pause_toggled = false;
    if (!headless) {
      if (software_present) {
#ifdef HAVE_RASTERDEMO_XCB_SW
        pause_toggled = sw_win->take_pause_toggle();
#endif
      } else {
        pause_toggled = win->take_pause_toggle();
      }
    }
    if (pause_toggled) {
      render_paused = !render_paused;
      std::printf("Rendering %s.\n", render_paused ? "paused" : "resumed");
      fps_frames = 0;
      fps_time = std::chrono::steady_clock::now();
    }
    if (render_paused) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    if (!software_present && !headless)
      win->wait_idle(current);

    auto now = std::chrono::steady_clock::now();
    float time = std::chrono::duration<float>(now - start).count();
    build_demo_workload(prims, width, height, time);
    if (!build_tile_bins(prims, width, height, tiles_x, tiles_y, tile_ranges,
                         tile_indices, max_tile_indices)) {
      std::fprintf(stderr,
                   "error: tile bin buffer too small for this frame; reduce "
                   "resolution or increase max_tile_indices\n");
      return 1;
    }
    std::memcpy(prim_buf.data(), prims.data(), prims.size() * sizeof(prims[0]));
    std::memcpy(tile_ranges_buf.data(), tile_ranges.data(),
                tile_ranges.size() * sizeof(tile_ranges[0]));
    std::memcpy(tile_indices_buf.data(), tile_indices.data(),
                tile_indices.size() * sizeof(tile_indices[0]));
    kfd::detail::memory_barrier();

    RasterArgs args{
        .prims = static_cast<const DemoPrimitive *>(prim_buf.data()),
        .tile_indices = static_cast<const uint32_t *>(tile_indices_buf.data()),
        .tile_ranges = static_cast<const TileRange *>(tile_ranges_buf.data()),
        .color = static_cast<uint32_t *>(fbs[current].color.data()),
        .depth = static_cast<float *>(depth.data()),
        .width = width,
        .height = height,
        .pitch = pitch,
        .tile_size = TILE_SIZE,
        .tiles_x = tiles_x,
        .clear_color = CLEAR_COLOR,
        .clear_depth = CLEAR_DEPTH,
    };

    if (kernel_mode == KernelMode::StaticGrid) {
      frame_kernel.fill(frame_kernarg, args, cfg);

      KFD_EXPECT(fbs[current].signal->reset());
      KFD_EXPECT(compute.dispatch(frame_kernel, cfg, frame_kernarg,
                                  *fbs[current].signal));
      auto waited =
          fbs[current].signal->wait(kfd::Condition::EQ, 0, gpu_timeout_ns);
      if (!waited) {
        std::fprintf(stderr,
                     "error: frame %u GPU wait timed out or failed: %s\n",
                     frame, kfd::strerror(waited));
        return 1;
      }
    } else if (kernel_mode == KernelMode::PerFrame) {
      auto *active_cursor = &control->active_cursor;
      __atomic_store_n(active_cursor, 0u, __ATOMIC_RELEASE);
      kfd::detail::memory_barrier();

      ClaimFrameArgs claim_args{
          .prims = args.prims,
          .tile_indices = args.tile_indices,
          .tile_ranges = args.tile_ranges,
          .active_tiles =
              static_cast<const uint32_t *>(active_tiles_buf.data()),
          .active_cursor = active_cursor,
          .color = args.color,
          .depth = args.depth,
          .width = args.width,
          .height = args.height,
          .pitch = args.pitch,
          .tile_size = args.tile_size,
          .tiles_x = tiles_x,
          .tiles_y = tiles_y,
          .clear_color = args.clear_color,
          .clear_depth = args.clear_depth,
          .clear_only = clear_only ? 1u : 0u,
          .active_count = tile_count,
      };
      claim_frame_kernel.fill(claim_kernarg, claim_args, claim_cfg);

      KFD_EXPECT(fbs[current].signal->reset());
      KFD_EXPECT(compute.dispatch(claim_frame_kernel, claim_cfg, claim_kernarg,
                                  *fbs[current].signal));
      auto waited =
          fbs[current].signal->wait(kfd::Condition::EQ, 0, gpu_timeout_ns);
      if (!waited) {
        std::fprintf(stderr,
                     "error: frame %u per-frame GPU wait timed out or failed: "
                     "%s\n",
                     frame, kfd::strerror(waited));
        return 1;
      }
    } else {
      uint32_t epoch = frame + 1;
      persistent_args_device->color = args.color;
      kfd::detail::memory_barrier();
      __atomic_store_n(&control->active_cursor, 0u, __ATOMIC_RELEASE);
      __atomic_store_n(&control->render_done, 0u, __ATOMIC_RELEASE);
      __atomic_store_n(&control->active_count, tile_count, __ATOMIC_RELEASE);
      __atomic_store_n(&control->current_epoch, epoch, __ATOMIC_RELEASE);
      kfd::detail::memory_barrier();
      __atomic_store_n(&control->sealed_epoch, epoch, __ATOMIC_RELEASE);

      auto deadline = std::chrono::steady_clock::now() +
                      std::chrono::nanoseconds(gpu_timeout_ns);
      while (__atomic_load_n(&control->completed_epoch, __ATOMIC_ACQUIRE) !=
             epoch) {
        if (std::chrono::steady_clock::now() > deadline) {
          std::fprintf(
              stderr,
              "error: frame %u persistent GPU wait timed out "
              "(current_epoch=%u sealed_epoch=%u closing_epoch=%u "
              "completed_epoch=%u active_cursor=%u active_count=%u "
              "render_done=%u)\n",
              frame, __atomic_load_n(&control->current_epoch, __ATOMIC_ACQUIRE),
              __atomic_load_n(&control->sealed_epoch, __ATOMIC_ACQUIRE),
              __atomic_load_n(&control->closing_epoch, __ATOMIC_ACQUIRE),
              __atomic_load_n(&control->completed_epoch, __ATOMIC_ACQUIRE),
              __atomic_load_n(&control->active_cursor, __ATOMIC_ACQUIRE),
              __atomic_load_n(&control->active_count, __ATOMIC_ACQUIRE),
              __atomic_load_n(&control->render_done, __ATOMIC_ACQUIRE));
          __atomic_store_n(&control->terminate, 1u, __ATOMIC_RELEASE);
          KFD_EXPECT(compute.signal(shutdown_signal));
          (void)shutdown_signal.wait(kfd::Condition::EQ, 0, gpu_timeout_ns);
          return 1;
        }
        kfd::detail::spin_hint();
      }
    }
    if (headless) {
      // No presentation; this mode isolates the kernel and buffer setup.
    } else if (software_present) {
#ifdef HAVE_RASTERDEMO_XCB_SW
      sw_win->present(static_cast<const uint32_t *>(fbs[current].color.data()),
                      pitch);
#endif
    } else {
      win->present(current);
    }

    current = (current + 1) % NUM_BUFFERS;
    ++frame;
    ++fps_frames;
    if (max_frames != 0 && frame >= max_frames)
      break;

    double elapsed = std::chrono::duration<double>(now - fps_time).count();
    if (elapsed >= 2.0) {
      std::printf("%.1f FPS (%u frames in %.1fs)\n",
                  static_cast<double>(fps_frames) / elapsed, fps_frames,
                  elapsed);
      fps_frames = 0;
      fps_time = now;
    }
  }

  if (kernel_mode == KernelMode::Persistent) {
    __atomic_store_n(&control->terminate, 1u, __ATOMIC_RELEASE);
    KFD_EXPECT(compute.signal(shutdown_signal));
    auto stopped = shutdown_signal.wait(kfd::Condition::EQ, 0, gpu_timeout_ns);
    if (!stopped) {
      std::fprintf(stderr, "error: persistent kernel shutdown timed out: %s\n",
                   kfd::strerror(stopped));
      return 1;
    }
  }

  std::printf("Exiting after %u frames.\n", frame);
  return 0;
}
