//===-- tools/computetoy/drm_window.cpp - DRM/KMS window interface --------===//
//
// DRM atomic modesetting backend. Imports DMA-buf framebuffers and page-flips
// them to a CRTC primary plane with vblank sync.
//
// Uses mailbox triple-buffering: three slots track buffer ownership.
//
//   displaying  - on screen now.
//   pending     - submitted for the next vblank, or -1.
//   ready       - latest GPU output parked in the mailbox, or -1.
//                 Silently overwritten each time the GPU finishes a frame,
//                 so only the most recent result survives to the next vblank.
//
// With 3 buffers, at most 2 are locked (displaying + pending), so the GPU
// always has one free and never stalls on vsync.
//
// Based on David Rheinsberg's DRM atomic modesetting howto.
//  - https://github.com/dvdhrm/docs
//
//===----------------------------------------------------------------------===//

#include "window.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/kd.h>
#include <memory>
#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#include <utility>

#include <drm_fourcc.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

namespace {

class ObjProps {
public:
  ObjProps(int fd, uint32_t obj_id, uint32_t obj_type) {
    props = drmModeObjectGetProperties(fd, obj_id, obj_type);
    if (!props)
      return;
    info = std::make_unique<drmModePropertyPtr[]>(props->count_props);
    for (uint32_t i = 0; i < props->count_props; ++i)
      info[i] = drmModeGetProperty(fd, props->props[i]);
  }

  ~ObjProps() {
    if (!props)
      return;
    for (uint32_t i = 0; i < props->count_props; ++i)
      drmModeFreeProperty(info[i]);
    drmModeFreeObjectProperties(props);
  }

  ObjProps(const ObjProps &) = delete;
  ObjProps &operator=(const ObjProps &) = delete;

  explicit operator bool() const { return props != nullptr; }

  uint32_t id(const char *name) const {
    if (!props)
      return 0;
    for (uint32_t i = 0; i < props->count_props; ++i)
      if (info[i] && std::strcmp(info[i]->name, name) == 0)
        return info[i]->prop_id;
    return 0;
  }

  uint64_t value(const char *name) const {
    if (!props)
      return 0;
    for (uint32_t i = 0; i < props->count_props; ++i)
      if (info[i] && std::strcmp(info[i]->name, name) == 0)
        return props->prop_values[i];
    return 0;
  }

private:
  drmModeObjectPropertiesPtr props = nullptr;
  std::unique_ptr<drmModePropertyPtr[]> info;
};

struct Output {
  uint32_t connector_id;
  uint32_t crtc_id;
  uint32_t crtc_index;
  uint32_t plane_id;
  drmModeModeInfo mode;
};

// Cached property IDs for atomic commits.
struct PropIDs {
  uint32_t conn_crtc_id;
  uint32_t crtc_mode_id;
  uint32_t crtc_active;
  uint32_t plane_fb_id;
  uint32_t plane_crtc_id;
  uint32_t plane_src_x, plane_src_y, plane_src_w, plane_src_h;
  uint32_t plane_crtc_x, plane_crtc_y, plane_crtc_w, plane_crtc_h;
};

drmModeModeInfo *pick_mode(drmModeConnectorPtr conn, uint32_t want_w,
                           uint32_t want_h) {
  drmModeModeInfo *preferred = nullptr;
  drmModeModeInfo *exact = nullptr;
  drmModeModeInfo *first = nullptr;
  for (int i = 0; i < conn->count_modes; ++i) {
    auto &m = conn->modes[i];
    if (m.flags & DRM_MODE_FLAG_INTERLACE)
      continue;
    if (!exact && m.hdisplay == want_w && m.vdisplay == want_h)
      exact = &conn->modes[i];
    if (!preferred && (m.type & DRM_MODE_TYPE_PREFERRED))
      preferred = &conn->modes[i];
    if (!first)
      first = &conn->modes[i];
  }
  if (preferred)
    return preferred;
  if (exact)
    return exact;
  return first ? first : (conn->count_modes ? &conn->modes[0] : nullptr);
}

uint32_t find_primary_plane(int fd, uint32_t crtc_index) {
  auto *res = drmModeGetPlaneResources(fd);
  if (!res)
    return 0;
  uint32_t result = 0;
  for (uint32_t i = 0; i < res->count_planes && !result; ++i) {
    auto *plane = drmModeGetPlane(fd, res->planes[i]);
    if (!plane)
      continue;
    if (plane->possible_crtcs & (1u << crtc_index)) {
      ObjProps props(fd, res->planes[i], DRM_MODE_OBJECT_PLANE);
      if (props.value("type") == DRM_PLANE_TYPE_PRIMARY)
        result = res->planes[i];
    }
    drmModeFreePlane(plane);
  }
  drmModeFreePlaneResources(res);
  return result;
}

std::expected<Output, kfd::Error> find_output(int fd, uint32_t want_w,
                                              uint32_t want_h) {
  drmModeResPtr res = drmModeGetResources(fd);
  if (!res)
    return kfd::unexpected(ENODEV, "drmModeGetResources failed");

  for (int c = 0; c < res->count_connectors; ++c) {
    drmModeConnectorPtr conn = drmModeGetConnector(fd, res->connectors[c]);
    if (!conn)
      continue;
    if (conn->connection != DRM_MODE_CONNECTED || conn->count_modes == 0) {
      drmModeFreeConnector(conn);
      continue;
    }

    drmModeModeInfo *mode = pick_mode(conn, want_w, want_h);
    if (!mode) {
      drmModeFreeConnector(conn);
      continue;
    }

    // Prefer the currently active encoder+CRTC to avoid a full modeset.
    uint32_t crtc_id = 0;
    uint32_t crtc_index = 0;
    if (conn->encoder_id) {
      drmModeEncoderPtr enc = drmModeGetEncoder(fd, conn->encoder_id);
      if (enc && enc->crtc_id) {
        for (int r = 0; r < res->count_crtcs; ++r) {
          if (res->crtcs[r] == enc->crtc_id) {
            crtc_id = enc->crtc_id;
            crtc_index = static_cast<uint32_t>(r);
            break;
          }
        }
      }
      drmModeFreeEncoder(enc);
    }

    // Fall back to searching all encoders.
    if (!crtc_id) {
      for (int e = 0; e < conn->count_encoders && !crtc_id; ++e) {
        drmModeEncoderPtr enc = drmModeGetEncoder(fd, conn->encoders[e]);
        if (!enc)
          continue;
        for (int r = 0; r < res->count_crtcs; ++r) {
          if (enc->possible_crtcs & (1u << r)) {
            crtc_id = res->crtcs[r];
            crtc_index = static_cast<uint32_t>(r);
            break;
          }
        }
        drmModeFreeEncoder(enc);
      }
    }

    if (!crtc_id) {
      drmModeFreeConnector(conn);
      continue;
    }

    uint32_t plane_id = find_primary_plane(fd, crtc_index);
    if (!plane_id) {
      drmModeFreeConnector(conn);
      continue;
    }

    Output out{res->connectors[c], crtc_id, crtc_index, plane_id, *mode};
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    return out;
  }

  drmModeFreeResources(res);
  return kfd::unexpected(ENODEV, "No connected DRM output found");
}

std::expected<PropIDs, kfd::Error> lookup_props(int fd, const Output &out) {
  ObjProps conn(fd, out.connector_id, DRM_MODE_OBJECT_CONNECTOR);
  ObjProps crtc(fd, out.crtc_id, DRM_MODE_OBJECT_CRTC);
  ObjProps plane(fd, out.plane_id, DRM_MODE_OBJECT_PLANE);

  if (!conn || !crtc || !plane)
    return kfd::unexpected(ENODEV, "Failed to get DRM object properties");

  PropIDs p{};
  p.conn_crtc_id = conn.id("CRTC_ID");
  p.crtc_mode_id = crtc.id("MODE_ID");
  p.crtc_active = crtc.id("ACTIVE");
  p.plane_fb_id = plane.id("FB_ID");
  p.plane_crtc_id = plane.id("CRTC_ID");
  p.plane_src_x = plane.id("SRC_X");
  p.plane_src_y = plane.id("SRC_Y");
  p.plane_src_w = plane.id("SRC_W");
  p.plane_src_h = plane.id("SRC_H");
  p.plane_crtc_x = plane.id("CRTC_X");
  p.plane_crtc_y = plane.id("CRTC_Y");
  p.plane_crtc_w = plane.id("CRTC_W");
  p.plane_crtc_h = plane.id("CRTC_H");

  if (!p.conn_crtc_id || !p.crtc_mode_id || !p.crtc_active || !p.plane_fb_id ||
      !p.plane_crtc_id || !p.plane_src_w || !p.plane_src_h || !p.plane_crtc_w ||
      !p.plane_crtc_h)
    return kfd::unexpected(ENOENT, "Missing required DRM atomic properties");

  return p;
}

// Global TTY state for signal-safe restoration. KD_GRAPHICS survives process
// death, so we must restore KD_TEXT even under SIGTERM/SIGSEGV/etc.
int g_tty_fd = -1;
struct termios g_saved_tty;

void restore_tty() {
  if (g_tty_fd < 0)
    return;
  std::fprintf(stderr, "Restored TTY on abnormal exit. You're welcome");
  ::ioctl(g_tty_fd, KDSETMODE, KD_TEXT);
  tcsetattr(g_tty_fd, TCSANOW, &g_saved_tty);
  g_tty_fd = -1;
}

void tty_signal_handler(int sig) {
  restore_tty();
  ::signal(sig, SIG_DFL);
  ::raise(sig);
}

void install_tty_guard(int fd, struct termios saved) {
  g_tty_fd = fd;
  g_saved_tty = saved;

  struct sigaction sa{};
  sa.sa_handler = tty_signal_handler;
  sa.sa_flags = static_cast<int>(SA_RESETHAND);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGHUP, &sa, nullptr);

  std::atexit(restore_tty);
}

std::expected<int, kfd::Error> open_card(int render_fd) {
  drmDevicePtr dev = nullptr;
  if (drmGetDevice2(render_fd, 0, &dev) != 0)
    return kfd::unexpected(errno, "drmGetDevice2 failed on render fd %d",
                           render_fd);

  if (!(dev->available_nodes & (1 << DRM_NODE_PRIMARY))) {
    drmFreeDevice(&dev);
    return kfd::unexpected(ENODEV, "No card node for render fd %d", render_fd);
  }

  const char *path = dev->nodes[DRM_NODE_PRIMARY];
  int fd = ::open(path, O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    int e = errno;
    drmFreeDevice(&dev);
    return kfd::unexpected(e, "Cannot open %s", path);
  }

  if (!drmIsMaster(fd) && drmSetMaster(fd) != 0) {
    ::close(fd);
    drmFreeDevice(&dev);
    return kfd::unexpected(
        EACCES, "%s: cannot become DRM master (not on a TTY?)", path);
  }

  if (drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) != 0 ||
      drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
    ::close(fd);
    drmFreeDevice(&dev);
    return kfd::unexpected(ENOTSUP, "%s: atomic modesetting not supported",
                           path);
  }

  drmFreeDevice(&dev);
  return fd;
}

class DRMWindow final : public Window {
public:
  ~DRMWindow() override {
    if (card_fd < 0)
      return;

    ready = -1;
    while (pending >= 0)
      handle_events(-1);

    // Restore the saved CRTC state before tearing down our framebuffers,
    // so the CRTC stops scanning our buffers before we remove them.
    if (saved_crtc) {
      drmModeSetCrtc(card_fd, saved_crtc->crtc_id, saved_crtc->buffer_id,
                     saved_crtc->x, saved_crtc->y, &output.connector_id, 1,
                     &saved_crtc->mode);
      drmModeFreeCrtc(saved_crtc);
    }

    for (uint32_t i = 0; i < num_buffers; ++i) {
      if (fb_ids[i])
        drmModeRmFB(card_fd, fb_ids[i]);
      if (gem_handles[i])
        drmCloseBufferHandle(card_fd, gem_handles[i]);
    }

    if (mode_blob_id)
      drmModeDestroyPropertyBlob(card_fd, mode_blob_id);

    if (tty_fd >= 0) {
      g_tty_fd = -1;
      ::ioctl(tty_fd, KDSETMODE, KD_TEXT);
      tcsetattr(tty_fd, TCSANOW, &saved_tty);
      ::close(tty_fd);
    }

    drmDropMaster(card_fd);
    ::close(card_fd);
  }

  DRMWindow(DRMWindow &&o)
      : Window(o.w, o.h), card_fd(std::exchange(o.card_fd, -1)),
        tty_fd(std::exchange(o.tty_fd, -1)), output(o.output), props(o.props),
        mode_blob_id(std::exchange(o.mode_blob_id, 0)),
        saved_crtc(std::exchange(o.saved_crtc, nullptr)),
        saved_tty(o.saved_tty), num_buffers(o.num_buffers),
        fb_ids(std::move(o.fb_ids)), gem_handles(std::move(o.gem_handles)),
        displaying(o.displaying), pending(o.pending), ready(o.ready),
        mode_set(o.mode_set), quit(o.quit), pause_toggle(o.pause_toggle),
        vblank_interval(o.vblank_interval) {}

  std::expected<void, kfd::Error> import_buffer(uint32_t index, int dmabuf_fd,
                                                size_t /*size*/,
                                                uint32_t stride) override {
    if (index >= num_buffers)
      return kfd::unexpected(EINVAL, "Buffer index %u exceeds num_buffers",
                             index);

    uint32_t handle = 0;
    if (drmPrimeFDToHandle(card_fd, dmabuf_fd, &handle))
      return kfd::unexpected(errno, "drmPrimeFDToHandle failed");
    gem_handles[index] = handle;

    uint32_t handles[4] = {handle, 0, 0, 0};
    uint32_t strides[4] = {stride, 0, 0, 0};
    uint32_t offsets[4] = {0, 0, 0, 0};
    uint32_t fb = 0;
    if (drmModeAddFB2(card_fd, w, h, DRM_FORMAT_XRGB8888, handles, strides,
                      offsets, &fb, 0))
      return kfd::unexpected(errno, "drmModeAddFB2 failed (%ux%u stride %u)", w,
                             h, stride);
    fb_ids[index] = fb;
    return {};
  }

  bool poll() override {
    if (quit)
      return false;
    if (tty_fd >= 0) {
      char c;
      while (::read(tty_fd, &c, 1) == 1) {
        if (c == 'q' || c == 'Q' || c == 27) {
          quit = true;
          return false;
        }
        if (c == ' ')
          pause_toggle = true;
      }
    }
    return true;
  }

  bool take_pause_toggle() override {
    return std::exchange(pause_toggle, false);
  }

  void wait_idle(uint32_t index) override {
    auto idx = static_cast<int32_t>(index);
    while (displaying == idx || pending == idx || ready == idx)
      handle_events(-1);
  }

  void present(uint32_t index) override {
    drain_events();

    if (!mode_set) {
      uint32_t flags = DRM_MODE_ATOMIC_ALLOW_MODESET | DRM_MODE_PAGE_FLIP_EVENT;
      int ret = atomic_commit(index, flags);
      if (ret) {
        std::fprintf(stderr, "modeset FAILED: %s\n", std::strerror(-ret));
      } else {
        pending = static_cast<int32_t>(index);
        mode_set = true;
      }
      return;
    }

    if (pending < 0) {
      uint32_t flags = DRM_MODE_PAGE_FLIP_EVENT | DRM_MODE_ATOMIC_NONBLOCK;
      int ret = atomic_commit(index, flags);
      if (ret) {
        std::fprintf(stderr,
                     "page-flip FAILED: %s (displaying=%d pending=%d)\n",
                     std::strerror(-ret), displaying, pending);
      } else {
        pending = static_cast<int32_t>(index);
      }
    } else {
      ready = static_cast<int32_t>(index);
    }
  }

  double frame_interval() const override { return vblank_interval; }

  static std::expected<DRMWindow, kfd::Error>
  create(uint32_t width, uint32_t height, uint32_t num_buffers, int render_fd);

private:
  DRMWindow(int card_fd, int tty_fd, struct termios saved_tty, Output output,
            PropIDs props, uint32_t mode_blob_id, drmModeCrtcPtr saved_crtc,
            uint32_t w, uint32_t h, uint32_t num_buffers,
            double vblank_interval)
      : Window(w, h), card_fd(card_fd), tty_fd(tty_fd), output(output),
        props(props), mode_blob_id(mode_blob_id), saved_crtc(saved_crtc),
        saved_tty(saved_tty), num_buffers(num_buffers),
        fb_ids(std::make_unique<uint32_t[]>(num_buffers)),
        gem_handles(std::make_unique<uint32_t[]>(num_buffers)),
        vblank_interval(vblank_interval) {}

  int atomic_commit(uint32_t buf_index, uint32_t flags) {
    drmModeAtomicReqPtr req = drmModeAtomicAlloc();
    if (!req)
      return -ENOMEM;

    auto add = [&](uint32_t obj, uint32_t prop, uint64_t val) {
      return drmModeAtomicAddProperty(req, obj, prop, val) >= 0;
    };

    bool ok = true;
    ok = ok && add(output.connector_id, props.conn_crtc_id, output.crtc_id);
    ok = ok && add(output.crtc_id, props.crtc_mode_id, mode_blob_id);
    ok = ok && add(output.crtc_id, props.crtc_active, 1);

    ok = ok && add(output.plane_id, props.plane_fb_id, fb_ids[buf_index]);
    ok = ok && add(output.plane_id, props.plane_crtc_id, output.crtc_id);
    ok = ok && add(output.plane_id, props.plane_src_x, 0);
    ok = ok && add(output.plane_id, props.plane_src_y, 0);
    ok = ok && add(output.plane_id, props.plane_src_w,
                   static_cast<uint64_t>(w) << 16);
    ok = ok && add(output.plane_id, props.plane_src_h,
                   static_cast<uint64_t>(h) << 16);
    ok = ok && add(output.plane_id, props.plane_crtc_x, 0);
    ok = ok && add(output.plane_id, props.plane_crtc_y, 0);
    ok = ok && add(output.plane_id, props.plane_crtc_w, w);
    ok = ok && add(output.plane_id, props.plane_crtc_h, h);

    if (!ok) {
      drmModeAtomicFree(req);
      return -ENOMEM;
    }

    int ret = drmModeAtomicCommit(card_fd, req, flags, this);
    drmModeAtomicFree(req);
    return ret;
  }

  bool handle_events(int timeout_ms) {
    struct pollfd pfd = {.fd = card_fd, .events = POLLIN};
    if (::poll(&pfd, 1, timeout_ms) <= 0)
      return false;
    drmEventContext ctx{};
    ctx.version = 3;
    ctx.page_flip_handler2 = on_page_flip;
    drmHandleEvent(card_fd, &ctx);
    return true;
  }

  void drain_events() {
    while (handle_events(0))
      ;
  }

  static void on_page_flip(int /*fd*/, unsigned int /*frame*/,
                           unsigned int /*sec*/, unsigned int /*usec*/,
                           unsigned int /*crtc_id*/, void *data) {
    auto *self = static_cast<DRMWindow *>(data);
    self->displaying = self->pending;

    if (self->ready >= 0) {
      auto ri = static_cast<uint32_t>(self->ready);
      uint32_t flags = DRM_MODE_PAGE_FLIP_EVENT | DRM_MODE_ATOMIC_NONBLOCK;
      int ret = self->atomic_commit(ri, flags);
      if (ret) {
        std::fprintf(stderr, "chain flip FAILED: %s\n", std::strerror(-ret));
        self->pending = -1;
      } else {
        self->pending = self->ready;
      }
      self->ready = -1;
    } else {
      self->pending = -1;
    }
  }

  int card_fd = -1;
  int tty_fd = -1;
  Output output{};
  PropIDs props{};
  uint32_t mode_blob_id = 0;
  drmModeCrtcPtr saved_crtc = nullptr;
  struct termios saved_tty{};
  uint32_t num_buffers = 0;

  std::unique_ptr<uint32_t[]> fb_ids;
  std::unique_ptr<uint32_t[]> gem_handles;

  int32_t displaying = -1;
  int32_t pending = -1;
  int32_t ready = -1;
  bool mode_set = false;
  bool quit = false;
  bool pause_toggle = false;
  double vblank_interval = 0.0;
};

std::expected<DRMWindow, kfd::Error> DRMWindow::create(uint32_t width,
                                                       uint32_t height,
                                                       uint32_t num_buffers,
                                                       int render_fd) {
  auto card_fd = open_card(render_fd);
  if (!card_fd)
    return std::unexpected(card_fd.error());

  auto out = find_output(*card_fd, width, height);
  if (!out) {
    ::close(*card_fd);
    return std::unexpected(out.error());
  }

  auto p = lookup_props(*card_fd, *out);
  if (!p) {
    ::close(*card_fd);
    return std::unexpected(p.error());
  }

  drmModeCrtcPtr saved = drmModeGetCrtc(*card_fd, out->crtc_id);

  uint32_t mode_w = out->mode.hdisplay;
  uint32_t mode_h = out->mode.vdisplay;
  std::printf("DRM/KMS: %s %ux%u @ %uHz on CRTC %u (atomic)\n", out->mode.name,
              mode_w, mode_h, out->mode.vrefresh, out->crtc_id);

  if (mode_w != width || mode_h != height)
    std::printf("DRM/KMS: requested %ux%u, using native %ux%u\n", width, height,
                mode_w, mode_h);

  uint32_t blob_id = 0;
  if (drmModeCreatePropertyBlob(*card_fd, &out->mode, sizeof(out->mode),
                                &blob_id)) {
    if (saved)
      drmModeFreeCrtc(saved);
    ::close(*card_fd);
    return kfd::unexpected(errno, "drmModeCreatePropertyBlob failed");
  }

  struct termios saved_term{};
  int tty = ::open("/dev/tty", O_RDWR | O_CLOEXEC | O_NONBLOCK);
  if (tty >= 0) {
    tcgetattr(tty, &saved_term);
    struct termios raw = saved_term;
    raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(tty, TCSANOW, &raw);

    if (::ioctl(tty, KDSETMODE, KD_GRAPHICS) < 0)
      std::fprintf(stderr, "KDSETMODE(KD_GRAPHICS) failed: %s\n",
                   std::strerror(errno));

    install_tty_guard(tty, saved_term);
  }

  double pixel_clock_hz = static_cast<double>(out->mode.clock) * 1000.0;
  double total_pixels =
      static_cast<double>(out->mode.htotal) * out->mode.vtotal;
  double vblank_interval = total_pixels / pixel_clock_hz;

  return DRMWindow(*card_fd, tty, saved_term, *out, *p, blob_id, saved, mode_w,
                   mode_h, num_buffers, vblank_interval);
}

} // namespace

std::expected<std::unique_ptr<Window>, kfd::Error>
create_drm_window(uint32_t width, uint32_t height, uint32_t num_buffers,
                  int render_fd) {
  auto win = DRMWindow::create(width, height, num_buffers, render_fd);
  if (!win)
    return std::unexpected(win.error());
  return std::make_unique<DRMWindow>(std::move(*win));
}
