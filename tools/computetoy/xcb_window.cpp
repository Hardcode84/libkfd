//===-- tools/computetoy/xcb_window.cpp - DRI3/Present X11 backend --------===//
//
// X11 display backend using XCB with DRI3 pixmap import and Present-based
// buffer management. All XCB, DRI3, and Present interaction is confined here.
//
//===----------------------------------------------------------------------===//

#include "window.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unistd.h>
#include <utility>

#include <xcb/dri3.h>
#include <xcb/present.h>
#include <xcb/xcb.h>

namespace {

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

xcb_visualtype_t *find_argb_visual(xcb_screen_t *screen) {
  for (auto di = xcb_screen_allowed_depths_iterator(screen); di.rem;
       xcb_depth_next(&di)) {
    if (di.data->depth != 32)
      continue;
    for (auto vi = xcb_depth_visuals_iterator(di.data); vi.rem;
         xcb_visualtype_next(&vi)) {
      if (vi.data->_class == XCB_VISUAL_CLASS_TRUE_COLOR)
        return vi.data;
    }
  }
  return nullptr;
}

class XCBWindow final : public Window {
public:
  ~XCBWindow() override {
    if (!conn)
      return;
    for (uint32_t i = 0; i < num_buffers; ++i)
      if (pixmaps[i])
        xcb_free_pixmap(conn, pixmaps[i]);
    if (present_special)
      xcb_unregister_for_special_event(conn, present_special);
    xcb_destroy_window(conn, win);
    if (colormap)
      xcb_free_colormap(conn, colormap);
    xcb_disconnect(conn);
  }

  XCBWindow(XCBWindow &&o)
      : Window(o.w, o.h), conn(std::exchange(o.conn, nullptr)),
        win(std::exchange(o.win, 0)), colormap(std::exchange(o.colormap, 0)),
        wm_delete(std::exchange(o.wm_delete, 0)),
        present_special(std::exchange(o.present_special, nullptr)),
        depth(o.depth), num_buffers(o.num_buffers),
        pixmaps(std::move(o.pixmaps)), busy(std::move(o.busy)),
        last_msc(o.last_msc) {}

  std::expected<void, kfd::Error> import_buffer(uint32_t index, int dmabuf_fd,
                                                size_t size,
                                                uint32_t stride) override {
    if (index >= num_buffers)
      return kfd::unexpected(EINVAL, "Buffer index %u exceeds num_buffers",
                             index);
    int fd = dup(dmabuf_fd);
    if (fd < 0)
      return kfd::unexpected(errno, "dup() failed for dmabuf fd");

    pixmaps[index] = xcb_generate_id(conn);
    auto cookie = xcb_dri3_pixmap_from_buffer_checked(
        conn, pixmaps[index], win, static_cast<uint32_t>(size),
        static_cast<uint16_t>(w), static_cast<uint16_t>(h),
        static_cast<uint16_t>(stride), depth, 32, fd);
    if (auto *err = xcb_request_check(conn, cookie)) {
      uint8_t code = err->error_code;
      std::free(err);
      return kfd::unexpected(
          EIO, "DRI3 pixmap_from_buffer failed (X error %u, %ux%u stride %u)",
          code, w, h, stride);
    }
    return {};
  }

  bool poll() override {
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

  void wait_idle(uint32_t index) override {
    while (busy[index])
      drain_present_events(/*block=*/true);
  }

  void present(uint32_t index) override {
    uint64_t target = last_msc + 1;
    xcb_present_pixmap(conn, win, pixmaps[index], 0, 0, 0, 0, 0, 0, 0, 0,
                       XCB_PRESENT_OPTION_NONE, target, 0, 0, 0, nullptr);
    busy[index] = true;
    xcb_flush(conn);
    while (last_msc < target)
      drain_present_events(/*block=*/true);
  }

  static std::expected<XCBWindow, kfd::Error> create(uint32_t width,
                                                     uint32_t height,
                                                     uint32_t num_buffers,
                                                     const char *title);

private:
  XCBWindow(xcb_connection_t *conn, xcb_window_t win, xcb_colormap_t colormap,
            xcb_atom_t wm_delete, xcb_special_event_t *present_special,
            uint8_t depth, uint32_t w, uint32_t h, uint32_t num_buffers)
      : Window(w, h), conn(conn), win(win), colormap(colormap),
        wm_delete(wm_delete), present_special(present_special), depth(depth),
        num_buffers(num_buffers),
        pixmaps(std::make_unique<xcb_pixmap_t[]>(num_buffers)),
        busy(std::make_unique<bool[]>(num_buffers)) {}

  void drain_present_events(bool block) {
    for (;;) {
      xcb_generic_event_t *sev =
          block ? xcb_wait_for_special_event(conn, present_special)
                : xcb_poll_for_special_event(conn, present_special);
      if (!sev)
        break;
      auto *ge = reinterpret_cast<xcb_ge_generic_event_t *>(sev);
      if (ge->event_type == XCB_PRESENT_EVENT_COMPLETE_NOTIFY) {
        auto *comp =
            reinterpret_cast<xcb_present_complete_notify_event_t *>(sev);
        last_msc = comp->msc;
      } else if (ge->event_type == XCB_PRESENT_EVENT_IDLE_NOTIFY) {
        auto *idle = reinterpret_cast<xcb_present_idle_notify_event_t *>(sev);
        for (uint32_t i = 0; i < num_buffers; ++i)
          if (pixmaps[i] == idle->pixmap)
            busy[i] = false;
      }
      std::free(sev);
      block = false;
    }
  }

  xcb_connection_t *conn = nullptr;
  xcb_window_t win = 0;
  xcb_colormap_t colormap = 0;
  xcb_atom_t wm_delete = 0;
  xcb_special_event_t *present_special = nullptr;
  uint8_t depth = 0;
  uint32_t num_buffers = 0;
  std::unique_ptr<xcb_pixmap_t[]> pixmaps;
  std::unique_ptr<bool[]> busy;
  uint64_t last_msc = 0;
};

std::expected<XCBWindow, kfd::Error> XCBWindow::create(uint32_t width,
                                                       uint32_t height,
                                                       uint32_t num_buffers,
                                                       const char *title) {
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

  xcb_visualtype_t *visual = find_argb_visual(screen);
  uint8_t win_depth = visual ? 32 : screen->root_depth;
  xcb_visualid_t visual_id = visual ? visual->visual_id : screen->root_visual;

  xcb_colormap_t cmap = 0;
  if (visual) {
    cmap = xcb_generate_id(conn);
    xcb_create_colormap(conn, XCB_COLORMAP_ALLOC_NONE, cmap, screen->root,
                        visual_id);
  }

  xcb_window_t window = xcb_generate_id(conn);
  uint32_t event_mask = XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_KEY_PRESS |
                        XCB_EVENT_MASK_STRUCTURE_NOTIFY;
  uint32_t mask = XCB_CW_BORDER_PIXEL | XCB_CW_EVENT_MASK | XCB_CW_COLORMAP;
  uint32_t values[] = {0, event_mask, cmap ? cmap : screen->default_colormap};
  xcb_create_window(conn, win_depth, window, screen->root, 0, 0,
                    static_cast<uint16_t>(width), static_cast<uint16_t>(height),
                    0, XCB_WINDOW_CLASS_INPUT_OUTPUT, visual_id, mask, values);

  xcb_atom_t wm_protocols = intern_atom(conn, "WM_PROTOCOLS");
  xcb_atom_t wm_delete = intern_atom(conn, "WM_DELETE_WINDOW");
  xcb_change_property(conn, XCB_PROP_MODE_REPLACE, window, wm_protocols,
                      XCB_ATOM_ATOM, 32, 1, &wm_delete);

  size_t title_len = std::strlen(title);
  xcb_change_property(conn, XCB_PROP_MODE_REPLACE, window, XCB_ATOM_WM_NAME,
                      XCB_ATOM_STRING, 8, static_cast<uint32_t>(title_len),
                      title);

  xcb_atom_t wm_type = intern_atom(conn, "_NET_WM_WINDOW_TYPE");
  xcb_atom_t wm_type_dialog = intern_atom(conn, "_NET_WM_WINDOW_TYPE_DIALOG");
  xcb_change_property(conn, XCB_PROP_MODE_REPLACE, window, wm_type,
                      XCB_ATOM_ATOM, 32, 1, &wm_type_dialog);

  xcb_map_window(conn, window);
  xcb_flush(conn);

  {
    auto cookie = xcb_dri3_query_version(conn, 1, 2);
    auto *reply = xcb_dri3_query_version_reply(conn, cookie, nullptr);
    if (!reply) {
      xcb_destroy_window(conn, window);
      xcb_disconnect(conn);
      return kfd::unexpected(ENOTSUP, "DRI3 extension not available");
    }
    std::printf("DRI3 %u.%u\n", reply->major_version, reply->minor_version);
    std::free(reply);
  }

  {
    auto cookie = xcb_present_query_version(conn, 1, 2);
    auto *reply = xcb_present_query_version_reply(conn, cookie, nullptr);
    if (!reply) {
      xcb_destroy_window(conn, window);
      xcb_disconnect(conn);
      return kfd::unexpected(ENOTSUP, "Present extension not available");
    }
    std::printf("Present %u.%u\n", reply->major_version, reply->minor_version);
    std::free(reply);
  }

  uint32_t present_eid = xcb_generate_id(conn);
  xcb_present_select_input(conn, present_eid, window,
                           XCB_PRESENT_EVENT_MASK_COMPLETE_NOTIFY |
                               XCB_PRESENT_EVENT_MASK_IDLE_NOTIFY);
  xcb_special_event_t *special =
      xcb_register_for_special_xge(conn, &xcb_present_id, present_eid, nullptr);
  xcb_flush(conn);

  return XCBWindow(conn, window, cmap, wm_delete, special, win_depth, width,
                   height, num_buffers);
}

} // namespace

std::expected<std::unique_ptr<Window>, kfd::Error>
create_xcb_window(uint32_t width, uint32_t height, uint32_t num_buffers,
                  const char *title) {
  if (!std::getenv("DISPLAY"))
    return kfd::unexpected(
        ENODEV, "No X11 DISPLAY available; XCB/DRI3 needs X11 or Xwayland");
  auto win = XCBWindow::create(width, height, num_buffers, title);
  if (!win)
    return std::unexpected(win.error());
  return std::make_unique<XCBWindow>(std::move(*win));
}
