//===-- tools/computetoy/window.cpp - Display backend factory -------------===//
//
// Window::create() tries each compiled-in backend and returns the first that
// succeeds. Backend availability is controlled by HAVE_XCB / HAVE_LIBDRM
// definitions set by CMake.
//
//===----------------------------------------------------------------------===//

#include "window.h"

#include <optional>

#ifdef HAVE_XCB
std::expected<std::unique_ptr<Window>, kfd::Error>
create_xcb_window(uint32_t, uint32_t, uint32_t, const char *);
#endif

#ifdef HAVE_LIBDRM
std::expected<std::unique_ptr<Window>, kfd::Error>
create_drm_window(uint32_t, uint32_t, uint32_t, int);
#endif

std::expected<std::unique_ptr<Window>, kfd::Error>
Window::create(uint32_t width, uint32_t height, uint32_t num_buffers,
               const char *title, int render_fd, WindowBackend backend) {
  std::optional<kfd::Error> first_error;
#ifdef HAVE_XCB
  if (backend == WindowBackend::Auto || backend == WindowBackend::Xcb) {
    if (auto win = create_xcb_window(width, height, num_buffers, title))
      return win;
    else if (!first_error)
      first_error = win.error();
    if (backend == WindowBackend::Xcb)
      return std::unexpected(*first_error);
  }
#else
  if (backend == WindowBackend::Xcb)
    return kfd::unexpected(ENODEV, "XCB display backend was not built");
#endif
#ifdef HAVE_LIBDRM
  if (backend == WindowBackend::Auto || backend == WindowBackend::Drm) {
    if (auto win = create_drm_window(width, height, num_buffers, render_fd))
      return win;
    else if (!first_error)
      first_error = win.error();
    if (backend == WindowBackend::Drm)
      return std::unexpected(*first_error);
  }
#else
  if (backend == WindowBackend::Drm)
    return kfd::unexpected(ENODEV, "DRM display backend was not built");
#endif
  if (first_error)
    return std::unexpected(*first_error);
  return kfd::unexpected(ENODEV, "No display backend available");
}
