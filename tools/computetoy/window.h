//===-- tools/computetoy/window.h - Display backend interface ----*- C++
//-*-===//
//
// Abstract interface for presenting GPU-rendered frames. Concrete backends
// implement this for X11 (DRI3/Present) and raw DRM/KMS console output.
//
//===----------------------------------------------------------------------===//

#ifndef COMPUTETOY_WINDOW_H
#define COMPUTETOY_WINDOW_H

#include "libkfd/error.h"

#include <cstdint>
#include <expected>
#include <memory>

enum class WindowBackend {
  Auto,
  Xcb,
  Drm,
};

class Window {
public:
  virtual ~Window() = default;

  Window(const Window &) = delete;
  Window &operator=(const Window &) = delete;

  // Import a DMA buffer file descriptor as a presentable buffer. The fd is
  // dup'd internally and the caller retains ownership of the original.
  virtual std::expected<void, kfd::Error> import_buffer(uint32_t index,
                                                        int dmabuf_fd,
                                                        size_t size,
                                                        uint32_t stride) = 0;

  // Poll for input events. Returns false when the session should end.
  virtual bool poll() = 0;

  // Block until the given buffer is no longer in use by the display.
  virtual void wait_idle(uint32_t index) = 0;

  // Present the given buffer to the display.
  virtual void present(uint32_t index) = 0;

  uint32_t width() const { return w; }
  uint32_t height() const { return h; }

  // Steady-state frame interval in seconds, or 0 if unknown (free-running).
  // DRM reports the vblank period; XCB/Present runs unthrottled.
  virtual double frame_interval() const { return 0.0; }

  static std::expected<std::unique_ptr<Window>, kfd::Error>
  create(uint32_t width, uint32_t height, uint32_t num_buffers,
         const char *title, int render_fd = -1,
         WindowBackend backend = WindowBackend::Auto);

protected:
  Window(uint32_t w, uint32_t h) : w(w), h(h) {}

  uint32_t w;
  uint32_t h;
};

#endif // COMPUTETOY_WINDOW_H
