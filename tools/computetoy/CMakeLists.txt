find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(XCB xcb xcb-dri3 xcb-present)
  pkg_check_modules(LIBDRM libdrm)
endif()
if(NOT XCB_FOUND AND NOT LIBDRM_FOUND)
  message(WARNING "computetoy: neither xcb nor libdrm found, skipping build")
  return()
endif()

# Compile a GPU shader for every target architecture.
function(add_shader name source)
  foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
    string(REPLACE "-" "_" suffix "${arch}")
    set(target "${name}_${suffix}")

    add_executable(${target} ${source})
    set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/kernels")

    target_compile_options(${target} PRIVATE
      --target=amdgcn--amdhsa
      -mcpu=${arch}
      -nogpulib
      -nostdlibinc
      -O2
      "SHELL:-Xclang -target-feature"
      "SHELL:-Xclang +kernarg-preload"
      "SHELL:-mllvm -amdgpu-kernarg-preload-count=16"
    )

    target_link_options(${target} PRIVATE
      --target=amdgcn--amdhsa
      -mcpu=${arch}
    )
  endforeach()
endfunction()

add_shader(plasma shaders/plasma.c)
add_shader(tunnel shaders/tunnel.c)

add_library(computetoy_window OBJECT window.cpp)
target_link_libraries(computetoy_window PUBLIC libkfd::kfd)
target_compile_options(computetoy_window PRIVATE ${common_compile_options})
target_include_directories(computetoy_window PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(XCB_FOUND)
  target_sources(computetoy_window PRIVATE xcb_window.cpp)
  target_compile_definitions(computetoy_window PRIVATE HAVE_XCB)
  target_link_libraries(computetoy_window PUBLIC ${XCB_LIBRARIES})
  target_include_directories(computetoy_window PRIVATE ${XCB_INCLUDE_DIRS})
endif()

if(LIBDRM_FOUND)
  target_sources(computetoy_window PRIVATE drm_window.cpp)
  target_compile_definitions(computetoy_window PRIVATE HAVE_LIBDRM)
  target_link_libraries(computetoy_window PUBLIC ${LIBDRM_LIBRARIES})
  target_include_directories(computetoy_window PRIVATE ${LIBDRM_INCLUDE_DIRS})
endif()

add_executable(computetoy main.cpp)
target_link_libraries(computetoy PRIVATE computetoy_window)
target_compile_options(computetoy PRIVATE ${common_compile_options})
