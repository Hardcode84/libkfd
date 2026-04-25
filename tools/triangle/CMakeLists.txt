find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(XCB xcb xcb-dri3 xcb-present)
endif()
if(NOT XCB_FOUND)
  message(WARNING "triangle: xcb/xcb-dri3/xcb-present not found, skipping build")
  return()
endif()

set(triangle_kernel_entries "")
set(triangle_kernel_targets "")

foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
  string(REPLACE "-" "_" suffix "${arch}")
  set(target "colored_triangle_${suffix}")

  add_executable(${target} shaders/colored_triangle.c)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/kernels")

  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

  target_compile_options(${target} PRIVATE
    --target=amdgcn--amdhsa
    -mcpu=${arch}
    -nogpulib
    -nostdlibinc
    -O2)

  target_link_options(${target} PRIVATE
    --target=amdgcn--amdhsa
    -mcpu=${arch})

  list(APPEND triangle_kernel_targets ${target})
  string(APPEND triangle_kernel_entries
    "  {\"$<TARGET_FILE:${target}>\", \"${arch}\"},\n")
endforeach()

file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/triangle_kernels.inc"
  CONTENT "${triangle_kernel_entries}")

add_executable(triangle
  main.cpp
  ../computetoy/window.cpp)
target_link_libraries(triangle PRIVATE libkfd::kfd ${XCB_LIBRARIES})
target_include_directories(triangle PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}
  ../computetoy
  ${XCB_INCLUDE_DIRS})
target_compile_options(triangle PRIVATE ${common_compile_options})
add_dependencies(triangle ${triangle_kernel_targets})
