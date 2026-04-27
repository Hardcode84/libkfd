if(NOT TARGET computetoy_window)
  message(WARNING "rasterdemo: display backend unavailable, skipping build")
  return()
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(RASTERDEMO_XCB xcb)
endif()

set(rasterdemo_kernel_entries "")
foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
  string(REPLACE "-" "_" suffix "${arch}")
  set(target "rasterdemo_kernel_${suffix}")

  add_executable(${target} raster_kernel.c)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/kernels")

  target_compile_options(${target} PRIVATE
    --target=amdgcn--amdhsa
    -mcpu=${arch}
    -nogpulib
    -nostdlibinc
    -O2
  )

  target_link_options(${target} PRIVATE
    --target=amdgcn--amdhsa
    -mcpu=${arch}
  )

  string(APPEND rasterdemo_kernel_entries
    "  {\"$<TARGET_FILE:${target}>\", \"${arch}\"},\n")
  list(APPEND rasterdemo_kernel_targets ${target})
endforeach()

file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/rasterdemo_kernels.inc"
  CONTENT "${rasterdemo_kernel_entries}"
)

add_executable(rasterdemo main.cpp)
target_link_libraries(rasterdemo PRIVATE libkfd::kfd computetoy_window)
target_compile_options(rasterdemo PRIVATE ${common_compile_options})
target_include_directories(rasterdemo PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(rasterdemo ${rasterdemo_kernel_targets})

if(RASTERDEMO_XCB_FOUND)
  target_compile_definitions(rasterdemo PRIVATE HAVE_RASTERDEMO_XCB_SW)
  target_link_libraries(rasterdemo PRIVATE ${RASTERDEMO_XCB_LIBRARIES})
  target_include_directories(rasterdemo PRIVATE ${RASTERDEMO_XCB_INCLUDE_DIRS})
endif()
