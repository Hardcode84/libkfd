cmake_minimum_required(VERSION 3.28)
project(libkfd VERSION 0.1.0 LANGUAGES C CXX ASM)

foreach(lang C CXX)
  if(NOT CMAKE_${lang}_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR
      "This project requires Clang as the ${lang} compiler, "
      "found ${CMAKE_${lang}_COMPILER_ID}")
  endif()
  if(CMAKE_${lang}_COMPILER_VERSION VERSION_LESS "21.0")
    message(FATAL_ERROR
      "Clang >= 21 is required for ${lang}, "
      "found ${CMAKE_${lang}_COMPILER_VERSION}")
  endif()
endforeach()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" OFF)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  else()
    message(WARNING "clang-tidy requested but not found")
  endif()
endif()

option(LIBKFD_BUILD_TESTS "Build unit and device tests" ${PROJECT_IS_TOP_LEVEL})

# AMDGPU architectures to compile for (test kernels, trap handlers, etc.).
set(LIBKFD_GPU_ALL_ARCHS
  gfx908
  gfx90a
  gfx9-generic
  gfx9-4-generic
  gfx10-1-generic
  gfx10-3-generic
  gfx11-generic
  gfx1170
  gfx1171
  gfx1172
  gfx12-generic
  gfx12-5-generic
)

set(LIBKFD_GPU_ARCHS "")
foreach(arch IN LISTS LIBKFD_GPU_ALL_ARCHS)
  try_compile(_arch_ok
    SOURCE_FROM_CONTENT test_${arch}.c "void _start(void) {}"
    COMPILE_DEFINITIONS --target=amdgcn--amdhsa -mcpu=${arch} -nogpulib
    LINK_OPTIONS --target=amdgcn--amdhsa -mcpu=${arch}
  )
  if(_arch_ok)
    list(APPEND LIBKFD_GPU_ARCHS ${arch})
  else()
    message(STATUS "Skipping unsupported GPU arch: ${arch}")
  endif()
  unset(_arch_ok CACHE)
endforeach()
message(STATUS "AMDGPU architectures: ${LIBKFD_GPU_ARCHS}")

set(common_compile_options
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wsign-conversion
  -Wshadow
  -Wformat=2
  -Wno-unused-parameter
  -Wno-missing-designated-field-initializers
  -Wno-gnu-statement-expression
  -fno-rtti
  -fno-exceptions
)
set(library_compile_options
  -Werror=global-constructors
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBDRM_AMDGPU REQUIRED libdrm_amdgpu)

add_library(kfd STATIC)
add_library(libkfd::kfd ALIAS kfd)

target_include_directories(kfd PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(kfd PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${LIBDRM_AMDGPU_INCLUDE_DIRS}
)
target_compile_features(kfd PUBLIC cxx_std_23)
target_link_libraries(kfd PUBLIC ${LIBDRM_AMDGPU_LIBRARIES})
target_compile_options(kfd PRIVATE ${common_compile_options}
                       ${library_compile_options})

# Suppress C++ standard library assertions to avoid their symbols.
target_compile_definitions(kfd PRIVATE _GLIBCXX_NO_ASSERTIONS)
target_compile_options(kfd PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wp,-U_GLIBCXX_ASSERTIONS>"
  "$<$<COMPILE_LANGUAGE:CXX>:-Wp,-U_LIBCPP_ENABLE_ASSERTIONS>"
)

add_library(quake_raster STATIC)
add_library(libkfd::quake_raster ALIAS quake_raster)
target_sources(quake_raster PRIVATE lib/quake_raster.c)
target_include_directories(quake_raster PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(quake_raster PUBLIC c_std_99)
set_target_properties(quake_raster PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)
target_compile_options(quake_raster PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  -Wconversion
  -Wsign-conversion
  -Wshadow
  -Wformat=2
  -Wstrict-prototypes
  -Wmissing-prototypes
)
target_link_libraries(quake_raster PUBLIC kfd)

function(quake_raster_add_gpu_kernel name source_file)
  set(targets "")
  set(entries "")

  foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
    string(REPLACE "-" "_" suffix "${arch}")
    set(target "quake_raster_${name}_${suffix}")

    add_executable(${target} ${source_file})
    set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/quake_raster/kernels"
    )
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
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

    list(APPEND targets ${target})
    string(APPEND entries
      "  {\"$<TARGET_FILE:${target}>\", \"${arch}\"},\n")
  endforeach()

  set(${name}_targets ${targets} PARENT_SCOPE)
  file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/quake_raster_${name}_kernels.inc"
    CONTENT "${entries}"
  )
endfunction()

quake_raster_add_gpu_kernel(clear_indexed
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/quake_raster/kernels/clear_indexed.c)
quake_raster_add_gpu_kernel(resolve_xrgb
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/quake_raster/kernels/resolve_xrgb.c)
target_include_directories(quake_raster PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
if(clear_indexed_targets)
  add_dependencies(quake_raster ${clear_indexed_targets})
endif()
if(resolve_xrgb_targets)
  add_dependencies(quake_raster ${resolve_xrgb_targets})
endif()

add_subdirectory(lib)

if(PROJECT_IS_TOP_LEVEL)
  add_subdirectory(tools)
endif()

if(LIBKFD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

install(TARGETS kfd quake_raster
  EXPORT libkfdTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT libkfdTargets
  FILE libkfdTargets.cmake
  NAMESPACE libkfd::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libkfd
)

configure_package_config_file(
  cmake/libkfdConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libkfdConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libkfd
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/libkfdConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/libkfdConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/libkfdConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libkfd
)

configure_file(
  cmake/libkfd.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libkfd.pc
  @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libkfd.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
