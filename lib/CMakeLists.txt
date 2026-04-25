target_sources(kfd PRIVATE
  device.cpp
  event.cpp
  gpu.cpp
  context.cpp
  loader.cpp
  memory.cpp
  queue.cpp
  signal.cpp
  topology.cpp
  trap_handler.c

  detail/mutex.cpp
  detail/scratch.cpp
  detail/elf.cpp
  detail/mapped_region.cpp
  detail/pool_allocator.cpp
)

# Build per-architecture AMDGCN trap handler executables and pass HAS_<arch>
# defines so trap_handler.c knows which #embed blobs to pull in.
set(trap_targets "")
set(trap_arch_defines "")
foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
    set(target trap_handler_${arch}.bin)
    add_executable(${target} device/trap_handler.S)
    target_compile_options(${target} PRIVATE
      --target=amdgcn--amdhsa
      -mcpu=${arch}
    )
    target_link_options(${target} PRIVATE
      --target=amdgcn--amdhsa
      -mcpu=${arch}
      -Wl,--entry=trap_entry
    )
    list(APPEND trap_targets ${target})

    string(REPLACE "-" "_" arch_id "${arch}")
    string(TOUPPER "${arch_id}" arch_id)
    list(APPEND trap_arch_defines "HAS_${arch_id}")
endforeach()

target_compile_options(kfd PRIVATE --embed-dir=${CMAKE_CURRENT_BINARY_DIR})
target_compile_definitions(kfd PRIVATE ${trap_arch_defines})
add_dependencies(kfd ${trap_targets})
