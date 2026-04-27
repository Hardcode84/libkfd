add_subdirectory(input)

get_property(device_kernels GLOBAL PROPERTY LIBKFD_GPU_KERNEL_TARGETS)

# Helper: create a device test and link it against the generated kernel binaries.
function(add_device_kernel_test name source)
  kfd_add_device_test(${name} ${source})
  target_include_directories(${name} PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/input")
  foreach(target IN LISTS device_kernels)
    add_dependencies(${name} ${target})
  endforeach()
endfunction()

add_device_kernel_test(loader_test loader_test.cpp)
add_device_kernel_test(dispatch_test dispatch_test.cpp)
add_device_kernel_test(gpu_memory_test memory_test.cpp)
add_device_kernel_test(launch_stress_test launch_stress_test.cpp)
add_device_kernel_test(scratch_stress_test scratch_stress_test.cpp)
add_device_kernel_test(signal_stress_test signal_stress_test.cpp)
add_device_kernel_test(sync_test sync_test.cpp)
add_device_kernel_test(preempt_test preempt_test.cpp)
add_device_kernel_test(preload_args_test preload_args_test.cpp)
add_device_kernel_test(streaming_raster_demo_test streaming_raster_demo_test.cpp)
