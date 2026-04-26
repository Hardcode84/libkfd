add_executable(quake_raster_bench main.cpp)
target_compile_features(quake_raster_bench PRIVATE cxx_std_23)
target_compile_options(quake_raster_bench PRIVATE ${common_compile_options}
                       ${library_compile_options})
target_link_libraries(quake_raster_bench PRIVATE quake_raster)
