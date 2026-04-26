add_executable(quake_raster_diff main.cpp)
target_compile_features(quake_raster_diff PRIVATE cxx_std_23)
target_compile_options(quake_raster_diff PRIVATE ${common_compile_options}
                       ${library_compile_options})
