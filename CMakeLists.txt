cmake_minimum_required(VERSION 3.16 ) 
project(pointpillars_preprocess LANGUAGES CXX CUDA) 

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CXX_STANDARD 17 ) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pointpillars_preprocess 
    src/main.cpp
    src/kitti_reader.cpp
    src/cpu_range_filter.cpp
    )

target_include_directories( 
    pointpillars_preprocess PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cpu_range_filter
    tests/test_cpu_range_filter.cpp
    src/cpu_range_filter.cpp
    )

target_include_directories( 
    test_cpu_range_filter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cpu_pillar 
    tests/test_cpu_pillar.cpp
    src/cpu_pillar.cpp
    )

target_include_directories( 
    test_cpu_pillar PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cpu_pillar_scatter
    tests/test_cpu_pillar_scatter.cpp
    src/cpu_pillar_scatter.cpp 
    src/cpu_pillar.cpp
    )

target_include_directories(
    test_cpu_pillar_scatter PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cpu_pillar_storage
    tests/test_cpu_pillar_storage.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    )

target_include_directories(
    test_cpu_pillar_storage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cpu_feature
    tests/test_cpu_feature.cpp
    src/cpu_feature.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    )

target_include_directories(
    test_cpu_feature PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cpu_bev
    tests/test_cpu_bev.cpp
    src/cpu_bev.cpp
    src/cpu_feature.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar.cpp
    )

target_include_directories(
    test_cpu_bev PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cuda_range_filter
    tests/test_cuda_range_filter.cpp
    src/cpu_range_filter.cpp    
    src/cuda_range_filter.cu
)

target_include_directories(
    test_cuda_range_filter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

add_executable(test_cuda_range_filter_prefix
    tests/test_cuda_range_filter_prefix.cpp
    src/cpu_range_filter.cpp
    src/cuda_range_filter.cu
)

target_include_directories(
    test_cuda_range_filter_prefix PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

add_executable(test_cuda_pillar_coord
    tests/test_cuda_pillar_coord.cpp
    src/cpu_pillar.cpp
    src/cuda_pillar_coord.cu
)

target_include_directories(
    test_cuda_pillar_coord PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cuda_pillar_scatter
    tests/test_cuda_pillar_scatter.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cuda_pillar_scatter.cu
)

target_include_directories(
    test_cuda_pillar_scatter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cuda_pillar_storage
    tests/test_cuda_pillar_storage.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar.cpp
    src/cuda_pillar_storage.cu
)

target_include_directories(
    test_cuda_pillar_storage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cuda_feature
    tests/test_cuda_feature.cpp
    src/cpu_feature.cpp
    src/cpu_pillar_storage.cpp
    src/cuda_feature.cu
)

target_include_directories(
    test_cuda_feature PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cuda_bev
    tests/test_cuda_bev.cpp
    src/cpu_bev.cpp
    src/cpu_feature.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar.cpp
    src/cuda_bev.cu
)

target_include_directories(
    test_cuda_bev PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(benchmark
    src/benchmark.cpp
    src/kitti_reader.cpp
    src/cpu_bev.cpp
    src/cpu_feature.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar.cpp
    src/cpu_range_filter.cpp
    src/cpu_pillar_scatter.cpp
    src/cuda_bev.cu
    src/cuda_feature.cu
    src/cuda_pillar_coord.cu
    src/cuda_pillar_scatter.cu
    src/cuda_pillar_storage.cu
    src/cuda_range_filter.cu
)

target_include_directories(
    benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(benchmark_cuda_timing
    src/benchmark_cuda_timing.cu
    src/kitti_reader.cpp
    src/cpu_bev.cpp
    src/cpu_feature.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar.cpp
    src/cpu_range_filter.cpp
    src/cpu_pillar_scatter.cpp
)

target_include_directories(
    benchmark_cuda_timing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(test_cuda_pipelinev1
    tests/test_cuda_pipeline.cpp
    src/cuda_pipelinev1.cu
    src/cpu_pillar.cpp
    src/cpu_range_filter.cpp
    src/cpu_pillar_scatter.cpp
)

target_include_directories(
    test_cuda_pipelinev1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cuda_pipelinev2
    tests/test_cuda_pipelinev2.cpp
    src/cuda_pipelinev2.cu
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cpu_range_filter.cpp
)

target_include_directories(
    test_cuda_pipelinev2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cuda_pipelinev3
    tests/test_cuda_pipelinev3.cpp
    src/cuda_pipelinev3.cu
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cpu_range_filter.cpp
)

target_include_directories(
    test_cuda_pipelinev3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_cuda_pipelinev4
    tests/test_cuda_pipelinev4.cpp
    src/cuda_pipelinev4.cu
    src/cpu_bev.cpp
    src/cpu_feature.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cpu_range_filter.cpp
)

target_include_directories(
    test_cuda_pipelinev4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(benchmark_cuda_pipelinev1
    src/benchmark_cuda_pipelinev1.cpp
    src/kitti_reader.cpp
    src/cpu_range_filter.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cuda_range_filter.cu
    src/cuda_pillar_coord.cu
    src/cuda_pillar_scatter.cu
    src/cuda_pipelinev1.cu
)

target_include_directories(
    benchmark_cuda_pipelinev1 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(benchmark_cuda_pipelinev2
    src/benchmark_cuda_pipelinev2.cpp
    src/kitti_reader.cpp
    src/cpu_range_filter.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cuda_range_filter.cu
    src/cuda_pillar_coord.cu
    src/cuda_pillar_scatter.cu
    src/cuda_pillar_storage.cu
    src/cuda_pipelinev2.cu
)

target_include_directories(
    benchmark_cuda_pipelinev2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(benchmark_cuda_pipelinev3
    src/benchmark_cuda_pipelinev3.cpp
    src/kitti_reader.cpp
    src/cpu_range_filter.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cuda_range_filter.cu
    src/cuda_pillar_coord.cu
    src/cuda_pillar_scatter.cu
    src/cuda_pillar_storage.cu
    src/cuda_feature.cu
    src/cuda_pipelinev3.cu
)

target_include_directories(
    benchmark_cuda_pipelinev3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(benchmark_cuda_pipelinev4
    src/benchmark_cuda_pipelinev4.cpp
    src/kitti_reader.cpp
    src/cpu_range_filter.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cuda_range_filter.cu
    src/cuda_pillar_coord.cu
    src/cuda_pillar_scatter.cu
    src/cuda_pillar_storage.cu
    src/cuda_feature.cu
    src/cuda_bev.cu
    src/cuda_pipelinev4.cu
)

target_include_directories(
    benchmark_cuda_pipelinev4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(benchmark_multi_frame
    src/benchmark_multi_frame.cpp
    src/kitti_reader.cpp
    src/cpu_bev.cpp
    src/cpu_feature.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cpu_range_filter.cpp
    src/cuda_range_filter.cu
    src/cuda_pillar_coord.cu
    src/cuda_pillar_scatter.cu
    src/cuda_pillar_storage.cu
    src/cuda_feature.cu
    src/cuda_bev.cu
    src/cuda_pipelinev4.cu
)

target_include_directories(
    benchmark_multi_frame PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(profile_cuda_modular_v4
    src/profile_cuda_modular_v4.cpp
    src/kitti_reader.cpp
    src/cpu_pillar.cpp
    src/cuda_range_filter.cu
    src/cuda_pillar_coord.cu
    src/cuda_pillar_scatter.cu
    src/cuda_pillar_storage.cu
    src/cuda_feature.cu
    src/cuda_bev.cu
)

target_include_directories(
    profile_cuda_modular_v4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(profile_cuda_pipeline_v4
    src/profile_cuda_pipeline_v4.cpp
    src/kitti_reader.cpp
    src/cpu_pillar.cpp
    src/cuda_pipelinev4.cu
)

target_include_directories(
    profile_cuda_pipeline_v4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(validate_cuda_pipelinev4_kitti
    src/validate_cuda_pipelinev4_kitti.cpp
    src/kitti_reader.cpp
    src/cpu_bev.cpp
    src/cpu_feature.cpp
    src/cpu_pillar_storage.cpp
    src/cpu_pillar_scatter.cpp
    src/cpu_pillar.cpp
    src/cpu_range_filter.cpp
    src/cuda_pipelinev4.cu
)

target_include_directories(
    validate_cuda_pipelinev4_kitti PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
