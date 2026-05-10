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
