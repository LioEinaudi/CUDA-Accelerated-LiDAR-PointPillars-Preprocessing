cmake_minimum_required(VERSION 3.16 ) 
project(pointpillars_preprocess LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 17 ) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pointpillars_preprocess 
    src/main.cpp
    src/kitti_reader.cpp
    src/cpu_range_filter.cpp)

target_include_directories( 
    pointpillars_preprocess PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)