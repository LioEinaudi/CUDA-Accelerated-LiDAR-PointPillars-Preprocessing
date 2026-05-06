#include"cuda_preprocess.cuh"
#include"cpu_preprocess.hpp"
#include"config.hpp"
#include"point.hpp"

#include<cassert>
#include<vector>
#include<iostream>
#include<stdexcept>
#include<string>

int main(){
    RangeConfig range;
    std::vector<PointXYZI> points = {
        {10.0f, 0.0f, 0.0f, 0.5f},
        {-1.0f, 0.0f, 0.0f, 0.5f},
        {69.12f, 0.0f, 0.0f, 0.5f},
        {10.0f, 40.0f, 0.0f, 0.5f},
        {10.0f, 0.0f, 2.0f, 0.5f},
        {20.0f, 1.0f, -1.0f, 0.7f}};

    auto cpu_filtered = cpuRangeFilter(points, range );
    std::vector<PointXYZI> gpu_filtered;
    try {
        gpu_filtered = cudaRangeFilterAtomic(points, range);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_range_filter skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    assert(cpu_filtered.size() == 2);
    assert(gpu_filtered.size() == cpu_filtered.size());

    std::cout << "test_cuda_range_filter passed" << std::endl; 
    return 0;
}
