#include"cuda_preprocess.cuh"
#include"cpu_preprocess.hpp"
#include"config.hpp"
#include"point.hpp"

#include<cassert>
#include<iostream>
#include<stdexcept>
#include<string>
#include<vector>

int main() {
    RangeConfig range;
    std::vector<PointXYZI> points = {
        {10.0f, 0.0f, 0.0f, 0.5f},
        {-1.0f, 0.0f, 0.0f, 0.5f},
        {69.12f, 0.0f, 0.0f, 0.5f},
        {10.0f, 40.0f, 0.0f, 0.5f},
        {10.0f, 0.0f, 2.0f, 0.5f},
        {20.0f, 1.0f, -1.0f, 0.7f}
    };

    auto cpu_filtered = cpuRangeFilter(points, range);

    std::vector<PointXYZI> cuda_filtered;
    try {
        cuda_filtered = cudaRangeFilterPrefixSum(points, range);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_range_filter_prefix skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    assert(cpu_filtered.size() == 2);
    assert(cuda_filtered.size() == cpu_filtered.size());

    for (size_t i = 0; i < cpu_filtered.size(); ++i) {
        assert(cuda_filtered[i].x == cpu_filtered[i].x);
        assert(cuda_filtered[i].y == cpu_filtered[i].y);
        assert(cuda_filtered[i].z == cpu_filtered[i].z);
        assert(cuda_filtered[i].intensity == cpu_filtered[i].intensity);
    }

    std::cout << "test_cuda_range_filter_prefix passed" << std::endl;
    return 0;
}
