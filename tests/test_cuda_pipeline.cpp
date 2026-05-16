#include "cuda_pipeline.cuh"
#include "cpu_pillar_scatter.hpp"
#include "cpu_preprocess.hpp"
#include "config.hpp"
#include "point.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main()
{
    RangeConfig range;
    PillarConfig pillar;
    pillar.max_pillars = 20;

    std::vector<PointXYZI> points = {
        {1.00f, -39.60f, 0.0f, 0.1f},
        {1.05f, -39.55f, 0.0f, 0.2f},
        {2.00f, -39.60f, 0.0f, 0.3f},
        {-1.0f, -39.60f, 0.0f, 0.4f},
        {70.0f, -39.60f, 0.0f, 0.5f},
        {1.00f, -39.60f, 2.0f, 0.6f}
    };

    auto cpu_filtered = cpuRangeFilter(points, range);
    auto cpu_scatter = cpuPillarScatter(cpu_filtered, range, pillar);

    CudaPipelineSummary summary;
    try {
        summary = cudaPreprocessPipelineV1(points, range, pillar);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_pipeline skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    assert(summary.filtered_count == static_cast<int>(cpu_filtered.size()));
    assert(summary.num_pillars == static_cast<int>(cpu_scatter.pillars.size()));

    std::cout << "test_cuda_pipeline passed" << std::endl;
    return 0;
}
