#include "cuda_pipeline.cuh"
#include "cpu_pillar_scatter.hpp"
#include "cpu_pillar_storage.hpp"
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
    pillar.max_points_per_pillar = 2;

    std::vector<PointXYZI> points = {
        {1.00f, -39.60f, 0.0f, 0.1f},
        {1.05f, -39.55f, 0.0f, 0.2f},
        {1.08f, -39.58f, 0.0f, 0.3f},
        {2.00f, -39.60f, 0.0f, 0.4f},
        {-1.0f, -39.60f, 0.0f, 0.5f},
        {1.00f, -39.60f, 2.0f, 0.6f}
    };

    auto cpu_filtered = cpuRangeFilter(points, range);
    auto cpu_scatter = cpuPillarScatter(cpu_filtered, range, pillar);
    auto cpu_storage = cpuBuildPillarPointStorage(cpu_filtered, cpu_scatter, pillar);

    int cpu_stored_points = 0;
    for (int count : cpu_storage.pillar_point_count) {
        cpu_stored_points += count;
    }

    CudaPipelineSummary summary;
    try {
        summary = cudaPreprocessPipelineV2(points, range, pillar);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_pipelinev2 skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    assert(summary.filtered_count == static_cast<int>(cpu_filtered.size()));
    assert(summary.num_pillars == static_cast<int>(cpu_scatter.pillars.size()));
    assert(summary.stored_points == cpu_stored_points);

    std::cout << "test_cuda_pipelinev2 passed" << std::endl;
    return 0;
}
