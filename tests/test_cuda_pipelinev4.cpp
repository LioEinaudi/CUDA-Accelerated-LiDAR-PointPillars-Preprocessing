#include "cuda_pipeline.cuh"
#include "cpu_bev.hpp"
#include "cpu_feature.hpp"
#include "cpu_pillar_scatter.hpp"
#include "cpu_pillar_storage.hpp"
#include "cpu_preprocess.hpp"
#include "config.hpp"
#include "point.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main()
{
    RangeConfig range;
    range.min_x = 0.0f;
    range.max_x = 0.64f;
    range.min_y = 0.0f;
    range.max_y = 0.64f;
    range.min_z = -1.0f;
    range.max_z = 1.0f;

    PillarConfig pillar;
    pillar.voxel_x = 0.16f;
    pillar.voxel_y = 0.16f;
    pillar.max_pillars = 20;
    pillar.max_points_per_pillar = 2;

    std::vector<PointXYZI> points = {
        {0.01f, 0.01f, 0.0f, 0.1f},
        {0.02f, 0.02f, 0.0f, 0.2f},
        {0.03f, 0.03f, 0.0f, 0.3f},
        {0.20f, 0.20f, 0.0f, 0.4f},
        {-1.0f, 0.01f, 0.0f, 0.5f},
        {0.20f, 0.20f, 2.0f, 0.6f}
    };

    auto cpu_filtered = cpuRangeFilter(points, range);
    auto cpu_scatter = cpuPillarScatter(cpu_filtered, range, pillar);
    auto cpu_storage = cpuBuildPillarPointStorage(cpu_filtered, cpu_scatter, pillar);
    auto cpu_features = cpuGeneratePillarFeatures(cpu_storage, range, pillar);
    auto cpu_bev = cpuBuildBevPseudoImage(cpu_features, cpu_storage, range, pillar);

    int cpu_stored_points = 0;
    for (int count : cpu_storage.pillar_point_count) {
        cpu_stored_points += count;
    }

    CudaPipelineV4DebugResult cuda_result;
    try {
        cuda_result = cudaPreprocessPipelineV4Debug(points, range, pillar);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_pipelinev4 skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    assert(cuda_result.summary.filtered_count == static_cast<int>(cpu_filtered.size()));
    assert(cuda_result.summary.num_pillars == static_cast<int>(cpu_scatter.pillars.size()));
    assert(cuda_result.summary.stored_points == cpu_stored_points);

    assert(cuda_result.bev.channels == cpu_bev.channels);
    assert(cuda_result.bev.height == cpu_bev.height);
    assert(cuda_result.bev.width == cpu_bev.width);
    assert(cuda_result.bev.data.size() == cpu_bev.data.size());

    for (size_t i = 0; i < cpu_bev.data.size(); ++i) {
        assert(std::fabs(cuda_result.bev.data[i] - cpu_bev.data[i]) < 1e-4f);
    }

    std::cout << "test_cuda_pipelinev4 passed" << std::endl;
    return 0;
}
