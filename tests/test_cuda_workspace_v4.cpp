#include "cuda_pipeline.cuh"
#include "cuda_workspace.cuh"
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

    try {
        auto expected = cudaPreprocessPipelineV4(points, range, pillar);
        auto workspace = createCudaPreprocessWorkspace(static_cast<int>(points.size()), range, pillar);
        auto actual = cudaPreprocessPipelineV4Workspace(points, range, pillar, workspace);
        destroyCudaPreprocessWorkspace(workspace);

        assert(actual.filtered_count == expected.filtered_count);
        assert(actual.num_pillars == expected.num_pillars);
        assert(actual.stored_points == expected.stored_points);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_workspace_v4 skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    std::cout << "test_cuda_workspace_v4 passed" << std::endl;
    return 0;
}
