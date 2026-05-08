#include "cuda_preprocess.cuh"
#include "cpu_pillar.hpp"
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

    std::vector<PointXYZI> points = {
        {0.0f, -39.68f, 0.0f, 0.5f},
        {0.20f, -39.68f, 0.0f, 0.5f},
        {0.0f, -39.50f, 0.0f, 0.5f},
        {1.70f, -38.00f, 0.0f, 0.5f},
        {0.31f, 0.0f, 0.0f, 0.5f},
        {10.0f, 1.0f, -1.0f, 0.7f}
    };

    std::vector<PillarCoord> cpu_coords;
    for (const auto &point : points) {
        cpu_coords.push_back(computePillarCoord(point, range, pillar));
    }

    std::vector<PillarCoord> cuda_coords;
    try {
        cuda_coords = cudaComputePillarCoords(points, range, pillar);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_pillar_coord skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    assert(cuda_coords.size() == cpu_coords.size());

    for (size_t i = 0; i < cpu_coords.size(); ++i) {
        assert(cuda_coords[i].x == cpu_coords[i].x);
        assert(cuda_coords[i].y == cpu_coords[i].y);
    }

    assert(cuda_coords[0].x == 0);
    assert(cuda_coords[0].y == 0);
    assert(cuda_coords[1].x == 1);
    assert(cuda_coords[1].y == 0);
    assert(cuda_coords[2].x == 0);
    assert(cuda_coords[2].y == 1);
    assert(cuda_coords[3].x == 10);
    assert(cuda_coords[3].y == 10);

    std::cout << "test_cuda_pillar_coord passed" << std::endl;
    return 0;
}
