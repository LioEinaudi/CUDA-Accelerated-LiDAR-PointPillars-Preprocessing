#include "cuda_preprocess.cuh"
#include "cpu_pillar_scatter.hpp"
#include "config.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int findPillarByCoord(const PillarScatterResult &result, const PillarCoord &coord)
{
    for (size_t i = 0; i < result.pillars.size(); ++i) {
        if (result.pillars[i].coord.x == coord.x &&
            result.pillars[i].coord.y == coord.y) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

int main()
{
    RangeConfig range;
    PillarConfig pillar;

    std::vector<PillarCoord> coords = {
        {1, 2},
        {1, 2},
        {3, 2},
        {1, 2},
        {3, 2}
    };

    PillarScatterResult result;
    try {
        result = cudaPillarScatter(coords, range, pillar);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_pillar_scatter skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    assert(result.pillars.size() == 2);
    assert(result.point_to_pillar.size() == coords.size());

    int pillar_1_2 = findPillarByCoord(result, PillarCoord{1, 2});
    int pillar_3_2 = findPillarByCoord(result, PillarCoord{3, 2});

    assert(pillar_1_2 >= 0);
    assert(pillar_3_2 >= 0);
    assert(result.pillars[pillar_1_2].point_count == 3);
    assert(result.pillars[pillar_3_2].point_count == 2);

    for (size_t i = 0; i < coords.size(); ++i) {
        int pillar_index = result.point_to_pillar[i];
        assert(pillar_index >= 0);
        assert(pillar_index < static_cast<int>(result.pillars.size()));
        assert(result.pillars[pillar_index].coord.x == coords[i].x);
        assert(result.pillars[pillar_index].coord.y == coords[i].y);
    }

    std::cout << "test_cuda_pillar_scatter passed" << std::endl;
    return 0;
}
