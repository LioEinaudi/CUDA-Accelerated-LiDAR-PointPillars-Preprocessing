#include "cuda_preprocess.cuh"
#include "cpu_pillar_storage.hpp"
#include "config.hpp"
#include "point.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

bool samePoint(const PointXYZI &a, const PointXYZI &b)
{
    return a.x == b.x &&
           a.y == b.y &&
           a.z == b.z &&
           a.intensity == b.intensity;
}

bool containsPoint(
    const std::vector<PointXYZI> &points,
    const PointXYZI &target)
{
    for (const auto &point : points) {
        if (samePoint(point, target)) {
            return true;
        }
    }
    return false;
}

int main()
{
    PillarConfig pillar;

    std::vector<PointXYZI> points = {
        {0.01f, -39.67f, -1.0f, 0.5f},
        {0.02f, -39.66f, -1.0f, 0.6f},
        {0.20f, -39.67f, -1.0f, 0.7f}
    };

    PillarScatterResult scatter;
    scatter.pillars = {
        {PillarCoord{0, 0}, 2},
        {PillarCoord{1, 0}, 1}
    };
    scatter.point_to_pillar = {0, 0, 1};

    PillarPointStorage storage;
    try {
        storage = cudaBuildPillarPointStorage(points, scatter, pillar);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_pillar_storage skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    assert(storage.num_pillar == 2);
    assert(storage.pillar_coords.size() == 2);
    assert(storage.pillar_point_count.size() == 2);
    assert(storage.pillar_points.size() == 2 * pillar.max_points_per_pillar);

    assert(storage.pillar_coords[0].x == 0);
    assert(storage.pillar_coords[0].y == 0);
    assert(storage.pillar_coords[1].x == 1);
    assert(storage.pillar_coords[1].y == 0);

    assert(storage.pillar_point_count[0] == 2);
    assert(storage.pillar_point_count[1] == 1);

    std::vector<PointXYZI> pillar0_points = {
        storage.pillar_points[getPillarPointIndex(0, 0, pillar)],
        storage.pillar_points[getPillarPointIndex(0, 1, pillar)]
    };

    assert(containsPoint(pillar0_points, points[0]));
    assert(containsPoint(pillar0_points, points[1]));

    int pillar1_point0 = getPillarPointIndex(1, 0, pillar);
    assert(samePoint(storage.pillar_points[pillar1_point0], points[2]));

    PillarConfig small_pillar;
    small_pillar.max_points_per_pillar = 2;

    std::vector<PointXYZI> same_pillar_points = {
        {0.01f, -39.67f, -1.0f, 0.1f},
        {0.02f, -39.66f, -1.0f, 0.2f},
        {0.03f, -39.65f, -1.0f, 0.3f}
    };

    PillarScatterResult same_pillar_scatter;
    same_pillar_scatter.pillars = {
        {PillarCoord{0, 0}, 3}
    };
    same_pillar_scatter.point_to_pillar = {0, 0, 0};

    PillarPointStorage same_pillar_storage =
        cudaBuildPillarPointStorage(same_pillar_points, same_pillar_scatter, small_pillar);

    assert(same_pillar_storage.num_pillar == 1);
    assert(same_pillar_storage.pillar_point_count[0] == 2);
    assert(same_pillar_storage.pillar_points.size() == 2);

    std::vector<PointXYZI> saved_points = {
        same_pillar_storage.pillar_points[0],
        same_pillar_storage.pillar_points[1]
    };

    for (const auto &saved_point : saved_points) {
        assert(containsPoint(same_pillar_points, saved_point));
    }

    assert(!samePoint(saved_points[0], saved_points[1]));

    std::cout << "test_cuda_pillar_storage passed" << std::endl;
    return 0;
}
