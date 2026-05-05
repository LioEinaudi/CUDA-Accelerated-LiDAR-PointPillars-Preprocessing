#include"cpu_pillar_storage.hpp"
#include"cpu_pillar_scatter.hpp"
#include"config.hpp"
#include"point.hpp"

#include<cassert>
#include<iostream>
#include<vector>

int main() {
    RangeConfig range;
    PillarConfig pillar;

    assert(getPillarPointIndex(3, 7, pillar) == 307);

    std::vector<PointXYZI> points = {
        {0.01f, -39.67f, -1.0f, 0.5f},
        {0.02f, -39.66f, -1.0f, 0.6f},
        {0.20f, -39.67f, -1.0f, 0.7f}
    };

    PillarScatterResult scatter = cpuPillarScatter(points, range, pillar);
    PillarPointStorage storage = cpuBuildPillarPointStorage(points, scatter, pillar);

    assert(storage.num_pillar == 2);
    assert(storage.pillar_coords.size() == 2);
    assert(storage.pillar_point_count.size() == 2);
    assert(storage.pillar_points.size() == 2 * pillar.max_points_per_pillar);

    assert(storage.pillar_point_count[0] == 2);
    assert(storage.pillar_point_count[1] == 1);

    int point0_index = getPillarPointIndex(0, 0, pillar);
    int point1_index = getPillarPointIndex(0, 1, pillar);
    int point2_index = getPillarPointIndex(1, 0, pillar);

    assert(storage.pillar_points[point0_index].x == points[0].x);
    assert(storage.pillar_points[point0_index].y == points[0].y);
    assert(storage.pillar_points[point0_index].z == points[0].z);
    assert(storage.pillar_points[point0_index].intensity == points[0].intensity);

    assert(storage.pillar_points[point1_index].x == points[1].x);
    assert(storage.pillar_points[point1_index].y == points[1].y);
    assert(storage.pillar_points[point1_index].z == points[1].z);
    assert(storage.pillar_points[point1_index].intensity == points[1].intensity);

    assert(storage.pillar_points[point2_index].x == points[2].x);
    assert(storage.pillar_points[point2_index].y == points[2].y);
    assert(storage.pillar_points[point2_index].z == points[2].z);
    assert(storage.pillar_points[point2_index].intensity == points[2].intensity);

    PillarConfig small_pillar;
    small_pillar.max_points_per_pillar = 2;

    std::vector<PointXYZI> same_pillar_points = {
        {0.01f, -39.67f, -1.0f, 0.1f},
        {0.02f, -39.66f, -1.0f, 0.2f},
        {0.03f, -39.65f, -1.0f, 0.3f}
    };

    PillarScatterResult same_pillar_scatter =
        cpuPillarScatter(same_pillar_points, range, small_pillar);

    PillarPointStorage same_pillar_storage =
        cpuBuildPillarPointStorage(same_pillar_points, same_pillar_scatter, small_pillar);

    assert(same_pillar_scatter.pillars.size() == 1);
    assert(same_pillar_scatter.pillars[0].point_count == 3);

    assert(same_pillar_storage.num_pillar == 1);
    assert(same_pillar_storage.pillar_point_count[0] == 2);
    assert(same_pillar_storage.pillar_points.size() == 2);

    assert(same_pillar_storage.pillar_points[0].intensity == 0.1f);
    assert(same_pillar_storage.pillar_points[1].intensity == 0.2f);

    std::cout << "test_cpu_pillar_storage passed" << std::endl;
    return 0;
}
