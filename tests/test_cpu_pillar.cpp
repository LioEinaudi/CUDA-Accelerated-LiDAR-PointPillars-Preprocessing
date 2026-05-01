#include "cpu_pillar.hpp"
#include "config.hpp"
#include "point.hpp"

#include <cassert>
#include <iostream>

int main()
{
    RangeConfig range;
    PillarConfig pillar;
    int grid_x = getGridX(range, pillar);
    int grid_y = getGridY(range, pillar);
    assert(grid_x == 432);
    assert(grid_y == 496);

    PointXYZI point = {0.0f, -39.68f, 0.0f, 0.5f};
    PillarCoord coord = computePillarCoord(point, range, pillar);
    assert(coord.x == 0);
    assert(coord.y == 0);

    point = {0.16f, 0.0f, 0.0f, 0.5f};
    coord = computePillarCoord(point, range, pillar);
    assert(coord.x == 1);
    assert(coord.y == 248);

    point = {0.31f, 0.0f, 0.0f, 0.5f};
    coord = computePillarCoord(point, range, pillar);
    assert(coord.x == 1);

    std ::cout << "test_cpu_pillar passed" << std::endl;
    return 0;
}
