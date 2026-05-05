#pragma once
#include"point.hpp"
#include"cpu_pillar.hpp"
#include"cpu_pillar_scatter.hpp"
#include<vector>

struct PillarPointStorage{
    std::vector<PointXYZI> pillar_points;
    std::vector<int> pillar_point_count;
    std::vector<PillarCoord> pillar_coords;
    int num_pillar;
};

int getPillarPointIndex(
    int pillar_index,
    int point_offset,
    const PillarConfig &pillar
);

PillarPointStorage cpuBuildPillarPointStorage(
    const std::vector<PointXYZI> &points,
    const PillarScatterResult &scatter,
    const PillarConfig &pillar  
);
