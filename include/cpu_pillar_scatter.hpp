#pragma once 

#include"cpu_pillar.hpp"
#include"point.hpp" 
#include"config.hpp"
#include<vector>

struct PillarInfo{
    PillarCoord coord;
    int point_count;
};

struct PillarScatterResult{
    std::vector<PillarInfo> pillars;
    std::vector<int> point_to_pillar;
};

int makePillarKey(const PillarCoord &coord, int grid_x);

PillarScatterResult cpuPillarScatter(const std ::vector<PointXYZI> &points, const RangeConfig &range, const PillarConfig &pillar);