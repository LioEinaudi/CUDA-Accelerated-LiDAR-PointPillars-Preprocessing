#pragma once  
#include"config.hpp"
#include"point.hpp"
struct PillarCoord{
    int x;
    int y;
};

int getGridX(const RangeConfig &range, const PillarConfig &pillar);

int getGridY(const RangeConfig &range, const PillarConfig &pillar);

PillarCoord computePillarCoord(
    const PointXYZI &point,
    const RangeConfig &range,
    const PillarConfig &pillar);