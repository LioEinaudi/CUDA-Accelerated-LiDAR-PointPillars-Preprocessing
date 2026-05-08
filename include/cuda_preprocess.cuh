#pragma once 
#include<vector>
#include"point.hpp"
#include"config.hpp"
#include"cpu_pillar.hpp"

std::vector<PointXYZI> cudaRangeFilterAtomic(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range);

std::vector<PointXYZI> cudaRangeFilterPrefixSum(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range);

std::vector<PillarCoord> cudaComputePillarCoords(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar);
