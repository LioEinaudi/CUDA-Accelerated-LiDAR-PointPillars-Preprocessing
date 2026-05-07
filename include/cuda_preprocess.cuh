#pragma once 
#include<vector>
#include"point.hpp"
#include"config.hpp"

std::vector<PointXYZI> cudaRangeFilterAtomic(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range);

std::vector<PointXYZI> cudaRangeFilterPrefixSum(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range);