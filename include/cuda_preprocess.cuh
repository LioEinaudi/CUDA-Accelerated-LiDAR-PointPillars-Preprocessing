#pragma once 
#include<vector>
#include"point.hpp"
#include"config.hpp"

std::vector<PointXYZI> cudaRangeFilterAtomic(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range); 
