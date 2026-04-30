#pragma once 

#include<vector> 

#include"point.hpp"
#include"config.hpp"

bool isInRangeConfig(const PointXYZI &point, const RangeConfig &range);

std ::vector<PointXYZI> cpuRangeFilter(const std ::vector<PointXYZI> &points, const RangeConfig &range); 
