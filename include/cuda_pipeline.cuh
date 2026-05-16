#pragma once
#include"point.hpp"
#include"config.hpp"
#include<vector>
struct CudaPipelineSummary{
    int filtered_count=0;
    int num_pillars=0;
    int stored_points = 0;
};

CudaPipelineSummary cudaPreprocessPipelineV1(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar);

CudaPipelineSummary cudaPreprocessPipelineV2(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar);
