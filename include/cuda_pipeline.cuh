#pragma once
#include"point.hpp"
#include"config.hpp"
#include"cpu_bev.hpp"
#include<vector>

struct CudaPipelineSummary{
    int filtered_count=0;
    int num_pillars=0;
    int stored_points = 0;
    // float feature_values = 0.0f;
};

struct CudaPipelineV4DebugResult{
    CudaPipelineSummary summary;
    BevPseudoImage bev;
};

CudaPipelineSummary cudaPreprocessPipelineV1(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar);

CudaPipelineSummary cudaPreprocessPipelineV2(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar);

CudaPipelineSummary cudaPreprocessPipelineV3(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar);

CudaPipelineSummary cudaPreprocessPipelineV4(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar);

CudaPipelineV4DebugResult cudaPreprocessPipelineV4Debug(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar);
