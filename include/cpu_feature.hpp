#pragma once 
#include"cpu_pillar_storage.hpp"
#include"config.hpp"

#include<vector>

constexpr int kPillarFeatureDim = 9;

struct PillarFeatureTensor{
    std::vector<float> features;
    int num_pillars;
    int max_points_per_pillar;
    int feature_dim;
};

int getFeatureIndex(
    int pillar_index,
    int point_offset,
    int feature_index,
    int max_points_per_pillar,
    int feature_dim);

PillarFeatureTensor cpuGeneratePillarFeatures(
    const PillarPointStorage &storage,
    const RangeConfig &range,
    const PillarConfig &pillar);