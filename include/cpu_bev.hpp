#pragma once 
#include"cpu_feature.hpp"
#include"config.hpp" 
#include"cpu_pillar_storage.hpp"
#include<vector>
struct BevPseudoImage
{
    std::vector<float> data;
    int channels;
    int height;
    int width;
};

int getBevIndex(
    int channel,
    int y,
    int x,
    int height,
    int width);

BevPseudoImage cpuBuildBevPseudoImage(
    const PillarFeatureTensor &features,
    const PillarPointStorage &storage,
    const RangeConfig &range,
    const PillarConfig &pillar);
