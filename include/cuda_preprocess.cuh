#pragma once 
#include<vector>
#include"point.hpp"
#include"config.hpp"
#include"cpu_pillar.hpp"
#include"cpu_pillar_scatter.hpp"
#include"cpu_pillar_storage.hpp"
#include"cpu_feature.hpp"

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

PillarScatterResult cudaPillarScatter(
    const std ::vector<PillarCoord> &coords,
    const RangeConfig &range,
    const PillarConfig &pillar);

PillarPointStorage cudaBuildPillarPointStorage(
    const std::vector<PointXYZI>&points,
    const PillarScatterResult & scatter,
    const PillarConfig & pillar
);

PillarFeatureTensor cudaGeneratePillarFeatures(
    const PillarPointStorage &storage,
    const RangeConfig &range,
    const PillarConfig &pillar);
