#include"cpu_bev.hpp"
#include"cpu_feature.hpp"
#include"cpu_pillar_storage.hpp"
#include"config.hpp"

#include<cassert>
#include<cmath>
#include<iostream>

int main() {
    RangeConfig range;
    range.min_x = 0.0f;
    range.max_x = 0.32f;
    range.min_y = 0.0f;
    range.max_y = 0.32f;

    PillarConfig pillar;
    pillar.voxel_x = 0.16f;
    pillar.voxel_y = 0.16f;
    pillar.max_points_per_pillar = 2;

    assert(getBevIndex(1, 1, 1, 2, 2) == 7);

    PillarPointStorage storage;
    storage.num_pillar = 2;
    storage.pillar_coords = {
        PillarCoord{0, 0},
        PillarCoord{1, 1}
    };
    storage.pillar_point_count = {2, 1};

    PillarFeatureTensor features;
    features.num_pillars = 2;
    features.max_points_per_pillar = 2;
    features.feature_dim = 2;
    features.features.resize(2 * 2 * 2, 0.0f);

    features.features[getFeatureIndex(0, 0, 0, 2, 2)] = 2.0f;
    features.features[getFeatureIndex(0, 0, 1, 2, 2)] = 10.0f;
    features.features[getFeatureIndex(0, 1, 0, 2, 2)] = 4.0f;
    features.features[getFeatureIndex(0, 1, 1, 2, 2)] = 20.0f;

    features.features[getFeatureIndex(1, 0, 0, 2, 2)] = 8.0f;
    features.features[getFeatureIndex(1, 0, 1, 2, 2)] = 30.0f;
    features.features[getFeatureIndex(1, 1, 0, 2, 2)] = 100.0f;
    features.features[getFeatureIndex(1, 1, 1, 2, 2)] = 200.0f;

    BevPseudoImage bev = cpuBuildBevPseudoImage(features, storage, range, pillar);

    assert(bev.channels == 2);
    assert(bev.height == 2);
    assert(bev.width == 2);
    assert(bev.data.size() == 2 * 2 * 2);

    int pillar0_channel0 = getBevIndex(0, 0, 0, bev.height, bev.width);
    int pillar0_channel1 = getBevIndex(1, 0, 0, bev.height, bev.width);
    int pillar1_channel0 = getBevIndex(0, 1, 1, bev.height, bev.width);
    int pillar1_channel1 = getBevIndex(1, 1, 1, bev.height, bev.width);
    int empty_cell = getBevIndex(0, 0, 1, bev.height, bev.width);

    assert(std::fabs(bev.data[pillar0_channel0] - 3.0f) < 1e-5f);
    assert(std::fabs(bev.data[pillar0_channel1] - 15.0f) < 1e-5f);
    assert(std::fabs(bev.data[pillar1_channel0] - 8.0f) < 1e-5f);
    assert(std::fabs(bev.data[pillar1_channel1] - 30.0f) < 1e-5f);
    assert(bev.data[empty_cell] == 0.0f);

    std::cout << "test_cpu_bev passed" << std::endl;
    return 0;
}
