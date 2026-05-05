#include"cpu_feature.hpp"
#include"cpu_pillar_storage.hpp"
#include"config.hpp"
#include"point.hpp"

#include<cassert>
#include<cmath>
#include<iostream>

int main() {
    RangeConfig range;
    PillarConfig pillar;
    pillar.max_points_per_pillar = 3;

    assert(getFeatureIndex(2, 3, 4, 100, 9) == 1831);

    PillarPointStorage storage;
    storage.num_pillar = 1;
    storage.pillar_coords.push_back(PillarCoord{0, 0});
    storage.pillar_point_count.push_back(2);
    storage.pillar_points.resize(storage.num_pillar * pillar.max_points_per_pillar);

    storage.pillar_points[0] = PointXYZI{0.01f, -39.67f, -1.0f, 0.5f};
    storage.pillar_points[1] = PointXYZI{0.03f, -39.65f, 1.0f, 0.7f};

    PillarFeatureTensor tensor = cpuGeneratePillarFeatures(storage, range, pillar);

    assert(tensor.num_pillars == 1);
    assert(tensor.max_points_per_pillar == 3);
    assert(tensor.feature_dim == kPillarFeatureDim);
    assert(tensor.features.size() == 1 * 3 * kPillarFeatureDim);

    int point0_feature0 = getFeatureIndex(0, 0, 0, pillar.max_points_per_pillar, kPillarFeatureDim);
    int point0_feature1 = getFeatureIndex(0, 0, 1, pillar.max_points_per_pillar, kPillarFeatureDim);
    int point0_feature2 = getFeatureIndex(0, 0, 2, pillar.max_points_per_pillar, kPillarFeatureDim);
    int point0_feature3 = getFeatureIndex(0, 0, 3, pillar.max_points_per_pillar, kPillarFeatureDim);
    int point0_feature4 = getFeatureIndex(0, 0, 4, pillar.max_points_per_pillar, kPillarFeatureDim);
    int point0_feature5 = getFeatureIndex(0, 0, 5, pillar.max_points_per_pillar, kPillarFeatureDim);
    int point0_feature6 = getFeatureIndex(0, 0, 6, pillar.max_points_per_pillar, kPillarFeatureDim);
    int point0_feature7 = getFeatureIndex(0, 0, 7, pillar.max_points_per_pillar, kPillarFeatureDim);
    int point0_feature8 = getFeatureIndex(0, 0, 8, pillar.max_points_per_pillar, kPillarFeatureDim);

    assert(std::fabs(tensor.features[point0_feature0] - 0.01f) < 1e-5f);
    assert(std::fabs(tensor.features[point0_feature1] - (-39.67f)) < 1e-5f);
    assert(std::fabs(tensor.features[point0_feature2] - (-1.0f)) < 1e-5f);
    assert(std::fabs(tensor.features[point0_feature3] - 0.5f) < 1e-5f);

    assert(std::fabs(tensor.features[point0_feature4] - (-0.01f)) < 1e-5f);
    assert(std::fabs(tensor.features[point0_feature5] - (-0.01f)) < 1e-5f);
    assert(std::fabs(tensor.features[point0_feature6] - (-1.0f)) < 1e-5f);

    assert(std::fabs(tensor.features[point0_feature7] - (-0.07f)) < 1e-5f);
    assert(std::fabs(tensor.features[point0_feature8] - (-0.07f)) < 1e-5f);

    int point2_feature0 = getFeatureIndex(0, 2, 0, pillar.max_points_per_pillar, kPillarFeatureDim);
    int point2_feature8 = getFeatureIndex(0, 2, 8, pillar.max_points_per_pillar, kPillarFeatureDim);

    assert(tensor.features[point2_feature0] == 0.0f);
    assert(tensor.features[point2_feature8] == 0.0f);

    std::cout << "test_cpu_feature passed" << std::endl;
    return 0;
}
