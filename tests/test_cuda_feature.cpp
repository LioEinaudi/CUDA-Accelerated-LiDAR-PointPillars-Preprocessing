#include "cuda_preprocess.cuh"
#include "cpu_feature.hpp"
#include "config.hpp"
#include "point.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    RangeConfig range;
    PillarConfig pillar;
    pillar.max_points_per_pillar = 3;

    PillarPointStorage storage;
    storage.num_pillar = 1;
    storage.pillar_coords.push_back(PillarCoord{0, 0});
    storage.pillar_point_count.push_back(2);
    storage.pillar_points.resize(storage.num_pillar * pillar.max_points_per_pillar);

    storage.pillar_points[0] = PointXYZI{0.01f, -39.67f, -1.0f, 0.5f};
    storage.pillar_points[1] = PointXYZI{0.03f, -39.65f, 1.0f, 0.7f};

    PillarFeatureTensor cpu_tensor = cpuGeneratePillarFeatures(storage, range, pillar);

    PillarFeatureTensor cuda_tensor;
    try {
        cuda_tensor = cudaGeneratePillarFeatures(storage, range, pillar);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "test_cuda_feature skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    assert(cuda_tensor.num_pillars == cpu_tensor.num_pillars);
    assert(cuda_tensor.max_points_per_pillar == cpu_tensor.max_points_per_pillar);
    assert(cuda_tensor.feature_dim == cpu_tensor.feature_dim);
    assert(cuda_tensor.features.size() == cpu_tensor.features.size());

    for (size_t i = 0; i < cpu_tensor.features.size(); ++i) {
        assert(std::fabs(cuda_tensor.features[i] - cpu_tensor.features[i]) < 1e-5f);
    }

    int point2_feature0 = getFeatureIndex(
        0,
        2,
        0,
        pillar.max_points_per_pillar,
        kPillarFeatureDim);
    int point2_feature8 = getFeatureIndex(
        0,
        2,
        8,
        pillar.max_points_per_pillar,
        kPillarFeatureDim);

    assert(cuda_tensor.features[point2_feature0] == 0.0f);
    assert(cuda_tensor.features[point2_feature8] == 0.0f);

    std::cout << "test_cuda_feature passed" << std::endl;
    return 0;
}
