#include "cuda_preprocess.cuh"
#include "cpu_pillar_storage.hpp"
#include "kitti_reader.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int sumStoredPoints(const PillarPointStorage &storage)
{
    int stored_points = 0;
    for (int count : storage.pillar_point_count) {
        stored_points += count;
    }
    return stored_points;
}

int parseRepeats(int argc, char **argv)
{
    if (argc < 3) {
        return 20;
    }
    int repeats = std::atoi(argv[2]);
    return repeats > 0 ? repeats : 20;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./profile_cuda_modular_v4 path/to/000000.bin [repeats]\n";
        return 1;
    }

    RangeConfig range;
    PillarConfig pillar;
    pillar.max_pillars = 20000;

    int repeats = parseRepeats(argc, argv);
    auto points = readKittiBin(argv[1]);

    try {
        std::cout << "Loaded points: " << points.size() << std::endl;
        std::cout << "Repeats: " << repeats << std::endl;

        auto warmup_filtered = cudaRangeFilterPrefixSum(points, range);
        auto warmup_coords = cudaComputePillarCoords(warmup_filtered, range, pillar);
        auto warmup_scatter = cudaPillarScatter(warmup_coords, range, pillar);
        auto warmup_storage = cudaBuildPillarPointStorage(warmup_filtered, warmup_scatter, pillar);
        auto warmup_features = cudaGeneratePillarFeatures(warmup_storage, range, pillar);
        auto warmup_bev = cudaBuildBevPseudoImage(warmup_features, warmup_storage, range, pillar);
        (void)warmup_bev;
        std::cout << "CUDA modular V4 warmup done" << std::endl;

        long long checksum = 0;
        for (int i = 0; i < repeats; ++i) {
            auto filtered = cudaRangeFilterPrefixSum(points, range);
            auto coords = cudaComputePillarCoords(filtered, range, pillar);
            auto scatter = cudaPillarScatter(coords, range, pillar);
            auto storage = cudaBuildPillarPointStorage(filtered, scatter, pillar);
            auto features = cudaGeneratePillarFeatures(storage, range, pillar);
            auto bev = cudaBuildBevPseudoImage(features, storage, range, pillar);

            checksum += static_cast<long long>(filtered.size());
            checksum += static_cast<long long>(storage.num_pillar);
            checksum += static_cast<long long>(sumStoredPoints(storage));
            checksum += static_cast<long long>(features.features.size());
            checksum += static_cast<long long>(bev.data.size());
        }

        std::cout << "CUDA modular V4 profile done" << std::endl;
        std::cout << "Checksum: " << checksum << std::endl;
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "profile_cuda_modular_v4 skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    return 0;
}
