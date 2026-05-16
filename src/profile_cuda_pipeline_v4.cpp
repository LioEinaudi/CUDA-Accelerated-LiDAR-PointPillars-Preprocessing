#include "cuda_pipeline.cuh"
#include "kitti_reader.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

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
        std::cerr << "Usage: ./profile_cuda_pipeline_v4 path/to/000000.bin [repeats]\n";
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

        auto warmup_summary = cudaPreprocessPipelineV4(points, range, pillar);
        (void)warmup_summary;
        std::cout << "CUDA pipeline V4 warmup done" << std::endl;

        long long checksum = 0;
        for (int i = 0; i < repeats; ++i) {
            auto summary = cudaPreprocessPipelineV4(points, range, pillar);
            checksum += static_cast<long long>(summary.filtered_count);
            checksum += static_cast<long long>(summary.num_pillars);
            checksum += static_cast<long long>(summary.stored_points);
        }

        std::cout << "CUDA pipeline V4 profile done" << std::endl;
        std::cout << "Checksum: " << checksum << std::endl;
        std::cout << "Note: feature and BEV tensors stay on GPU; full D2H export is skipped." << std::endl;
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "profile_cuda_pipeline_v4 skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    return 0;
}
