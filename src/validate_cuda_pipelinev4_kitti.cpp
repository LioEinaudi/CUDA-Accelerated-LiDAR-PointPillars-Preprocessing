#include "cuda_pipeline.cuh"
#include "cpu_bev.hpp"
#include "cpu_feature.hpp"
#include "cpu_pillar_scatter.hpp"
#include "cpu_pillar_storage.hpp"
#include "cpu_preprocess.hpp"
#include "kitti_reader.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./validate_cuda_pipelinev4_kitti path/to/000000.bin\n";
        return 1;
    }

    RangeConfig range;
    PillarConfig pillar;
    pillar.max_pillars = 20000;

    auto points = readKittiBin(argv[1]);

    auto cpu_filtered = cpuRangeFilter(points, range);
    auto cpu_scatter = cpuPillarScatter(cpu_filtered, range, pillar);
    auto cpu_storage = cpuBuildPillarPointStorage(cpu_filtered, cpu_scatter, pillar);
    auto cpu_features = cpuGeneratePillarFeatures(cpu_storage, range, pillar);
    auto cpu_bev = cpuBuildBevPseudoImage(cpu_features, cpu_storage, range, pillar);

    int cpu_stored_points = 0;
    for (int count : cpu_storage.pillar_point_count) {
        cpu_stored_points += count;
    }

    CudaPipelineV4DebugResult cuda_result;
    try {
        cuda_result = cudaPreprocessPipelineV4Debug(points, range, pillar);
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "validate_cuda_pipelinev4_kitti skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    int mismatches = 0;
    float max_abs_diff = 0.0f;
    const float tolerance = 1e-3f;

    if (cuda_result.summary.filtered_count != static_cast<int>(cpu_filtered.size())) {
        ++mismatches;
    }
    if (cuda_result.summary.num_pillars != static_cast<int>(cpu_scatter.pillars.size())) {
        ++mismatches;
    }
    if (cuda_result.summary.stored_points != cpu_stored_points) {
        ++mismatches;
    }
    if (cuda_result.bev.channels != cpu_bev.channels ||
        cuda_result.bev.height != cpu_bev.height ||
        cuda_result.bev.width != cpu_bev.width ||
        cuda_result.bev.data.size() != cpu_bev.data.size()) {
        ++mismatches;
    }

    if (cuda_result.bev.data.size() == cpu_bev.data.size()) {
        for (size_t i = 0; i < cpu_bev.data.size(); ++i) {
            float diff = std::fabs(cuda_result.bev.data[i] - cpu_bev.data[i]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
            if (diff > tolerance) {
                ++mismatches;
            }
        }
    }

    std::cout << "Loaded points: " << points.size() << std::endl;
    std::cout << "CPU filtered count: " << cpu_filtered.size() << std::endl;
    std::cout << "CUDA filtered count: " << cuda_result.summary.filtered_count << std::endl;
    std::cout << "CPU pillars: " << cpu_scatter.pillars.size() << std::endl;
    std::cout << "CUDA pillars: " << cuda_result.summary.num_pillars << std::endl;
    std::cout << "CPU stored points: " << cpu_stored_points << std::endl;
    std::cout << "CUDA stored points: " << cuda_result.summary.stored_points << std::endl;
    std::cout << "BEV values: " << cpu_bev.data.size() << std::endl;
    std::cout << "Max abs diff: " << max_abs_diff << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;

    if (mismatches != 0) {
        std::cerr << "validate_cuda_pipelinev4_kitti failed, mismatches: " << mismatches << std::endl;
        return 1;
    }

    std::cout << "validate_cuda_pipelinev4_kitti passed" << std::endl;
    return 0;
}
