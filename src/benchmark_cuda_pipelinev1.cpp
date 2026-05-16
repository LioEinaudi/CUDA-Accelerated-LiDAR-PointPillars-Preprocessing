#include "cuda_pipeline.cuh"
#include "cuda_preprocess.cuh"
#include "cpu_pillar_scatter.hpp"
#include "cpu_preprocess.hpp"
#include "kitti_reader.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double elapsedMs(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./benchmark_cuda_pipeline path/to/000000.bin\n";
        return 1;
    }

    RangeConfig range;
    PillarConfig pillar;
    pillar.max_pillars = 20000;

    std::string bin_path = argv[1];
    auto points = readKittiBin(bin_path);

    auto cpu_filtered = cpuRangeFilter(points, range);
    auto cpu_scatter = cpuPillarScatter(cpu_filtered, range, pillar);

    std::cout << "Loaded points: " << points.size() << std::endl;
    std::cout << std::endl;

    std::cout << "CPU sanity:" << std::endl;
    std::cout << "filtered count: " << cpu_filtered.size() << std::endl;
    std::cout << "pillars:        " << cpu_scatter.pillars.size() << std::endl;
    std::cout << std::endl;

    try {
        std::cout << "CUDA warmup..." << std::flush;
        auto warmup_filtered = cudaRangeFilterPrefixSum(points, range);
        auto warmup_coords = cudaComputePillarCoords(warmup_filtered, range, pillar);
        auto warmup_scatter = cudaPillarScatter(warmup_coords, range, pillar);
        (void)warmup_scatter;
        auto warmup_summary = cudaPreprocessPipelineV1(points, range, pillar);
        (void)warmup_summary;
        std::cout << " done" << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto modular_filtered = cudaRangeFilterPrefixSum(points, range);
        auto t1 = std::chrono::high_resolution_clock::now();
        double modular_range_ms = elapsedMs(t0, t1);

        t0 = std::chrono::high_resolution_clock::now();
        auto modular_coords = cudaComputePillarCoords(modular_filtered, range, pillar);
        t1 = std::chrono::high_resolution_clock::now();
        double modular_coord_ms = elapsedMs(t0, t1);

        t0 = std::chrono::high_resolution_clock::now();
        auto modular_scatter = cudaPillarScatter(modular_coords, range, pillar);
        t1 = std::chrono::high_resolution_clock::now();
        double modular_scatter_ms = elapsedMs(t0, t1);

        double modular_total_ms = modular_range_ms + modular_coord_ms + modular_scatter_ms;

        t0 = std::chrono::high_resolution_clock::now();
        auto pipeline_summary = cudaPreprocessPipelineV1(points, range, pillar);
        t1 = std::chrono::high_resolution_clock::now();
        double pipeline_total_ms = elapsedMs(t0, t1);

        std::cout << "CUDA modular V1:" << std::endl;
        std::cout << "range filter:   " << modular_range_ms << " ms" << std::endl;
        std::cout << "coord:          " << modular_coord_ms << " ms" << std::endl;
        std::cout << "scatter:        " << modular_scatter_ms << " ms" << std::endl;
        std::cout << "total:          " << modular_total_ms << " ms" << std::endl;
        std::cout << "filtered count: " << modular_filtered.size() << std::endl;
        std::cout << "pillars:        " << modular_scatter.pillars.size() << std::endl;
        std::cout << std::endl;

        std::cout << "CUDA pipeline V1:" << std::endl;
        std::cout << "total:          " << pipeline_total_ms << " ms" << std::endl;
        std::cout << "filtered count: " << pipeline_summary.filtered_count << std::endl;
        std::cout << "pillars:        " << pipeline_summary.num_pillars << std::endl;
        std::cout << std::endl;

        if (pipeline_total_ms > 0.0) {
            std::cout << "Speedup over modular V1: "
                      << modular_total_ms / pipeline_total_ms << "x" << std::endl;
        }

    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "benchmark_cuda_pipeline skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    return 0;
}
