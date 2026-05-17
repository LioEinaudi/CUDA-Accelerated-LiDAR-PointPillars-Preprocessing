#include "cuda_pipeline.cuh"
#include "cuda_workspace.cuh"
#include "kitti_reader.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

double elapsedMs(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
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
        std::cerr << "Usage: ./benchmark_cuda_workspace_v4 path/to/000000.bin [repeats]\n";
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
        std::cout << std::endl;

        auto workspace = createCudaPreprocessWorkspace(static_cast<int>(points.size()), range, pillar);

        auto normal_warmup = cudaPreprocessPipelineV4(points, range, pillar);
        auto workspace_warmup = cudaPreprocessPipelineV4Workspace(points, range, pillar, workspace);

        if (normal_warmup.filtered_count != workspace_warmup.filtered_count ||
            normal_warmup.num_pillars != workspace_warmup.num_pillars ||
            normal_warmup.stored_points != workspace_warmup.stored_points) {
            throw std::runtime_error("workspace warmup summary mismatch");
        }

        double normal_total_ms = 0.0;
        CudaPipelineSummary normal_summary;
        for (int i = 0; i < repeats; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            normal_summary = cudaPreprocessPipelineV4(points, range, pillar);
            auto t1 = std::chrono::high_resolution_clock::now();
            normal_total_ms += elapsedMs(t0, t1);
        }

        double workspace_total_ms = 0.0;
        CudaPipelineSummary workspace_summary;
        for (int i = 0; i < repeats; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            workspace_summary = cudaPreprocessPipelineV4Workspace(points, range, pillar, workspace);
            auto t1 = std::chrono::high_resolution_clock::now();
            workspace_total_ms += elapsedMs(t0, t1);
        }

        destroyCudaPreprocessWorkspace(workspace);

        if (normal_summary.filtered_count != workspace_summary.filtered_count ||
            normal_summary.num_pillars != workspace_summary.num_pillars ||
            normal_summary.stored_points != workspace_summary.stored_points) {
            throw std::runtime_error("workspace benchmark summary mismatch");
        }

        double normal_avg_ms = normal_total_ms / repeats;
        double workspace_avg_ms = workspace_total_ms / repeats;

        std::cout << "Normal pipeline V4:" << std::endl;
        std::cout << "avg:            " << normal_avg_ms << " ms" << std::endl;
        std::cout << "filtered count: " << normal_summary.filtered_count << std::endl;
        std::cout << "pillars:        " << normal_summary.num_pillars << std::endl;
        std::cout << "stored points:  " << normal_summary.stored_points << std::endl;
        std::cout << std::endl;

        std::cout << "Workspace pipeline V4:" << std::endl;
        std::cout << "avg:            " << workspace_avg_ms << " ms" << std::endl;
        std::cout << "filtered count: " << workspace_summary.filtered_count << std::endl;
        std::cout << "pillars:        " << workspace_summary.num_pillars << std::endl;
        std::cout << "stored points:  " << workspace_summary.stored_points << std::endl;
        std::cout << std::endl;

        if (workspace_avg_ms > 0.0) {
            std::cout << "Speedup over normal pipeline V4: "
                      << normal_avg_ms / workspace_avg_ms << "x" << std::endl;
        }
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "benchmark_cuda_workspace_v4 skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    return 0;
}
