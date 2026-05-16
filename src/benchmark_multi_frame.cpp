#include "cuda_pipeline.cuh"
#include "cuda_preprocess.cuh"
#include "cpu_bev.hpp"
#include "cpu_feature.hpp"
#include "cpu_pillar_scatter.hpp"
#include "cpu_pillar_storage.hpp"
#include "cpu_preprocess.hpp"
#include "kitti_reader.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Stats {
    double sum = 0.0;
    double min = std::numeric_limits<double>::max();
    double max = 0.0;
    int count = 0;

    void add(double value)
    {
        sum += value;
        min = std::min(min, value);
        max = std::max(max, value);
        ++count;
    }

    double avg() const
    {
        return count == 0 ? 0.0 : sum / count;
    }
};

struct FrameResult {
    std::string name;
    int points = 0;
    int filtered = 0;
    int pillars = 0;
    int stored_points = 0;
    double cpu_ms = 0.0;
    double cuda_modular_ms = 0.0;
    double cuda_pipeline_ms = 0.0;
};

double elapsedMs(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int sumStoredPoints(const PillarPointStorage &storage)
{
    int stored_points = 0;
    for (int count : storage.pillar_point_count) {
        stored_points += count;
    }
    return stored_points;
}

std::vector<std::filesystem::path> collectBinFiles(const std::filesystem::path &input_path)
{
    std::vector<std::filesystem::path> files;

    if (std::filesystem::is_regular_file(input_path)) {
        if (input_path.extension() == ".bin") {
            files.push_back(input_path);
        }
        return files;
    }

    for (const auto &entry : std::filesystem::directory_iterator(input_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (entry.path().extension() == ".bin") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

void printStatsRow(const std::string &name, const Stats &stats)
{
    std::cout << std::left << std::setw(20) << name
              << std::right << std::setw(12) << stats.avg()
              << std::setw(12) << stats.min
              << std::setw(12) << stats.max << std::endl;
}

void runCudaWarmup(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar)
{
    auto filtered = cudaRangeFilterPrefixSum(points, range);
    auto coords = cudaComputePillarCoords(filtered, range, pillar);
    auto scatter = cudaPillarScatter(coords, range, pillar);
    auto storage = cudaBuildPillarPointStorage(filtered, scatter, pillar);
    auto features = cudaGeneratePillarFeatures(storage, range, pillar);
    auto bev = cudaBuildBevPseudoImage(features, storage, range, pillar);
    (void)bev;

    auto summary = cudaPreprocessPipelineV4(points, range, pillar);
    (void)summary;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./benchmark_multi_frame path/to/bin_or_directory\n";
        return 1;
    }

    std::filesystem::path input_path = argv[1];
    auto bin_files = collectBinFiles(input_path);
    if (bin_files.empty()) {
        std::cerr << "No .bin files found: " << input_path << std::endl;
        return 1;
    }

    RangeConfig range;
    PillarConfig pillar;
    pillar.max_pillars = 20000;

    std::cout << "Frames: " << bin_files.size() << std::endl;
    std::cout << "Input:  " << input_path << std::endl;
    std::cout << std::endl;

    std::vector<FrameResult> results;
    results.reserve(bin_files.size());

    bool warmed_up = false;

    try {
        for (size_t frame_index = 0; frame_index < bin_files.size(); ++frame_index) {
            const auto &path = bin_files[frame_index];
            auto points = readKittiBin(path.string());

            if (!warmed_up) {
                std::cout << "CUDA warmup..." << std::flush;
                runCudaWarmup(points, range, pillar);
                warmed_up = true;
                std::cout << " done" << std::endl;
            }

            FrameResult result;
            result.name = path.filename().string();
            result.points = static_cast<int>(points.size());

            auto t0 = std::chrono::high_resolution_clock::now();
            auto cpu_filtered = cpuRangeFilter(points, range);
            auto cpu_scatter = cpuPillarScatter(cpu_filtered, range, pillar);
            auto cpu_storage = cpuBuildPillarPointStorage(cpu_filtered, cpu_scatter, pillar);
            auto cpu_features = cpuGeneratePillarFeatures(cpu_storage, range, pillar);
            auto cpu_bev = cpuBuildBevPseudoImage(cpu_features, cpu_storage, range, pillar);
            auto t1 = std::chrono::high_resolution_clock::now();
            result.cpu_ms = elapsedMs(t0, t1);
            result.filtered = static_cast<int>(cpu_filtered.size());
            result.pillars = static_cast<int>(cpu_scatter.pillars.size());
            result.stored_points = sumStoredPoints(cpu_storage);

            t0 = std::chrono::high_resolution_clock::now();
            auto modular_filtered = cudaRangeFilterPrefixSum(points, range);
            auto modular_coords = cudaComputePillarCoords(modular_filtered, range, pillar);
            auto modular_scatter = cudaPillarScatter(modular_coords, range, pillar);
            auto modular_storage = cudaBuildPillarPointStorage(modular_filtered, modular_scatter, pillar);
            auto modular_features = cudaGeneratePillarFeatures(modular_storage, range, pillar);
            auto modular_bev = cudaBuildBevPseudoImage(modular_features, modular_storage, range, pillar);
            t1 = std::chrono::high_resolution_clock::now();
            result.cuda_modular_ms = elapsedMs(t0, t1);

            int modular_stored_points = sumStoredPoints(modular_storage);
            if (static_cast<int>(modular_filtered.size()) != result.filtered ||
                modular_storage.num_pillar != result.pillars ||
                modular_stored_points != result.stored_points ||
                modular_bev.data.size() != cpu_bev.data.size()) {
                throw std::runtime_error("CUDA modular V4 summary mismatch on " + result.name);
            }

            t0 = std::chrono::high_resolution_clock::now();
            auto pipeline_summary = cudaPreprocessPipelineV4(points, range, pillar);
            t1 = std::chrono::high_resolution_clock::now();
            result.cuda_pipeline_ms = elapsedMs(t0, t1);

            if (pipeline_summary.filtered_count != result.filtered ||
                pipeline_summary.num_pillars != result.pillars ||
                pipeline_summary.stored_points != result.stored_points) {
                throw std::runtime_error("CUDA pipeline V4 summary mismatch on " + result.name);
            }

            results.push_back(result);

            std::cout << "[" << (frame_index + 1) << "/" << bin_files.size() << "] "
                      << result.name
                      << " points=" << result.points
                      << " filtered=" << result.filtered
                      << " pillars=" << result.pillars
                      << " CPU=" << result.cpu_ms << " ms"
                      << " CUDA modular=" << result.cuda_modular_ms << " ms"
                      << " CUDA pipeline=" << result.cuda_pipeline_ms << " ms";
            if (result.cuda_pipeline_ms > 0.0) {
                std::cout << " speedup=" << result.cuda_modular_ms / result.cuda_pipeline_ms << "x";
            }
            std::cout << std::endl;
        }
    } catch (const std::runtime_error &error) {
        std::string message = error.what();
        if (message.find("no CUDA-capable device") != std::string::npos) {
            std::cout << "benchmark_multi_frame skipped: no CUDA-capable device" << std::endl;
            return 0;
        }
        throw;
    }

    Stats point_stats;
    Stats filtered_stats;
    Stats pillar_stats;
    Stats stored_stats;
    Stats cpu_stats;
    Stats modular_stats;
    Stats pipeline_stats;
    Stats speedup_stats;

    for (const auto &result : results) {
        point_stats.add(result.points);
        filtered_stats.add(result.filtered);
        pillar_stats.add(result.pillars);
        stored_stats.add(result.stored_points);
        cpu_stats.add(result.cpu_ms);
        modular_stats.add(result.cuda_modular_ms);
        pipeline_stats.add(result.cuda_pipeline_ms);
        if (result.cuda_pipeline_ms > 0.0) {
            speedup_stats.add(result.cuda_modular_ms / result.cuda_pipeline_ms);
        }
    }

    std::cout << std::endl;
    std::cout << "Latency summary (ms):" << std::endl;
    std::cout << std::left << std::setw(20) << "Pipeline"
              << std::right << std::setw(12) << "avg"
              << std::setw(12) << "min"
              << std::setw(12) << "max" << std::endl;
    printStatsRow("CPU full", cpu_stats);
    printStatsRow("CUDA modular V4", modular_stats);
    printStatsRow("CUDA pipeline V4", pipeline_stats);

    std::cout << std::endl;
    std::cout << "Count summary:" << std::endl;
    std::cout << std::left << std::setw(20) << "Metric"
              << std::right << std::setw(12) << "avg"
              << std::setw(12) << "min"
              << std::setw(12) << "max" << std::endl;
    printStatsRow("points", point_stats);
    printStatsRow("filtered", filtered_stats);
    printStatsRow("pillars", pillar_stats);
    printStatsRow("stored points", stored_stats);

    std::cout << std::endl;
    std::cout << "Speedup over CUDA modular V4:" << std::endl;
    std::cout << "avg: " << speedup_stats.avg()
              << "x, min: " << speedup_stats.min
              << "x, max: " << speedup_stats.max << "x" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: CUDA pipeline V4 keeps feature and BEV tensors on GPU and skips full D2H export." << std::endl;

    return 0;
}
