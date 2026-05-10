#include "config.hpp"
#include "cpu_bev.hpp"
#include "cpu_feature.hpp"
#include "cpu_pillar_scatter.hpp"
#include "cpu_pillar_storage.hpp"
#include "cpu_preprocess.hpp"
#include "kitti_reader.hpp"
#include "point.hpp"

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

namespace {

double elapsedMs(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void checkCuda(cudaError_t err, const char *message)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(err));
    }
}

float eventElapsedMs(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime failed");
    return ms;
}

__device__ bool isPointInRangeTimingDevice(const PointXYZI &point, const RangeConfig &range)
{
    return point.x >= range.min_x && point.x < range.max_x &&
           point.y >= range.min_y && point.y < range.max_y &&
           point.z >= range.min_z && point.z < range.max_z;
}

__global__ void markValidTimingKernel(
    const PointXYZI *input,
    int num_points,
    RangeConfig range,
    int *valid_flags)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points) {
        return;
    }

    PointXYZI point = input[idx];
    valid_flags[idx] = isPointInRangeTimingDevice(point, range) ? 1 : 0;
}

__global__ void scatterValidTimingKernel(
    const PointXYZI *input,
    int num_points,
    const int *valid_flags,
    const int *prefix_sum,
    PointXYZI *output)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points) {
        return;
    }

    if (valid_flags[idx]) {
        int out_idx = prefix_sum[idx];
        output[out_idx] = input[idx];
    }
}

__global__ void buildPillarPointStorageTimingKernel(
    const PointXYZI *points,
    const int *point_to_pillar,
    int num_points,
    int max_points_per_pillar,
    PointXYZI *pillar_points,
    int *raw_point_count,
    int *pillar_point_count)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points) {
        return;
    }

    int pillar_index = point_to_pillar[idx];
    if (pillar_index < 0) {
        return;
    }

    int offset = atomicAdd(&raw_point_count[pillar_index], 1);
    if (offset >= max_points_per_pillar) {
        return;
    }

    int out_index = pillar_index * max_points_per_pillar + offset;
    pillar_points[out_index] = points[idx];
    atomicAdd(&pillar_point_count[pillar_index], 1);
}

__global__ void computePillarMeanTimingKernel(
    const PointXYZI *pillar_points,
    const int *pillar_point_count,
    int num_pillars,
    int max_points_per_pillar,
    float *mean_x,
    float *mean_y,
    float *mean_z)
{
    unsigned int pillar_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (pillar_index >= num_pillars) {
        return;
    }

    int count = pillar_point_count[pillar_index];
    if (count == 0) {
        mean_x[pillar_index] = 0.0f;
        mean_y[pillar_index] = 0.0f;
        mean_z[pillar_index] = 0.0f;
        return;
    }

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    for (int point_offset = 0; point_offset < count; point_offset++) {
        int point_index = pillar_index * max_points_per_pillar + point_offset;
        PointXYZI point = pillar_points[point_index];
        sum_x += point.x;
        sum_y += point.y;
        sum_z += point.z;
    }

    mean_x[pillar_index] = sum_x / count;
    mean_y[pillar_index] = sum_y / count;
    mean_z[pillar_index] = sum_z / count;
}

__global__ void generatePillarFeatureTimingKernel(
    const PointXYZI *pillar_points,
    const int *pillar_point_count,
    const PillarCoord *pillar_coords,
    const float *mean_x,
    const float *mean_y,
    const float *mean_z,
    int num_pillars,
    RangeConfig range,
    PillarConfig pillar,
    float *features)
{
    unsigned int linear_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = num_pillars * pillar.max_points_per_pillar;
    if (linear_idx >= total) {
        return;
    }

    int pillar_index = linear_idx / pillar.max_points_per_pillar;
    int point_offset = linear_idx % pillar.max_points_per_pillar;
    int count = pillar_point_count[pillar_index];
    if (point_offset >= count) {
        return;
    }

    PillarCoord coord = pillar_coords[pillar_index];
    float center_x = range.min_x + (coord.x + 0.5f) * pillar.voxel_x;
    float center_y = range.min_y + (coord.y + 0.5f) * pillar.voxel_y;

    int point_index = pillar_index * pillar.max_points_per_pillar + point_offset;
    PointXYZI point = pillar_points[point_index];
    int base = (pillar_index * pillar.max_points_per_pillar + point_offset) * kPillarFeatureDim;

    features[base + 0] = point.x;
    features[base + 1] = point.y;
    features[base + 2] = point.z;
    features[base + 3] = point.intensity;
    features[base + 4] = point.x - mean_x[pillar_index];
    features[base + 5] = point.y - mean_y[pillar_index];
    features[base + 6] = point.z - mean_z[pillar_index];
    features[base + 7] = point.x - center_x;
    features[base + 8] = point.y - center_y;
}

__global__ void buildBevPseudoImageTimingKernel(
    const float *features,
    const PillarCoord *pillar_coords,
    const int *pillar_point_count,
    int num_pillars,
    int max_points_per_pillar,
    int feature_dim,
    int height,
    int width,
    float *bev)
{
    unsigned int linear_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = num_pillars * feature_dim;
    if (linear_idx >= total) {
        return;
    }

    int pillar_index = linear_idx / feature_dim;
    int channel = linear_idx % feature_dim;
    int count = pillar_point_count[pillar_index];
    if (count == 0) {
        return;
    }

    float sum = 0.0f;
    for (int point_offset = 0; point_offset < count; point_offset++) {
        int feature_index =
            (pillar_index * max_points_per_pillar + point_offset) * feature_dim + channel;
        sum += features[feature_index];
    }

    PillarCoord coord = pillar_coords[pillar_index];
    int bev_index = (channel * height + coord.y) * width + coord.x;
    bev[bev_index] = sum / count;
}

struct RangeFilterTiming {
    int output_count = 0;
    double malloc_ms = 0.0;
    double h2d_copy_ms = 0.0;
    float flag_kernel_ms = 0.0f;
    float scan_ms = 0.0f;
    double d2h_count_copy_ms = 0.0;
    float scatter_kernel_ms = 0.0f;
    double host_output_alloc_ms = 0.0;
    double d2h_points_copy_ms = 0.0;
    double free_ms = 0.0;
    double total_ms = 0.0;
};

struct StorageTiming {
    int num_pillars = 0;
    double malloc_ms = 0.0;
    double memset_ms = 0.0;
    double h2d_copy_ms = 0.0;
    float kernel_ms = 0.0f;
    double host_output_alloc_ms = 0.0;
    double d2h_copy_ms = 0.0;
    double free_ms = 0.0;
    double total_ms = 0.0;
};

struct FeatureTiming {
    int num_pillars = 0;
    int feature_count = 0;
    double malloc_ms = 0.0;
    double memset_ms = 0.0;
    double h2d_copy_ms = 0.0;
    float mean_kernel_ms = 0.0f;
    float feature_kernel_ms = 0.0f;
    double host_output_alloc_ms = 0.0;
    double d2h_copy_ms = 0.0;
    double free_ms = 0.0;
    double total_ms = 0.0;
};

struct BevTiming {
    int bev_count = 0;
    double malloc_ms = 0.0;
    double memset_ms = 0.0;
    double h2d_copy_ms = 0.0;
    float kernel_ms = 0.0f;
    double host_output_alloc_ms = 0.0;
    double d2h_copy_ms = 0.0;
    double free_ms = 0.0;
    double total_ms = 0.0;
};

RangeFilterTiming benchmarkRangeFilterPrefixSum(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range)
{
    RangeFilterTiming timing;
    if (points.empty()) {
        return timing;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    int num_points = static_cast<int>(points.size());
    size_t point_bytes = points.size() * sizeof(PointXYZI);
    size_t int_bytes = points.size() * sizeof(int);

    PointXYZI *d_input = nullptr;
    PointXYZI *d_output = nullptr;
    int *d_valid_flags = nullptr;
    int *d_prefix_sum = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMalloc((void **)&d_input, point_bytes), "cudaMalloc d_input failed");
    checkCuda(cudaMalloc((void **)&d_output, point_bytes), "cudaMalloc d_output failed");
    checkCuda(cudaMalloc((void **)&d_valid_flags, int_bytes), "cudaMalloc d_valid_flags failed");
    checkCuda(cudaMalloc((void **)&d_prefix_sum, int_bytes), "cudaMalloc d_prefix_sum failed");
    auto t1 = std::chrono::high_resolution_clock::now();
    timing.malloc_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(d_input, points.data(), point_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy H2D input failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.h2d_copy_ms = elapsedMs(t0, t1);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start failed");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop failed");

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    checkCuda(cudaEventRecord(start), "cudaEventRecord flag start failed");
    markValidTimingKernel<<<blocks, threads>>>(d_input, num_points, range, d_valid_flags);
    checkCuda(cudaGetLastError(), "markValidTimingKernel launch failed");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord flag stop failed");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize flag failed");
    timing.flag_kernel_ms = eventElapsedMs(start, stop);

    thrust::device_ptr<int> flags_ptr(d_valid_flags);
    thrust::device_ptr<int> prefix_ptr(d_prefix_sum);

    checkCuda(cudaEventRecord(start), "cudaEventRecord scan start failed");
    thrust::exclusive_scan(flags_ptr, flags_ptr + num_points, prefix_ptr);
    checkCuda(cudaEventRecord(stop), "cudaEventRecord scan stop failed");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize scan failed");
    timing.scan_ms = eventElapsedMs(start, stop);

    int last_flag = 0;
    int last_prefix = 0;
    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(&last_flag, d_valid_flags + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H last flag failed");
    checkCuda(cudaMemcpy(&last_prefix, d_prefix_sum + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H last prefix failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.d2h_count_copy_ms = elapsedMs(t0, t1);
    timing.output_count = last_flag + last_prefix;

    checkCuda(cudaEventRecord(start), "cudaEventRecord scatter start failed");
    scatterValidTimingKernel<<<blocks, threads>>>(d_input, num_points, d_valid_flags, d_prefix_sum, d_output);
    checkCuda(cudaGetLastError(), "scatterValidTimingKernel launch failed");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord scatter stop failed");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize scatter failed");
    timing.scatter_kernel_ms = eventElapsedMs(start, stop);

    t0 = std::chrono::high_resolution_clock::now();
    std::vector<PointXYZI> filtered(timing.output_count);
    t1 = std::chrono::high_resolution_clock::now();
    timing.host_output_alloc_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    if (timing.output_count > 0) {
        checkCuda(cudaMemcpy(filtered.data(), d_output, timing.output_count * sizeof(PointXYZI),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H filtered points failed");
    }
    t1 = std::chrono::high_resolution_clock::now();
    timing.d2h_points_copy_ms = elapsedMs(t0, t1);

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start failed");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop failed");

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaFree(d_input), "cudaFree d_input failed");
    checkCuda(cudaFree(d_output), "cudaFree d_output failed");
    checkCuda(cudaFree(d_valid_flags), "cudaFree d_valid_flags failed");
    checkCuda(cudaFree(d_prefix_sum), "cudaFree d_prefix_sum failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.free_ms = elapsedMs(t0, t1);

    auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_ms = elapsedMs(total_start, total_end);

    return timing;
}

StorageTiming benchmarkPillarStorage(
    const std::vector<PointXYZI> &points,
    const PillarScatterResult &scatter,
    const PillarConfig &pillar)
{
    StorageTiming timing;
    if (scatter.pillars.empty()) {
        return timing;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    int num_points = static_cast<int>(points.size());
    int num_pillars = static_cast<int>(scatter.pillars.size());
    timing.num_pillars = num_pillars;

    size_t points_bytes = points.size() * sizeof(PointXYZI);
    size_t point_to_pillar_bytes = scatter.point_to_pillar.size() * sizeof(int);
    size_t pillar_points_bytes =
        static_cast<size_t>(num_pillars) * pillar.max_points_per_pillar * sizeof(PointXYZI);
    size_t count_bytes = static_cast<size_t>(num_pillars) * sizeof(int);

    PointXYZI *d_points = nullptr;
    int *d_point_to_pillar = nullptr;
    PointXYZI *d_pillar_points = nullptr;
    int *d_raw_point_count = nullptr;
    int *d_pillar_point_count = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMalloc((void **)&d_points, points_bytes), "cudaMalloc d_points failed");
    checkCuda(cudaMalloc((void **)&d_point_to_pillar, point_to_pillar_bytes),
              "cudaMalloc d_point_to_pillar failed");
    checkCuda(cudaMalloc((void **)&d_pillar_points, pillar_points_bytes),
              "cudaMalloc d_pillar_points failed");
    checkCuda(cudaMalloc((void **)&d_raw_point_count, count_bytes),
              "cudaMalloc d_raw_point_count failed");
    checkCuda(cudaMalloc((void **)&d_pillar_point_count, count_bytes),
              "cudaMalloc d_pillar_point_count failed");
    auto t1 = std::chrono::high_resolution_clock::now();
    timing.malloc_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemset(d_raw_point_count, 0, count_bytes), "cudaMemset raw count failed");
    checkCuda(cudaMemset(d_pillar_point_count, 0, count_bytes),
              "cudaMemset pillar count failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.memset_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(d_points, points.data(), points_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy H2D storage points failed");
    checkCuda(cudaMemcpy(d_point_to_pillar, scatter.point_to_pillar.data(), point_to_pillar_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D point_to_pillar failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.h2d_copy_ms = elapsedMs(t0, t1);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate storage start failed");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate storage stop failed");

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;
    checkCuda(cudaEventRecord(start), "cudaEventRecord storage start failed");
    buildPillarPointStorageTimingKernel<<<blocks, threads>>>(
        d_points, d_point_to_pillar, num_points, pillar.max_points_per_pillar, d_pillar_points,
        d_raw_point_count, d_pillar_point_count);
    checkCuda(cudaGetLastError(), "buildPillarPointStorageTimingKernel launch failed");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord storage stop failed");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize storage failed");
    timing.kernel_ms = eventElapsedMs(start, stop);

    t0 = std::chrono::high_resolution_clock::now();
    std::vector<PointXYZI> pillar_points(num_pillars * pillar.max_points_per_pillar);
    std::vector<int> pillar_point_count(num_pillars);
    t1 = std::chrono::high_resolution_clock::now();
    timing.host_output_alloc_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(pillar_points.data(), d_pillar_points, pillar_points_bytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H pillar_points failed");
    checkCuda(cudaMemcpy(pillar_point_count.data(), d_pillar_point_count, count_bytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H pillar_point_count failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.d2h_copy_ms = elapsedMs(t0, t1);

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy storage start failed");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy storage stop failed");

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaFree(d_pillar_point_count), "cudaFree d_pillar_point_count failed");
    checkCuda(cudaFree(d_pillar_points), "cudaFree d_pillar_points failed");
    checkCuda(cudaFree(d_point_to_pillar), "cudaFree d_point_to_pillar failed");
    checkCuda(cudaFree(d_points), "cudaFree d_points failed");
    checkCuda(cudaFree(d_raw_point_count), "cudaFree d_raw_point_count failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.free_ms = elapsedMs(t0, t1);

    auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_ms = elapsedMs(total_start, total_end);
    return timing;
}

FeatureTiming benchmarkPillarFeature(
    const PillarPointStorage &storage,
    const RangeConfig &range,
    const PillarConfig &pillar)
{
    FeatureTiming timing;
    if (storage.pillar_coords.empty()) {
        return timing;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    int num_pillars = storage.num_pillar;
    int feature_dim = kPillarFeatureDim;
    size_t feature_count =
        static_cast<size_t>(num_pillars) * pillar.max_points_per_pillar * feature_dim;
    timing.num_pillars = num_pillars;
    timing.feature_count = static_cast<int>(feature_count);

    size_t pillar_points_bytes = storage.pillar_points.size() * sizeof(PointXYZI);
    size_t count_bytes = storage.pillar_point_count.size() * sizeof(int);
    size_t coord_bytes = storage.pillar_coords.size() * sizeof(PillarCoord);
    size_t mean_bytes = static_cast<size_t>(num_pillars) * sizeof(float);
    size_t feature_bytes = feature_count * sizeof(float);

    PointXYZI *d_pillar_points = nullptr;
    int *d_pillar_point_count = nullptr;
    PillarCoord *d_pillar_coords = nullptr;
    float *d_mean_x = nullptr;
    float *d_mean_y = nullptr;
    float *d_mean_z = nullptr;
    float *d_features = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMalloc((void **)&d_pillar_points, pillar_points_bytes),
              "cudaMalloc d_pillar_points failed");
    checkCuda(cudaMalloc((void **)&d_pillar_point_count, count_bytes),
              "cudaMalloc d_pillar_point_count failed");
    checkCuda(cudaMalloc((void **)&d_pillar_coords, coord_bytes),
              "cudaMalloc d_pillar_coords failed");
    checkCuda(cudaMalloc((void **)&d_mean_x, mean_bytes), "cudaMalloc d_mean_x failed");
    checkCuda(cudaMalloc((void **)&d_mean_y, mean_bytes), "cudaMalloc d_mean_y failed");
    checkCuda(cudaMalloc((void **)&d_mean_z, mean_bytes), "cudaMalloc d_mean_z failed");
    checkCuda(cudaMalloc((void **)&d_features, feature_bytes), "cudaMalloc d_features failed");
    auto t1 = std::chrono::high_resolution_clock::now();
    timing.malloc_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemset(d_features, 0, feature_bytes), "cudaMemset d_features failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.memset_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(d_pillar_points, storage.pillar_points.data(), pillar_points_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D feature pillar_points failed");
    checkCuda(cudaMemcpy(d_pillar_point_count, storage.pillar_point_count.data(), count_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D feature count failed");
    checkCuda(cudaMemcpy(d_pillar_coords, storage.pillar_coords.data(), coord_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D feature coords failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.h2d_copy_ms = elapsedMs(t0, t1);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate feature start failed");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate feature stop failed");

    int threads = 256;
    int mean_blocks = (num_pillars + threads - 1) / threads;
    checkCuda(cudaEventRecord(start), "cudaEventRecord mean start failed");
    computePillarMeanTimingKernel<<<mean_blocks, threads>>>(
        d_pillar_points, d_pillar_point_count, num_pillars, pillar.max_points_per_pillar,
        d_mean_x, d_mean_y, d_mean_z);
    checkCuda(cudaGetLastError(), "computePillarMeanTimingKernel launch failed");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord mean stop failed");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize mean failed");
    timing.mean_kernel_ms = eventElapsedMs(start, stop);

    int total_feature_points = num_pillars * pillar.max_points_per_pillar;
    int feature_blocks = (total_feature_points + threads - 1) / threads;
    checkCuda(cudaEventRecord(start), "cudaEventRecord feature start failed");
    generatePillarFeatureTimingKernel<<<feature_blocks, threads>>>(
        d_pillar_points, d_pillar_point_count, d_pillar_coords, d_mean_x, d_mean_y, d_mean_z,
        num_pillars, range, pillar, d_features);
    checkCuda(cudaGetLastError(), "generatePillarFeatureTimingKernel launch failed");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord feature stop failed");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize feature failed");
    timing.feature_kernel_ms = eventElapsedMs(start, stop);

    t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> features(feature_count);
    t1 = std::chrono::high_resolution_clock::now();
    timing.host_output_alloc_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(features.data(), d_features, feature_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H features failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.d2h_copy_ms = elapsedMs(t0, t1);

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy feature start failed");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy feature stop failed");

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaFree(d_features), "cudaFree d_features failed");
    checkCuda(cudaFree(d_mean_x), "cudaFree d_mean_x failed");
    checkCuda(cudaFree(d_mean_y), "cudaFree d_mean_y failed");
    checkCuda(cudaFree(d_mean_z), "cudaFree d_mean_z failed");
    checkCuda(cudaFree(d_pillar_coords), "cudaFree d_pillar_coords failed");
    checkCuda(cudaFree(d_pillar_point_count), "cudaFree d_pillar_point_count failed");
    checkCuda(cudaFree(d_pillar_points), "cudaFree d_pillar_points failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.free_ms = elapsedMs(t0, t1);

    auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_ms = elapsedMs(total_start, total_end);
    return timing;
}

BevTiming benchmarkBev(
    const PillarFeatureTensor &features,
    const PillarPointStorage &storage,
    const RangeConfig &range,
    const PillarConfig &pillar)
{
    BevTiming timing;
    if (features.features.empty()) {
        return timing;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    int width = getGridX(range, pillar);
    int height = getGridY(range, pillar);
    int channels = features.feature_dim;
    size_t bev_count = static_cast<size_t>(channels) * width * height;
    timing.bev_count = static_cast<int>(bev_count);

    size_t feature_bytes = features.features.size() * sizeof(float);
    size_t coord_bytes = storage.pillar_coords.size() * sizeof(PillarCoord);
    size_t count_bytes = storage.pillar_point_count.size() * sizeof(int);
    size_t bev_bytes = bev_count * sizeof(float);

    float *d_features = nullptr;
    PillarCoord *d_pillar_coords = nullptr;
    int *d_pillar_point_count = nullptr;
    float *d_bev = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMalloc((void **)&d_features, feature_bytes), "cudaMalloc d_features failed");
    checkCuda(cudaMalloc((void **)&d_pillar_coords, coord_bytes),
              "cudaMalloc d_pillar_coords failed");
    checkCuda(cudaMalloc((void **)&d_pillar_point_count, count_bytes),
              "cudaMalloc d_pillar_point_count failed");
    checkCuda(cudaMalloc((void **)&d_bev, bev_bytes), "cudaMalloc d_bev failed");
    auto t1 = std::chrono::high_resolution_clock::now();
    timing.malloc_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemset(d_bev, 0, bev_bytes), "cudaMemset d_bev failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.memset_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(d_features, features.features.data(), feature_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D bev features failed");
    checkCuda(cudaMemcpy(d_pillar_coords, storage.pillar_coords.data(), coord_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D bev coords failed");
    checkCuda(cudaMemcpy(d_pillar_point_count, storage.pillar_point_count.data(), count_bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D bev counts failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.h2d_copy_ms = elapsedMs(t0, t1);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate bev start failed");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate bev stop failed");

    int threads = 256;
    int total = features.num_pillars * features.feature_dim;
    int blocks = (total + threads - 1) / threads;
    checkCuda(cudaEventRecord(start), "cudaEventRecord bev start failed");
    buildBevPseudoImageTimingKernel<<<blocks, threads>>>(
        d_features, d_pillar_coords, d_pillar_point_count, features.num_pillars,
        pillar.max_points_per_pillar, features.feature_dim, height, width, d_bev);
    checkCuda(cudaGetLastError(), "buildBevPseudoImageTimingKernel launch failed");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord bev stop failed");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize bev failed");
    timing.kernel_ms = eventElapsedMs(start, stop);

    t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> bev(bev_count);
    t1 = std::chrono::high_resolution_clock::now();
    timing.host_output_alloc_ms = elapsedMs(t0, t1);

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaMemcpy(bev.data(), d_bev, bev_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H bev failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.d2h_copy_ms = elapsedMs(t0, t1);

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy bev start failed");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy bev stop failed");

    t0 = std::chrono::high_resolution_clock::now();
    checkCuda(cudaFree(d_bev), "cudaFree d_bev failed");
    checkCuda(cudaFree(d_features), "cudaFree d_features failed");
    checkCuda(cudaFree(d_pillar_coords), "cudaFree d_pillar_coords failed");
    checkCuda(cudaFree(d_pillar_point_count), "cudaFree d_pillar_point_count failed");
    t1 = std::chrono::high_resolution_clock::now();
    timing.free_ms = elapsedMs(t0, t1);

    auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_ms = elapsedMs(total_start, total_end);
    return timing;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./benchmark_cuda_timing path/to/000000.bin\n";
        return 1;
    }

    int device_count = 0;
    cudaError_t device_err = cudaGetDeviceCount(&device_count);
    if (device_err != cudaSuccess || device_count == 0) {
        std::cout << "No CUDA device found, skip benchmark_cuda_timing";
        if (device_err != cudaSuccess) {
            std::cout << ": " << cudaGetErrorString(device_err);
        }
        std::cout << std::endl;
        return 0;
    }

    void *warmup_ptr = nullptr;
    checkCuda(cudaMalloc(&warmup_ptr, 1), "warmup cudaMalloc failed");
    checkCuda(cudaFree(warmup_ptr), "warmup cudaFree failed");

    std::string bin_path = argv[1];
    RangeConfig range;
    PillarConfig pillar;
    pillar.max_pillars = 20000;

    auto points = readKittiBin(bin_path);
    auto cpu_filtered = cpuRangeFilter(points, range);
    auto cpu_scatter = cpuPillarScatter(cpu_filtered, range, pillar);
    auto cpu_storage = cpuBuildPillarPointStorage(cpu_filtered, cpu_scatter, pillar);
    auto cpu_features = cpuGeneratePillarFeatures(cpu_storage, range, pillar);

    std::cout << "CUDA timing warmup..." << std::flush;
    benchmarkRangeFilterPrefixSum(points, range);
    benchmarkPillarStorage(cpu_filtered, cpu_scatter, pillar);
    benchmarkPillarFeature(cpu_storage, range, pillar);
    benchmarkBev(cpu_features, cpu_storage, range, pillar);
    std::cout << " done" << std::endl;

    auto range_timing = benchmarkRangeFilterPrefixSum(points, range);
    auto storage_timing = benchmarkPillarStorage(cpu_filtered, cpu_scatter, pillar);
    auto feature_timing = benchmarkPillarFeature(cpu_storage, range, pillar);
    auto bev_timing = benchmarkBev(cpu_features, cpu_storage, range, pillar);

    std::cout << "Loaded points: " << points.size() << std::endl;
    std::cout << "CPU filtered count: " << cpu_filtered.size() << std::endl;
    std::cout << "CUDA filtered count: " << range_timing.output_count << std::endl;
    std::cout << "CPU pillars: " << cpu_storage.num_pillar << std::endl;
    std::cout << std::endl;

    std::cout << "CUDA range filter timing:" << std::endl;
    std::cout << "malloc:          " << range_timing.malloc_ms << " ms" << std::endl;
    std::cout << "H2D copy:        " << range_timing.h2d_copy_ms << " ms" << std::endl;
    std::cout << "flag kernel:     " << range_timing.flag_kernel_ms << " ms" << std::endl;
    std::cout << "scan:            " << range_timing.scan_ms << " ms" << std::endl;
    std::cout << "D2H count copy:  " << range_timing.d2h_count_copy_ms << " ms" << std::endl;
    std::cout << "scatter kernel:  " << range_timing.scatter_kernel_ms << " ms" << std::endl;
    std::cout << "host out alloc:  " << range_timing.host_output_alloc_ms << " ms" << std::endl;
    std::cout << "D2H points copy: " << range_timing.d2h_points_copy_ms << " ms" << std::endl;
    std::cout << "free:            " << range_timing.free_ms << " ms" << std::endl;
    std::cout << "total:           " << range_timing.total_ms << " ms" << std::endl;
    std::cout << std::endl;

    std::cout << "CUDA pillar storage timing:" << std::endl;
    std::cout << "malloc:          " << storage_timing.malloc_ms << " ms" << std::endl;
    std::cout << "memset:          " << storage_timing.memset_ms << " ms" << std::endl;
    std::cout << "H2D copy:        " << storage_timing.h2d_copy_ms << " ms" << std::endl;
    std::cout << "storage kernel:  " << storage_timing.kernel_ms << " ms" << std::endl;
    std::cout << "host out alloc:  " << storage_timing.host_output_alloc_ms << " ms" << std::endl;
    std::cout << "D2H copy:        " << storage_timing.d2h_copy_ms << " ms" << std::endl;
    std::cout << "free:            " << storage_timing.free_ms << " ms" << std::endl;
    std::cout << "total:           " << storage_timing.total_ms << " ms" << std::endl;
    std::cout << std::endl;

    std::cout << "CUDA feature timing:" << std::endl;
    std::cout << "malloc:          " << feature_timing.malloc_ms << " ms" << std::endl;
    std::cout << "memset:          " << feature_timing.memset_ms << " ms" << std::endl;
    std::cout << "H2D copy:        " << feature_timing.h2d_copy_ms << " ms" << std::endl;
    std::cout << "mean kernel:     " << feature_timing.mean_kernel_ms << " ms" << std::endl;
    std::cout << "feature kernel:  " << feature_timing.feature_kernel_ms << " ms" << std::endl;
    std::cout << "host out alloc:  " << feature_timing.host_output_alloc_ms << " ms" << std::endl;
    std::cout << "D2H copy:        " << feature_timing.d2h_copy_ms << " ms" << std::endl;
    std::cout << "free:            " << feature_timing.free_ms << " ms" << std::endl;
    std::cout << "total:           " << feature_timing.total_ms << " ms" << std::endl;
    std::cout << std::endl;

    std::cout << "CUDA BEV timing:" << std::endl;
    std::cout << "malloc:          " << bev_timing.malloc_ms << " ms" << std::endl;
    std::cout << "memset:          " << bev_timing.memset_ms << " ms" << std::endl;
    std::cout << "H2D copy:        " << bev_timing.h2d_copy_ms << " ms" << std::endl;
    std::cout << "BEV kernel:      " << bev_timing.kernel_ms << " ms" << std::endl;
    std::cout << "host out alloc:  " << bev_timing.host_output_alloc_ms << " ms" << std::endl;
    std::cout << "D2H copy:        " << bev_timing.d2h_copy_ms << " ms" << std::endl;
    std::cout << "free:            " << bev_timing.free_ms << " ms" << std::endl;
    std::cout << "total:           " << bev_timing.total_ms << " ms" << std::endl;

    return 0;
}
