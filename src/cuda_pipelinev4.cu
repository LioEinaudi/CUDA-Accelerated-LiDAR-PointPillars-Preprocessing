#include "cuda_pipeline.cuh"
#include "cpu_pillar.hpp"
#include "cpu_pillar_scatter.hpp"
#include "cpu_feature.hpp"
#include "cpu_bev.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <stdexcept>
#include <string>

namespace
{

    void checkCuda(cudaError_t err, const char *message)
    {
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(err));
        }
    }

    __device__ bool pipelineIsPointInRange(
        const PointXYZI &point,
        const RangeConfig &range)
    {
        return point.x >= range.min_x && point.x < range.max_x && point.y >= range.min_y && point.y < range.max_y && point.z >= range.min_z && point.z < range.max_z;
    }
    __global__ void pipelineMarkValidKernel(
        const PointXYZI *input,
        int num_points,
        RangeConfig range,
        int *valid_flags)
    {
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_points)
            return;

        PointXYZI point = input[idx];
        valid_flags[idx] = pipelineIsPointInRange(point, range) ? 1 : 0;
    }
    __global__ void pipelineScatterValidKernel(
        const PointXYZI *input,
        int num_points,
        const int *valid_flags,
        const int *prefix_sum,
        PointXYZI *output)
    {
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_points)
            return;

        if (valid_flags[idx])
        {
            int out_idx = prefix_sum[idx];
            output[out_idx] = input[idx];
        }
    }
    __global__ void pipelineComputePillarCoordKernel(
        const PointXYZI *points,
        int num_points,
        RangeConfig range,
        PillarConfig pillar,
        PillarCoord *coords)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_points)
            return;

        PointXYZI point = points[idx];

        int pillar_x = static_cast<int>(floorf((point.x - range.min_x) / pillar.voxel_x));

        int pillar_y = static_cast<int>(floorf((point.y - range.min_y) / pillar.voxel_y));

        coords[idx] = PillarCoord{pillar_x, pillar_y};
    }
    __global__ void pipelinePillarScatterKernel(
        const PillarCoord *coords,
        int num_points,
        int grid_x,
        int grid_size,
        int max_pillars,
        int *key_to_pillar,
        PillarInfo *pillars,
        int *point_to_pillar,
        int *num_pillars)
    {
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_points)
            return;

        PillarCoord coord = coords[idx];
        int key = coord.x + grid_x * coord.y;
        if (key < 0 || key >= grid_size)
        {
            point_to_pillar[idx] = -1;
            return;
        }

        int old = atomicCAS(&key_to_pillar[key], -1, -2);
        int pillar_index;
        if (old == -1)
        {
            pillar_index = atomicAdd(num_pillars, 1);
            if (pillar_index < max_pillars)
            {
                pillars[pillar_index].coord = coord;
                pillars[pillar_index].point_count = 0;
                __threadfence();
                atomicExch(&key_to_pillar[key], pillar_index);
            }
            else
            {
                atomicExch(&key_to_pillar[key], -1);
                point_to_pillar[idx] = -1;
                return;
            }
        }
        else if (old == -2)
        {
            do
            {
                pillar_index = atomicAdd(&key_to_pillar[key], 0);
            } while (pillar_index == -2);
        }
        else
        {
            pillar_index = old;
        }
        if (pillar_index < 0)
        {
            point_to_pillar[idx] = -1;
            return;
        }

        atomicAdd(&pillars[pillar_index].point_count, 1);
        point_to_pillar[idx] = pillar_index;
    }
    __global__ void pipelineBuildPillarPointStorageKernel(
        const PointXYZI *points,
        const int *point_to_pillar,
        int num_points,
        int max_points_per_pillar,
        PointXYZI *pillar_points,
        int *raw_point_count,
        int *pillar_point_count)
    {
        unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= num_points)
            return;

        int pillar_index = point_to_pillar[idx];
        if (pillar_index < 0)
            return;

        int offset = atomicAdd(&raw_point_count[pillar_index], 1);

        if (offset >= max_points_per_pillar)
            return;

        int out_index = pillar_index * max_points_per_pillar + offset;
        pillar_points[out_index] = points[idx];
        atomicAdd(&pillar_point_count[pillar_index], 1);
    }
    __global__ void pipelineComputePillarMeanKernel(
        const PointXYZI *pillar_points,
        const int *pillar_point_count,
        int num_pillars,
        int max_points_per_pillar,
        float *mean_x,
        float *mean_y,
        float *mean_z)
    {
        unsigned int pillar_index = blockDim.x * blockIdx.x + threadIdx.x;
        if (pillar_index >= num_pillars)
            return;

        int count = pillar_point_count[pillar_index];

        if (count == 0)
        {
            mean_x[pillar_index] = 0.0f;
            mean_y[pillar_index] = 0.0f;
            mean_z[pillar_index] = 0.0f;
            return;
        }

        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;

        for (int point_offset = 0; point_offset < count; point_offset++)
        {
            int point_index = pillar_index * max_points_per_pillar + point_offset;
            PointXYZI p = pillar_points[point_index];
            sum_x += p.x;
            sum_y += p.y;
            sum_z += p.z;
        }
        mean_x[pillar_index] = sum_x / count;
        mean_y[pillar_index] = sum_y / count;
        mean_z[pillar_index] = sum_z / count;
    }
    __global__ void pipelineGeneratePillarFeatureKernel(
        const PointXYZI *pillar_points,
        const int *pillar_point_count,
        const PillarInfo *pillars,
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

        if (linear_idx >= total)
            return;

        int pillar_index = linear_idx / pillar.max_points_per_pillar;
        int point_offset = linear_idx % pillar.max_points_per_pillar;

        int count = pillar_point_count[pillar_index];
        if (point_offset >= count)
            return;

        PillarCoord coord = pillars[pillar_index].coord;
        float center_x = range.min_x + (coord.x + 0.5f) * pillar.voxel_x;
        float center_y = range.min_y + (coord.y + 0.5f) * pillar.voxel_y;

        int point_index = pillar_index * pillar.max_points_per_pillar + point_offset;
        PointXYZI p = pillar_points[point_index];

        int base = (pillar_index * pillar.max_points_per_pillar + point_offset) * kPillarFeatureDim;

        features[base + 0] = p.x;
        features[base + 1] = p.y;
        features[base + 2] = p.z;
        features[base + 3] = p.intensity;
        features[base + 4] = p.x - mean_x[pillar_index];
        features[base + 5] = p.y - mean_y[pillar_index];
        features[base + 6] = p.z - mean_z[pillar_index];
        features[base + 7] = p.x - center_x;
        features[base + 8] = p.y - center_y;
    }
    __global__ void pipelineBuildBevPseudoImageKernel(
        const float *features,
        const PillarInfo *pillars,
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
        if (linear_idx >= total)
            return;

        int pillar_index = linear_idx / feature_dim;
        int channel = linear_idx % feature_dim;

        int count = pillar_point_count[pillar_index];
        if (count == 0)
            return;

        float sum = 0.0f;
        for (int point_offset = 0; point_offset < count; point_offset++)
        {
            int feature_index = (pillar_index * max_points_per_pillar + point_offset) * feature_dim + channel;
            sum += features[feature_index];
        }
        float mean = sum / count;

        PillarCoord coord = pillars[pillar_index].coord;
        int bev_index = (channel * height + coord.y) * width + coord.x;
        bev[bev_index] = mean;
    }
}

CudaPipelineSummary cudaPreprocessPipelineV4(const std::vector<PointXYZI> &points, const RangeConfig &range, const PillarConfig &pillar)
{
    if (points.empty())
        return {};
    int num_points = points.size();
    int grid_x = getGridX(range, pillar);
    int grid_y = getGridY(range, pillar);
    CudaPipelineSummary result;
    // V4 keeps the BEV pseudo-image on GPU for resident pipeline benchmarking.
    // BevPseudoImage bev;
    // bev.channels = kPillarFeatureDim;
    // bev.height = grid_y;
    // bev.width = grid_x;
    // bev.data.resize(kPillarFeatureDim * grid_x * grid_y, 0.0f);

    PointXYZI *d_input_points = nullptr;
    size_t points_bytes = sizeof(PointXYZI) * num_points;
    checkCuda(cudaMalloc((void **)&d_input_points, points_bytes), "cudaMalloc d_input_points failed");

    int *d_valid_flags = nullptr;
    checkCuda(cudaMalloc((void **)&d_valid_flags, sizeof(int) * num_points), "cudaMalloc d_valid_flags failed");

    int *d_prefix_sum = nullptr;
    checkCuda(cudaMalloc((void **)&d_prefix_sum, sizeof(int) * num_points), "cudaMalloc d_prefix_sum failed");

    PointXYZI *d_filtered_points = nullptr;
    checkCuda(cudaMalloc((void **)&d_filtered_points, sizeof(PointXYZI) * num_points), "cudaMalloc d_filtered_points failed");

    PillarCoord *d_coords = nullptr;
    checkCuda(cudaMalloc((void **)&d_coords, sizeof(PillarCoord) * num_points), "cudaMalloc d_coords failed");

    int *d_key_to_pillar = nullptr;
    checkCuda(cudaMalloc((void **)&d_key_to_pillar, sizeof(int) * grid_x * grid_y), "cudaMalloc d_key_to_pillar failed");

    PillarInfo *d_pillars = nullptr;
    checkCuda(cudaMalloc((void **)&d_pillars, sizeof(PillarInfo) * pillar.max_pillars), "cudaMalloc d_pillars failed");

    int *d_point_to_pillar = nullptr;
    checkCuda(cudaMalloc((void **)&d_point_to_pillar, sizeof(int) * num_points), "cudaMalloc d_point_to_pillar failed");

    int *d_num_pillars = nullptr;
    checkCuda(cudaMalloc((void **)&d_num_pillars, sizeof(int)), "cudaMalloc d_num_pillars failed");

    PointXYZI *d_pillar_points = nullptr;
    checkCuda(cudaMalloc((void **)&d_pillar_points, sizeof(PointXYZI) * pillar.max_pillars * pillar.max_points_per_pillar), "cudaMalloc d_pillar_points failed");

    int *d_raw_point_count = nullptr;
    checkCuda(cudaMalloc((void **)&d_raw_point_count, sizeof(int) * pillar.max_pillars), "cudaMalloc d_raw_point_count failed");

    int *d_pillar_point_count = nullptr;
    checkCuda(cudaMalloc((void **)&d_pillar_point_count, sizeof(int) * pillar.max_pillars), "cudaMalloc d_pillar_point_count failed");

    float *d_mean_x = nullptr, *d_mean_y = nullptr, *d_mean_z = nullptr;
    checkCuda(cudaMalloc((void **)&d_mean_x, sizeof(float) * pillar.max_pillars), "cudaMalloc d_mean_x failed");
    checkCuda(cudaMalloc((void **)&d_mean_y, sizeof(float) * pillar.max_pillars), "cudaMalloc d_mean_y failed");
    checkCuda(cudaMalloc((void **)&d_mean_z, sizeof(float) * pillar.max_pillars), "cudaMalloc d_mean_z failed");

    float *d_features = nullptr;
    checkCuda(cudaMalloc((void **)&d_features, sizeof(float) * pillar.max_points_per_pillar * kPillarFeatureDim * pillar.max_pillars), "cudaMalloc d_features failed");

    float *d_bev = nullptr;
    checkCuda(cudaMalloc((void **)&d_bev, sizeof(float) * kPillarFeatureDim * grid_x * grid_y), "cudaMalloc d_bev failed");

    checkCuda(cudaMemset(d_key_to_pillar, -1, sizeof(int) * grid_x * grid_y), "cudaMemset d_key_to_pillar failed");

    checkCuda(cudaMemset(d_num_pillars, 0, sizeof(int)), "cudaMemset d_num_pillars failed");

    checkCuda(cudaMemset(d_raw_point_count, 0, sizeof(int) * pillar.max_pillars), "cudaMemset d_raw_point_count failed");

    checkCuda(cudaMemset(d_pillar_point_count, 0, sizeof(int) * pillar.max_pillars), "cudaMemset d_pillar_point_count");

    checkCuda(cudaMemcpy(d_input_points, points.data(), sizeof(PointXYZI) * num_points, cudaMemcpyHostToDevice), "cudaMemcpy input points H2D failed");

    checkCuda(cudaMemset(d_features, 0, sizeof(float) * pillar.max_pillars * pillar.max_points_per_pillar * kPillarFeatureDim), "cudaMemset d_features failed");

    checkCuda(cudaMemset(d_bev, 0, sizeof(float) * grid_x * grid_y * kPillarFeatureDim), "cudaMemset d_bev failed");

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    pipelineMarkValidKernel<<<blocks, threads>>>(d_input_points, num_points, range, d_valid_flags);
    checkCuda(cudaGetLastError(), "pipelineMarkValidKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "pipelineMarkValidKernel failed");

    thrust::device_ptr<int> flags_ptr(d_valid_flags);
    thrust::device_ptr<int> scan_ptr(d_prefix_sum);
    thrust::exclusive_scan(flags_ptr, flags_ptr + num_points, scan_ptr);
    checkCuda(cudaDeviceSynchronize(), "thrust exclusive_scan failed");
    int last_flag = 0, last_prefix = 0;
    checkCuda(cudaMemcpy(&last_flag, d_valid_flags + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_flag D2H failed");
    checkCuda(cudaMemcpy(&last_prefix, d_prefix_sum + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_prefix D2H failed");

    result.filtered_count = last_flag + last_prefix;

    pipelineScatterValidKernel<<<blocks, threads>>>(d_input_points, num_points, d_valid_flags, d_prefix_sum, d_filtered_points);
    checkCuda(cudaGetLastError(), "pipelineScatterValidKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "pipelineScatterValidKernel failed");

    if (result.filtered_count == 0)
    {
        checkCuda(cudaFree(d_coords), "cudaFree d_coords failed");
        checkCuda(cudaFree(d_filtered_points), "cudaFree d_filtered_points failed");
        checkCuda(cudaFree(d_input_points), "cudaFree d_input_points failed");
        checkCuda(cudaFree(d_key_to_pillar), "cudaFree d_key_to_pillar failed");
        checkCuda(cudaFree(d_num_pillars), "cudaFree d_num_pillars failed");
        checkCuda(cudaFree(d_pillars), "cudaFree d_pillars failed");
        checkCuda(cudaFree(d_point_to_pillar), "cudaFree d_point_to_pillar failed");
        checkCuda(cudaFree(d_prefix_sum), "cudaFree d_prefix_sum failed");
        checkCuda(cudaFree(d_valid_flags), "cudaFree d_valid_flags failed");
        checkCuda(cudaFree(d_pillar_point_count), "cudaFree d_pillar_point_count failed");
        checkCuda(cudaFree(d_pillar_points), "cudaFree d_pillar_points failed");
        checkCuda(cudaFree(d_raw_point_count), "cudaFree d_raw_point_count failed");
        checkCuda(cudaFree(d_features), "cudaFree d_features failed");
        checkCuda(cudaFree(d_mean_x), "cudaFree d_mean_x failed");
        checkCuda(cudaFree(d_mean_y), "cudaFree d_mean_y failed");
        checkCuda(cudaFree(d_mean_z), "cudaFree d_mean_z failed");
        checkCuda(cudaFree(d_bev), "cudaFree d_bev failed");
        return result;
    }

    int filtered_blocks = (result.filtered_count + threads - 1) / threads;
    pipelineComputePillarCoordKernel<<<filtered_blocks, threads>>>(d_filtered_points, result.filtered_count, range, pillar, d_coords);
    checkCuda(cudaGetLastError(), "pipelineComputePillarCoordKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "pipelineComputePillarCoordKernel failed");

    pipelinePillarScatterKernel<<<filtered_blocks, threads>>>(d_coords, result.filtered_count, grid_x, grid_x * grid_y, pillar.max_pillars, d_key_to_pillar, d_pillars, d_point_to_pillar, d_num_pillars);
    checkCuda(cudaGetLastError(), "pipelinePillarScatterKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "pipelinePillarScatterKernel failed");

    checkCuda(cudaMemcpy(&result.num_pillars, d_num_pillars, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy num_pillars D2H failed");

    pipelineBuildPillarPointStorageKernel<<<filtered_blocks, threads>>>(d_filtered_points, d_point_to_pillar, result.filtered_count, pillar.max_points_per_pillar, d_pillar_points, d_raw_point_count, d_pillar_point_count);
    checkCuda(cudaGetLastError(), "pipelineBuildPillarPointStorageKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "pipelineBuildPillarPointStorageKernel failed");

    std::vector<int> h_pillar_point_count(result.num_pillars);
    checkCuda(cudaMemcpy(h_pillar_point_count.data(), d_pillar_point_count, sizeof(int) * result.num_pillars, cudaMemcpyDeviceToHost), "cudaMemcpy pillar_point_count D2H failed");

    for (int count : h_pillar_point_count)
        result.stored_points += count;

    int pillar_blocks = (result.num_pillars + threads - 1) / threads;
    pipelineComputePillarMeanKernel<<<pillar_blocks, threads>>>(d_pillar_points, d_pillar_point_count, result.num_pillars, pillar.max_points_per_pillar, d_mean_x, d_mean_y, d_mean_z);
    checkCuda(cudaGetLastError(), "pipelineComputePillarMeanKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "pipelineComputePillarMeanKernel failed");

    int total_feature_points = result.num_pillars * pillar.max_points_per_pillar;
    int feature_blocks = (total_feature_points + threads - 1) / threads;
    pipelineGeneratePillarFeatureKernel<<<feature_blocks, threads>>>(d_pillar_points, d_pillar_point_count, d_pillars, d_mean_x, d_mean_y, d_mean_z, result.num_pillars, range, pillar, d_features);
    checkCuda(cudaGetLastError(), "pipelineGeneratePillarFeatureKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "pipelineGeneratePillarFeatureKernel failed");

    int bev_total = result.num_pillars * kPillarFeatureDim;
    int bev_blocks = (bev_total + threads - 1) / threads;
    pipelineBuildBevPseudoImageKernel<<<bev_blocks, threads>>>(d_features, d_pillars, d_pillar_point_count, result.num_pillars, pillar.max_points_per_pillar, kPillarFeatureDim, grid_y, grid_x, d_bev);
    checkCuda(cudaGetLastError(), "pipelineBuildBevPseudoImageKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "pipelineBuildBevPseudoImageKernel failed");
    // checkCuda(cudaMemcpy(bev.data.data(), d_bev, sizeof(float) * kPillarFeatureDim * grid_y * grid_x, cudaMemcpyDeviceToHost), "cudaMemcpy bev D2H failed");

    // V4 benchmark skips full feature D2H copy.
    // std::vector<float> h_features(result.num_pillars * pillar.max_points_per_pillar * kPillarFeatureDim);
    // checkCuda(cudaMemcpy(h_features.data(), d_features, sizeof(float) * result.num_pillars * pillar.max_points_per_pillar * kPillarFeatureDim, cudaMemcpyDeviceToHost), "cudaMemcpy features D2H failed");
    // for (float feature : h_features)
    //     result.feature_values += feature;

    checkCuda(cudaFree(d_coords), "cudaFree d_coords failed");
    checkCuda(cudaFree(d_filtered_points), "cudaFree d_filtered_points failed");
    checkCuda(cudaFree(d_input_points), "cudaFree d_input_points failed");
    checkCuda(cudaFree(d_key_to_pillar), "cudaFree d_key_to_pillar failed");
    checkCuda(cudaFree(d_num_pillars), "cudaFree d_num_pillars failed");
    checkCuda(cudaFree(d_pillars), "cudaFree d_pillars failed");
    checkCuda(cudaFree(d_point_to_pillar), "cudaFree d_point_to_pillar failed");
    checkCuda(cudaFree(d_prefix_sum), "cudaFree d_prefix_sum failed");
    checkCuda(cudaFree(d_valid_flags), "cudaFree d_valid_flags failed");
    checkCuda(cudaFree(d_pillar_point_count), "cudaFree d_pillar_point_count failed");
    checkCuda(cudaFree(d_pillar_points), "cudaFree d_pillar_points failed");
    checkCuda(cudaFree(d_raw_point_count), "cudaFree d_raw_point_count failed");
    checkCuda(cudaFree(d_features), "cudaFree d_features failed");
    checkCuda(cudaFree(d_mean_x), "cudaFree d_mean_x failed");
    checkCuda(cudaFree(d_mean_y), "cudaFree d_mean_y failed");
    checkCuda(cudaFree(d_mean_z), "cudaFree d_mean_z failed");
    checkCuda(cudaFree(d_bev), "cudaFree d_bev failed");
    return result;
}
