#include "cuda_workspace.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void checkCuda(cudaError_t err, const char *message)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(err));
    }
}

__device__ bool workspaceIsPointInRange(
    const PointXYZI &point,
    const RangeConfig &range)
{
    return point.x >= range.min_x && point.x < range.max_x &&
           point.y >= range.min_y && point.y < range.max_y &&
           point.z >= range.min_z && point.z < range.max_z;
}

__global__ void workspaceMarkValidKernel(
    const PointXYZI *input,
    int num_points,
    RangeConfig range,
    int *valid_flags)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points)
        return;

    PointXYZI point = input[idx];
    valid_flags[idx] = workspaceIsPointInRange(point, range) ? 1 : 0;
}

__global__ void workspaceScatterValidKernel(
    const PointXYZI *input,
    int num_points,
    const int *valid_flags,
    const int *prefix_sum,
    PointXYZI *output)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points)
        return;

    if (valid_flags[idx]) {
        int out_idx = prefix_sum[idx];
        output[out_idx] = input[idx];
    }
}

__global__ void workspaceComputePillarCoordKernel(
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

__global__ void workspacePillarScatterKernel(
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
    if (key < 0 || key >= grid_size) {
        point_to_pillar[idx] = -1;
        return;
    }

    int old = atomicCAS(&key_to_pillar[key], -1, -2);
    int pillar_index;
    if (old == -1) {
        pillar_index = atomicAdd(num_pillars, 1);
        if (pillar_index < max_pillars) {
            pillars[pillar_index].coord = coord;
            pillars[pillar_index].point_count = 0;
            __threadfence();
            atomicExch(&key_to_pillar[key], pillar_index);
        } else {
            atomicExch(&key_to_pillar[key], -1);
            point_to_pillar[idx] = -1;
            return;
        }
    } else if (old == -2) {
        do {
            pillar_index = atomicAdd(&key_to_pillar[key], 0);
        } while (pillar_index == -2);
    } else {
        pillar_index = old;
    }

    if (pillar_index < 0) {
        point_to_pillar[idx] = -1;
        return;
    }

    atomicAdd(&pillars[pillar_index].point_count, 1);
    point_to_pillar[idx] = pillar_index;
}

__global__ void workspaceBuildPillarPointStorageKernel(
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

__global__ void workspaceComputePillarMeanKernel(
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
    if (count == 0) {
        mean_x[pillar_index] = 0.0f;
        mean_y[pillar_index] = 0.0f;
        mean_z[pillar_index] = 0.0f;
        return;
    }

    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
    for (int point_offset = 0; point_offset < count; ++point_offset) {
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

__global__ void workspaceGeneratePillarFeatureKernel(
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

__global__ void workspaceBuildBevPseudoImageKernel(
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
    for (int point_offset = 0; point_offset < count; ++point_offset) {
        int feature_index = (pillar_index * max_points_per_pillar + point_offset) * feature_dim + channel;
        sum += features[feature_index];
    }

    PillarCoord coord = pillars[pillar_index].coord;
    int bev_index = (channel * height + coord.y) * width + coord.x;
    bev[bev_index] = sum / count;
}

template <typename T>
void freePointer(T *&ptr)
{
    if (ptr != nullptr) {
        checkCuda(cudaFree(ptr), "cudaFree workspace pointer failed");
        ptr = nullptr;
    }
}

} // namespace

CudaPreprocessWorkspace createCudaPreprocessWorkspace(
    int max_points,
    const RangeConfig &range,
    const PillarConfig &pillar)
{
    CudaPreprocessWorkspace workspace{};
    workspace.max_points = max_points;
    workspace.max_pillars = pillar.max_pillars;
    workspace.max_points_per_pillar = pillar.max_points_per_pillar;
    workspace.grid_x = getGridX(range, pillar);
    workspace.grid_y = getGridY(range, pillar);
    workspace.grid_size = workspace.grid_x * workspace.grid_y;

    checkCuda(cudaMalloc((void **)&workspace.d_input_points, sizeof(PointXYZI) * workspace.max_points), "cudaMalloc d_input_points failed");
    checkCuda(cudaMalloc((void **)&workspace.d_valid_flags, sizeof(int) * workspace.max_points), "cudaMalloc d_valid_flags failed");
    checkCuda(cudaMalloc((void **)&workspace.d_prefix_sum, sizeof(int) * workspace.max_points), "cudaMalloc d_prefix_sum failed");
    checkCuda(cudaMalloc((void **)&workspace.d_filtered_points, sizeof(PointXYZI) * workspace.max_points), "cudaMalloc d_filtered_points failed");
    checkCuda(cudaMalloc((void **)&workspace.d_coords, sizeof(PillarCoord) * workspace.max_points), "cudaMalloc d_coords failed");

    checkCuda(cudaMalloc((void **)&workspace.d_key_to_pillar, sizeof(int) * workspace.grid_size), "cudaMalloc d_key_to_pillar failed");
    checkCuda(cudaMalloc((void **)&workspace.d_pillars, sizeof(PillarInfo) * workspace.max_pillars), "cudaMalloc d_pillars failed");
    checkCuda(cudaMalloc((void **)&workspace.d_point_to_pillar, sizeof(int) * workspace.max_points), "cudaMalloc d_point_to_pillar failed");
    checkCuda(cudaMalloc((void **)&workspace.d_num_pillars, sizeof(int)), "cudaMalloc d_num_pillars failed");

    checkCuda(cudaMalloc((void **)&workspace.d_pillar_points, sizeof(PointXYZI) * workspace.max_points_per_pillar * workspace.max_pillars), "cudaMalloc d_pillar_points failed");
    checkCuda(cudaMalloc((void **)&workspace.d_raw_point_count, sizeof(int) * workspace.max_pillars), "cudaMalloc d_raw_point_count failed");
    checkCuda(cudaMalloc((void **)&workspace.d_pillar_point_count, sizeof(int) * workspace.max_pillars), "cudaMalloc d_pillar_point_count failed");

    checkCuda(cudaMalloc((void **)&workspace.d_mean_x, sizeof(float) * workspace.max_pillars), "cudaMalloc d_mean_x failed");
    checkCuda(cudaMalloc((void **)&workspace.d_mean_y, sizeof(float) * workspace.max_pillars), "cudaMalloc d_mean_y failed");
    checkCuda(cudaMalloc((void **)&workspace.d_mean_z, sizeof(float) * workspace.max_pillars), "cudaMalloc d_mean_z failed");

    checkCuda(cudaMalloc((void **)&workspace.d_features, sizeof(float) * workspace.max_pillars * workspace.max_points_per_pillar * kPillarFeatureDim), "cudaMalloc d_features failed");
    checkCuda(cudaMalloc((void **)&workspace.d_bev, sizeof(float) * workspace.grid_size * kPillarFeatureDim), "cudaMalloc d_bev failed");

    return workspace;
}

CudaPipelineSummary cudaPreprocessPipelineV4Workspace(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar,
    CudaPreprocessWorkspace &workspace)
{
    CudaPipelineSummary result;
    if (points.empty())
        return result;

    if (static_cast<int>(points.size()) > workspace.max_points) {
        throw std::runtime_error("points.size() exceeds workspace.max_points");
    }
    if (pillar.max_pillars > workspace.max_pillars ||
        pillar.max_points_per_pillar > workspace.max_points_per_pillar) {
        throw std::runtime_error("pillar config exceeds workspace capacity");
    }

    int num_points = static_cast<int>(points.size());
    int grid_x = getGridX(range, pillar);
    int grid_y = getGridY(range, pillar);
    int grid_size = grid_x * grid_y;

    if (grid_x != workspace.grid_x || grid_y != workspace.grid_y || grid_size != workspace.grid_size) {
        throw std::runtime_error("range/pillar grid does not match workspace grid");
    }

    checkCuda(cudaMemset(workspace.d_key_to_pillar, -1, sizeof(int) * workspace.grid_size), "cudaMemset d_key_to_pillar failed");
    checkCuda(cudaMemset(workspace.d_num_pillars, 0, sizeof(int)), "cudaMemset d_num_pillars failed");
    checkCuda(cudaMemset(workspace.d_raw_point_count, 0, sizeof(int) * workspace.max_pillars), "cudaMemset d_raw_point_count failed");
    checkCuda(cudaMemset(workspace.d_pillar_point_count, 0, sizeof(int) * workspace.max_pillars), "cudaMemset d_pillar_point_count failed");
    //checkCuda(cudaMemset(workspace.d_features, 0, sizeof(float) * workspace.max_pillars * workspace.max_points_per_pillar * kPillarFeatureDim), "cudaMemset d_features failed");
    checkCuda(cudaMemset(workspace.d_bev, 0, sizeof(float) * workspace.grid_size * kPillarFeatureDim), "cudaMemset d_bev failed");

    checkCuda(cudaMemcpy(workspace.d_input_points, points.data(), sizeof(PointXYZI) * num_points, cudaMemcpyHostToDevice), "cudaMemcpy input points H2D failed");

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    workspaceMarkValidKernel<<<blocks, threads>>>(workspace.d_input_points, num_points, range, workspace.d_valid_flags);
    checkCuda(cudaGetLastError(), "workspaceMarkValidKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "workspaceMarkValidKernel failed");

    thrust::device_ptr<int> flags_ptr(workspace.d_valid_flags);
    thrust::device_ptr<int> scan_ptr(workspace.d_prefix_sum);
    thrust::exclusive_scan(flags_ptr, flags_ptr + num_points, scan_ptr);
    checkCuda(cudaDeviceSynchronize(), "thrust exclusive_scan failed");

    int last_flag = 0, last_prefix = 0;
    checkCuda(cudaMemcpy(&last_flag, workspace.d_valid_flags + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_flag D2H failed");
    checkCuda(cudaMemcpy(&last_prefix, workspace.d_prefix_sum + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_prefix D2H failed");
    result.filtered_count = last_flag + last_prefix;

    workspaceScatterValidKernel<<<blocks, threads>>>(workspace.d_input_points, num_points, workspace.d_valid_flags, workspace.d_prefix_sum, workspace.d_filtered_points);
    checkCuda(cudaGetLastError(), "workspaceScatterValidKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "workspaceScatterValidKernel failed");

    if (result.filtered_count == 0)
        return result;

    int filtered_blocks = (result.filtered_count + threads - 1) / threads;
    workspaceComputePillarCoordKernel<<<filtered_blocks, threads>>>(workspace.d_filtered_points, result.filtered_count, range, pillar, workspace.d_coords);
    checkCuda(cudaGetLastError(), "workspaceComputePillarCoordKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "workspaceComputePillarCoordKernel failed");

    workspacePillarScatterKernel<<<filtered_blocks, threads>>>(workspace.d_coords, result.filtered_count, grid_x, grid_size, workspace.max_pillars, workspace.d_key_to_pillar, workspace.d_pillars, workspace.d_point_to_pillar, workspace.d_num_pillars);
    checkCuda(cudaGetLastError(), "workspacePillarScatterKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "workspacePillarScatterKernel failed");

    int raw_num_pillars = 0;
    checkCuda(cudaMemcpy(&raw_num_pillars, workspace.d_num_pillars, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy num_pillars D2H failed");
    result.num_pillars = std::min(raw_num_pillars, workspace.max_pillars);

    workspaceBuildPillarPointStorageKernel<<<filtered_blocks, threads>>>(workspace.d_filtered_points, workspace.d_point_to_pillar, result.filtered_count, workspace.max_points_per_pillar, workspace.d_pillar_points, workspace.d_raw_point_count, workspace.d_pillar_point_count);
    checkCuda(cudaGetLastError(), "workspaceBuildPillarPointStorageKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "workspaceBuildPillarPointStorageKernel failed");

    std::vector<int> h_pillar_point_count(result.num_pillars);
    checkCuda(cudaMemcpy(h_pillar_point_count.data(), workspace.d_pillar_point_count, sizeof(int) * result.num_pillars, cudaMemcpyDeviceToHost), "cudaMemcpy pillar_point_count D2H failed");
    for (int count : h_pillar_point_count) {
        result.stored_points += count;
    }

    int pillar_blocks = (result.num_pillars + threads - 1) / threads;
    workspaceComputePillarMeanKernel<<<pillar_blocks, threads>>>(workspace.d_pillar_points, workspace.d_pillar_point_count, result.num_pillars, workspace.max_points_per_pillar, workspace.d_mean_x, workspace.d_mean_y, workspace.d_mean_z);
    checkCuda(cudaGetLastError(), "workspaceComputePillarMeanKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "workspaceComputePillarMeanKernel failed");

    int total_feature_points = result.num_pillars * workspace.max_points_per_pillar;
    int feature_blocks = (total_feature_points + threads - 1) / threads;
    workspaceGeneratePillarFeatureKernel<<<feature_blocks, threads>>>(workspace.d_pillar_points, workspace.d_pillar_point_count, workspace.d_pillars, workspace.d_mean_x, workspace.d_mean_y, workspace.d_mean_z, result.num_pillars, range, pillar, workspace.d_features);
    checkCuda(cudaGetLastError(), "workspaceGeneratePillarFeatureKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "workspaceGeneratePillarFeatureKernel failed");

    int bev_total = result.num_pillars * kPillarFeatureDim;
    int bev_blocks = (bev_total + threads - 1) / threads;
    workspaceBuildBevPseudoImageKernel<<<bev_blocks, threads>>>(workspace.d_features, workspace.d_pillars, workspace.d_pillar_point_count, result.num_pillars, workspace.max_points_per_pillar, kPillarFeatureDim, grid_y, grid_x, workspace.d_bev);
    checkCuda(cudaGetLastError(), "workspaceBuildBevPseudoImageKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "workspaceBuildBevPseudoImageKernel failed");

    return result;
}

void destroyCudaPreprocessWorkspace(
    CudaPreprocessWorkspace &workspace)
{
    freePointer(workspace.d_input_points);
    freePointer(workspace.d_valid_flags);
    freePointer(workspace.d_prefix_sum);
    freePointer(workspace.d_filtered_points);
    freePointer(workspace.d_coords);

    freePointer(workspace.d_key_to_pillar);
    freePointer(workspace.d_pillars);
    freePointer(workspace.d_point_to_pillar);
    freePointer(workspace.d_num_pillars);

    freePointer(workspace.d_pillar_points);
    freePointer(workspace.d_raw_point_count);
    freePointer(workspace.d_pillar_point_count);

    freePointer(workspace.d_mean_x);
    freePointer(workspace.d_mean_y);
    freePointer(workspace.d_mean_z);
    freePointer(workspace.d_features);
    freePointer(workspace.d_bev);

    workspace.max_points = 0;
    workspace.max_pillars = 0;
    workspace.max_points_per_pillar = 0;
    workspace.grid_x = 0;
    workspace.grid_y = 0;
    workspace.grid_size = 0;
}
