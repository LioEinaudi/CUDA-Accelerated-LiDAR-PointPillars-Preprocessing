#include"cuda_pipeline.cuh"
#include"cpu_pillar.hpp"
#include"cpu_pillar_scatter.hpp"
#include<cuda_runtime.h>
#include<thrust/device_ptr.h>
#include<thrust/scan.h>
#include<stdexcept>
#include<string>


namespace {

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
}



CudaPipelineSummary cudaPreprocessPipelineV1 (const std::vector<PointXYZI> & points , const RangeConfig & range , const PillarConfig & pillar ) {
    if ( points.empty () )
        return {};
    int num_points = points.size();
    int grid_x = getGridX(range, pillar);
    int grid_y = getGridY(range, pillar);
    CudaPipelineSummary result;

    PointXYZI *d_input_points = nullptr;
    size_t points_bytes = sizeof(PointXYZI) * num_points;
    checkCuda(cudaMalloc((void **)&d_input_points, points_bytes), "cudaMalloc d_input_points failed");

    int * d_valid_flags = nullptr;
    checkCuda(cudaMalloc((void**)&d_valid_flags , sizeof(int) * num_points), "cudaMalloc d_valid_flags failed");

    int *d_prefix_sum = nullptr;
    checkCuda(cudaMalloc((void **)&d_prefix_sum, sizeof(int) * num_points), "cudaMalloc d_prefix_sum failed");

    PointXYZI *d_filtered_points = nullptr;
    checkCuda(cudaMalloc((void **)&d_filtered_points, sizeof(PointXYZI) * num_points), "cudaMalloc d_filtered_points failed");

    PillarCoord *d_coords = nullptr;
    checkCuda(cudaMalloc((void**)& d_coords,sizeof(PillarCoord) * num_points), "cudaMalloc d_coords failed");

    int *d_key_to_pillar = nullptr;
    checkCuda(cudaMalloc((void **)&d_key_to_pillar, sizeof(int) * grid_x * grid_y), "cudaMalloc d_key_to_pillar failed");

    PillarInfo *d_pillars = nullptr;
    checkCuda(cudaMalloc((void **)&d_pillars, sizeof(PillarInfo) * pillar.max_pillars), "cudaMalloc d_pillars failed");

    int *d_point_to_pillar = nullptr;
    checkCuda(cudaMalloc((void **)&d_point_to_pillar, sizeof(int) * num_points), "cudaMalloc d_point_to_pillar failed");

    int *d_num_pillars = nullptr;
    checkCuda(cudaMalloc((void **)&d_num_pillars, sizeof(int)), "cudaMalloc d_num_pillars failed");



    checkCuda(cudaMemset(d_key_to_pillar, -1, sizeof(int) * grid_x * grid_y), "cudaMemset d_key_to_pillar failed");

    checkCuda(cudaMemset(d_num_pillars , 0 , sizeof(int)), "cudaMemset d_num_pillars failed");

    checkCuda(cudaMemcpy(d_input_points, points.data(), sizeof(PointXYZI) * num_points, cudaMemcpyHostToDevice), "cudaMemcpy input points H2D failed");

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
    checkCuda(cudaMemcpy(&last_prefix, d_prefix_sum+num_points - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy last_prefix D2H failed");

    result.filtered_count = last_flag + last_prefix;


    pipelineScatterValidKernel<<<blocks, threads>>>(d_input_points, num_points, d_valid_flags, d_prefix_sum, d_filtered_points);
    checkCuda(cudaGetLastError(), "pipelineScatterValidKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "pipelineScatterValidKernel failed");

    if (result.filtered_count == 0) {
        checkCuda(cudaFree(d_coords), "cudaFree d_coords failed");
        checkCuda(cudaFree(d_filtered_points), "cudaFree d_filtered_points failed");
        checkCuda(cudaFree(d_input_points), "cudaFree d_input_points failed");
        checkCuda(cudaFree(d_key_to_pillar), "cudaFree d_key_to_pillar failed");
        checkCuda(cudaFree(d_num_pillars), "cudaFree d_num_pillars failed");
        checkCuda(cudaFree(d_pillars), "cudaFree d_pillars failed");
        checkCuda(cudaFree(d_point_to_pillar), "cudaFree d_point_to_pillar failed");
        checkCuda(cudaFree(d_prefix_sum), "cudaFree d_prefix_sum failed");
        checkCuda(cudaFree(d_valid_flags), "cudaFree d_valid_flags failed");
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
    checkCuda(cudaFree(d_coords), "cudaFree d_coords failed");
    checkCuda(cudaFree(d_filtered_points), "cudaFree d_filtered_points failed");
    checkCuda(cudaFree(d_input_points), "cudaFree d_input_points failed");
    checkCuda(cudaFree(d_key_to_pillar), "cudaFree d_key_to_pillar failed");
    checkCuda(cudaFree(d_num_pillars), "cudaFree d_num_pillars failed");
    checkCuda(cudaFree(d_pillars), "cudaFree d_pillars failed");
    checkCuda(cudaFree(d_point_to_pillar), "cudaFree d_point_to_pillar failed");
    checkCuda(cudaFree(d_prefix_sum), "cudaFree d_prefix_sum failed");
    checkCuda(cudaFree(d_valid_flags), "cudaFree d_valid_flags failed");
    return result;
}
