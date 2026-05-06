#include"cuda_preprocess.cuh"
#include"point.hpp"
#include"config.hpp"
#include<vector>
#include<cuda_runtime.h>
#include<string>
#include<stdexcept>


__device__ bool isPointInRangeDevice(const PointXYZI &point, const RangeConfig &range){
    return point.x >= range.min_x && point.x < range.max_x && point.y >= range.min_y && point.y < range.max_y && point.z >= range.min_z && point.z < range.max_z; 
}

__global__ void rangeFilterAtomicKernel(
    const PointXYZI * input ,
    int num_points ,
    RangeConfig range ,
    PointXYZI * output ,
    int * output_count 
){
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 

    if ( idx >= num_points )
        return;

    PointXYZI p = input[idx]; 
    if ( isPointInRangeDevice(p,range)){
        int out_idx = atomicAdd(output_count, 1);
        output[out_idx] = p; 
    }
    
}

std::vector<PointXYZI> cudaRangeFilterAtomic(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range){
    if ( points.empty ())
        return {};
    
    int num_points = static_cast<int>(points.size());
    size_t bytes = points.size() * sizeof(PointXYZI);

    PointXYZI *d_input, *d_output;
    int *d_output_count;
    cudaMalloc((void **)&d_input, bytes);
    cudaMalloc((void **)&d_output, bytes);
    cudaMalloc((void **)&d_output_count, sizeof(int));

    cudaMemcpy(d_input, points.data(), bytes, cudaMemcpyHostToDevice);

    cudaMemset(d_output_count, 0, sizeof(int));

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;

    rangeFilterAtomicKernel<<<blocks, threads>>>(d_input, num_points, range, d_output, d_output_count);

    cudaError err = cudaGetLastError();
    if ( err != cudaSuccess ) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    int h_output_count = 0;
    cudaMemcpy(&h_output_count, d_output_count, sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<PointXYZI> filtered(h_output_count);
    cudaMemcpy(filtered.data(), d_output, h_output_count *sizeof ( PointXYZI ) , cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_count);
    return filtered; 

}