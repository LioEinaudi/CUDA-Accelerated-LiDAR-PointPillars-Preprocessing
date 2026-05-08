#include"cuda_preprocess.cuh"
#include"point.hpp"
#include"config.hpp"
#include<vector>
#include<cuda_runtime.h>
#include<string>
#include<stdexcept>

#include<thrust/device_ptr.h>
#include<thrust/scan.h>

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

    cudaError_t err = cudaGetLastError();
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

__global__ void markValidKernel(
    const PointXYZI *input,
    int num_points,
    RangeConfig range,
    int *valid_flags){
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if ( idx >= num_points )
        return;

    PointXYZI point = input[idx];
    valid_flags[idx] = isPointInRangeDevice(point, range) ? 1 : 0;
}

__global__ void scatterValidKernel(
    const PointXYZI *input,
    int num_points,
    const int *valid_flags,
    const int *prefix_sum,
    PointXYZI *output){
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if ( idx >= num_points )
        return; 
    
    if ( valid_flags[idx] ) {
        int out_idx = prefix_sum[idx];
        output[out_idx] = input[idx];   
    }
}

std::vector<PointXYZI>cudaRangeFilterPrefixSum (
    const std::vector<PointXYZI>&points , 
    const RangeConfig&range 
){
    if (points.empty())
        return{};

    int num_points = static_cast<int>(points.size());
    size_t nbytes = points.size() * sizeof(PointXYZI);

    PointXYZI *d_input, *d_output;
    int *d_valid_flags, *d_prefix_sum  ;
    
    cudaMalloc((void **)&d_input, nbytes);
    cudaMalloc((void **)&d_output, nbytes);
    cudaMalloc((void **)&d_valid_flags, points.size() * sizeof(int));
    cudaMalloc((void **)&d_prefix_sum, points.size() * sizeof(int));

    cudaMemcpy(d_input, points.data(), nbytes, cudaMemcpyHostToDevice);


    int threads = 256;
    int blocks = ((num_points + threads - 1) / threads);
    markValidKernel<<<blocks, threads>>>(d_input, num_points, range, d_valid_flags);
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess) 
        throw std::runtime_error (cudaGetErrorString(err)) ;

    err = cudaDeviceSynchronize(); 
    if ( err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err)); 

    thrust::device_ptr<int>flags_ptr(d_valid_flags) ;
    thrust::device_ptr<int> scan_ptr(d_prefix_sum);
    thrust::exclusive_scan(flags_ptr, flags_ptr + num_points, scan_ptr);

    int last_flag = 0, last_prefix = 0;
    cudaMemcpy(&last_flag, d_valid_flags + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_prefix, d_prefix_sum + num_points - 1, sizeof(int), cudaMemcpyDeviceToHost);

    int h_output_count = last_flag + last_prefix;

    scatterValidKernel<<<blocks, threads>>>(d_input, num_points, d_valid_flags, d_prefix_sum, d_output);

    err = cudaGetLastError(); 
    if ( err != cudaSuccess )
        throw std::runtime_error(cudaGetErrorString(err));

    err = cudaDeviceSynchronize(); 
    if ( err != cudaSuccess )
        throw std::runtime_error(cudaGetErrorString(err)); 
        

    std::vector<PointXYZI> filtered(h_output_count);

    cudaMemcpy(filtered.data(), d_output, h_output_count * sizeof(PointXYZI), cudaMemcpyDeviceToHost); 

    cudaFree(d_input) ;
    cudaFree(d_output);
    cudaFree(d_prefix_sum); 
    cudaFree(d_valid_flags) ;
    return filtered; 
}
