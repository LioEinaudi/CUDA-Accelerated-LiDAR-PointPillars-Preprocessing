#include"cuda_preprocess.cuh"
#include"point.hpp"
#include"config.hpp"
#include"cpu_pillar_scatter.hpp"
#include<cuda_runtime.h>
#include<stdexcept> 
#include<vector>

__global__ void pillarScatterKernel(
    const PillarCoord *coords,
    int num_points,
    int grid_x,
    int grid_size,
    int max_pillars,
    int *key_to_pillar,
    PillarInfo *pillars,
    int *point_to_pillar,
    int *num_pillars){
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx >= num_points)
        return;

    PillarCoord coord = coords[idx];
    int key = coord.x + grid_x * coord.y;
    if (key < 0 || key >= grid_size) {
        point_to_pillar[idx] = -1;
        return;
    }

    int old = atomicCAS(&key_to_pillar[key], -1, -2);
    int pillar_index; 
    if (old == -1)
    {
        pillar_index = atomicAdd(num_pillars, 1); 
        if ( pillar_index < max_pillars ){
            pillars[pillar_index].coord = coord;
            pillars[pillar_index].point_count = 0;
            __threadfence();
            atomicExch(&key_to_pillar[key], pillar_index);
        }
        else {
            atomicExch(&key_to_pillar[key], -1);
            point_to_pillar[idx] = -1;
            return;
        }
    }
    else if ( old== -2 ){
        do{
            pillar_index = atomicAdd(&key_to_pillar[key], 0);
        } while (pillar_index == -2);
    }
    else {
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

PillarScatterResult cudaPillarScatter(
    const std ::vector<PillarCoord> &coords,
    const RangeConfig &range,
    const PillarConfig &pillar){
    if ( coords.empty())
        return {};
    int grid_x = getGridX(range, pillar);
    int grid_y = getGridY(range, pillar);
    int grid_size = grid_x * grid_y;
    int num_points = coords.size();
    PillarScatterResult result; 

    PillarCoord *d_coords;
    cudaMalloc((void **)&d_coords, num_points * sizeof(PillarCoord));
    int *d_key_to_pillar;
    cudaMalloc((void **)&d_key_to_pillar, grid_size * sizeof(int));
    PillarInfo *d_pillars;
    cudaMalloc((void **)&d_pillars, pillar.max_pillars* sizeof(PillarInfo));
    int *d_point_to_pillar;
    cudaMalloc((void **)&d_point_to_pillar, num_points * sizeof(int));
    int *d_num_pillars;
    cudaMalloc((void **)&d_num_pillars,  sizeof(int));

    cudaMemset(d_key_to_pillar, 0xff, grid_size * sizeof(int));
    cudaMemset(d_num_pillars, 0, sizeof(int));

    cudaMemcpy(d_coords, coords.data(), num_points * sizeof(PillarCoord), cudaMemcpyHostToDevice); 


    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;
    pillarScatterKernel<<<blocks, threads>>>(d_coords, num_points, grid_x, grid_size, pillar.max_pillars, d_key_to_pillar, d_pillars, d_point_to_pillar, d_num_pillars);
    
    cudaError_t err = cudaGetLastError(); 
    if ( err != cudaSuccess )
        throw std::runtime_error(cudaGetErrorString(err));
    err = cudaDeviceSynchronize(); 
    if ( err != cudaSuccess )
        throw std::runtime_error(cudaGetErrorString(err));

    int h_num_pillars = 0;
    cudaMemcpy(&h_num_pillars, d_num_pillars, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_num_pillars > pillar.max_pillars)
    {
        h_num_pillars = pillar.max_pillars;
    }

    result.pillars.resize(h_num_pillars);
    result.point_to_pillar.resize(num_points);
    cudaMemcpy(result.pillars.data(), d_pillars, h_num_pillars * sizeof(PillarInfo), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.point_to_pillar.data(), d_point_to_pillar, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_coords);
    cudaFree(d_key_to_pillar);
    cudaFree(d_num_pillars);
    cudaFree(d_pillars);
    cudaFree(d_point_to_pillar); 

    return result;
}
