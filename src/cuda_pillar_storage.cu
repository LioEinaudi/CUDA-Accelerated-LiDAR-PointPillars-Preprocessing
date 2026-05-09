#include"cuda_preprocess.cuh"
#include<cuda_runtime.h>
#include<stdexcept>
#include<vector>

__global__ void buildPillarPointStorageKernel(
    const PointXYZI*points ,
    const int * point_to_pillar,
    int num_points , 
    int max_points_per_pillar , 
    PointXYZI * pillar_points , 
    int * raw_point_count,
    int * pillar_point_count 
){
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if ( idx >= num_points )
        return;

    int pillar_index = point_to_pillar[idx]; 
    if ( pillar_index < 0 )
        return;

    int offset = atomicAdd(&raw_point_count[pillar_index], 1); 

    if ( offset >= max_points_per_pillar)
        return;

    int out_index = pillar_index * max_points_per_pillar + offset;
    pillar_points[out_index] = points[idx];
    atomicAdd(&pillar_point_count[pillar_index], 1); 
}

PillarPointStorage cudaBuildPillarPointStorage(
    const std::vector<PointXYZI> &points,
    const PillarScatterResult &scatter,
    const PillarConfig &pillar){
    if ( scatter.pillars.empty())    
        return {};

    int num_pillars = scatter.pillars.size();
    
    PointXYZI *d_points;
    cudaMalloc((void**)&d_points , sizeof (PointXYZI) * points.size()) ;

    int *d_point_to_pillar;
    cudaMalloc((void**)&d_point_to_pillar,sizeof(int) * points.size() ) ;

    PointXYZI *d_pillar_points;
    cudaMalloc((void **)&d_pillar_points, sizeof(PointXYZI) * num_pillars*pillar.max_points_per_pillar);

    int *d_raw_point_count;
    cudaMalloc((void **)&d_raw_point_count, sizeof(int)*num_pillars);

    int *d_pillar_point_count;
    cudaMalloc((void **)&d_pillar_point_count, num_pillars*sizeof(int));

    cudaMemset(d_raw_point_count, 0, sizeof(int) * num_pillars);
    cudaMemset(d_pillar_point_count, 0, num_pillars * sizeof(int));

    cudaMemcpy(d_points, points.data(), sizeof(PointXYZI) * points.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_point_to_pillar, scatter.point_to_pillar.data(), sizeof(int) * points.size() , cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (points.size() + threads - 1) / threads;
    buildPillarPointStorageKernel<<<blocks, threads>>>(d_points,d_point_to_pillar,points.size(),pillar.max_points_per_pillar,d_pillar_points,d_raw_point_count,d_pillar_point_count);

    cudaError_t err = cudaGetLastError(); 
    if ( err != cudaSuccess )
        throw std::runtime_error(cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess )
        throw std::runtime_error(cudaGetErrorString(err));
    
    PillarPointStorage storage;
    storage.num_pillar = num_pillars;
    storage.pillar_points.resize(num_pillars * pillar.max_points_per_pillar);
    storage.pillar_point_count.resize(num_pillars);

    cudaMemcpy(storage.pillar_points.data(), d_pillar_points, sizeof(PointXYZI) * num_pillars * pillar.max_points_per_pillar, cudaMemcpyDeviceToHost);
    
    cudaMemcpy(storage.pillar_point_count.data(), d_pillar_point_count, sizeof(int) * num_pillars, cudaMemcpyDeviceToHost);
    
    for ( const auto &pillar_info :scatter.pillars )
        storage.pillar_coords.push_back(pillar_info.coord);

    cudaFree(d_pillar_point_count);
    cudaFree(d_pillar_points);
    cudaFree(d_point_to_pillar);
    cudaFree(d_points);
    cudaFree(d_raw_point_count); 
    return storage;
}