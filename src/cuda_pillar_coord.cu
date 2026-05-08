#include"cuda_preprocess.cuh"
#include<vector>
#include<string>
#include<stdexcept>
#include<cuda_runtime.h>
#include<cmath>
__global__ void computePillarCoordKernel (
    const PointXYZI*points , 
    int num_points , 
    RangeConfig range ,
    PillarConfig pillar ,
    PillarCoord * coords 
){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    if ( idx >= num_points )
        return;

    PointXYZI point = points[idx];

    int pillar_x = static_cast<int>(floorf((point.x - range.min_x) / pillar.voxel_x));

    int pillar_y = static_cast<int>(floorf((point.y - range.min_y) / pillar.voxel_y)); 

    coords[idx] = PillarCoord{pillar_x,pillar_y} ; 
}

std::vector<PillarCoord>cudaComputePillarCoords(
    const std::vector<PointXYZI>&points,
    const RangeConfig &range, 
    const PillarConfig & pillar 
){
    if ( points.empty())
        return {};

    
    int num_points = static_cast<int>(points.size());
    size_t points_bytes = points.size() * sizeof(PointXYZI);
    size_t coord_bytes = points.size() * sizeof(PillarCoord);
    std::vector<PillarCoord> coords(num_points); 

    PointXYZI *d_points;
    PillarCoord * d_coords ;

    cudaMalloc((void **)&d_points, points_bytes);
    cudaMalloc((void**) & d_coords , coord_bytes ) ;

    cudaMemcpy(d_points, points.data(), points_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads; 

    computePillarCoordKernel <<< blocks , threads>>>(d_points,num_points,range,pillar,d_coords ) ;
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
        throw std::runtime_error(cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    if ( err != cudaSuccess )
        throw std::runtime_error(cudaGetErrorString(err));

    cudaMemcpy(coords.data(), d_coords, coord_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_coords);
    cudaFree(d_points);
    return coords; 
}