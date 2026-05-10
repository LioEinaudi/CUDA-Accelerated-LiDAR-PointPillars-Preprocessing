#include"cuda_preprocess.cuh"
#include<stdexcept>
#include<vector>
#include<cuda_runtime.h>

__global__  void buildBevPseudoImageKernel(
    const float * features , 
    const PillarCoord * pillar_coords , 
    const int * pillar_point_count ,
    int num_pillars , 
    int max_points_per_pillar,
    int feature_dim , 
    int height,
    int width,
    float * bev 
){
    unsigned int linear_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = num_pillars * feature_dim; 
    if ( linear_idx >= total ) 
        return ;

    int pillar_index = linear_idx / feature_dim;
    int channel = linear_idx % feature_dim;

    int count = pillar_point_count[pillar_index];
    if (count == 0)
        return;

    float sum = 0.0f;
    for (int point_offset = 0; point_offset < count; point_offset ++ ) {
        int feature_index = (pillar_index * max_points_per_pillar + point_offset) * feature_dim + channel;
        sum += features[feature_index]; 
    }
    float mean = sum / count;

    PillarCoord coord = pillar_coords[pillar_index];
    int bev_index = (channel * height + coord.y) * width + coord.x;
    bev[bev_index] = mean; 
}

BevPseudoImage cudaBuildBevPseudoImage(
    const PillarFeatureTensor &features,
    const PillarPointStorage &storage,
    const RangeConfig &range,
    const PillarConfig &pillar){
    if(features.features.empty())
        return {};
    int width = getGridX(range, pillar);
    int height = getGridY(range, pillar);
    int channels = features.feature_dim;
    BevPseudoImage bev;
    bev.channels = channels;
    bev.height = height;
    bev.width = width;
    bev.data.resize(channels * width * height,0.0f); 

    size_t feature_bytes = features.features.size() * sizeof(float);
    size_t coord_bytes = storage.pillar_coords.size() * sizeof(PillarCoord);
    size_t count_bytes = storage.pillar_point_count.size() * sizeof(int);
    size_t bev_bytes = bev.data.size() * sizeof(float);

    float *d_features;
    cudaMalloc((void **)&d_features, feature_bytes);

    PillarCoord *d_pillar_coords;
    cudaMalloc((void **)&d_pillar_coords, coord_bytes);

    int *d_pillar_point_count;
    cudaMalloc((void **)&d_pillar_point_count, count_bytes);

    float *d_bev;
    cudaMalloc((void **)&d_bev, bev_bytes);

    cudaMemset(d_bev, 0, bev_bytes);

    cudaMemcpy(d_pillar_coords, storage.pillar_coords.data(), coord_bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_features, features.features.data(), feature_bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_pillar_point_count, storage.pillar_point_count.data(), count_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int total = features.num_pillars * features.feature_dim;
    int blocks = (total + threads - 1) / threads;

    buildBevPseudoImageKernel<<<blocks, threads>>>(d_features, d_pillar_coords, d_pillar_point_count, features.num_pillars, pillar.max_points_per_pillar, features.feature_dim, height, width, d_bev);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    cudaMemcpy(bev.data.data(), d_bev, bev_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_bev);
    cudaFree(d_features);
    cudaFree(d_pillar_coords);
    cudaFree(d_pillar_point_count);
    return bev; 
}
