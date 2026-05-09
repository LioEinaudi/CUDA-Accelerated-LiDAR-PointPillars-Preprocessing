#include"cuda_preprocess.cuh"
#include<stdexcept>
#include<vector>

__global__ void computePillarMeanKernel(
    const PointXYZI* pillar_points , 
    const int * pillar_point_count ,
    int num_pillars ,
    int max_points_per_pillar ,
    float * mean_x , 
    float *mean_y , 
    float * mean_z
){
    unsigned int pillar_index = blockDim.x * blockIdx.x + threadIdx.x; 
    if ( pillar_index >= num_pillars )
        return;

    int count = pillar_point_count[pillar_index]; 

    if ( count == 0 ) {
        mean_x[pillar_index] = 0.0f;
        mean_y[pillar_index] = 0.0f;
        mean_z[pillar_index] = 0.0f;
        return; 
    }

    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;

    for (int point_offset = 0; point_offset < count; point_offset ++ ) {
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
__global__ void generatePillarFeatureKernel(
    const PointXYZI *pillar_points,
    const int *pillar_point_count,
    const PillarCoord *pillar_coords,
    const float *mean_x,
    const float *mean_y,
    const float *mean_z,
    int num_pillars ,
    RangeConfig range ,
    PillarConfig pillar , 
    float * features 
){
    unsigned int linear_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = num_pillars * pillar.max_points_per_pillar; 

    if ( linear_idx >= total)
        return;

    int pillar_index = linear_idx / pillar.max_points_per_pillar;
    int point_offset = linear_idx % pillar.max_points_per_pillar;

    int count = pillar_point_count[pillar_index]; 
    if ( point_offset >= count )
        return;

    PillarCoord coord = pillar_coords[pillar_index];
    float center_x = range.min_x + (coord.x + 0.5f) * pillar.voxel_x;
    float center_y = range.min_y + (coord.y + 0.5f) * pillar.voxel_y;

    int point_index = pillar_index * pillar.max_points_per_pillar + point_offset;
    PointXYZI p = pillar_points[point_index];

    int base = (pillar_index * pillar.max_points_per_pillar + point_offset) * kPillarFeatureDim;

    features[base + 0] = p.x;
    features[base + 1] = p.y;
    features[base + 2] = p.z;
    features[base + 3] = p.intensity;
    features[base + 4] = p.x-mean_x[pillar_index];
    features[base + 5] = p.y-mean_y[pillar_index];
    features[base + 6] = p.z-mean_z[pillar_index];
    features[base + 7] = p.x-center_x;
    features[base + 8] = p.y - center_y; 
}

PillarFeatureTensor cudaGeneratePillarFeatures(
    const PillarPointStorage &storage,
    const RangeConfig &range,
    const PillarConfig &pillar){
        if (storage.pillar_coords.empty())
            return {};

        PillarFeatureTensor tensor;
        tensor.num_pillars = storage.num_pillar;
        tensor.max_points_per_pillar = pillar.max_points_per_pillar;
        tensor.feature_dim = kPillarFeatureDim;
        tensor.features.resize(tensor.num_pillars * tensor.max_points_per_pillar * tensor.feature_dim, 0.0f); 

        PointXYZI *d_pillar_points;
        cudaMalloc((void **)&d_pillar_points, storage.pillar_points.size() * sizeof(PointXYZI));

        int *d_pillar_point_count;
        cudaMalloc((void **)&d_pillar_point_count, sizeof(int) * storage.pillar_point_count.size());

        PillarCoord *d_pillar_coords;
        cudaMalloc((void **)&d_pillar_coords, sizeof(PillarCoord) * storage.pillar_coords.size());

        float *d_mean_x, *d_mean_y, *d_mean_z;
        cudaMalloc((void **)&d_mean_x, sizeof(float) * tensor.num_pillars);
        cudaMalloc((void **)&d_mean_y, sizeof(float) * tensor.num_pillars);
        cudaMalloc((void **)&d_mean_z, sizeof(float) * tensor.num_pillars);

        float *d_features;
        cudaMalloc((void **)&d_features,tensor.max_points_per_pillar*tensor.feature_dim * tensor.num_pillars * sizeof(float));

        cudaMemset(d_features, 0, tensor.feature_dim * tensor.num_pillars*tensor.max_points_per_pillar*sizeof(float ));

        cudaMemcpy(d_pillar_coords, storage.pillar_coords.data(), storage.pillar_coords.size() * sizeof(PillarCoord), cudaMemcpyHostToDevice);

        cudaMemcpy(d_pillar_point_count, storage.pillar_point_count.data(), sizeof(int) * storage.pillar_point_count.size(), cudaMemcpyHostToDevice);

        cudaMemcpy(d_pillar_points, storage.pillar_points.data(), sizeof(PointXYZI) * storage.pillar_points.size(), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (tensor.num_pillars + threads - 1) / threads;

        computePillarMeanKernel<<<blocks,threads >>>(d_pillar_points, d_pillar_point_count, tensor.num_pillars, tensor.max_points_per_pillar, d_mean_x, d_mean_y, d_mean_z);

        cudaError_t err = cudaGetLastError(); 
        if ( err != cudaSuccess )
            throw std::runtime_error(cudaGetErrorString(err));

        err = cudaDeviceSynchronize(); 
        if ( err != cudaSuccess )
            throw std::runtime_error(cudaGetErrorString(err));

        int total_feature_points = tensor.num_pillars * tensor.max_points_per_pillar;
        int feature_blocks = (total_feature_points + threads - 1) / threads;
        generatePillarFeatureKernel<<<feature_blocks, threads>>>(d_pillar_points, d_pillar_point_count, d_pillar_coords, d_mean_x, d_mean_y, d_mean_z, tensor.num_pillars, range, pillar, d_features);

        err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));

        cudaMemcpy(tensor.features.data(), d_features, sizeof(float) * tensor.feature_dim * tensor.num_pillars*tensor.max_points_per_pillar, cudaMemcpyDeviceToHost);
        cudaFree(d_features);
        cudaFree(d_mean_x);
        cudaFree(d_mean_y);
        cudaFree(d_mean_z);
        cudaFree(d_pillar_coords);
        cudaFree(d_pillar_point_count);
        cudaFree(d_pillar_points);
        return  tensor; 
}
