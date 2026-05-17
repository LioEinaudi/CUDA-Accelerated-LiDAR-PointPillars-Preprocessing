#pragma once
#include"point.hpp"
#include"cpu_preprocess.hpp"
#include"config.hpp"
#include"cpu_pillar.hpp"
#include"cpu_pillar_scatter.hpp"
#include"cpu_pillar_storage.hpp"
#include"cpu_feature.hpp"
#include"cuda_pipeline.cuh"
#include<vector>
struct CudaPreprocessWorkspace{
    int max_points;
    int max_pillars;
    int max_points_per_pillar;
    int grid_x;
    int grid_y;
    int grid_size;


    PointXYZI *d_input_points;
    int *d_valid_flags;
    int *d_prefix_sum;
    PointXYZI *d_filtered_points;
    PillarCoord *d_coords;

    int *d_key_to_pillar;
    PillarInfo *d_pillars;
    int *d_point_to_pillar;
    int *d_num_pillars;

    PointXYZI *d_pillar_points;
    int *d_raw_point_count;
    int *d_pillar_point_count;

    float *d_mean_x;
    float *d_mean_y;
    float *d_mean_z;
    float *d_features;
    float *d_bev;
};
CudaPreprocessWorkspace createCudaPreprocessWorkspace(
    int max_points,
    const RangeConfig &range,
    const PillarConfig &pillar);

CudaPipelineSummary cudaPreprocessPipelineV4Workspace(
    const std::vector<PointXYZI> &points,
    const RangeConfig &range,
    const PillarConfig &pillar,
    CudaPreprocessWorkspace &workspace);

void destroyCudaPreprocessWorkspace(
    CudaPreprocessWorkspace &workspace);
