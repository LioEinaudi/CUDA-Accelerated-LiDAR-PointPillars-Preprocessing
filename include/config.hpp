#pragma once 
struct RangeConfig{
    float min_x = 0.0f;
    float max_x = 69.12f;

    float min_y = -39.68f;
    float max_y = 39.68f;

    float min_z = -3.0f;
    float max_z = 1.0f;
}; 

struct PillarConfig {
    float voxel_x = 0.16f;
    float voxel_y = 0.16f;
    float voxel_z = 4.0f;

    int max_pillars = 12000;
    int max_points_per_pillar = 100;
};