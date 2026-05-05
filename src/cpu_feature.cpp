#include"cpu_feature.hpp"

int getFeatureIndex(
    int pillar_index,
    int point_offset,
    int feature_index,
    int max_points_per_pillar,
    int feature_dim)
{
    return (pillar_index * max_points_per_pillar + point_offset) * feature_dim + feature_index;
}

PillarFeatureTensor cpuGeneratePillarFeatures(
    const PillarPointStorage &storage,
    const RangeConfig &range,
    const PillarConfig &pillar){
    PillarFeatureTensor tensor;
    tensor.num_pillars = storage.num_pillar;
    tensor.max_points_per_pillar = pillar.max_points_per_pillar;
    tensor.feature_dim = kPillarFeatureDim;

    tensor.features.resize(kPillarFeatureDim * storage.num_pillar*pillar.max_points_per_pillar,0.0f)   ;

    for (int pillar_index = 0; pillar_index < storage.num_pillar; pillar_index ++ ) {
        int count = storage.pillar_point_count[pillar_index];
        if ( count == 0 )
            continue;
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
        for (int point_offset = 0; point_offset< count; point_offset ++ ){
            int point_index = getPillarPointIndex(pillar_index, point_offset, pillar);
            PointXYZI p = storage.pillar_points[point_index];
            sum_x += p.x;
            sum_y += p.y;
            sum_z += p.z; 
        }
        float mean_x = sum_x * 1.0f / count;
        float mean_y = sum_y * 1.0f / count;
        float mean_z = sum_z * 1.0f / count;

        PillarCoord coord = storage.pillar_coords[pillar_index];
        float center_x = range.min_x + (coord.x + 0.5f) * pillar.voxel_x;
        float center_y = range.min_y + (coord.y + 0.5f) * pillar.voxel_y;

        for (int point_offset = 0; point_offset < count; point_offset ++ ) {
            int point_index = getPillarPointIndex(pillar_index, point_offset, pillar);
            PointXYZI p = storage.pillar_points[point_index];
            std::vector<float> feature(9);
            feature[0] = p.x;
            feature[1] = p.y;
            feature[2] = p.z;
            feature[3] = p.intensity;
            feature[4] = p.x - mean_x;
            feature[5] = p.y - mean_y;
            feature[6] = p.z - mean_z;
            feature[7] = p.x - center_x;
            feature[8] = p.y - center_y ;
            for (int i = 0; i < kPillarFeatureDim; i ++ ) {
                int featureindex = getFeatureIndex(pillar_index, point_offset, i, pillar.max_points_per_pillar, kPillarFeatureDim);
                tensor.features[featureindex] = feature[i]; 
            }
        }
    }
    return tensor; 
}
