#include"cpu_bev.hpp"

int getBevIndex(
    int channel,
    int y,
    int x,
    int height,
    int width){
    return (channel * height + y) * width + x; 
}

BevPseudoImage cpuBuildBevPseudoImage(
    const PillarFeatureTensor &features,
    const PillarPointStorage &storage,
    const RangeConfig &range,
    const PillarConfig &pillar){

    BevPseudoImage bev; 

    int width = getGridX(range, pillar);
    int height = getGridY(range, pillar);
    int channels = features.feature_dim;

    int data_size = channels * width * height;
    bev.channels = channels;
    bev.height = height;
    bev.width = width;
    bev.data.resize(data_size, 0.0f);

    for (int pillar_index = 0; pillar_index < storage.num_pillar; pillar_index ++ ){
        PillarCoord coord = storage.pillar_coords[pillar_index];
        int count = storage.pillar_point_count[pillar_index]; 

        if ( count == 0 )
            continue;

        for (int channel = 0; channel < features.feature_dim; channel ++ ) {
        
            float sum = 0.0f;
            for (int point_offset = 0; point_offset < count; point_offset ++ ) {
                int features_index = getFeatureIndex(pillar_index, point_offset, channel, pillar.max_points_per_pillar, features.feature_dim);
                sum += features.features[features_index]; 
            }
            float mean = sum / count;
            int bev_index = getBevIndex(channel, coord.y, coord.x, height, width);

            bev.data[bev_index] = mean; 
        }
    }
    return bev; 
}
