#include"cpu_pillar.hpp"
#include<cmath>

int getGridX ( const RangeConfig & range , const PillarConfig & pillar ) {
    return static_cast<int>(std::round( (range.max_x - range.min_x  )/ pillar.voxel_x)); 
}

int getGridY( const RangeConfig & range , const PillarConfig & pillar ) {
    return static_cast<int>(std::round((range.max_y - range.min_y) / pillar.voxel_y));
}

PillarCoord computePillarCoord ( const PointXYZI & point , const RangeConfig & range , const PillarConfig &pillar) {
    int pillar_x = static_cast<int>(std::floor((point.x - range.min_x) / pillar.voxel_x));
    int pillar_y = static_cast<int>( std::floor((point.y - range.min_y) / pillar.voxel_y));
    return PillarCoord{pillar_x, pillar_y}; 
}