#include"cpu_preprocess.hpp"

bool isInRangeConfig(const PointXYZI&point , const RangeConfig & range ) {
    return point.x >= range.min_x && point.x < range.max_x && point.y >= range.min_y && point.y < range.max_y && point.z >= range.min_z && point.z < range.max_z; 
}

std::vector< PointXYZI > cpuRangeFilter ( const std :: vector < PointXYZI> & points , const RangeConfig & range ) {
    std::vector<PointXYZI> pointreturn;
    pointreturn.reserve(points.size());
    for ( auto & p : points ) {
        if ( isInRangeConfig ( p , range ) )
            pointreturn.push_back(p); 
    }
    return pointreturn; 
}