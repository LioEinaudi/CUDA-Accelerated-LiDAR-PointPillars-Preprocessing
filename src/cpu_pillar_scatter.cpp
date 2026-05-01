#include"cpu_pillar_scatter.hpp"
#include<unordered_map> 
int makePillarKey ( const PillarCoord &coord ,int grid_x ) {
    return coord.y * grid_x + coord.x; 
}

PillarScatterResult cpuPillarScatter(const std ::vector<PointXYZI> &points, const RangeConfig &range, const PillarConfig &pillar){

    int grid_x = getGridX(range, pillar);
    
    std::unordered_map<int, int> key_to_pillar;
    
    PillarScatterResult result;
    
    result.point_to_pillar.resize(points.size()); 
    
    for ( size_t i = 0 ; i < points.size( ) ; ++i ) {
        PillarCoord coord = computePillarCoord(points[i], range, pillar);

        int key = makePillarKey ( coord , grid_x ) ; 
        auto it = key_to_pillar.find(key) ;
        int pillar_index; 

        if ( it == key_to_pillar.end()) {
            pillar_index = result.pillars.size();
            key_to_pillar[key] = pillar_index;
            result.pillars.push_back({coord, 1});
        }
        else {
            pillar_index = it->second;
            result.pillars[pillar_index].point_count += 1; 
        }
        result.point_to_pillar[i]=pillar_index; 
    }

    return result; 
}