#include"cpu_pillar_storage.hpp"
#include<vector>

int getPillarPointIndex( int pillar_index , int point_offset , const PillarConfig & pillar ) {
    return pillar_index * pillar.max_points_per_pillar + point_offset; 
}

PillarPointStorage cpuBuildPillarPointStorage ( 
    const std::vector<PointXYZI> &points ,
    const PillarScatterResult & scatter , 
    const PillarConfig & pillar 
){
    PillarPointStorage storage;
    int num_pillar = scatter.pillars.size();
    
    storage.num_pillar = num_pillar; 
    storage.pillar_point_count.resize(num_pillar,0);
    storage.pillar_points.resize(num_pillar * pillar.max_points_per_pillar);
    for ( auto & pillarinfo : scatter.pillars ) {
        storage.pillar_coords.push_back(pillarinfo.coord); 
    }

    for (int i = 0; i < scatter.point_to_pillar.size(); i ++ ) {
        int pillar_index = scatter.point_to_pillar[i];
        int offset = storage.pillar_point_count[pillar_index]; 

        if ( offset< pillar.max_points_per_pillar) {
            int index = getPillarPointIndex(pillar_index, offset, pillar);
            storage.pillar_points[index] = points[i];
            storage.pillar_point_count[pillar_index] += 1;
        }
    }
    return storage; 
}