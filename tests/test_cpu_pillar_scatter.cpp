#include"cpu_pillar_scatter.hpp"
#include"config.hpp"
#include"point.hpp"

#include<cassert>
#include<iostream>
#include<vector>

int main() {
    RangeConfig range ;
    PillarConfig pillar; 
    assert(makePillarKey(PillarCoord{1, 2}, 10) == 21);

    std::vector<PointXYZI> points = {
        {0.01f, -39.67f, -1.0f, 0.5f},
        {0.02f, -39.66f, -1.0f, 0.6f},
        {0.20f, -39.67f, -1.0f, 0.7f}};

    PillarScatterResult result = cpuPillarScatter(points, range, pillar);
    
    assert(result.pillars.size() == 2);
    assert(result.pillars[0].point_count == 2);
    assert(result.pillars[1].point_count == 1);

    assert(result.point_to_pillar.size() == 3);
    assert(result.point_to_pillar[0] == 0);
    assert(result.point_to_pillar[1] == 0);
    assert(result.point_to_pillar[2] == 1);

    std::cout << "test_cpu_pillar_scatter passed " << std ::endl;

    return 0; 
}