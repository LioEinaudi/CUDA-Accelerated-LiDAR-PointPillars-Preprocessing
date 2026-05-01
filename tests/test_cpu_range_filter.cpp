#include"point.hpp"
#include"config.hpp"
#include"cpu_preprocess.hpp"

#include<cassert> 
#include<iostream> 
#include<vector>

int main(){
    RangeConfig range ;
    PointXYZI valid{10.0f, 0.0f, 0.0f, 0.5f};
    PointXYZI x_small{-10.0f, 0.0f, 0.0f, 0.6f};
    PointXYZI x_big{100.0f, 0.0f, 0.0f, 0.6f};
    PointXYZI y_small{10.0f, -40.0f, 0.0f, 0.4f};
    PointXYZI y_big{10.0f, 100.0f, 0.0f, 0.3f};
    PointXYZI z_big{10.0f, 0.0f, 1.0f, 0.7f};
    PointXYZI z_small{10.0f, 0.0f, -4.0f, 0.3f};

    assert(isInRangeConfig(valid, range) == true);
    assert(isInRangeConfig(x_small, range) == false);
    assert(isInRangeConfig(x_big, range) == false);
    assert(isInRangeConfig(y_big, range) == false);
    assert(isInRangeConfig(y_small, range) == false);
    assert(isInRangeConfig(z_big, range) == false);
    assert(isInRangeConfig(z_small, range) == false);

    std::vector<PointXYZI> points = {
        valid, x_big, x_small, y_big, y_small, z_big, z_small, PointXYZI{20.0f, 1.0f, -1.0f, 0.7f}
    };
    auto filtered = cpuRangeFilter(points, range);
    assert(filtered.size() == 2);

    std::cout << "test_cpu_range_filter passed"<<std::endl;
    return 0;
}