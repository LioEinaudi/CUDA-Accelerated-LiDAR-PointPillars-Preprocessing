#include"config.hpp" 
#include"cpu_preprocess.hpp"
#include"kitti_reader.hpp"

#include<iostream>
#include<string>
#include<exception>

int main(int argc , char ** argv ) {
    if ( argc < 2 )
    {
        std::cerr << "Usage: ./pointpillars_preprocess path/to/000000.bin\n";
        return 1;
    }
    std :: string bin_path = argv[1];
    RangeConfig range;
    try{

        std :: vector<PointXYZI> points = readKittiBin(bin_path);

        std ::vector<PointXYZI> point_after_filter = cpuRangeFilter(points, range);
        
        std::cout << "Loaded Points: " << points.size() << std::endl;
        std::cout << "Filtered Points: " << point_after_filter.size() << std ::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
        
    }return 0;
}