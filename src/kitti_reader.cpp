#include"kitti_reader.hpp"

#include<fstream>
#include<stdexcept>

std :: vector < PointXYZI > readKittiBin ( const std :: string& path ){
    std ::ifstream file(path, std ::ios ::binary | std ::ios ::ate);

    if ( !file ) {
        throw std ::runtime_error("Failed to open KITTI bin file: " + path);
    }
    
    const std ::streamsize bytes = file.tellg();

    file.seekg(0, std ::ios ::beg); 

    if ( bytes % ( 4 * sizeof ( float )) != 0 ) {
        throw std::runtime_error("Invalid KITTI bin size: " + path); 
    }

    const size_t num_points = bytes / (4 * sizeof(float));

    std ::vector<PointXYZI> points(num_points);

    file.read(reinterpret_cast<char *>(points.data()),bytes); 

    if ( !file ){
        throw std ::runtime_error("Failed to read KITTI bin file: " + path); 
    }

    return points; 
}