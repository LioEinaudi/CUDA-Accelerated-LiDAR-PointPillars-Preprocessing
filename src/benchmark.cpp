#include"config.hpp"
#include"point.hpp"
#include"kitti_reader.hpp"
#include"cpu_bev.hpp"
#include"cpu_feature.hpp"
#include"cpu_pillar_scatter.hpp"
#include"cpu_pillar.hpp"
#include"cpu_pillar_storage.hpp"
#include"cpu_preprocess.hpp"
#include"cuda_preprocess.cuh"

#include<iostream>
#include<chrono>
#include<exception>
#include<string>

double elaspsedMs(
    std::chrono::high_resolution_clock::time_point start,
    std::chrono::high_resolution_clock::time_point end
){
    return std::chrono::duration<double, std::milli>(end - start).count();
}
int main(int argc , char**argv){
    if (argc < 2)
    {
        std::cerr << "Usage: ./pointpillars_preprocess path/to/000000.bin\n";
        return 1;
    }
    std ::string bin_path = argv[1];
    RangeConfig range;
    PillarConfig pillar;

    pillar.max_pillars = 20000;

    auto points = readKittiBin(bin_path);
    std::cout << "Loaded points: " << points.size() << std::endl;

    std::cout << "[CPU] range filter..." << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto cpu_filtered = cpuRangeFilter(points, range);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_filter_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;

    std::cout << "[CPU] scatter..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cpu_scatter = cpuPillarScatter(cpu_filtered, range , pillar );
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_scatter_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;

    std::cout << "[CPU] storage..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cpu_storage = cpuBuildPillarPointStorage(cpu_filtered, cpu_scatter, pillar);
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_storage_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;

    std::cout << "[CPU] feature..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cpu_features = cpuGeneratePillarFeatures(cpu_storage, range, pillar);
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_features_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;

    std::cout << "[CPU] bev..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cpu_bev = cpuBuildBevPseudoImage(cpu_features, cpu_storage, range, pillar); 
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_bev_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;


    std::cout << "[CUDA] warmup..." << std::flush;
    auto warmup = cudaRangeFilterPrefixSum(points, range);
    std::cout << " done" << std::endl;

    std::cout << "[CUDA] range filter..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cuda_filtered = cudaRangeFilterPrefixSum(points, range);
    t1 = std::chrono::high_resolution_clock::now();
    double cuda_filter_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;
    
    std::cout << "[CUDA] coord..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cuda_coords = cudaComputePillarCoords(cuda_filtered, range, pillar); 
    t1 = std::chrono::high_resolution_clock::now();
    double cuda_coords_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;

    std::cout << "[CUDA] scatter..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cuda_scatter = cudaPillarScatter(cuda_coords, range, pillar); 
    t1 = std::chrono::high_resolution_clock::now();
    double cuda_scatter_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;

    std::cout << "[CUDA] storage..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cuda_storage= cudaBuildPillarPointStorage(cuda_filtered,cuda_scatter,pillar) ;
    t1 = std::chrono::high_resolution_clock::now();
    double cuda_storage_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;

    std::cout << "[CUDA] feature..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cuda_features = cudaGeneratePillarFeatures(cuda_storage, range, pillar); 
    t1 = std::chrono::high_resolution_clock::now();
    double cuda_features_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;

    std::cout << "[CUDA] bev..." << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto cuda_bev = cudaBuildBevPseudoImage(cuda_features, cuda_storage, range, pillar); 
    t1 = std::chrono::high_resolution_clock::now();
    double cuda_bev_ms = elaspsedMs(t0, t1);
    std::cout << " done" << std::endl;

    std::cout << "CPU filtered: " << cpu_filtered.size() << std::endl;
    std::cout << "CUDA filtered: " << cuda_filtered.size() << std::endl;
    std::cout << "CPU pillars: " << cpu_storage.num_pillar << std::endl;
    std::cout << "CUDA pillars: " << cuda_storage.num_pillar << std::endl;
    std::cout << "CPU BEV size: " << cpu_bev.data.size() << std::endl;
    std::cout << "CUDA BEV size: " << cuda_bev.data.size() << std::endl;

    std::cout << "Points size:" << points.size() << std::endl;
    
    std::cout << "CPU pipeline:" << std::endl;
    std::cout << "range filter:" << cpu_filter_ms << " ms" << std::endl;
    std::cout << "scatter:" << cpu_scatter_ms << " ms" << std::endl;
    std::cout << "storage:" << cpu_storage_ms << " ms" << std::endl;
    std::cout << "feature:" << cpu_features_ms << " ms" << std::endl;
    std::cout << "bev:" << cpu_bev_ms << " ms" << std::endl;
    std::cout << "total:" << cpu_filter_ms +cpu_bev_ms+cpu_scatter_ms+cpu_storage_ms+cpu_features_ms<< " ms" << std::endl<<std::endl ;

    std::cout << "CUDA pipeline:" << std::endl;
    std::cout << "range filter:" << cuda_filter_ms << " ms" << std::endl;
    std::cout << "coord:" << cuda_coords_ms << " ms" << std::endl;
    std::cout << "scatter:" << cuda_scatter_ms << " ms" << std::endl;
    std::cout << "storage:" << cuda_storage_ms << " ms" << std::endl;
    std::cout << "feature:" << cuda_features_ms << " ms" << std::endl;
    std::cout << "bev:" << cuda_bev_ms << " ms" << std::endl;
    std::cout << "total:" << cuda_filter_ms + cuda_bev_ms + cuda_scatter_ms + cuda_storage_ms + cuda_features_ms +cuda_coords_ms<< " ms" << std::endl
              << std::endl;

    return 0; 
}
