# CUDA-Accelerated LiDAR PointPillars Preprocessing

This project is a step-by-step CUDA learning project for the LiDAR preprocessing stage used by PointPillars-style 3D object detection pipelines.

The current implementation focuses on the first CPU baseline:

- Define a KITTI point type: `PointXYZI`
- Load KITTI-style `.bin` point cloud files
- Apply a CPU range filter
- Print the original and filtered point counts
- Test CPU range-filter boundary behavior with hand-written points

The neural network part of PointPillars is intentionally not included yet. The goal is to first understand and accelerate the preprocessing pipeline.

## Current Pipeline

```text
KITTI .bin
  -> readKittiBin
  -> std::vector<PointXYZI>
  -> cpuRangeFilter
  -> filtered points
  -> point count summary
```

## KITTI Point Format

KITTI Velodyne `.bin` files store point clouds as a flat binary array of `float32` values:

```text
x y z intensity x y z intensity ...
```

Each point is represented as:

```cpp
struct PointXYZI {
    float x;
    float y;
    float z;
    float intensity;
};
```

## Default Range Filter

The current CPU range filter keeps points inside the following region:

```text
x: [0.0, 69.12)
y: [-39.68, 39.68)
z: [-3.0, 1.0)
```

The upper bound is exclusive so that later pillar coordinate computation does not produce out-of-range grid indices.

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

This creates:

```text
build/pointpillars_preprocess
build/test_cpu_range_filter
```

## Run

Run the program with a KITTI-style `.bin` file:

```bash
./build/pointpillars_preprocess data/000000.bin
```

Example output:

```text
Loaded Points: 120000
Filtered Points: 37920
```

The exact filtered count depends on the input point cloud.

## Tests

The current test target verifies the CPU range filter with hand-written points instead of reading a KITTI file. It checks in-range points, out-of-range points, and exclusive upper-bound behavior such as `x == max_x`.

Build and run:

```bash
cmake --build build
./build/test_cpu_range_filter
```

Expected output:

```text
test_cpu_range_filter passed
```

## Repository Layout

```text
.
├── CMakeLists.txt
├── README.md
├── include/
│   ├── config.hpp
│   ├── cpu_preprocess.hpp
│   ├── kitti_reader.hpp
│   └── point.hpp
├── src/
│   ├── cpu_range_filter.cpp
│   ├── kitti_reader.cpp
│   └── main.cpp
├── tests/
│   └── test_cpu_range_filter.cpp
└── data/
    └── 000000.bin
```

Generated build files, local learning notes, and binary point cloud data are not committed.

## Next Steps

Planned preprocessing stages:

1. CPU pillar coordinate computation
2. CUDA range filter with atomic compaction
3. CUDA range filter with prefix-sum compaction
4. Pillar scatter / hash
5. Per-pillar point counting
6. Pillar feature generation
7. BEV pseudo-image generation
8. CPU vs CUDA benchmark and Nsight Compute analysis
