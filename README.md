# CUDA-Accelerated LiDAR PointPillars Preprocessing

This project is a step-by-step CUDA learning project for the LiDAR preprocessing stage used by PointPillars-style 3D object detection pipelines.

The current implementation focuses on the first CPU baseline:

- Define a KITTI point type: `PointXYZI`
- Load KITTI-style `.bin` point cloud files
- Apply a CPU range filter
- Compute CPU pillar grid dimensions and point-to-pillar coordinates
- Print the original and filtered point counts
- Test CPU range-filter boundary behavior with hand-written points
- Test CPU pillar coordinate behavior with hand-written points

The neural network part of PointPillars is intentionally not included yet. The goal is to first understand and accelerate the preprocessing pipeline.

## Current Pipeline

```text
KITTI .bin
  -> readKittiBin
  -> std::vector<PointXYZI>
  -> cpuRangeFilter
  -> filtered points
  -> computePillarCoord
  -> pillar coordinates
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

## CPU Pillar Coordinates

The current CPU pillar utilities map filtered points from continuous metric coordinates into discrete BEV grid coordinates:

```text
pillar_x = floor((x - min_x) / voxel_x)
pillar_y = floor((y - min_y) / voxel_y)
```

With the default KITTI/PointPillars-style settings:

```text
x range: 0.0 to 69.12, voxel_x: 0.16 -> grid_x = 432
y range: -39.68 to 39.68, voxel_y: 0.16 -> grid_y = 496
```

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
build/test_cpu_pillar
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

The current test targets use hand-written points instead of reading KITTI files.

`test_cpu_range_filter` verifies in-range points, out-of-range points, and exclusive upper-bound behavior such as `x == max_x`.

`test_cpu_pillar` verifies default grid dimensions and point-to-pillar coordinate mapping.

Build and run:

```bash
cmake --build build
./build/test_cpu_range_filter
./build/test_cpu_pillar
```

Expected output:

```text
test_cpu_range_filter passed
test_cpu_pillar passed
```

## Repository Layout

```text
.
├── CMakeLists.txt
├── README.md
├── include/
│   ├── config.hpp
│   ├── cpu_pillar.hpp
│   ├── cpu_preprocess.hpp
│   ├── kitti_reader.hpp
│   └── point.hpp
├── src/
│   ├── cpu_pillar.cpp
│   ├── cpu_range_filter.cpp
│   ├── kitti_reader.cpp
│   └── main.cpp
├── tests/
│   ├── test_cpu_pillar.cpp
│   └── test_cpu_range_filter.cpp
└── data/
    └── 000000.bin
```

Generated build files, local learning notes, and binary point cloud data are not committed.

## Next Steps

Planned preprocessing stages:

1. CPU pillar scatter / grouping
2. CUDA range filter with atomic compaction
3. CUDA range filter with prefix-sum compaction
4. Pillar scatter / hash
5. Per-pillar point counting
6. Pillar feature generation
7. BEV pseudo-image generation
8. CPU vs CUDA benchmark and Nsight Compute analysis
