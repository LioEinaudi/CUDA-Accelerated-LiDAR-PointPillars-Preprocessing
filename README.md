# CUDA-Accelerated LiDAR PointPillars Preprocessing

This project is a step-by-step CUDA learning project for the LiDAR preprocessing stage used by PointPillars-style 3D object detection pipelines.

The current implementation focuses on the first CPU baseline:

- Define a KITTI point type: `PointXYZI`
- Load KITTI-style `.bin` point cloud files
- Apply a CPU range filter
- Compute CPU pillar grid dimensions and point-to-pillar coordinates
- Group filtered points into unique CPU pillars with point counts
- Store points in fixed-size per-pillar CPU buffers
- Generate 9D CPU pillar features
- Print the original and filtered point counts
- Test CPU range-filter boundary behavior with hand-written points
- Test CPU pillar coordinate behavior with hand-written points
- Test CPU pillar scatter/grouping behavior with hand-written points
- Test CPU pillar point storage and max-points truncation
- Test CPU pillar feature indexing, feature values, and padding

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
  -> cpuPillarScatter
  -> unique pillars and point_to_pillar mapping
  -> cpuBuildPillarPointStorage
  -> fixed per-pillar point buffers
  -> cpuGeneratePillarFeatures
  -> 9D per-point pillar features
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

## CPU Pillar Scatter

The CPU scatter step groups filtered points by their pillar coordinates. A 2D pillar coordinate is converted into a 1D key:

```text
key = pillar_y * grid_x + pillar_x
```

The CPU implementation uses an `std::unordered_map` to map each pillar key to a unique pillar index. It returns:

```text
pillars         unique non-empty pillars with point counts
point_to_pillar per-point mapping into the pillars array
```

## CPU Pillar Point Storage

The CPU storage step uses the scatter result to place points into a fixed-size per-pillar buffer:

```text
pillar_points[pillar_index][point_offset]
```

The implementation stores this logical 2D layout in a 1D contiguous vector:

```text
index = pillar_index * max_points_per_pillar + point_offset
```

It also tracks:

```text
pillar_point_count actual number of stored points per pillar
pillar_coords       BEV coordinate for each pillar
num_pillar          number of unique pillars
```

If more than `max_points_per_pillar` points fall into the same pillar, extra points are truncated.

## CPU Pillar Features

The CPU feature step converts stored `PointXYZI` values into a fixed 9D feature tensor:

```text
[num_pillars, max_points_per_pillar, feature_dim]
```

The current feature dimension is:

```text
0: x
1: y
2: z
3: intensity
4: x - mean_x
5: y - mean_y
6: z - mean_z
7: x - pillar_center_x
8: y - pillar_center_y
```

The tensor is stored as a 1D contiguous vector:

```text
index = (pillar_index * max_points_per_pillar + point_offset) * feature_dim + feature_index
```

Padding positions remain zero.

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
build/test_cpu_pillar_scatter
build/test_cpu_pillar_storage
build/test_cpu_feature
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

`test_cpu_pillar_scatter` verifies pillar key generation, unique pillar grouping, point counts, and per-point pillar indices.

`test_cpu_pillar_storage` verifies fixed-layout indexing, stored point positions, and `max_points_per_pillar` truncation.

`test_cpu_feature` verifies feature indexing, 9D feature values, and zero padding.

Build and run:

```bash
cmake --build build
./build/test_cpu_range_filter
./build/test_cpu_pillar
./build/test_cpu_pillar_scatter
./build/test_cpu_pillar_storage
./build/test_cpu_feature
```

Expected output:

```text
test_cpu_range_filter passed
test_cpu_pillar passed
test_cpu_pillar_scatter passed
test_cpu_pillar_storage passed
test_cpu_feature passed
```

## Repository Layout

```text
.
├── CMakeLists.txt
├── README.md
├── include/
│   ├── config.hpp
│   ├── cpu_feature.hpp
│   ├── cpu_pillar.hpp
│   ├── cpu_pillar_scatter.hpp
│   ├── cpu_pillar_storage.hpp
│   ├── cpu_preprocess.hpp
│   ├── kitti_reader.hpp
│   └── point.hpp
├── src/
│   ├── cpu_feature.cpp
│   ├── cpu_pillar.cpp
│   ├── cpu_pillar_scatter.cpp
│   ├── cpu_pillar_storage.cpp
│   ├── cpu_range_filter.cpp
│   ├── kitti_reader.cpp
│   └── main.cpp
├── tests/
│   ├── test_cpu_feature.cpp
│   ├── test_cpu_pillar.cpp
│   ├── test_cpu_pillar_scatter.cpp
│   ├── test_cpu_pillar_storage.cpp
│   └── test_cpu_range_filter.cpp
└── data/
    └── 000000.bin
```

Generated build files, local learning notes, and binary point cloud data are not committed.

## Next Steps

Planned preprocessing stages:

1. BEV pseudo-image layout
2. CUDA range filter with atomic compaction
3. CUDA range filter with prefix-sum compaction
4. CUDA pillar coordinate computation
5. CUDA pillar scatter / hash
6. CUDA pillar feature generation
7. CPU vs CUDA benchmark and Nsight Compute analysis
