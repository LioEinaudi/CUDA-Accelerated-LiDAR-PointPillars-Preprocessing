# CUDA-Accelerated LiDAR PointPillars Preprocessing

This project implements a C++/CUDA LiDAR preprocessing pipeline for PointPillars-style 3D object detection. It includes a CPU reference implementation, CUDA baseline kernels, correctness tests, stage-wise wrapper-level benchmarks, and profiling-oriented optimization planning.

The project is developed with an emphasis on correctness, benchmark transparency, and pipeline-level bottleneck analysis. The neural network part of PointPillars is intentionally not included yet; the focus is the preprocessing path from KITTI point clouds to BEV pseudo-image tensors.

## Highlights

- End-to-end LiDAR preprocessing pipeline from KITTI `.bin` point clouds to BEV pseudo-image.
- CPU reference implementation for range filtering, pillarization, fixed-size pillar storage, 9D feature generation, and BEV construction.
- CUDA baseline implementation for each preprocessing stage, validated against the CPU baseline.
- Correctness tests for CPU and CUDA range filtering, pillar coordinates, scatter, storage, feature generation, and BEV output.
- Stage-wise wrapper-level benchmark on an RTX 4060 Laptop GPU, with explicit discussion of allocation, transfer, synchronization, host output allocation, and copy-back overhead.
- Detailed CUDA timing breakdown that separates wrapper-level costs from measured kernel/GPU execution time.

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
  -> cpuBuildBevPseudoImage
  -> dense [C, H, W] BEV pseudo-image
  -> point count summary
```

The CUDA side follows the same logical preprocessing order:

```text
KITTI .bin
  -> readKittiBin
  -> cudaRangeFilterPrefixSum
  -> cudaComputePillarCoords
  -> cudaPillarScatter
  -> cudaBuildPillarPointStorage
  -> cudaGeneratePillarFeatures
  -> cudaBuildBevPseudoImage
  -> CPU/CUDA count and timing summary
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

## CPU BEV Pseudo-Image

The CPU BEV step scatters sparse pillar features into a dense pseudo-image:

```text
[channels, height, width]
```

For the current simplified baseline, each BEV cell stores the mean feature value over the points inside that pillar:

```text
bev[channel][pillar_y][pillar_x] =
    mean(features[pillar_index][point_offset][channel])
```

The dense tensor is stored as a 1D vector:

```text
index = (channel * height + y) * width + x
```

With the default settings, the BEV shape is:

```text
[9, 496, 432]
```

Empty cells remain zero.

## CUDA Range Filter

The first CUDA preprocessing kernels implement range filtering on GPU.

The atomic-compaction version uses:

```text
one CUDA thread -> one input point
```

Each valid point uses:

```text
atomicAdd(output_count, 1)
```

to reserve a unique output slot before writing into the filtered point buffer.

This version is intentionally simple:

```text
input points on GPU
  -> rangeFilterAtomicKernel
  -> filtered points on GPU
  -> copy filtered count and points back to CPU
```

The first correctness check compares only filtered counts:

```text
CPU filtered count == CUDA filtered count
```

The atomic version may not preserve the exact same output ordering as the CPU `push_back` baseline, so strict point-by-point order comparison is left for a later validation step.

The prefix-sum compact version uses:

```text
valid flags -> exclusive scan -> scatter
```

This computes each valid point's output position from a prefix sum instead of using a shared atomic counter. It preserves input order, so the test compares both count and point values against the CPU baseline.

## CUDA Pillar Coordinates

The CUDA pillar-coordinate kernel maps each filtered point to its BEV grid coordinate:

```text
one CUDA thread -> one input point -> one PillarCoord
```

Unlike range filtering, this stage has a fixed-size output:

```text
N input points -> N output coordinates
```

So it does not need atomic operations or prefix-sum compaction. Each thread writes directly to `coords[idx]`.

## CUDA Pillar Scatter

The CUDA pillar-scatter kernel groups point coordinates into unique non-empty pillars:

```text
many PillarCoord values -> unique pillars + point_to_pillar
```

The first CUDA scatter version uses a dense BEV grid table:

```text
key = coord.y * grid_x + coord.x
key_to_pillar[key] = pillar_index
```

It uses `atomicCAS` to create each unique pillar once and `atomicAdd` to count points inside each pillar. The test validates semantic output instead of pillar creation order, because CUDA thread scheduling can create pillars in a different order than the CPU baseline.

## CUDA Pillar Storage

The CUDA pillar-storage kernel writes points into a fixed per-pillar layout:

```text
pillar_points[pillar_index][point_offset]
```

The implementation stores this as a flat contiguous array:

```text
index = pillar_index * max_points_per_pillar + point_offset
```

Each CUDA thread processes one point. It uses one atomic counter to reserve a raw offset inside the pillar, then only saves points whose offset is smaller than `max_points_per_pillar`. A second count records how many points were actually stored, matching the CPU storage semantics.

## CUDA Pillar Features

The CUDA feature stage converts stored pillar points into the same 9D feature layout as the CPU baseline:

```text
[num_pillars, max_points_per_pillar, 9]
```

The 9D feature layout is:

```text
x, y, z, intensity,
x - mean_x, y - mean_y, z - mean_z,
x - center_x, y - center_y
```

The first CUDA version uses two kernels: one kernel computes per-pillar mean values, and a second kernel writes the 9D feature vector for each valid pillar point. Padding positions remain zero.

## CUDA BEV Pseudo-Image

The CUDA BEV stage scatters sparse pillar features into a dense pseudo-image:

```text
[num_pillars, max_points_per_pillar, feature_dim]
  -> [feature_dim, grid_y, grid_x]
```

The current simplified BEV stage averages each pillar's point features per channel, then writes the result into:

```text
bev[channel][pillar_y][pillar_x]
```

Because pillar scatter already produces unique pillar coordinates, the first CUDA BEV version does not need atomics. Each CUDA thread handles one `(pillar_index, channel)` pair.

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
build/test_cpu_bev
build/test_cuda_range_filter
build/test_cuda_range_filter_prefix
build/test_cuda_pillar_coord
build/test_cuda_pillar_scatter
build/test_cuda_pillar_storage
build/test_cuda_feature
build/test_cuda_bev
build/benchmark
build/benchmark_cuda_timing
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

Run the benchmark with a KITTI-style `.bin` file:

```bash
./build/benchmark data/000000.bin
```

Run the detailed CUDA timing breakdown:

```bash
./build/benchmark_cuda_timing data/000000.bin
```

## Tests

The current test targets use hand-written points instead of reading KITTI files.

`test_cpu_range_filter` verifies in-range points, out-of-range points, and exclusive upper-bound behavior such as `x == max_x`.

`test_cpu_pillar` verifies default grid dimensions and point-to-pillar coordinate mapping.

`test_cpu_pillar_scatter` verifies pillar key generation, unique pillar grouping, point counts, and per-point pillar indices.

`test_cpu_pillar_storage` verifies fixed-layout indexing, stored point positions, and `max_points_per_pillar` truncation.

`test_cpu_feature` verifies feature indexing, 9D feature values, and zero padding.

`test_cpu_bev` verifies BEV indexing, mean aggregation into BEV cells, and zero-valued empty cells.

`test_cuda_range_filter` verifies that the CUDA atomic range filter returns the same filtered count as the CPU range filter.

`test_cuda_range_filter_prefix` verifies that the CUDA prefix-sum range filter matches CPU filtered count and output order.

`test_cuda_pillar_coord` verifies that CUDA pillar-coordinate computation matches the CPU coordinate baseline.

`test_cuda_pillar_scatter` verifies unique pillar grouping, point counts, and per-point pillar mappings for CUDA scatter.

`test_cuda_pillar_storage` verifies fixed-layout CUDA point storage and `max_points_per_pillar` truncation.

`test_cuda_feature` verifies CUDA 9D pillar feature values and zero padding against the CPU feature baseline.

`test_cuda_bev` verifies CUDA BEV pseudo-image values and empty cells against the CPU BEV baseline.

The CUDA tests have been validated on a local NVIDIA GeForce RTX 4060 Laptop GPU. The tests still handle environments without a visible CUDA device by printing a skip message and exiting successfully.

Build and run:

```bash
cmake --build build
./build/test_cpu_range_filter
./build/test_cpu_pillar
./build/test_cpu_pillar_scatter
./build/test_cpu_pillar_storage
./build/test_cpu_feature
./build/test_cpu_bev
./build/test_cuda_range_filter
./build/test_cuda_range_filter_prefix
./build/test_cuda_pillar_coord
./build/test_cuda_pillar_scatter
./build/test_cuda_pillar_storage
./build/test_cuda_feature
./build/test_cuda_bev
./build/benchmark data/000000.bin
./build/benchmark_cuda_timing data/000000.bin
```

Expected output:

```text
test_cpu_range_filter passed
test_cpu_pillar passed
test_cpu_pillar_scatter passed
test_cpu_pillar_storage passed
test_cpu_feature passed
test_cpu_bev passed
test_cuda_range_filter passed
test_cuda_range_filter_prefix passed
test_cuda_pillar_coord passed
test_cuda_pillar_scatter passed
test_cuda_pillar_storage passed
test_cuda_feature passed
test_cuda_bev passed
```

## Benchmark: Stage-wise Wrapper-level Latency

The benchmark currently compares the CPU preprocessing pipeline with the CUDA preprocessing pipeline on one KITTI-style frame. It uses `pillar.max_pillars = 20000` in the benchmark path so the CPU and CUDA pillar counts can match on the tested frame.

The current benchmark reports stage-wise C++ wrapper-level latency. Each CUDA stage may include `cudaMalloc`, H2D copies, kernel execution, synchronization, D2H copies, host output allocation, and `cudaFree`. Therefore, this table is intended to expose pipeline-level overhead rather than pure kernel latency. The benchmark also runs a small CUDA warmup before timing to avoid counting first-use CUDA context initialization.

Example result on a local NVIDIA GeForce RTX 4060 Laptop GPU:

```text
Input points: 120000
Filtered points: CPU 17353, CUDA 17353
Pillars: CPU 16645, CUDA 16645
BEV size: CPU 1928448, CUDA 1928448
```

| Stage | CPU ms | CUDA wrapper ms |
| --- | ---: | ---: |
| Range filter | 1.55536 | 0.622642 |
| Pillar coord | included in scatter path | 0.192323 |
| Scatter | 3.73143 | 0.318826 |
| Storage | 7.63388 | 9.97151 |
| Feature | 21.4164 | 29.0623 |
| BEV | 5.02407 | 10.4525 |
| Total | 39.3611 | 50.6201 |

In this run, the early CUDA stages are faster than the CPU baseline, while the full CUDA wrapper-level pipeline is still slower overall. A likely reason is that the current implementation is intentionally modular: each stage owns its own memory allocation, transfer, kernel launch, copy-back, host output construction, and cleanup. The later stages also move larger intermediate buffers between CPU and GPU. The next optimization step is to keep the whole preprocessing pipeline GPU-resident and use CUDA events to separate kernel time from memory-management and transfer time.

## CUDA Timing Breakdown

After the first end-to-end benchmark, the CUDA benchmark was split into finer timing sections. The goal was to check whether the slower wrapper-level stages were slow because of the CUDA kernels themselves, or because of allocation, host-device transfers, device-host transfers, and CPU-side output buffer creation.

The detailed timing tool is:

```text
build/benchmark_cuda_timing
```

It currently breaks down:

```text
range filter
pillar storage
pillar feature generation
BEV pseudo-image generation
```

Example result on the same local NVIDIA GeForce RTX 4060 Laptop GPU:

| Stage | Wrapper ms | H2D ms | Kernel / GPU ms | Host output alloc ms | D2H ms | Free ms | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Range filter | 0.854504 | 0.127122 | flag 0.010304, scan 0.028672, scatter 0.006336 | 0.019655 | count 0.012071, points 0.0542 | 0.153998 | prefix-sum compaction |
| Pillar storage | 8.38971 | 0.057832 | 0.016288 | 4.86727 | 2.65549 | 0.686507 | atomicAdd per point |
| Pillar feature | 30.1683 | 2.72125 | mean 0.013024, feature 0.044032 | 19.5231 | 6.22539 | 1.48574 | mean kernel + feature kernel |
| BEV | 9.2195 | 5.74984 | 0.03152 | 1.00435 | 0.821527 | 1.35062 | dense output write |

This suggests that the current CUDA kernels are probably not the main bottleneck in these measured stages. The larger cost appears to come from the modular wrapper design: every stage copies intermediate results back to CPU, creates host output vectors, and owns its own temporary device allocations. In particular, feature generation spends about `0.057 ms` in its two CUDA kernels, while the wrapper-level time is about `30.168 ms`. The next project step is therefore to build a GPU-resident preprocessing pipeline, where intermediate tensors stay on GPU and only final debug or benchmark values are copied back.

The project can now be viewed as a staged profiling path:

```text
V0 CPU pipeline
V1 CUDA modular wrapper pipeline
V2 CUDA wrapper breakdown: allocation / H2D / kernel / D2H / free
V3 CUDA GPU-resident pipeline
```

## Repository Layout

```text
.
├── CMakeLists.txt
├── README.md
├── include/
│   ├── config.hpp
│   ├── cpu_bev.hpp
│   ├── cpu_feature.hpp
│   ├── cpu_pillar.hpp
│   ├── cpu_pillar_scatter.hpp
│   ├── cpu_pillar_storage.hpp
│   ├── cpu_preprocess.hpp
│   ├── cuda_preprocess.cuh
│   ├── kitti_reader.hpp
│   └── point.hpp
├── src/
│   ├── cpu_bev.cpp
│   ├── cpu_feature.cpp
│   ├── cpu_pillar.cpp
│   ├── cpu_pillar_scatter.cpp
│   ├── cpu_pillar_storage.cpp
│   ├── cpu_range_filter.cpp
│   ├── cuda_bev.cu
│   ├── cuda_feature.cu
│   ├── cuda_pillar_coord.cu
│   ├── cuda_pillar_scatter.cu
│   ├── cuda_pillar_storage.cu
│   ├── cuda_range_filter.cu
│   ├── benchmark.cpp
│   ├── benchmark_cuda_timing.cu
│   ├── kitti_reader.cpp
│   └── main.cpp
├── tests/
│   ├── test_cpu_bev.cpp
│   ├── test_cpu_feature.cpp
│   ├── test_cpu_pillar.cpp
│   ├── test_cpu_pillar_scatter.cpp
│   ├── test_cpu_pillar_storage.cpp
│   ├── test_cpu_range_filter.cpp
│   ├── test_cuda_bev.cpp
│   ├── test_cuda_feature.cpp
│   ├── test_cuda_pillar_coord.cpp
│   ├── test_cuda_pillar_scatter.cpp
│   ├── test_cuda_pillar_storage.cpp
│   ├── test_cuda_range_filter.cpp
│   └── test_cuda_range_filter_prefix.cpp
└── data/
    └── 000000.bin
```

## Next Steps

Planned optimization stages:

1. Keep the CUDA preprocessing pipeline GPU-resident instead of copying every stage back to CPU.
2. Start with a fused CUDA pipeline for range filter, pillar coordinate, and pillar scatter.
3. Extend the GPU-resident pipeline to storage, feature generation, and BEV pseudo-image generation.
4. Use Nsight Systems or Nsight Compute to inspect the remaining bottlenecks.
5. Run benchmark results on more KITTI frames and report average latency.
