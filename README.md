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
- GPU-resident CUDA pipeline prototypes that keep intermediate tensors on device and reduce CPU/GPU round trips.

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
build/test_cuda_pipelinev1
build/test_cuda_pipelinev2
build/test_cuda_pipelinev3
build/test_cuda_pipelinev4
build/benchmark
build/benchmark_cuda_timing
build/benchmark_cuda_pipelinev1
build/benchmark_cuda_pipelinev2
build/benchmark_cuda_pipelinev3
build/benchmark_cuda_pipelinev4
build/benchmark_multi_frame
build/profile_cuda_modular_v4
build/profile_cuda_pipeline_v4
build/validate_cuda_pipelinev4_kitti
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

Run the GPU-resident pipeline benchmarks:

```bash
./build/benchmark_cuda_pipelinev1 data/000000.bin
./build/benchmark_cuda_pipelinev2 data/000000.bin
./build/benchmark_cuda_pipelinev3 data/000000.bin
./build/benchmark_cuda_pipelinev4 data/000000.bin
```

Run full-frame V4 correctness validation:

```bash
./build/validate_cuda_pipelinev4_kitti data/000000.bin
```

Generate synthetic KITTI-format multi-frame data:

```bash
python3 make_data.py
```

This writes gradient test frames to:

```text
data/synthetic/
```

Run the multi-frame benchmark:

```bash
./build/benchmark_multi_frame data/synthetic
```

Run Nsight Systems profile targets:

```bash
mkdir -p results/nsight
nsys profile -o results/nsight/profile_cuda_modular_v4 ./build/profile_cuda_modular_v4 data/000000.bin 5
nsys profile -o results/nsight/profile_cuda_pipeline_v4 ./build/profile_cuda_pipeline_v4 data/000000.bin 5
```

The synthetic frames are designed as a controlled workload gradient rather than a real KITTI replacement. The generator increases total point count, in-range point ratio, and clustered in-range density across 10 frames:

| Frame | Total Points | In-range Ratio | Clustered In-range Ratio |
| --- | ---: | ---: | ---: |
| `000000.bin` | 60000 | 0.35 | 0.10 |
| `000001.bin` | 70000 | 0.40 | 0.15 |
| `000002.bin` | 80000 | 0.45 | 0.20 |
| `000003.bin` | 90000 | 0.50 | 0.25 |
| `000004.bin` | 100000 | 0.55 | 0.30 |
| `000005.bin` | 110000 | 0.60 | 0.35 |
| `000006.bin` | 120000 | 0.65 | 0.40 |
| `000007.bin` | 130000 | 0.70 | 0.45 |
| `000008.bin` | 140000 | 0.75 | 0.50 |
| `000009.bin` | 150000 | 0.80 | 0.55 |

This synthetic set is useful for validating benchmark stability and latency aggregation under gradually increasing preprocessing load. Real KITTI multi-frame evaluation is still the better final benchmark for reporting dataset-level performance.

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

`test_cuda_pipelinev1` verifies the GPU-resident range-filter, coordinate, and scatter pipeline summary against CPU filtered and pillar counts.

`test_cuda_pipelinev2` verifies the GPU-resident pipeline after adding pillar storage, including filtered count, pillar count, and stored point count.

`test_cuda_pipelinev3` verifies that the GPU-resident pipeline can extend through pillar feature generation while preserving the filtered count, pillar count, and stored point count. The full feature tensor is intentionally not copied back in the pipeline benchmark path.

`test_cuda_pipelinev4` uses a small deterministic point cloud and a debug V4 path that copies BEV output back to CPU, then compares the full BEV pseudo-image against the CPU baseline.

`validate_cuda_pipelinev4_kitti` runs the same V4 debug validation on a full KITTI-style frame. This tool is intended for manual correctness validation because it depends on local point cloud data and copies the full BEV output back to host memory.

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
./build/test_cuda_pipelinev1
./build/test_cuda_pipelinev2
./build/test_cuda_pipelinev3
./build/test_cuda_pipelinev4
./build/benchmark data/000000.bin
./build/benchmark_cuda_timing data/000000.bin
./build/benchmark_cuda_pipelinev1 data/000000.bin
./build/benchmark_cuda_pipelinev2 data/000000.bin
./build/benchmark_cuda_pipelinev3 data/000000.bin
./build/benchmark_cuda_pipelinev4 data/000000.bin
./build/benchmark_multi_frame data/synthetic
./build/validate_cuda_pipelinev4_kitti data/000000.bin
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
test_cuda_pipeline passed
test_cuda_pipelinev2 passed
test_cuda_pipelinev3 passed
test_cuda_pipelinev4 passed
validate_cuda_pipelinev4_kitti passed
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
V3/V4 CUDA GPU-resident pipeline
```

## GPU-resident Pipeline Benchmarks

The GPU-resident pipeline prototypes test the main optimization suggested by the timing breakdown: keep intermediate preprocessing tensors on the GPU and copy back only compact summary values needed for validation.

Pipeline V1 fuses:

```text
range filter -> pillar coordinate -> pillar scatter
```

Pipeline V2 extends V1 with:

```text
pillar storage
```

Pipeline V3 extends V2 with:

```text
pillar feature generation
```

Pipeline V4 extends V3 with:

```text
BEV pseudo-image generation
```

The pipeline benchmarks are validated against CPU reference summary counts before comparing latency. V3 and V4 intentionally keep large intermediate tensors on the GPU: V3 skips full feature D2H copy-back, and V4 skips both full feature and BEV D2H copy-back. Therefore, these speedups compare GPU-resident preprocessing latency against modular CUDA wrapper latency, not full host-visible output export latency.

For correctness, V4 also has a separate debug validation path. The benchmark path keeps BEV on GPU, while the debug path copies the BEV pseudo-image back to CPU and compares it against the CPU BEV baseline on both a small deterministic input and an optional full KITTI-style frame.

Summary on one KITTI-style frame:

| Pipeline | Stages Included | CUDA modular ms | GPU-resident ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| V1 | range + coord + scatter | 0.983315 | 0.623737 | 1.57649x |
| V2 | + storage | 12.1325 | 1.63467 | 7.42194x |
| V3 | + feature | 44.2121 | 4.10545 | 10.7691x |
| V4 | + BEV | 62.091 | 3.69822 | 16.7895x |

### Pipeline V1

V1 compares the modular CUDA wrappers:

```text
cudaRangeFilterPrefixSum
cudaComputePillarCoords
cudaPillarScatter
```

against:

```text
cudaPreprocessPipelineV1
```

Result on the same KITTI-style frame:

| Metric | CUDA modular V1 | CUDA pipeline V1 |
| --- | ---: | ---: |
| Total latency | 0.983315 ms | 0.623737 ms |
| Filtered count | 17353 | 17353 |
| Pillars | 16645 | 16645 |
| Speedup | - | 1.57649x |

### Pipeline V2

V2 compares the modular CUDA wrappers:

```text
cudaRangeFilterPrefixSum
cudaComputePillarCoords
cudaPillarScatter
cudaBuildPillarPointStorage
```

against:

```text
cudaPreprocessPipelineV2
```

Result on the same KITTI-style frame:

| Metric | CUDA modular V2 | CUDA pipeline V2 |
| --- | ---: | ---: |
| Total latency | 12.1325 ms | 1.63467 ms |
| Filtered count | 17353 | 17353 |
| Pillars | 16645 | 16645 |
| Stored points | 17353 | 17353 |
| Speedup | - | 7.42194x |

The much larger V2 gain is consistent with the earlier timing breakdown: the storage kernel itself is inexpensive, while the modular storage wrapper spends most of its time in host output allocation and device-host transfer. Keeping storage inside the GPU-resident pipeline removes that large intermediate round trip.

### Pipeline V3

V3 compares the modular CUDA wrappers:

```text
cudaRangeFilterPrefixSum
cudaComputePillarCoords
cudaPillarScatter
cudaBuildPillarPointStorage
cudaGeneratePillarFeatures
```

against:

```text
cudaPreprocessPipelineV3
```

Result on the same KITTI-style frame:

| Metric | CUDA modular V3 | CUDA pipeline V3 |
| --- | ---: | ---: |
| Total latency | 44.2121 ms | 4.10545 ms |
| Filtered count | 17353 | 17353 |
| Pillars | 16645 | 16645 |
| Stored points | 17353 | 17353 |
| Feature values | 14980500 | skipped copy-back |
| Speedup | - | 10.7691x |

The modular feature wrapper still materializes the full feature tensor on the CPU, while the V3 pipeline keeps that tensor on GPU. This makes the benchmark useful for measuring GPU-resident preprocessing latency, but it is not claiming that the full feature tensor has been exported to host memory.

### Pipeline V4

V4 compares the modular CUDA wrappers:

```text
cudaRangeFilterPrefixSum
cudaComputePillarCoords
cudaPillarScatter
cudaBuildPillarPointStorage
cudaGeneratePillarFeatures
cudaBuildBevPseudoImage
```

against:

```text
cudaPreprocessPipelineV4
```

Result on the same KITTI-style frame:

| Metric | CUDA modular V4 | CUDA pipeline V4 |
| --- | ---: | ---: |
| Total latency | 62.091 ms | 3.69822 ms |
| Filtered count | 17353 | 17353 |
| Pillars | 16645 | 16645 |
| Stored points | 17353 | 17353 |
| Feature values | 14980500 | skipped copy-back |
| BEV values | 1928448 | skipped copy-back |
| Speedup | - | 16.7895x |

V4 completes the current GPU-resident preprocessing path from KITTI points to BEV pseudo-image generation. The benchmark keeps both the feature tensor and BEV pseudo-image on the device, which removes the large intermediate CPU/GPU round trips present in the modular CUDA wrappers.

The separate full-frame V4 debug validation copies the complete BEV pseudo-image back to CPU and compares it against the CPU BEV baseline:

```text
Loaded points: 120000
CPU filtered count: 17353
CUDA filtered count: 17353
CPU pillars: 16645
CUDA pillars: 16645
CPU stored points: 17353
CUDA stored points: 17353
BEV values: 1928448
Max abs diff: 3.8147e-06
Tolerance: 0.001
validate_cuda_pipelinev4_kitti passed
```

This keeps the performance benchmark and correctness validation separate: the benchmark path measures GPU-resident latency without full BEV export, while the debug validation path proves that the generated BEV values match the CPU baseline within floating-point tolerance.

### Synthetic Multi-frame Benchmark

A synthetic KITTI-format 10-frame gradient set is generated by `make_data.py` under `data/synthetic/`. This is a controlled workload test, not a replacement for real KITTI multi-frame evaluation. The gradient increases total points, in-range points, and pillar density across frames while keeping the number of unique pillars below the current `max_pillars = 20000` benchmark capacity.

Run:

```bash
python3 make_data.py
cmake --build build --target benchmark_multi_frame
./build/benchmark_multi_frame ./data/synthetic
```

Synthetic workload gradient result on the same local NVIDIA GeForce RTX 4060 Laptop GPU:

| Frame | Points | Filtered | Pillars | CPU Full ms | CUDA Modular V4 ms | CUDA Pipeline V4 ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `000000.bin` | 60000 | 21000 | 3263 | 12.6746 | 14.2545 | 3.38861 | 4.20659x |
| `000001.bin` | 70000 | 28000 | 4337 | 15.3616 | 17.2542 | 3.46318 | 4.9822x |
| `000002.bin` | 80000 | 36000 | 5560 | 19.3257 | 20.4814 | 3.44754 | 5.94087x |
| `000003.bin` | 90000 | 45000 | 6963 | 21.1955 | 22.0269 | 3.56 | 6.18732x |
| `000004.bin` | 100000 | 55000 | 8470 | 24.2914 | 25.9473 | 3.60094 | 7.20569x |
| `000005.bin` | 110000 | 66000 | 10133 | 29.235 | 29.9599 | 3.65932 | 8.18729x |
| `000006.bin` | 120000 | 78000 | 11928 | 34.2948 | 35.0616 | 3.6629 | 9.57209x |
| `000007.bin` | 130000 | 91000 | 13758 | 39.5433 | 39.4106 | 3.77305 | 10.4453x |
| `000008.bin` | 140000 | 105000 | 15527 | 52.3337 | 47.9643 | 3.94112 | 12.1702x |
| `000009.bin` | 150000 | 120000 | 17460 | 56.0272 | 54.2256 | 4.00711 | 13.5323x |

Summary:

| Pipeline | Avg ms | Min ms | Max ms |
| --- | ---: | ---: | ---: |
| CPU full | 30.4283 | 12.6746 | 56.0272 |
| CUDA modular V4 | 30.6586 | 14.2545 | 54.2256 |
| CUDA pipeline V4 | 3.65038 | 3.38861 | 4.00711 |

| Metric | Avg | Min | Max |
| --- | ---: | ---: | ---: |
| Points | 105000 | 60000 | 150000 |
| Filtered points | 64500 | 21000 | 120000 |
| Pillars | 9739.9 | 3263 | 17460 |
| Stored points | 63685.6 | 20605 | 118103 |

Average speedup over CUDA modular V4 is `8.24299x`, with a range from `4.20659x` to `13.5323x`. The pipeline latency remains relatively stable because intermediate feature and BEV tensors stay on GPU and full D2H export is skipped, while the modular wrapper repeatedly moves large intermediate buffers across the CPU/GPU boundary.

## Nsight Systems Profiling

Nsight Systems was used to compare the modular CUDA V4 path with the GPU-resident pipeline V4 path. Two dedicated profile targets isolate each path:

```text
profile_cuda_modular_v4
profile_cuda_pipeline_v4
```

The modular V4 profile runs the standalone CUDA wrappers repeatedly:

```text
cudaRangeFilterPrefixSum
cudaComputePillarCoords
cudaPillarScatter
cudaBuildPillarPointStorage
cudaGeneratePillarFeatures
cudaBuildBevPseudoImage
```

The pipeline V4 profile repeatedly runs:

```text
cudaPreprocessPipelineV4
```

Both profiles were captured with 5 repeats on the same local NVIDIA GeForce RTX 4060 Laptop GPU.

| Profile | Kernel Timeline | Memory Timeline | HtoD Memcpy | DtoH Memcpy | Key Observation |
| --- | ---: | ---: | ---: | ---: | --- |
| CUDA modular V4 | 0.6% | 99.4% | 47.0% | 51.6% | Dominated by repeated intermediate H2D/D2H transfers and wrapper memory operations. |
| CUDA pipeline V4 | 21.7% | 78.3% | 34.4% | 1.9% | Full feature and BEV copy-back are removed; DtoH traffic drops sharply. |

The modular V4 timeline is almost entirely memory-transfer dominated. CUDA kernels account for only about `0.6%` of the captured GPU timeline, while HtoD and DtoH memory copies account for most of the activity. This matches the wrapper-level benchmark: the modular implementation repeatedly moves intermediate tensors between CPU and GPU after each stage.

The GPU-resident V4 timeline is more compact. Kernel share rises to about `21.7%`, and DtoH memcpy drops from about `51.6%` to about `1.9%`, because the feature tensor and BEV pseudo-image stay on the GPU during latency measurement. The remaining memory activity is mostly input H2D transfer and device-side initialization, especially `cudaMemset`, which points to the next optimization step: a reusable CUDA workspace and more careful buffer initialization.

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
│   ├── cuda_pipeline.cuh
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
│   ├── cuda_pipelinev1.cu
│   ├── cuda_pipelinev2.cu
│   ├── cuda_pipelinev3.cu
│   ├── cuda_pipelinev4.cu
│   ├── cuda_range_filter.cu
│   ├── benchmark.cpp
│   ├── benchmark_cuda_pipelinev1.cpp
│   ├── benchmark_cuda_pipelinev2.cpp
│   ├── benchmark_cuda_pipelinev3.cpp
│   ├── benchmark_cuda_pipelinev4.cpp
│   ├── benchmark_cuda_timing.cu
│   ├── benchmark_multi_frame.cpp
│   ├── kitti_reader.cpp
│   ├── main.cpp
│   ├── profile_cuda_modular_v4.cpp
│   ├── profile_cuda_pipeline_v4.cpp
│   └── validate_cuda_pipelinev4_kitti.cpp
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
│   ├── test_cuda_pipeline.cpp
│   ├── test_cuda_pipelinev2.cpp
│   ├── test_cuda_pipelinev3.cpp
│   ├── test_cuda_pipelinev4.cpp
│   ├── test_cuda_range_filter.cpp
│   └── test_cuda_range_filter_prefix.cpp
└── data/
    └── 000000.bin
```

## Next Steps

Planned optimization stages:

1. Replace repeated per-call allocations with a reusable CUDA workspace.
2. Reduce or combine repeated device buffer initialization, especially large `cudaMemset` calls.
3. Run benchmark results on real KITTI multi-frame data and report average latency.
4. Add optional full BEV export timing to separate GPU-resident latency from host-visible output latency.
