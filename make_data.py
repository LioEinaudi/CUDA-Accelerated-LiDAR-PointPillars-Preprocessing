import os
import random
import struct


OUTPUT_DIR = "data/synthetic"
RANDOM_SEED = 20260517

RANGE = {
    "min_x": 0.0,
    "max_x": 69.12,
    "min_y": -39.68,
    "max_y": 39.68,
    "min_z": -3.0,
    "max_z": 1.0,
}

FRAME_SPECS = [
    # filename, total points, in-range ratio, clustered in-range ratio
    ("000000.bin", 60000, 0.35, 0.10),
    ("000001.bin", 70000, 0.40, 0.15),
    ("000002.bin", 80000, 0.45, 0.20),
    ("000003.bin", 90000, 0.50, 0.25),
    ("000004.bin", 100000, 0.55, 0.30),
    ("000005.bin", 110000, 0.60, 0.35),
    ("000006.bin", 120000, 0.65, 0.40),
    ("000007.bin", 130000, 0.70, 0.45),
    ("000008.bin", 140000, 0.75, 0.50),
    ("000009.bin", 150000, 0.80, 0.55),
]


def clamp(value, low, high):
    return max(low, min(high, value))


def corridor_bounds(frame_index):
    max_x = 20.0 + frame_index * 3.1
    y_half_width = 2.0 + frame_index * 0.31
    max_x = min(max_x, RANGE["max_x"] - 1e-3)
    return max_x, y_half_width


def sample_uniform_in_range(frame_index):
    # Gradually widen a forward corridor. This increases workload while keeping
    # the number of unique pillars below the benchmark max_pillars capacity.
    max_x, y_half_width = corridor_bounds(frame_index)

    x = random.uniform(RANGE["min_x"], max_x)
    y = random.uniform(-y_half_width, y_half_width)
    z = random.uniform(RANGE["min_z"], RANGE["max_z"] - 1e-3)
    intensity = random.uniform(0.0, 1.0)
    return x, y, z, intensity


def sample_clustered_in_range(frame_index):
    centers = [
        (8.0 + frame_index * 1.2, -3.0),
        (28.0 + frame_index * 1.0, 0.0),
        (18.0 + frame_index * 2.1, 3.0),
    ]
    center_x, center_y = random.choice(centers)
    sigma_xy = max(0.8, 4.5 - frame_index * 0.25)
    max_x, y_half_width = corridor_bounds(frame_index)

    x = random.gauss(center_x, sigma_xy)
    y = random.gauss(center_y, sigma_xy)
    z = random.gauss(-1.0, 0.55)

    x = clamp(x, RANGE["min_x"], max_x)
    y = clamp(y, -y_half_width, y_half_width)
    z = clamp(z, RANGE["min_z"], RANGE["max_z"] - 1e-3)
    intensity = random.uniform(0.2, 1.0)
    return x, y, z, intensity


def sample_out_of_range():
    mode = random.randint(0, 3)
    if mode == 0:
        x = random.uniform(-50.0, -0.01)
        y = random.uniform(-50.0, 50.0)
        z = random.uniform(-5.0, 5.0)
    elif mode == 1:
        x = random.uniform(RANGE["max_x"], 100.0)
        y = random.uniform(-50.0, 50.0)
        z = random.uniform(-5.0, 5.0)
    elif mode == 2:
        x = random.uniform(-10.0, 90.0)
        y = random.choice([
            random.uniform(-60.0, RANGE["min_y"] - 0.01),
            random.uniform(RANGE["max_y"], 60.0),
        ])
        z = random.uniform(-5.0, 5.0)
    else:
        x = random.uniform(-10.0, 90.0)
        y = random.uniform(-50.0, 50.0)
        z = random.choice([
            random.uniform(-6.0, RANGE["min_z"] - 0.01),
            random.uniform(RANGE["max_z"], 4.0),
        ])
    intensity = random.uniform(0.0, 1.0)
    return x, y, z, intensity


def write_frame(path, frame_index, total_points, in_range_ratio, cluster_ratio):
    in_range_points = int(total_points * in_range_ratio)
    out_of_range_points = total_points - in_range_points
    clustered_points = int(in_range_points * cluster_ratio)
    uniform_points = in_range_points - clustered_points

    with open(path, "wb") as f:
        for _ in range(uniform_points):
            f.write(struct.pack("<4f", *sample_uniform_in_range(frame_index)))
        for _ in range(clustered_points):
            f.write(struct.pack("<4f", *sample_clustered_in_range(frame_index)))
        for _ in range(out_of_range_points):
            f.write(struct.pack("<4f", *sample_out_of_range()))

    return in_range_points, clustered_points, out_of_range_points


def main():
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest_lines = [
        "# Synthetic KITTI-format Point Cloud Gradient",
        "",
        "Each `.bin` file stores `float32 x y z intensity` points.",
        "The frames are generated with a workload gradient:",
        "",
        "- total point count increases from 60k to 150k",
        "- in-range point ratio increases from 35% to 80%",
        "- clustered in-range ratio increases from 10% to 55%",
        "- in-range points occupy a gradually wider forward corridor while staying below the default `max_pillars = 20000` benchmark capacity",
        "",
        "| file | total points | in-range ratio | clustered in-range ratio |",
        "| --- | ---: | ---: | ---: |",
    ]

    print(f"Generating synthetic KITTI-format frames in {OUTPUT_DIR}")
    for frame_index, (filename, total_points, in_range_ratio, cluster_ratio) in enumerate(FRAME_SPECS):
        path = os.path.join(OUTPUT_DIR, filename)
        in_range_points, clustered_points, out_of_range_points = write_frame(
            path,
            frame_index,
            total_points,
            in_range_ratio,
            cluster_ratio,
        )

        manifest_lines.append(
            f"| `{filename}` | {total_points} | {in_range_ratio:.2f} | {cluster_ratio:.2f} |"
        )
        print(
            f"{filename}: total={total_points}, "
            f"in_range={in_range_points}, clustered={clustered_points}, "
            f"out_of_range={out_of_range_points}"
        )

    readme_path = os.path.join(OUTPUT_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"Done. Manifest written to {readme_path}")


if __name__ == "__main__":
    main()
