import csv
import os
import subprocess
import sys

import open3d as o3d


ROOT = os.path.dirname(os.path.dirname(__file__))
THIS_DIR = os.path.dirname(__file__)
BATCH_OUTPUT_ROOT = os.environ.get("SLAM_BATCH_OUTPUT_ROOT", os.path.join(ROOT, "final"))
SUMMARY_DIR = os.path.join(BATCH_OUTPUT_ROOT, "outputs_summary_newcode", "point_only")
os.makedirs(SUMMARY_DIR, exist_ok=True)

DATASETS = [
    ("xyz", "outputs_xyz_newcode"),
    ("room", "outputs_room_newcode"),
    ("desk2", "outputs_desk2_newcode"),
    ("floor", "outputs_floor_newcode"),
    ("360", "outputs_360_newcode"),
    ("rgbd", "outputs_rgbd_newcode"),
    ("loop", "outputs_loop_newcode"),
    ("fr2desk", "outputs_fr2desk_newcode"),
]


def run_cmd(args, env):
    subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=True,
    )


def best_from_table(output_dir):
    table = os.path.join(output_dir, "point_only", "part_17_method_comparison_table.csv")
    with open(table, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        method_idx = header.index("method")
        prefix_idx = header.index("prefix")
        rmse_idx = header.index("corrected_ate_rmse_m")

        best_method = ""
        best_prefix = ""
        best_rmse = None
        for row in reader:
            rmse = float(row[rmse_idx])
            if best_rmse is None or rmse < best_rmse:
                best_method = row[method_idx]
                best_prefix = row[prefix_idx]
                best_rmse = rmse

    return best_method, best_prefix, best_rmse


def point_count_and_size(path):
    if not os.path.exists(path):
        return "", ""
    cloud = o3d.io.read_point_cloud(path)
    return len(cloud.points), os.stat(path).st_size


def main():
    rows = []
    for dataset_name, output_name in DATASETS:
        output_dir = os.path.join(BATCH_OUTPUT_ROOT, output_name)
        print(f"\ndataset: {dataset_name}")
        env = os.environ.copy()
        env["SLAM_DATASET"] = dataset_name
        env["SLAM_OUTPUT_DIR"] = output_dir
        env["SLAM_BATCH_OUTPUT_ROOT"] = BATCH_OUTPUT_ROOT

        run_cmd([sys.executable, os.path.join(THIS_DIR, "run_all_parts.py"), "--full"], env)

        sor_env = env.copy()
        sor_env["SLAM_MAP_SOR"] = "1"
        sor_env["SLAM_MAP_SOR_MEAN_K"] = "20"
        sor_env["SLAM_MAP_SOR_STD"] = "1.5"
        sor_env["SLAM_MAP_TAG"] = "dense_sor"
        run_cmd([sys.executable, os.path.join(THIS_DIR, "part_17_reconstruct_3d_with_estimated_trajectory.py")], sor_env)

        best_method, best_prefix, best_rmse = best_from_table(output_dir)
        base = os.path.join(output_dir, "point_only")
        dense_ply = os.path.join(base, f"part_19_{best_prefix}_corrected_dense_mapped_cloud.ply")
        sor_ply = os.path.join(base, f"part_19_{best_prefix}_corrected_dense_sor_mapped_cloud.ply")

        dense_points, dense_bytes = point_count_and_size(dense_ply)
        sor_points, sor_bytes = point_count_and_size(sor_ply)

        row = [
            dataset_name,
            os.path.relpath(output_dir, ROOT),
            best_method,
            best_prefix,
            f"{best_rmse:.6f}",
            dense_points,
            dense_bytes,
            sor_points,
            sor_bytes,
        ]
        rows.append(row)
        print(f"best: {best_method} {best_rmse:.6f} m")

    out_csv = os.path.join(SUMMARY_DIR, "newcode_map_variant_comparison.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "output_dir",
            "best_method",
            "best_prefix",
            "best_rmse_m",
            "dense_points",
            "dense_bytes",
            "dense_sor_points",
            "dense_sor_bytes",
        ])
        for row in rows:
            writer.writerow(row)

    summary_env = os.environ.copy()
    summary_env["SLAM_BATCH_OUTPUT_ROOT"] = BATCH_OUTPUT_ROOT
    run_cmd([sys.executable, os.path.join(THIS_DIR, "part_15_compare_multi_dataset.py")], summary_env)

    print("\nsaved:", os.path.basename(out_csv))


if __name__ == "__main__":
    main()
