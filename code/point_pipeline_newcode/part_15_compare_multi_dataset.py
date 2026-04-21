import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from part_01_ground_truth_given_data import read_tum_trajectory
from part_13_estimated_trajectory_calculated import align_trajectory_to_ground_truth


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
BATCH_OUTPUT_ROOT = os.environ.get("SLAM_BATCH_OUTPUT_ROOT", PROJECT_ROOT)
OUTPUT_DIR = os.path.join(BATCH_OUTPUT_ROOT, "outputs_summary_newcode", "point_only")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = [
    ("fr1_xyz", os.path.join(BATCH_OUTPUT_ROOT, "outputs_xyz_newcode", "point_only"), os.path.join(PROJECT_ROOT, "rgbd_dataset_freiburg1_xyz")),
    ("fr1_room", os.path.join(BATCH_OUTPUT_ROOT, "outputs_room_newcode", "point_only"), os.path.join(PROJECT_ROOT, "rgbd_dataset_freiburg1_room")),
    ("fr1_desk2", os.path.join(BATCH_OUTPUT_ROOT, "outputs_desk2_newcode", "point_only"), os.path.join(PROJECT_ROOT, "rgbd_dataset_freiburg1_desk2")),
    ("fr1_360", os.path.join(BATCH_OUTPUT_ROOT, "outputs_360_newcode", "point_only"), os.path.join(PROJECT_ROOT, "rgbd_dataset_freiburg1_360")),
    ("fr1_floor", os.path.join(BATCH_OUTPUT_ROOT, "outputs_floor_newcode", "point_only"), os.path.join(PROJECT_ROOT, "rgbd_dataset_freiburg1_floor")),
    ("tum_loop", os.path.join(BATCH_OUTPUT_ROOT, "outputs_loop_newcode", "point_only"), os.path.join(PROJECT_ROOT, "tum_loop")),
    ("tum_rgbd", os.path.join(BATCH_OUTPUT_ROOT, "outputs_rgbd_newcode", "point_only"), os.path.join(PROJECT_ROOT, "tum_rgbd")),
    ("fr2_desk", os.path.join(BATCH_OUTPUT_ROOT, "outputs_fr2desk_newcode", "point_only"), os.path.join(PROJECT_ROOT, "rgbd_dataset_freiburg2_desk")),
]

METHODS = [
    ("ICP/RANSAC", "icp_ransac"),
    ("ICP/LMedS", "icp_lmeds"),
    ("GICP/RANSAC", "gicp_ransac"),
    ("GICP/LMedS", "gicp_lmeds"),
]


def _rms_from_saved_outputs(base_dir, prefix, dataset_dir):
    traj_path = os.path.join(base_dir, f"{prefix}_corrected_trajectory.npy")
    time_path = os.path.join(base_dir, f"{prefix}_used_rgb_times.npy")
    gt_path = os.path.join(dataset_dir, "groundtruth.txt")
    if not (os.path.exists(traj_path) and os.path.exists(time_path) and os.path.exists(gt_path)):
        return None
    traj = np.load(traj_path)
    rgb_times = np.load(time_path)
    gt_data = read_tum_trajectory(gt_path)
    aligned = align_trajectory_to_ground_truth(
        traj,
        rgb_times,
        gt_data,
        dataset_dir=dataset_dir,
    )
    if aligned is None:
        return None
    return float(aligned["rms"])


def _loop_detected(base_dir, prefix):
    return "yes" if os.path.exists(os.path.join(base_dir, f"{prefix}_closure_info.npy")) else "no"


def _find_row(rows, dataset_name, method_name):
    for row in rows:
        if row["dataset"] == dataset_name and row["method"] == method_name:
            return row
    return None


def _collect_rows():
    rows = []
    for dataset_name, base_dir, dataset_dir in DATASETS:
        for method_name, prefix in METHODS:
            rmse = _rms_from_saved_outputs(base_dir, prefix, dataset_dir)
            if rmse is None:
                continue
            rows.append({
                "dataset": dataset_name,
                "method": method_name,
                "prefix": prefix,
                "corrected_ate_rmse_m": rmse,
                "loop_detected": _loop_detected(base_dir, prefix),
            })
    return rows


def _save_csv(rows):
    out_csv = os.path.join(OUTPUT_DIR, "part_17_multi_dataset_method_table.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "method",
            "prefix",
            "corrected_ate_rmse_m",
            "loop_detected",
        ])
        for row in rows:
            writer.writerow([
                row["dataset"],
                row["method"],
                row["prefix"],
                f"{row['corrected_ate_rmse_m']:.6f}",
                row["loop_detected"],
            ])
    return out_csv


def _plot_grouped_bar(rows):
    dataset_names = [name for name, _, _ in DATASETS]
    x = np.arange(len(dataset_names))
    width = 0.15

    plt.figure()
    for idx, (method_name, _) in enumerate(METHODS):
        values = []
        for dataset_name in dataset_names:
            row = _find_row(rows, dataset_name, method_name)
            values.append(np.nan if row is None else row["corrected_ate_rmse_m"])
        plt.bar(x + (idx - 1) * width, values, width=width, label=method_name)

    plt.yscale("log")
    plt.ylabel("ATE RMSE (m, log)")
    plt.title("ATE RMSE by method and dataset")
    plt.xticks(x, dataset_names, rotation=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "part_17_multi_dataset_grouped_bar.png")
    named_out_path = os.path.join(OUTPUT_DIR, "part_17_multi_dataset_grouped_bar__ICP_RANSAC__ICP_LMedS__GICP_RANSAC__GICP_LMedS.png")
    plt.savefig(out_path)
    plt.savefig(named_out_path)
    plt.close()
    return out_path, named_out_path


def _plot_heatmap(rows):
    dataset_names = [name for name, _, _ in DATASETS]
    method_names = [name for name, _ in METHODS]
    data = np.full((len(method_names), len(dataset_names)), np.nan, dtype=float)
    for r, method_name in enumerate(method_names):
        for c, dataset_name in enumerate(dataset_names):
            row = _find_row(rows, dataset_name, method_name)
            if row is not None:
                data[r, c] = row["corrected_ate_rmse_m"]

    plt.figure()
    plt.imshow(data, cmap="viridis_r", aspect="auto")
    plt.xticks(np.arange(len(dataset_names)), dataset_names, rotation=20)
    plt.yticks(np.arange(len(method_names)), method_names)
    plt.title("ATE RMSE heatmap")
    plt.colorbar(label="ATE RMSE (m)")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "part_17_multi_dataset_heatmap.png")
    named_out_path = os.path.join(OUTPUT_DIR, "part_17_multi_dataset_heatmap__ICP_RANSAC__ICP_LMedS__GICP_RANSAC__GICP_LMedS.png")
    plt.savefig(out_path)
    plt.savefig(named_out_path)
    plt.close()
    return out_path, named_out_path


def _plot_best_methods(rows):
    best_rows = []
    for dataset_name, _, _ in DATASETS:
        best = None
        for row in rows:
            if row["dataset"] != dataset_name:
                continue
            if best is None or row["corrected_ate_rmse_m"] < best["corrected_ate_rmse_m"]:
                best = row
        if best is None:
            continue
        best_rows.append(best)

    plt.figure()
    plt.bar(
        [row["dataset"] for row in best_rows],
        [row["corrected_ate_rmse_m"] for row in best_rows],
        color="#4C78A8",
    )
    plt.ylabel("Best ATE RMSE (m)")
    plt.title("Best ATE RMSE by dataset")
    plt.xticks(rotation=20)
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "part_17_best_method_per_dataset.png")
    plt.savefig(out_path)
    plt.close()
    return out_path, best_rows


def main():
    rows = _collect_rows()
    if not rows:
        print("multi-dataset outputs not found")
        return

    csv_path = _save_csv(rows)
    bar_path, bar_named_path = _plot_grouped_bar(rows)
    heatmap_path, heatmap_named_path = _plot_heatmap(rows)
    best_path, best_rows = _plot_best_methods(rows)

    print("\nPart 15")
    print(f"saved: {os.path.basename(csv_path)}, {os.path.basename(bar_path)}, {os.path.basename(heatmap_path)}, {os.path.basename(best_path)}")
    print("best:")
    for row in best_rows:
        print(f"- {row['dataset']}: {row['method']} ({row['corrected_ate_rmse_m']:.6f} m)")


if __name__ == "__main__":
    main()
