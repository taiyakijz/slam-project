import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import part_00_project_config as config
from part_01_ground_truth_given_data import read_tum_trajectory
from part_08_run_baseline_icp_ransac import PREFIX as PREFIX_BASELINE
from part_09_run_lmeds_rejection import PREFIX as PREFIX_LMEDS
from part_10_run_gicp_registration import PREFIX as PREFIX_GICP
from part_11_run_gicp_lmeds import PREFIX as PREFIX_GICP_LMEDS
from part_13_estimated_trajectory_calculated import align_trajectory_to_ground_truth


OUTPUT_DIR = config.OUTPUT_DIR
METHODS = [
    ("ICP/RANSAC", PREFIX_BASELINE),
    ("ICP/LMedS", PREFIX_LMEDS),
    ("GICP/RANSAC", PREFIX_GICP),
    ("GICP/LMedS", PREFIX_GICP_LMEDS),
]


def _rms(prefix, kind, gt_data):
    traj_path = os.path.join(OUTPUT_DIR, f"{prefix}_{kind}_trajectory.npy")
    time_path = os.path.join(OUTPUT_DIR, f"{prefix}_used_rgb_times.npy")
    if not (os.path.exists(traj_path) and os.path.exists(time_path)):
        return None
    traj = np.load(traj_path)
    rgb_times = np.load(time_path)
    aligned = align_trajectory_to_ground_truth(
        traj,
        rgb_times,
        gt_data,
        max_diff=config.MAX_GT_DIFF,
        dataset_dir=config.DATASET_DIR,
    )
    if aligned is None:
        return None
    return float(aligned["rms"])


def _loop_detected(prefix):
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{prefix}_closure_info.npy"))


def main():
    gt_data = read_tum_trajectory(config.GT_PATH)
    rows = []
    for label, prefix in METHODS:
        stats_path = os.path.join(OUTPUT_DIR, f"{prefix}_stats.npy")
        raw_traj_path = os.path.join(OUTPUT_DIR, f"{prefix}_raw_trajectory.npy")
        corrected_traj_path = os.path.join(OUTPUT_DIR, f"{prefix}_corrected_trajectory.npy")
        time_path = os.path.join(OUTPUT_DIR, f"{prefix}_used_rgb_times.npy")
        if not (
            os.path.exists(stats_path)
            and os.path.exists(raw_traj_path)
            and os.path.exists(corrected_traj_path)
            and os.path.exists(time_path)
        ):
            continue
        raw_rms = _rms(prefix, "raw", gt_data)
        corrected_rms = _rms(prefix, "corrected", gt_data)
        if raw_rms is None or corrected_rms is None:
            continue
        rows.append({
            "method": label,
            "prefix": prefix,
            "loop_detected": "yes" if _loop_detected(prefix) else "no",
            "corrected_ate_rmse_m": f"{corrected_rms:.6f}",
        })
    if len(rows) == 0:
        print("\nPart 14")
        print("method outputs not found")
        return

    out_csv = os.path.join(OUTPUT_DIR, "part_17_method_comparison_table.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "prefix", "loop_detected", "corrected_ate_rmse_m"])
        for row in rows:
            writer.writerow([
                row["method"],
                row["prefix"],
                row["loop_detected"],
                row["corrected_ate_rmse_m"],
            ])

    bar_out = os.path.join(OUTPUT_DIR, "part_17_method_comparison_bar.png")
    bar_named_out = os.path.join(OUTPUT_DIR, "part_17_method_comparison_bar__ICP_RANSAC__ICP_LMedS__GICP_RANSAC__GICP_LMedS.png")
    plt.figure()
    plt.bar(
        [row["method"] for row in rows],
        [float(row["corrected_ate_rmse_m"]) for row in rows],
    )
    plt.ylabel("ATE RMSE (m)")
    plt.title("ATE RMSE by method")
    plt.xticks(rotation=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(bar_out)
    plt.savefig(bar_named_out)
    plt.close()
    for old_name in [
        "part_17_method_comparison_overlay_xy.png",
        "part_17_method_comparison_overlay_3d.png",
    ]:
        old_path = os.path.join(OUTPUT_DIR, old_name)
        if os.path.exists(old_path):
            os.remove(old_path)

    print("\nPart 14")
    best_row = rows[0]
    for row in rows[1:]:
        if float(row["corrected_ate_rmse_m"]) < float(best_row["corrected_ate_rmse_m"]):
            best_row = row
    print(f"best: {best_row['method']}")
    print(f"rmse: {best_row['corrected_ate_rmse_m']} m")
    print(f"saved: {os.path.basename(out_csv)}, {os.path.basename(bar_out)}, {os.path.basename(bar_named_out)}")


if __name__ == "__main__":
    main()
