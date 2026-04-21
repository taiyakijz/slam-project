import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import part_00_project_config as config
from part_01_ground_truth_given_data import read_tum_trajectory
from best_prefix import get_best_prefix
from part_13_estimated_trajectory_calculated import align_trajectory_to_ground_truth


OUTPUT_DIR = config.OUTPUT_DIR


def moving_average_filter(traj_xyz, size=5):
    traj_xyz = np.asarray(traj_xyz, dtype=np.float64)
    if len(traj_xyz) == 0 or size <= 1:
        return traj_xyz.copy()

    step = max(1, size // 2)
    out = np.zeros_like(traj_xyz)
    for i in range(len(traj_xyz)):
        start = max(0, i - step)
        end = min(len(traj_xyz), i + step + 1)
        out[i] = np.mean(traj_xyz[start:end], axis=0)
    return out


def kalman_filter_trajectory(
    traj_xyz,
    pos_noise=1e-3,
    vel_noise=5e-3,
    meas_noise=2e-2,
):
    traj_xyz = np.asarray(traj_xyz, dtype=np.float64)
    if len(traj_xyz) == 0:
        return traj_xyz.copy()

    move = np.eye(6, dtype=np.float64)
    move[0, 3] = 1.0
    move[1, 4] = 1.0
    move[2, 5] = 1.0

    observe = np.zeros((3, 6), dtype=np.float64)
    observe[:, :3] = np.eye(3, dtype=np.float64)

    process = np.diag([pos_noise, pos_noise, pos_noise, vel_noise, vel_noise, vel_noise])
    measure = np.eye(3, dtype=np.float64) * meas_noise
    eye6 = np.eye(6, dtype=np.float64)

    state = np.zeros(6, dtype=np.float64)
    state[:3] = traj_xyz[0]
    cov = np.eye(6, dtype=np.float64)

    out = np.zeros_like(traj_xyz)
    out[0] = traj_xyz[0]

    for i in range(1, len(traj_xyz)):
        point = traj_xyz[i]
        state_pred = move @ state
        cov_pred = move @ cov @ move.T + process

        diff = point - (observe @ state_pred)
        s = observe @ cov_pred @ observe.T + measure
        gain = cov_pred @ observe.T @ np.linalg.inv(s)

        state = state_pred + gain @ diff
        cov = (eye6 - gain @ observe) @ cov_pred
        out[i] = state[:3]

    return out


def main():
    prefix = get_best_prefix("SLAM_FILTER_PREFIX", "gicp_ransac")
    traj_path = os.path.join(OUTPUT_DIR, f"{prefix}_corrected_trajectory.npy")
    times_path = os.path.join(OUTPUT_DIR, f"{prefix}_used_rgb_times.npy")
    if not (os.path.exists(traj_path) and os.path.exists(times_path)):
        print("\nPart 16")
        print("trajectory not found")
        return

    traj = np.load(traj_path)
    rgb_times = np.load(times_path)
    gt_data = read_tum_trajectory(config.GT_PATH)

    methods = [
        ("unfiltered", traj),
        ("moving average", moving_average_filter(traj, size=5)),
        ("kalman", kalman_filter_trajectory(traj)),
    ]

    rows = []
    overlays = []
    gt_overlay = None
    for label, arr in methods:
        aligned = align_trajectory_to_ground_truth(arr, rgb_times, gt_data, max_diff=config.MAX_GT_DIFF)
        if aligned is None:
            rms = "unavailable"
        else:
            rms = f"{aligned['rms']:.6f}"
            overlays.append((label, aligned["est_aligned"]))
            gt_overlay = aligned["gt_xyz0"]
        rows.append({"filter": label, "ate_rmse_m": rms})

    out_csv = os.path.join(OUTPUT_DIR, "part_18_filter_comparison_table.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filter", "ate_rmse_m"])
        for row in rows:
            writer.writerow([row["filter"], row["ate_rmse_m"]])

    out_img = os.path.join(OUTPUT_DIR, f"{prefix}_filter_comparison.png")
    named_out_img = os.path.join(OUTPUT_DIR, f"part_18_filter_comparison__{prefix}.png")
    if gt_overlay is not None and len(overlays) > 0:
        plt.figure()
        for label, est_aligned in overlays:
            plt.plot(est_aligned[:, 0], est_aligned[:, 1], label=label)
        plt.plot(gt_overlay[:, 0], gt_overlay[:, 1], label="ground truth")
        plt.plot(gt_overlay[0, 0], gt_overlay[0, 1], "o")
        plt.plot(gt_overlay[-1, 0], gt_overlay[-1, 1], "x")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Trajectory filter comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_img)
        plt.savefig(named_out_img)
        plt.close()
        out_img_3d = os.path.join(OUTPUT_DIR, f"{prefix}_filter_comparison_3d.png")
        if os.path.exists(out_img_3d):
            os.remove(out_img_3d)

        print("\nPart 16")
    best_row = None
    for row in rows:
        if row["ate_rmse_m"] == "unavailable":
            continue
        if best_row is None or float(row["ate_rmse_m"]) < float(best_row["ate_rmse_m"]):
            best_row = row
    if best_row is not None:
        print(f"best: {best_row['filter']}")
        print(f"rmse: {best_row['ate_rmse_m']} m")
    print(f"saved: {os.path.basename(out_csv)}")
    if gt_overlay is not None and len(overlays) > 0:
        print(f"saved: {os.path.basename(out_img)}, {os.path.basename(named_out_img)}")


if __name__ == "__main__":
    main()
