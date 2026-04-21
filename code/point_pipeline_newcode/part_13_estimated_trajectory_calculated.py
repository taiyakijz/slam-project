import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import part_00_project_config as config
from part_01_ground_truth_given_data import read_tum_trajectory
from best_prefix import get_best_prefix
from part_17_reconstruct_3d_with_estimated_trajectory import (
    align_points_rigid_kabsch,
    apply_rigid_transform,
    rms_position_error,
)
from tum_loop_eval.gt_eval import select_gt_positions


TRAJECTORY_KIND = "corrected"


def align_trajectory_to_ground_truth(traj_xyz, rgb_times, gt_data, max_diff=0.05, dataset_dir=None):
    gt_selected = select_gt_positions(
        rgb_times,
        gt_data,
        dataset_dir or config.DATASET_DIR,
        max_diff=max_diff,
    )
    gt_xyz = gt_selected["gt_xyz"]
    valid_indices = gt_selected["valid_indices"]
    if len(valid_indices) < 3:
        return None

    traj_xyz = np.asarray(traj_xyz, dtype=np.float64)
    est_xyz = traj_xyz[valid_indices].copy()
    gt_xyz0 = gt_xyz - gt_xyz[0]
    est_xyz0 = est_xyz - est_xyz[0]

    R, t = align_points_rigid_kabsch(est_xyz0, gt_xyz0)
    est_aligned = apply_rigid_transform(est_xyz0, R, t)
    rms, err = rms_position_error(est_aligned, gt_xyz0)
    return {
        "gt_xyz0": gt_xyz0,
        "est_aligned": est_aligned,
        "valid_indices": valid_indices,
        "rms": rms,
        "error": err,
        "gt_mode": gt_selected["mode"],
        "time_diffs": gt_selected["time_diffs"],
    }


def main():
    prefix = get_best_prefix("SLAM_RESULT_PREFIX", "gicp_ransac")
    traj_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_{TRAJECTORY_KIND}_trajectory.npy")
    times_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_used_rgb_times.npy")

    if not os.path.exists(traj_path) or not os.path.exists(times_path):
        print("\nPart 13")
        print("trajectory not found")
        return

    estimated_xyz = np.load(traj_path)
    rgb_times = np.load(times_path)
    gt_data = read_tum_trajectory(config.GT_PATH)
    aligned = align_trajectory_to_ground_truth(
        estimated_xyz,
        rgb_times,
        gt_data,
        max_diff=config.MAX_GT_DIFF,
        dataset_dir=config.DATASET_DIR,
    )
    if aligned is None:
        raise RuntimeError("Not enough ground-truth poses to evaluate the trajectory.")

    print("\nPart 13")
    print(f"poses: {len(estimated_xyz)}")
    print(f"matched gt: {len(aligned['valid_indices'])}")
    print(f"ate rmse: {aligned['rms']:.6f} m")

    est_aligned = aligned["est_aligned"]
    gt_xyz0 = aligned["gt_xyz0"]
    mins = gt_xyz0.min(axis=0)
    maxs = gt_xyz0.max(axis=0)
    spans = maxs - mins
    dims_text = "\n".join([
        f"x min/max/span: {mins[0]:.4f}, {maxs[0]:.4f}, {spans[0]:.4f} m",
        f"y min/max/span: {mins[1]:.4f}, {maxs[1]:.4f}, {spans[1]:.4f} m",
        f"z min/max/span: {mins[2]:.4f}, {maxs[2]:.4f}, {spans[2]:.4f} m",
    ])
    plt.figure()
    plt.suptitle("Estimated vs ground truth")

    plt.subplot(221)
    plt.plot(est_aligned[:, 0], est_aligned[:, 1], label="estimated")
    plt.plot(gt_xyz0[:, 0], gt_xyz0[:, 1], label="ground truth")
    plt.plot(est_aligned[0, 0], est_aligned[0, 1], "o")
    plt.plot(est_aligned[-1, 0], est_aligned[-1, 1], "x")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("x-y projection")
    plt.grid(True)
    plt.legend()

    plt.subplot(222)
    plt.plot(est_aligned[:, 0], est_aligned[:, 2])
    plt.plot(gt_xyz0[:, 0], gt_xyz0[:, 2])
    plt.plot(est_aligned[0, 0], est_aligned[0, 2], "o")
    plt.plot(est_aligned[-1, 0], est_aligned[-1, 2], "x")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("x-z projection")
    plt.grid(True)

    plt.subplot(223)
    plt.plot(est_aligned[:, 1], est_aligned[:, 2])
    plt.plot(gt_xyz0[:, 1], gt_xyz0[:, 2])
    plt.plot(est_aligned[0, 1], est_aligned[0, 2], "o")
    plt.plot(est_aligned[-1, 1], est_aligned[-1, 2], "x")
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.title("y-z projection")
    plt.grid(True)

    ax = plt.subplot(224, projection="3d")
    ax.plot(est_aligned[:, 0], est_aligned[:, 1], est_aligned[:, 2])
    ax.plot(gt_xyz0[:, 0], gt_xyz0[:, 1], gt_xyz0[:, 2])
    ax.plot([est_aligned[0, 0]], [est_aligned[0, 1]], [est_aligned[0, 2]], "o")
    ax.plot([est_aligned[-1, 0]], [est_aligned[-1, 1]], [est_aligned[-1, 2]], "x")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D trajectory")
    out_proj = os.path.join(config.OUTPUT_DIR, "part_16_estimated_vs_given_ground_truth_projections_3d.png")
    named_out_proj = os.path.join(
        config.OUTPUT_DIR,
        f"part_16_estimated_vs_given_ground_truth_projections_3d__{prefix}.png",
    )
    plt.figtext(
        0.02,
        0.02,
        dims_text,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9},
    )
    plt.tight_layout()
    plt.savefig(out_proj)
    plt.savefig(named_out_proj)
    plt.close()

    old_xy = os.path.join(config.OUTPUT_DIR, "part_16_estimated_vs_given_ground_truth_xy.png")
    if os.path.exists(old_xy):
        os.remove(old_xy)

    txt_out = os.path.join(config.OUTPUT_DIR, "part_16_matched_gt_summary.txt")
    lines = [
        f"Dataset: {config.DATASET_DIR}",
        f"Prefix: {prefix}",
        f"Estimated poses: {len(estimated_xyz)}",
        f"RGB times: {len(rgb_times)}",
        f"Matched GT: {len(aligned['valid_indices'])}",
        f"GT mode: {aligned['gt_mode']}",
        f"ATE RMSE: {aligned['rms']:.6f}",
    ]
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"saved: {os.path.basename(out_proj)}")
    print(f"Saved: {os.path.basename(named_out_proj)}")
    print(f"saved: {os.path.basename(txt_out)}")


if __name__ == "__main__":
    main()
