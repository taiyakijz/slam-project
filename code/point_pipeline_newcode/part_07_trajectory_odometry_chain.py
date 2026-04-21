import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import part_00_project_config as config
from part_01_ground_truth_given_data import (
    _USE_DEFAULT,
    associate_gt_with_diffs,
    read_tum_trajectory,
)
from part_02_orb_point_features import save_match_visual
from part_06_point_pose_step import estimate_point_transform
from part_17_reconstruct_3d_with_estimated_trajectory import (
    align_points_rigid_kabsch,
    apply_loop_correction,
    apply_rigid_transform,
    invert_transform,
    load_selected_frames,
    make_o3d_cloud,
    rms_position_error,
)
from tum_loop_eval.gt_eval import select_gt_positions
from tum_loop_eval.global_loop import search_global_loop_candidate, use_global_loop_fallback


def build_merged_cloud(frames, poses_arr):
    merged = o3d.geometry.PointCloud()
    for frame, pose in zip(frames, poses_arr):
        pcd_global = make_o3d_cloud(frame["points"], frame["colors"])
        pcd_global.transform(pose.copy())
        merged += pcd_global
    return merged.voxel_down_sample(voxel_size=config.VOXEL_SIZE)


def save_trajectory_plot(traj_arr, title, out_name, use_3d=False):
    if use_3d:
        plt.figure()
        ax = plt.subplot(111, projection="3d")
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], traj_arr[:, 2], marker="o")
        ax.plot([traj_arr[0, 0]], [traj_arr[0, 1]], [traj_arr[0, 2]], "o")
        ax.plot([traj_arr[-1, 0]], [traj_arr[-1, 1]], [traj_arr[-1, 2]], "x")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(title)
        plt.savefig(os.path.join(config.OUTPUT_DIR, out_name))
        plt.close()
        return

    plt.figure()
    plt.plot(traj_arr[:, 0], traj_arr[:, 1], marker="o")
    plt.plot(traj_arr[0, 0], traj_arr[0, 1], "o")
    plt.plot(traj_arr[-1, 0], traj_arr[-1, 1], "x")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, out_name))
    plt.close()


def save_raw_vs_corrected(raw_traj_arr, corrected_traj_arr, prefix):
    plt.figure()
    plt.plot(raw_traj_arr[:, 0], raw_traj_arr[:, 1], marker="o", label="Raw")
    plt.plot(corrected_traj_arr[:, 0], corrected_traj_arr[:, 1], marker="x", label="Loop-corrected")
    plt.plot(corrected_traj_arr[0, 0], corrected_traj_arr[0, 1], "o")
    plt.plot(corrected_traj_arr[-1, 0], corrected_traj_arr[-1, 1], "x")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Raw vs loop-corrected trajectory")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, f"{prefix}_raw_vs_corrected.png"), dpi=200)
    plt.close()


def save_gt_overlay(est_aligned, gt_xyz0, title, out_name, use_3d=False):
    if use_3d:
        plt.figure()
        ax = plt.subplot(111, projection="3d")
        ax.plot(est_aligned[:, 0], est_aligned[:, 1], est_aligned[:, 2], label="estimated")
        ax.plot(gt_xyz0[:, 0], gt_xyz0[:, 1], gt_xyz0[:, 2], label="ground truth")
        ax.plot([est_aligned[0, 0]], [est_aligned[0, 1]], [est_aligned[0, 2]], "o")
        ax.plot([est_aligned[-1, 0]], [est_aligned[-1, 1]], [est_aligned[-1, 2]], "x")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(title)
        ax.legend()
        plt.savefig(os.path.join(config.OUTPUT_DIR, out_name))
        plt.close()
        return

    plt.figure()
    plt.plot(est_aligned[:, 0], est_aligned[:, 1], label="estimated")
    plt.plot(gt_xyz0[:, 0], gt_xyz0[:, 1], label="ground truth")
    plt.plot(est_aligned[0, 0], est_aligned[0, 1], "o")
    plt.plot(est_aligned[-1, 0], est_aligned[-1, 1], "x")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, out_name))
    plt.close()


def save_gt_comparison_overlay(trajectories, gt_xyz0, title, out_name, use_3d=False):
    if use_3d:
        plt.figure()
        ax = plt.subplot(111, projection="3d")
        for label, arr, marker in trajectories:
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], label=label)
        ax.plot(gt_xyz0[:, 0], gt_xyz0[:, 1], gt_xyz0[:, 2], label="ground truth")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(title)
        ax.legend()
        plt.savefig(os.path.join(config.OUTPUT_DIR, out_name))
        plt.close()
        return

    plt.figure()
    for label, arr, marker in trajectories:
        plt.plot(arr[:, 0], arr[:, 1], label=label)
    plt.plot(gt_xyz0[:, 0], gt_xyz0[:, 1], label="ground truth")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, out_name))
    plt.close()


def save_raw_outputs(frames, poses_arr, used_rgb_times, used_pair_indices, odom_stats, prefix):
    raw_traj_arr = poses_arr[:, :3, 3].copy()
    raw_map = build_merged_cloud(frames, poses_arr)
    raw_map_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_raw_map.ply")
    o3d.io.write_point_cloud(raw_map_path, raw_map)

    np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_raw_poses.npy"), poses_arr)
    np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_raw_trajectory.npy"), raw_traj_arr)
    np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_used_rgb_times.npy"), used_rgb_times)
    np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_used_pair_indices.npy"), used_pair_indices)
    np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_stats.npy"), odom_stats)

    save_trajectory_plot(raw_traj_arr, "Raw trajectory (x-y projection)", f"{prefix}_raw_trajectory.png")
    save_trajectory_plot(raw_traj_arr, "Raw trajectory (3D)", f"{prefix}_raw_trajectory_3d.png", use_3d=True)
    return raw_traj_arr, raw_map_path


def save_gt_comparison_csv(
    prefix,
    used_rgb_times,
    used_pair_indices,
    valid_indices,
    est_xyz,
    est_aligned,
    gt_xyz0,
    position_error,
):
    _ = est_xyz
    out_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_gt_compare.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rgb_time",
            "pair_idx",
            "est_x",
            "est_y",
            "est_z",
            "gt_x",
            "gt_y",
            "gt_z",
            "error_m",
        ])
        for row_idx, pose_idx in enumerate(valid_indices):
            writer.writerow([
                float(used_rgb_times[pose_idx]),
                int(used_pair_indices[pose_idx]),
                float(est_aligned[row_idx, 0]),
                float(est_aligned[row_idx, 1]),
                float(est_aligned[row_idx, 2]),
                float(gt_xyz0[row_idx, 0]),
                float(gt_xyz0[row_idx, 1]),
                float(gt_xyz0[row_idx, 2]),
                float(position_error[row_idx]),
            ])
    return out_path


def choose_saved_match_steps(num_pairs):
    if num_pairs <= 0:
        return set()
    if num_pairs <= 3:
        return set(range(1, num_pairs + 1))

    selected = {1, num_pairs}
    selected.add(max(1, (num_pairs + 1) // 2))
    return selected


def run_odometry_with_options(
    frontend_method="orb",
    registration_method="icp",
    feature_method="ransac",
    prefix=None,
    frame_step=None,
    max_frames=_USE_DEFAULT,
    show_details=False,
):
    prefix = prefix or config.PREFIX
    frames, _, _ = load_selected_frames(frame_step=frame_step, max_frames=max_frames)

    poses = [np.eye(4, dtype=np.float64)]
    used_rgb_times = [frames[0]["t_rgb"]]
    used_pair_indices = [frames[0]["pair_idx"]]
    odom_stats = []
    prev_rel_T = np.eye(4, dtype=np.float64)

    if show_details:
        print("\nOdometry")

    num_pairs = max(0, len(frames) - 1)
    saved_match_steps = choose_saved_match_steps(num_pairs)

    for i in range(1, len(frames)):
        frame_a = frames[i - 1]
        frame_b = frames[i]
        result = estimate_point_transform(
            frame_a,
            frame_b,
            prev_rel_T=prev_rel_T,
            frontend_method=frontend_method,
            registration_method=registration_method,
            feature_method=feature_method,
        )
        T_rel = result["T_rel"]

        if show_details:
            print(f"\nPair {frame_a['pair_idx']} -> {frame_b['pair_idx']}")
            print("  init source:", result["init_source"])
            print("  feature ok :", result["feature_ok"])
            if result["feature_ok"]:
                feat = result["feature_result"]
                print("  Frontend      :", feat["frontend_method"])
                print("  Feature reject:", feat["feature_method"])
                print("  Raw kNN       :", feat["num_raw_knn"])
                print(f"  {feat['frontend_stage_label']:<14}:", feat["num_frontend"])
                print("  After geometry:", feat["num_geometric"])
                print("  Valid 3D matches:", len(feat["pts3d_A"]))
            print("  Registration :", result["registration_method"])
            print("  Accepted     :", result["icp_accepted"])
            print("  Fitness      :", result["fitness"])
            print("  RMSE         :", result["rmse"])

        poses.append(poses[-1] @ T_rel)
        used_rgb_times.append(frame_b["t_rgb"])
        used_pair_indices.append(frame_b["pair_idx"])
        prev_rel_T = T_rel.copy()

        odom_stats.append([
            frame_a["pair_idx"],
            frame_b["pair_idx"],
            1.0 if result["feature_ok"] else 0.0,
            float(result["feature_result"]["num_ransac"]) if result["feature_ok"] else 0.0,
            float(result["fitness"]),
            float(result["rmse"]),
            float(result["score"]),
        ])

        if result["feature_ok"] and i in saved_match_steps:
            save_match_visual(
                frame_a,
                frame_b,
                result["feature_result"],
                f"{prefix}_odom_matches_{frame_a['pair_idx']}_{frame_b['pair_idx']}.jpg",
            )

    poses_arr = np.stack(poses, axis=0)
    used_rgb_times = np.array(used_rgb_times, dtype=np.float64)
    used_pair_indices = np.array(used_pair_indices, dtype=np.int32)
    odom_stats = np.array(odom_stats, dtype=np.float64) if len(odom_stats) > 0 else np.empty((0, 7), dtype=np.float64)

    raw_traj_arr, raw_map_path = save_raw_outputs(
        frames,
        poses_arr,
        used_rgb_times,
        used_pair_indices,
        odom_stats,
        prefix,
    )

    if show_details:
        print("\nSaved:")
        print(" -", os.path.join(config.OUTPUT_DIR, f"{prefix}_raw_poses.npy"))
        print(" -", raw_map_path)

    return {
        "frames": frames,
        "poses_arr": poses_arr,
        "raw_traj_arr": raw_traj_arr,
        "used_rgb_times": used_rgb_times,
        "used_pair_indices": used_pair_indices,
        "odom_stats": odom_stats,
        "prefix": prefix,
        "frontend_method": frontend_method,
        "registration_method": registration_method,
        "feature_method": feature_method,
        "frame_step": config.FRAME_STEP if frame_step is None else frame_step,
        "max_frames": None if max_frames is _USE_DEFAULT else max_frames,
    }


def load_saved_odometry_inputs(prefix=None):
    prefix = prefix or config.PREFIX
    frames, _, _ = load_selected_frames()
    poses_arr = np.load(os.path.join(config.OUTPUT_DIR, f"{prefix}_raw_poses.npy"))
    used_rgb_times = np.load(os.path.join(config.OUTPUT_DIR, f"{prefix}_used_rgb_times.npy"))
    used_pair_indices = np.load(os.path.join(config.OUTPUT_DIR, f"{prefix}_used_pair_indices.npy"))
    stats_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_stats.npy")
    odom_stats = np.load(stats_path) if os.path.exists(stats_path) else np.empty((0, 7), dtype=np.float64)
    return {
        "frames": frames,
        "poses_arr": poses_arr,
        "raw_traj_arr": poses_arr[:, :3, 3].copy(),
        "used_rgb_times": used_rgb_times,
        "used_pair_indices": used_pair_indices,
        "odom_stats": odom_stats,
        "prefix": prefix,
    }


def run_loop_closure(
    odometry_result=None,
    frontend_method="orb",
    registration_method="icp",
    feature_method="ransac",
    prefix=None,
    show_details=False,
):
    if odometry_result is None:
        odometry_result = load_saved_odometry_inputs(prefix=prefix)

    frames = odometry_result["frames"]
    poses_arr = odometry_result["poses_arr"]
    raw_traj_arr = odometry_result["raw_traj_arr"]
    used_rgb_times = odometry_result["used_rgb_times"]
    used_pair_indices = odometry_result["used_pair_indices"]
    prefix = odometry_result.get("prefix", prefix or config.PREFIX)
    frontend_method = odometry_result.get("frontend_method", frontend_method)
    feature_method = odometry_result.get("feature_method", feature_method)
    selected_frame_step = odometry_result.get("frame_step", config.FRAME_STEP)
    selected_max_frames = odometry_result.get("max_frames", config.MAX_FRAMES)

    if show_details:
        print("\nLoop search")

    best_loop = None
    num_frames = len(frames)
    head_limit = min(config.LOOP_HEAD_FRAMES, num_frames - 1)
    query_start_idx = max(
        config.MIN_LOOP_SEPARATION + 1,
        int(np.floor((num_frames - 1) * config.LOOP_QUERY_START_FRACTION)),
    )
    query_stride = max(1, int(config.LOOP_QUERY_STRIDE))

    for query_idx in range(query_start_idx, num_frames, query_stride):
        query_frame = frames[query_idx]
        for candidate_idx in range(head_limit):
            if query_idx - candidate_idx < config.MIN_LOOP_SEPARATION:
                continue
            candidate_frame = frames[candidate_idx]
            result = estimate_point_transform(
                candidate_frame,
                query_frame,
                prev_rel_T=None,
                allow_large_translation=True,
                frontend_method=frontend_method,
                registration_method=registration_method,
                feature_method=feature_method,
            )
            feat_inliers = 0 if not result["feature_ok"] else result["feature_result"]["num_ransac"]
            if show_details:
                print(
                    f"Candidate {candidate_frame['pair_idx']} <- {query_frame['pair_idx']} | "
                    f"feature={result['feature_ok']} "
                    f"inliers={feat_inliers} fitness={result['fitness']:.4f} "
                    f"rmse={result['rmse']:.4f} score={result['score']:.2f}"
                )

            feature_valid = result["feature_ok"] and feat_inliers >= config.MIN_LOOP_RANSAC_INLIERS
            if not feature_valid:
                continue
            if best_loop is None or result["score"] > best_loop["score"]:
                best_loop = {
                    "candidate_idx": candidate_idx,
                    "query_idx": query_idx,
                    "candidate_pair_idx": candidate_frame["pair_idx"],
                    "query_pair_idx": query_frame["pair_idx"],
                    "score": result["score"],
                    "result": result,
                    "source": "feature",
                }

    if (best_loop is None or best_loop["score"] < config.MIN_LOOP_SCORE) and use_global_loop_fallback(config.DATASET_DIR):
        global_loop = search_global_loop_candidate(frames, registration_method=registration_method)
        if global_loop is not None:
            best_loop = {
                "candidate_idx": global_loop["candidate_idx"],
                "query_idx": global_loop["query_idx"],
                "candidate_pair_idx": global_loop["candidate_pair_idx"],
                "query_pair_idx": global_loop["query_pair_idx"],
                "score": global_loop["score"],
                "result": {
                    "T_rel": global_loop["T_rel"],
                    "fitness": global_loop["fitness"],
                    "rmse": global_loop["rmse"],
                    "feature_ok": False,
                    "feature_result": None,
                },
                "source": global_loop["source"],
            }

    corrected_poses_arr = poses_arr.copy()
    loop_detected = False
    loop_summary = {}
    if best_loop is not None and best_loop["score"] >= config.MIN_LOOP_SCORE:
        candidate_idx = best_loop["candidate_idx"]
        query_idx = best_loop["query_idx"]
        T_loop = best_loop["result"]["T_rel"]
        desired_query_pose = poses_arr[candidate_idx] @ T_loop
        correction_T = desired_query_pose @ invert_transform(poses_arr[query_idx])
        corrected_poses_arr = apply_loop_correction(
            poses_arr,
            candidate_idx,
            correction_T,
            closure_idx=query_idx,
        )
        loop_detected = True

        feature_result = best_loop["result"].get("feature_result")
        if feature_result is not None:
            save_match_visual(
                frames[candidate_idx],
                frames[query_idx],
                feature_result,
                f"{prefix}_closure_match_{best_loop['candidate_pair_idx']}_{best_loop['query_pair_idx']}.jpg",
            )

        loop_summary = {
            "candidate_idx": candidate_idx,
            "query_idx": query_idx,
            "candidate_pair_idx": best_loop["candidate_pair_idx"],
            "query_pair_idx": best_loop["query_pair_idx"],
            "score": best_loop["score"],
            "fitness": best_loop["result"]["fitness"],
            "rmse": best_loop["result"]["rmse"],
            "source": best_loop.get("source", "feature"),
        }
        if show_details:
            print("\nLoop found.")
            print("  Candidate pair index:", best_loop["candidate_pair_idx"])
            print("  Query pair index    :", best_loop["query_pair_idx"])
            print("  Loop score          :", best_loop["score"])
            print("  Loop source         :", best_loop.get("source", "feature"))
    elif show_details:
        print("\nNo loop used.")

    corrected_traj_arr = corrected_poses_arr[:, :3, 3].copy()
    corrected_map = build_merged_cloud(frames, corrected_poses_arr)
    corrected_map_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_corrected_map.ply")
    o3d.io.write_point_cloud(corrected_map_path, corrected_map)

    np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_corrected_poses.npy"), corrected_poses_arr)
    np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_corrected_trajectory.npy"), corrected_traj_arr)

    if loop_detected:
        np.save(
            os.path.join(config.OUTPUT_DIR, f"{prefix}_closure_info.npy"),
            np.array([
                loop_summary["candidate_idx"],
                loop_summary["query_idx"],
                loop_summary["candidate_pair_idx"],
                loop_summary["query_pair_idx"],
                loop_summary["score"],
                loop_summary["fitness"],
                loop_summary["rmse"],
            ], dtype=np.float64),
        )

    lines = [
        f"Dataset: {config.DATASET_DIR}",
        f"Frontend: {frontend_method}",
        f"Registration: {registration_method}",
        f"Reject: {feature_method}",
        f"Frame step: {selected_frame_step}",
        f"Max frames: {selected_max_frames}",
        f"Frames: {len(frames)}",
        f"Loop: {loop_detected}",
    ]
    if loop_detected:
        lines.append(f"Candidate: {loop_summary['candidate_pair_idx']}")
        lines.append(f"Query: {loop_summary['query_pair_idx']}")

    save_trajectory_plot(corrected_traj_arr, "Corrected trajectory (x-y projection)", f"{prefix}_corrected_trajectory.png")
    save_trajectory_plot(corrected_traj_arr, "Corrected trajectory (3D)", f"{prefix}_corrected_trajectory_3d.png", use_3d=True)
    save_raw_vs_corrected(raw_traj_arr, corrected_traj_arr, prefix)

    raw_rms = None
    corrected_rms = None
    if os.path.exists(config.GT_PATH):
        gt_data = read_tum_trajectory(config.GT_PATH)
        gt_selected = select_gt_positions(
            used_rgb_times,
            gt_data,
            config.DATASET_DIR,
            max_diff=config.MAX_GT_DIFF,
        )
        gt_xyz_eval = gt_selected["gt_xyz"]
        valid_indices_eval = gt_selected["valid_indices"]
        gt_time_diffs = gt_selected["time_diffs"]
        gt_mode = gt_selected["mode"]
        if gt_mode == "nearest-only":
            valid_indices_nearest = valid_indices_eval
        else:
            _, valid_indices_nearest, _ = associate_gt_with_diffs(
                used_rgb_times,
                gt_data,
                max_diff=config.MAX_GT_DIFF,
            )
        if len(valid_indices_eval) >= 3:
            raw_xyz = raw_traj_arr[valid_indices_eval].copy()
            corrected_xyz = corrected_traj_arr[valid_indices_eval].copy()
            gt_xyz0 = gt_xyz_eval - gt_xyz_eval[0]
            raw_xyz0 = raw_xyz - raw_xyz[0]
            corrected_xyz0 = corrected_xyz - corrected_xyz[0]

            R_raw, t_raw = align_points_rigid_kabsch(raw_xyz0, gt_xyz0)
            raw_aligned = apply_rigid_transform(raw_xyz0, R_raw, t_raw)
            raw_rms, raw_err = rms_position_error(raw_aligned, gt_xyz0)

            R_corr, t_corr = align_points_rigid_kabsch(corrected_xyz0, gt_xyz0)
            corrected_aligned = apply_rigid_transform(corrected_xyz0, R_corr, t_corr)
            corrected_rms, corrected_err = rms_position_error(corrected_aligned, gt_xyz0)

            np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_gt_xyz.npy"), gt_xyz0)
            np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_gt_interp_valid_indices.npy"), valid_indices_eval)
            np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_gt_nearest_valid_indices.npy"), valid_indices_nearest)
            np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_raw_est_xyz_aligned.npy"), raw_aligned)
            np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_corrected_est_xyz_aligned.npy"), corrected_aligned)
            np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_raw_position_error.npy"), raw_err)
            np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_corrected_position_error.npy"), corrected_err)
            np.save(os.path.join(config.OUTPUT_DIR, f"{prefix}_gt_time_diffs.npy"), gt_time_diffs)
            raw_csv_path = save_gt_comparison_csv(
                f"{prefix}_raw",
                used_rgb_times,
                used_pair_indices,
                valid_indices_eval,
                raw_xyz0,
                raw_aligned,
                gt_xyz0,
                raw_err,
            )
            corrected_csv_path = save_gt_comparison_csv(
                f"{prefix}_corrected",
                used_rgb_times,
                used_pair_indices,
                valid_indices_eval,
                corrected_xyz0,
                corrected_aligned,
                gt_xyz0,
                corrected_err,
            )

            save_gt_overlay(raw_aligned, gt_xyz0, "Raw vs ground truth (x-y projection)", f"{prefix}_raw_vs_groundtruth.png")
            save_gt_overlay(corrected_aligned, gt_xyz0, "Corrected vs ground truth (x-y projection)", f"{prefix}_corrected_vs_groundtruth.png")
            save_gt_overlay(corrected_aligned, gt_xyz0, "Corrected vs ground truth (3D trajectory)", f"{prefix}_corrected_vs_groundtruth_3d.png", use_3d=True)
            save_gt_comparison_overlay(
                [
                    ("Raw aligned", raw_aligned, "o"),
                    ("Corrected aligned", corrected_aligned, "^"),
                ],
                gt_xyz0,
                "Raw and corrected vs ground truth (x-y projection)",
                f"{prefix}_raw_corrected_vs_groundtruth.png",
            )
            save_gt_comparison_overlay(
                [
                    ("Raw aligned", raw_aligned, "o"),
                    ("Corrected aligned", corrected_aligned, "^"),
                ],
                gt_xyz0,
                "Raw and corrected vs ground truth (3D trajectory)",
                f"{prefix}_raw_corrected_vs_groundtruth_3d.png",
                use_3d=True,
            )

            if show_details:
                print("\nGround truth")
                print("  Evaluation mode    :", gt_mode)
                print("  Matched poses      :", len(corrected_aligned))
                print("  Nearest matches    :", len(valid_indices_nearest))
                if len(gt_time_diffs) > 0:
                    print("  Mean nearest |dt| (s):", float(np.mean(gt_time_diffs)))
                    print("  Max nearest |dt| (s) :", float(np.max(gt_time_diffs)))
                print("  Raw RMS error (m)   :", raw_rms)
                print("  Corrected RMS error :", corrected_rms)
                print("  Raw GT CSV          :", raw_csv_path)
                print("  Corrected GT CSV    :", corrected_csv_path)
        elif show_details:
            print("\nNot enough GT poses.")
    elif show_details:
        print("\nNo groundtruth.txt found.")

    if raw_rms is not None and corrected_rms is not None:
        lines.append(f"GT mode: {gt_mode}")
        lines.append(f"Matched GT: {len(valid_indices_eval)}")
        lines.append(f"Raw RMSE: {raw_rms:.6f}")
        lines.append(f"Corrected RMSE: {corrected_rms:.6f}")
    else:
        lines.append("GT: unavailable")

    with open(os.path.join(config.OUTPUT_DIR, f"{prefix}_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if show_details:
        print("\nSaved:")
        print(" -", os.path.join(config.OUTPUT_DIR, f"{prefix}_corrected_poses.npy"))
        print(" -", corrected_map_path)

    return {
        "frames": frames,
        "poses_arr": poses_arr,
        "raw_traj_arr": raw_traj_arr,
        "used_rgb_times": used_rgb_times,
        "used_pair_indices": used_pair_indices,
        "corrected_poses_arr": corrected_poses_arr,
        "corrected_traj_arr": corrected_traj_arr,
        "loop_detected": loop_detected,
        "loop_summary": loop_summary,
        "prefix": prefix,
        "frontend_method": frontend_method,
        "registration_method": registration_method,
        "feature_method": feature_method,
        "raw_rms": raw_rms,
        "corrected_rms": corrected_rms,
        "frame_step": selected_frame_step,
        "max_frames": selected_max_frames,
    }


def run_full_pipeline(
    frontend_method="orb",
    registration_method="icp",
    feature_method="ransac",
    prefix=None,
    frame_step=None,
    max_frames=_USE_DEFAULT,
    show_details=False,
):
    odometry_result = run_odometry_with_options(
        frontend_method=frontend_method,
        registration_method=registration_method,
        feature_method=feature_method,
        prefix=prefix,
        frame_step=frame_step,
        max_frames=max_frames,
        show_details=show_details,
    )
    return run_loop_closure(
        odometry_result=odometry_result,
        frontend_method=frontend_method,
        registration_method=registration_method,
        feature_method=feature_method,
        prefix=odometry_result["prefix"],
        show_details=show_details,
    )


def main():
    print("\nPart 07 is a shared module.")
    print("Run it through:")
    print("part_08_run_baseline_icp_ransac.py")
    print("part_09_run_lmeds_rejection.py")
    print("part_10_run_gicp_registration.py")
    print("No separate demo file is saved.")


if __name__ == "__main__":
    main()
