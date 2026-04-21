import os

import cv2
import numpy as np
import open3d as o3d

import part_00_project_config as config
from part_01_ground_truth_given_data import (
    _USE_DEFAULT,
    associate_lists,
    load_selected_pairs,
    read_tum_file_list,
)
from best_prefix import get_best_prefix


DENSE_PIXEL_STEP = 2
DENSE_VOXEL_SIZE = 0.015


def _read_map_tag():
    value = os.environ.get("SLAM_MAP_TAG", "").strip()
    if value:
        return value
    return "dense"


def _read_map_bool(name, default=False):
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _read_map_int(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return int(value)


def _read_map_float(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return float(value)


def pixel_to_3d(u, v, d):
    z = float(d) / config.depth_scale
    if z <= 0 or z > config.MAX_DEPTH_M:
        return None
    x = (u - config.cx) * z / config.fx
    y = (v - config.cy) * z / config.fy
    return np.array([x, y, z], dtype=np.float64)


def depth_to_points_and_colors(rgb_img, depth_img, pixel_step=4):
    h, w = depth_img.shape
    points = []
    colors = []
    for v in range(0, h, pixel_step):
        for u in range(0, w, pixel_step):
            d = depth_img[v, u]
            if d <= 0:
                continue
            z = float(d) / config.depth_scale
            if z <= 0 or z > config.MAX_DEPTH_M:
                continue
            x = (u - config.cx) * z / config.fx
            y = (v - config.cy) * z / config.fy
            b, g, r = rgb_img[v, u]
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)


def make_o3d_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    if len(points) > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def preprocess_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) == 0:
        return pcd_down
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.0,
            max_nn=30,
        )
    )
    return pcd_down


def make_transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def align_points_rigid_kabsch(src, dst):
    src_cent = np.mean(src, axis=0)
    dst_cent = np.mean(dst, axis=0)
    src_c = src - src_cent
    dst_c = dst - dst_cent
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_cent - R @ src_cent
    return R, t


def apply_rigid_transform(points, R, t):
    return (R @ points.T).T + t


def rms_position_error(est_xyz, gt_xyz):
    diff = est_xyz[: len(gt_xyz)] - gt_xyz[: len(est_xyz)]
    err = np.linalg.norm(diff, axis=1)
    return np.sqrt(np.mean(err ** 2)), err


def interpolate_transform(T, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    R_alpha, _ = cv2.Rodrigues(rvec * alpha)
    T_alpha = np.eye(4, dtype=np.float64)
    T_alpha[:3, :3] = R_alpha
    T_alpha[:3, 3] = alpha * T[:3, 3]
    return T_alpha


def apply_loop_correction(poses_arr, anchor_idx, correction_T, closure_idx=None):
    corrected = poses_arr.copy()
    if anchor_idx >= len(corrected) - 1:
        return corrected
    if closure_idx is None:
        closure_idx = len(corrected) - 1
    closure_idx = int(np.clip(closure_idx, anchor_idx + 1, len(corrected) - 1))
    denom = float(max(1, closure_idx - anchor_idx))
    for i in range(anchor_idx + 1, len(corrected)):
        if i <= closure_idx:
            alpha = (i - anchor_idx) / denom
        else:
            alpha = 1.0
        corrected[i] = interpolate_transform(correction_T, alpha) @ corrected[i]
    return corrected


def load_frame(frame_list_idx, pair_idx, pair_entry, pixel_step, voxel_size):
    t_rgb, rgb_rel, _, depth_rel = pair_entry
    rgb_path = os.path.join(config.DATASET_DIR, rgb_rel)
    depth_path = os.path.join(config.DATASET_DIR, depth_rel)
    color = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise RuntimeError(f"Failed to load pair index {pair_idx}")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    points, colors = depth_to_points_and_colors(color, depth, pixel_step=pixel_step)
    if len(points) == 0:
        return None
    pcd_down = preprocess_cloud(make_o3d_cloud(points, colors), voxel_size)
    return {
        "frame_list_idx": frame_list_idx,
        "pair_idx": pair_idx,
        "t_rgb": t_rgb,
        "rgb_rel": rgb_rel,
        "depth_rel": depth_rel,
        "color": color,
        "gray": gray,
        "depth": depth,
        "points": points,
        "colors": colors,
        "pcd_down": pcd_down,
    }


def load_selected_frames(show_info=True, frame_step=None, max_frames=_USE_DEFAULT):
    rgbd_pairs, pair_indices = load_selected_pairs(show_info=show_info, frame_step=frame_step, max_frames=max_frames)
    frames = []
    kept_pair_indices = []
    skipped = []
    for i, pair_idx in enumerate(pair_indices):
        frame = load_frame(i, pair_idx, rgbd_pairs[pair_idx], config.PIXEL_STEP, config.VOXEL_SIZE)
        if frame is None:
            skipped.append(int(pair_idx))
            continue
        frames.append(frame)
        kept_pair_indices.append(int(pair_idx))
    if len(frames) < 2:
        raise RuntimeError("Need at least 2 valid frames after filtering empty point clouds.")
    if show_info and skipped:
        print("skipped:", skipped[:10], "..." if len(skipped) > 10 else "")
        print("frames:", len(frames))
    return frames, rgbd_pairs, np.array(kept_pair_indices, dtype=np.int32)


def load_frames_by_pair_indices(pair_indices, show_info=True):
    rgb_list = read_tum_file_list(config.RGB_TXT)
    depth_list = read_tum_file_list(config.DEPTH_TXT)
    rgbd_pairs = associate_lists(rgb_list, depth_list, max_diff=config.MAX_ASSOC_DIFF)
    frames = []
    for i, pair_idx in enumerate(pair_indices):
        frame = load_frame(i, int(pair_idx), rgbd_pairs[int(pair_idx)], config.PIXEL_STEP, config.VOXEL_SIZE)
        if frame is None:
            raise RuntimeError(f"Empty point cloud at pair index {pair_idx}")
        frames.append(frame)
    if show_info:
        print("frames:", len(frames))
    return frames


def load_map_frames_by_pair_indices(pair_indices, show_info=True):
    rgb_list = read_tum_file_list(config.RGB_TXT)
    depth_list = read_tum_file_list(config.DEPTH_TXT)
    rgbd_pairs = associate_lists(rgb_list, depth_list, max_diff=config.MAX_ASSOC_DIFF)
    frames = []
    for i, pair_idx in enumerate(pair_indices):
        frame = load_frame(i, int(pair_idx), rgbd_pairs[int(pair_idx)], DENSE_PIXEL_STEP, DENSE_VOXEL_SIZE)
        if frame is None:
            raise RuntimeError(f"Empty point cloud at pair index {pair_idx}")
        frames.append(frame)
    if show_info:
        print("frames:", len(frames))
    return frames


def build_merged_cloud(frames, poses_arr, voxel_size):
    merged = o3d.geometry.PointCloud()
    for frame, pose in zip(frames, poses_arr):
        pcd_global = make_o3d_cloud(frame["points"], frame["colors"])
        pcd_global.transform(pose.copy())
        merged += pcd_global
    return merged.voxel_down_sample(voxel_size=voxel_size)


def apply_statistical_outlier_removal(pcd, mean_k, std_ratio):
    if len(pcd.points) == 0:
        return pcd
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=mean_k,
        std_ratio=std_ratio,
    )
    return filtered


def load_saved_map_inputs(prefix="gicp_ransac", use_corrected=True):
    pose_kind = "corrected" if use_corrected else "raw"
    poses_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_{pose_kind}_poses.npy")
    pair_idx_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_used_pair_indices.npy")
    if not os.path.exists(poses_path) or not os.path.exists(pair_idx_path):
        raise FileNotFoundError(f"missing saved output: {prefix}")
    poses_arr = np.load(poses_path)
    pair_indices = np.load(pair_idx_path).astype(np.int32)
    frames = load_map_frames_by_pair_indices(pair_indices, show_info=True)
    if len(frames) != len(poses_arr):
        raise RuntimeError("Saved poses and saved pair indices do not have the same length.")
    return frames, poses_arr, pair_indices, pose_kind


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    prefix = get_best_prefix("SLAM_MAP_PREFIX", "gicp_ransac")
    use_corrected = os.environ.get("SLAM_MAP_POSE_KIND", "corrected").strip().lower() != "raw"
    pixel_step = DENSE_PIXEL_STEP
    voxel_size = DENSE_VOXEL_SIZE
    tag = _read_map_tag()
    use_sor = _read_map_bool("SLAM_MAP_SOR", False)
    sor_mean_k = _read_map_int("SLAM_MAP_SOR_MEAN_K", 20)
    sor_std_ratio = _read_map_float("SLAM_MAP_SOR_STD", 1.5)

    pose_kind = "corrected" if use_corrected else "raw"
    poses_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_{pose_kind}_poses.npy")
    pair_idx_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_used_pair_indices.npy")
    if not os.path.exists(poses_path) or not os.path.exists(pair_idx_path):
        print("\nPart 17")
        print("saved map input not found")
        return

    frames, poses_arr, pair_indices, pose_kind = load_saved_map_inputs(prefix=prefix, use_corrected=use_corrected)

    merged = build_merged_cloud(frames, poses_arr, voxel_size)
    points_before = len(merged.points)
    if use_sor:
        merged = apply_statistical_outlier_removal(merged, sor_mean_k, sor_std_ratio)
    points_after = len(merged.points)

    print("\nPart 17")
    print(f"prefix: {prefix}")
    print(f"kind: {pose_kind}")
    print(f"frames: {len(frames)}")
    print(f"pixel step: {pixel_step}")
    print(f"voxel size: {voxel_size:.3f}")
    print(f"sor: {'on' if use_sor else 'off'}")
    if use_sor:
        print(f"mean k: {sor_mean_k}")
        print(f"std: {sor_std_ratio:.3f}")

    name = f"part_19_{prefix}_{pose_kind}"
    if tag:
        name += f"_{tag}"
    pcd_path = os.path.join(config.OUTPUT_DIR, f"{name}_mapped_cloud.ply")
    o3d.io.write_point_cloud(pcd_path, merged)

    print(f"saved: {os.path.basename(pcd_path)}")
    if use_sor:
        print(f"points before: {points_before}")
    print(f"points: {points_after}")


if __name__ == "__main__":
    main()
