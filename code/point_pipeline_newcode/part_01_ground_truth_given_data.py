import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import part_00_project_config as config
from best_prefix import get_best_prefix


def read_tum_file_list(txt_path):
    data = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                data.append((float(parts[0]), parts[1]))
    return data


def read_tum_trajectory(txt_path):
    data = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    return np.array(data, dtype=np.float64)


def associate_lists(list_a, list_b, max_diff=0.02):
    result = []
    b_times = np.array([x[0] for x in list_b], dtype=np.float64)
    for t_a, path_a in list_a:
        idx = np.argmin(np.abs(b_times - t_a))
        t_b, path_b = list_b[idx]
        if abs(t_a - t_b) <= max_diff:
            result.append((t_a, path_a, t_b, path_b))
    return result


def associate_gt_with_diffs(rgb_times, gt_data, max_diff=0.05):
    gt_times = gt_data[:, 0]
    gt_xyz_all = gt_data[:, 1:4]
    matched_gt = []
    valid_indices = []
    time_diffs = []
    for i, t in enumerate(rgb_times):
        idx = np.argmin(np.abs(gt_times - t))
        dt = abs(gt_times[idx] - t)
        if dt <= max_diff:
            matched_gt.append(gt_xyz_all[idx])
            valid_indices.append(i)
            time_diffs.append(dt)
    return (
        np.array(matched_gt, dtype=np.float64),
        np.array(valid_indices, dtype=np.int32),
        np.array(time_diffs, dtype=np.float64),
    )


def interpolate_gt_positions(rgb_times, gt_data):
    rgb_times = np.asarray(rgb_times, dtype=np.float64)
    gt_times = gt_data[:, 0].astype(np.float64)
    gt_xyz_all = gt_data[:, 1:4].astype(np.float64)

    in_range = (rgb_times >= gt_times[0]) & (rgb_times <= gt_times[-1])
    valid_indices = np.flatnonzero(in_range).astype(np.int32)
    if len(valid_indices) == 0:
        return np.empty((0, 3), dtype=np.float64), valid_indices

    valid_times = rgb_times[valid_indices]
    x = np.interp(valid_times, gt_times, gt_xyz_all[:, 0])
    y = np.interp(valid_times, gt_times, gt_xyz_all[:, 1])
    z = np.interp(valid_times, gt_times, gt_xyz_all[:, 2])
    gt_interp = np.stack([x, y, z], axis=1)
    return gt_interp.astype(np.float64), valid_indices


def split_gt_segments(gt_xyz0, gt_times, gap_threshold=1.0):
    gt_xyz0 = np.asarray(gt_xyz0, dtype=np.float64)
    gt_times = np.asarray(gt_times, dtype=np.float64)
    if len(gt_xyz0) == 0:
        return []
    parts = []
    start = 0
    for i in range(1, len(gt_xyz0)):
        if gt_times[i] - gt_times[i - 1] > gap_threshold:
            parts.append(gt_xyz0[start:i])
            start = i
    parts.append(gt_xyz0[start:])
    return parts


def select_gt_for_evaluation(rgb_times, gt_data, dataset_dir, max_diff):
    name = os.path.basename(os.path.normpath(dataset_dir)).lower()
    if name == "tum_loop":
        gt_xyz, valid_indices, time_diffs = associate_gt_with_diffs(
            rgb_times,
            gt_data,
            max_diff=max_diff,
        )
        return gt_xyz, valid_indices, time_diffs, "nearest-only"

    gt_xyz, valid_indices = interpolate_gt_positions(rgb_times, gt_data)
    time_diffs = np.empty((0,), dtype=np.float64)
    return gt_xyz, valid_indices, time_diffs, "interpolated"


def load_eval_rgb_times():
    prefix = get_best_prefix("SLAM_RESULT_PREFIX", "gicp_ransac")
    times_path = os.path.join(config.OUTPUT_DIR, f"{prefix}_used_rgb_times.npy")
    if os.path.exists(times_path):
        return np.load(times_path).astype(np.float64), prefix, times_path

    rgbd_pairs, pair_indices = load_selected_pairs(show_info=False)
    rgb_times = np.array([rgbd_pairs[i][0] for i in pair_indices], dtype=np.float64)
    return rgb_times, None, None


_USE_DEFAULT = object()


def load_selected_pairs(show_info=True, frame_step=None, max_frames=_USE_DEFAULT):
    rgb_list = read_tum_file_list(config.RGB_TXT)
    depth_list = read_tum_file_list(config.DEPTH_TXT)
    rgbd_pairs = associate_lists(rgb_list, depth_list, max_diff=config.MAX_ASSOC_DIFF)
    if len(rgbd_pairs) == 0:
        raise RuntimeError("No RGB-depth pairs found.")

    frame_step = config.FRAME_STEP if frame_step is None else frame_step
    max_frames = config.MAX_FRAMES if max_frames is _USE_DEFAULT else max_frames
    pair_indices = list(range(0, len(rgbd_pairs), frame_step))
    if max_frames is not None:
        pair_indices = pair_indices[:max_frames]
    if len(pair_indices) < 2:
        raise RuntimeError("Need at least 2 selected RGB-D pairs.")

    if show_info:
        print(f"{config.DATASET_DIR}: {len(pair_indices)} selected")

    return rgbd_pairs, pair_indices


def _join_segments(gt_segments):
    parts = []
    for seg in gt_segments:
        if len(seg) > 0:
            parts.append(seg)
    if len(parts) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(parts)


def _dims_text(xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    spans = maxs - mins
    return "\n".join([
        f"x min/max/span: {mins[0]:.4f}, {maxs[0]:.4f}, {spans[0]:.4f} m",
        f"y min/max/span: {mins[1]:.4f}, {maxs[1]:.4f}, {spans[1]:.4f} m",
        f"z min/max/span: {mins[2]:.4f}, {maxs[2]:.4f}, {spans[2]:.4f} m",
    ])


def save_full_ground_truth_xy(gt_segments, out_path):
    gt_xyz0 = _join_segments(gt_segments)
    plt.figure()
    for seg in gt_segments:
        plt.plot(seg[:, 0], seg[:, 1], linewidth=1.2)
    if len(gt_xyz0) > 0:
        plt.plot(gt_xyz0[0, 0], gt_xyz0[0, 1], "o")
        plt.plot(gt_xyz0[-1, 0], gt_xyz0[-1, 1], "x")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Ground truth path")
    plt.grid(True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_full_ground_truth_projections_and_3d(gt_segments, out_path):
    gt_xyz0 = _join_segments(gt_segments)
    plt.figure()
    plt.suptitle("Ground truth path")

    plt.subplot(221)
    for seg in gt_segments:
        plt.plot(seg[:, 0], seg[:, 1], linewidth=1.2)
    if len(gt_xyz0) > 0:
        plt.plot(gt_xyz0[0, 0], gt_xyz0[0, 1], "o")
        plt.plot(gt_xyz0[-1, 0], gt_xyz0[-1, 1], "x")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("x-y projection")
    plt.grid(True)

    plt.subplot(222)
    for seg in gt_segments:
        plt.plot(seg[:, 0], seg[:, 2], linewidth=1.2)
    if len(gt_xyz0) > 0:
        plt.plot(gt_xyz0[0, 0], gt_xyz0[0, 2], "o")
        plt.plot(gt_xyz0[-1, 0], gt_xyz0[-1, 2], "x")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("x-z projection")
    plt.grid(True)

    plt.subplot(223)
    for seg in gt_segments:
        plt.plot(seg[:, 1], seg[:, 2], linewidth=1.2)
    if len(gt_xyz0) > 0:
        plt.plot(gt_xyz0[0, 1], gt_xyz0[0, 2], "o")
        plt.plot(gt_xyz0[-1, 1], gt_xyz0[-1, 2], "x")
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.title("y-z projection")
    plt.grid(True)

    ax_3d = plt.subplot(224, projection="3d")
    for seg in gt_segments:
        ax_3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=1.2)
    if len(gt_xyz0) > 0:
        ax_3d.plot([gt_xyz0[0, 0]], [gt_xyz0[0, 1]], [gt_xyz0[0, 2]], "o")
        ax_3d.plot([gt_xyz0[-1, 0]], [gt_xyz0[-1, 1]], [gt_xyz0[-1, 2]], "x")
    ax_3d.set_xlabel("x (m)")
    ax_3d.set_ylabel("y (m)")
    ax_3d.set_zlabel("z (m)")
    ax_3d.set_title("3D trajectory")

    if len(gt_xyz0) > 0:
        plt.figtext(
            0.02,
            0.02,
            _dims_text(gt_xyz0),
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.9},
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_selected_ground_truth_projections_and_3d(gt_xyz0, out_path):
    plt.figure()
    plt.suptitle("Selected ground truth path")

    plt.subplot(221)
    plt.plot(gt_xyz0[:, 0], gt_xyz0[:, 1], linewidth=1.2)
    plt.plot(gt_xyz0[0, 0], gt_xyz0[0, 1], "o")
    plt.plot(gt_xyz0[-1, 0], gt_xyz0[-1, 1], "x")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("x-y projection")
    plt.grid(True)

    plt.subplot(222)
    plt.plot(gt_xyz0[:, 0], gt_xyz0[:, 2], linewidth=1.2)
    plt.plot(gt_xyz0[0, 0], gt_xyz0[0, 2], "o")
    plt.plot(gt_xyz0[-1, 0], gt_xyz0[-1, 2], "x")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("x-z projection")
    plt.grid(True)

    plt.subplot(223)
    plt.plot(gt_xyz0[:, 1], gt_xyz0[:, 2], linewidth=1.2)
    plt.plot(gt_xyz0[0, 1], gt_xyz0[0, 2], "o")
    plt.plot(gt_xyz0[-1, 1], gt_xyz0[-1, 2], "x")
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.title("y-z projection")
    plt.grid(True)

    ax_3d = plt.subplot(224, projection="3d")
    ax_3d.plot(gt_xyz0[:, 0], gt_xyz0[:, 1], gt_xyz0[:, 2], linewidth=1.2)
    ax_3d.plot([gt_xyz0[0, 0]], [gt_xyz0[0, 1]], [gt_xyz0[0, 2]], "o")
    ax_3d.plot([gt_xyz0[-1, 0]], [gt_xyz0[-1, 1]], [gt_xyz0[-1, 2]], "x")
    ax_3d.set_xlabel("x (m)")
    ax_3d.set_ylabel("y (m)")
    ax_3d.set_zlabel("z (m)")
    ax_3d.set_title("3D trajectory")

    plt.figtext(
        0.02,
        0.02,
        _dims_text(gt_xyz0),
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9},
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    rgb_times, best_prefix, times_path = load_eval_rgb_times()
    gt_data = read_tum_trajectory(config.GT_PATH)
    gt_times = gt_data[:, 0]
    gt_xyz_full = gt_data[:, 1:4]
    gt_xyz_full0 = gt_xyz_full - gt_xyz_full[0]
    gt_segments = split_gt_segments(gt_xyz_full0, gt_times, gap_threshold=1.0)
    gt_min = gt_xyz_full0.min(axis=0)
    gt_max = gt_xyz_full0.max(axis=0)
    gt_dims = gt_max - gt_min
    gt_xyz_selected, _, _, gt_mode = select_gt_for_evaluation(
        rgb_times,
        gt_data,
        config.DATASET_DIR,
        config.MAX_GT_DIFF,
    )
    gt_xyz_selected0 = np.empty((0, 3), dtype=np.float64)
    if len(gt_xyz_selected) > 0:
        gt_xyz_selected0 = gt_xyz_selected - gt_xyz_selected[0]

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    xy_out = os.path.join(config.OUTPUT_DIR, "part_01_full_ground_truth_xy.png")
    proj_out = os.path.join(config.OUTPUT_DIR, "part_01_full_ground_truth_projections_3d.png")
    selected_proj_out = os.path.join(config.OUTPUT_DIR, "part_01_selected_ground_truth_projections_3d.png")
    txt_out = os.path.join(config.OUTPUT_DIR, "part_01_full_ground_truth_dimensions.txt")
    save_full_ground_truth_xy(gt_segments, xy_out)
    save_full_ground_truth_projections_and_3d(gt_segments, proj_out)
    if len(gt_xyz_selected0) > 0:
        save_selected_ground_truth_projections_and_3d(gt_xyz_selected0, selected_proj_out)

    lines = [
        f"Dataset: {config.DATASET_DIR}",
        f"GT poses: {len(gt_xyz_full0)}",
        f"GT segments: {len(gt_segments)}",
        f"Selected RGB: {len(rgb_times)}",
        f"Matched GT: {len(gt_xyz_selected0)}",
        f"GT mode: {gt_mode}",
        f"x span (m): {gt_dims[0]:.6f}",
        f"y span (m): {gt_dims[1]:.6f}",
        f"z span (m): {gt_dims[2]:.6f}",
    ]
    if times_path:
        lines.append(f"Times file: {times_path}")
    if best_prefix:
        lines.append(f"Prefix: {best_prefix}")

    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\nPart 01")
    if len(gt_xyz_selected0) > 0:
        print(f"saved: {os.path.basename(xy_out)}, {os.path.basename(proj_out)}, {os.path.basename(selected_proj_out)}, {os.path.basename(txt_out)}")
    else:
        print(f"saved: {os.path.basename(xy_out)}, {os.path.basename(proj_out)}, {os.path.basename(txt_out)}")

if __name__ == "__main__":
    main()
