import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import part_00_project_config as config
from part_17_reconstruct_3d_with_estimated_trajectory import load_selected_frames


PAIR_SPECS = [
    ("early", 2, 3),
    ("middle", 8, 9),
    ("later", 14, 15),
]


def icp_register(source, target, threshold=config.ICP_THRESH, init=np.eye(4)):
    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return reg.transformation, reg.fitness, reg.inlier_rmse


def save_registration_overlay(source, target, transform, out_name, title):
    src = np.asarray(source.points)
    dst = np.asarray(target.points)
    src_h = np.hstack([src, np.ones((len(src), 1), dtype=np.float64)])
    src_aligned = (transform @ src_h.T).T[:, :3]
    keep = min(4000, len(dst), len(src_aligned))
    dst = dst[:keep]
    src = src[:keep]
    src_aligned = src_aligned[:keep]

    plt.figure()
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.scatter(dst[:, 0], dst[:, 2], s=1.0, label="target")
    plt.scatter(src[:, 0], src[:, 2], s=1.0, label="source")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Before registration")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(dst[:, 0], dst[:, 2], s=1.0, label="target")
    plt.scatter(src_aligned[:, 0], src_aligned[:, 2], s=1.0, label="aligned")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("After ICP")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(config.OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def save_summary_plot(rows, out_name):
    labels = [row["label"] for row in rows]
    values = [row["rmse"] for row in rows]
    plt.figure()
    bars = plt.bar(labels, values, color="#4C78A8")
    plt.ylabel("RMSE (m)")
    plt.title("ICP RMSE on three frame pairs")
    plt.grid(True)
    for bar, row in zip(bars, rows):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y, row["pair"], ha="center", va="bottom")
    out_path = os.path.join(config.OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main():
    frames, _, _ = load_selected_frames(show_info=False, frame_step=5, max_frames=16)

    print("\nPart 04: ICP")
    rows = []
    saved_example = None
    for label, idx_a, idx_b in PAIR_SPECS:
        frame_a, frame_b = frames[idx_a], frames[idx_b]
        T_icp, fitness, rmse = icp_register(
            source=frame_b["pcd_down"],
            target=frame_a["pcd_down"],
            init=np.eye(4),
        )
        if label == "later":
            out_name = "part_06_icp_example.png"
            title = f"ICP example ({frame_a['pair_idx']} -> {frame_b['pair_idx']})"
            saved_example = save_registration_overlay(frame_b["pcd_down"], frame_a["pcd_down"], T_icp, out_name, title)
        rows.append({
            "label": label,
            "pair": f"{frame_a['pair_idx']} -> {frame_b['pair_idx']}",
            "fitness": fitness,
            "rmse": rmse,
        })
        print(f"{label}: {rmse:.6f} m, {fitness:.6f}")

    summary_path = save_summary_plot(rows, "part_06_icp_registration_summary.png")
    for old_name in [
        "part_06_icp_registration_overlay_early.png",
        "part_06_icp_registration_overlay_middle.png",
        "part_06_icp_registration_overlay_later.png",
    ]:
        old_path = os.path.join(config.OUTPUT_DIR, old_name)
        if os.path.exists(old_path):
            os.remove(old_path)
    print(f"saved: {os.path.basename(summary_path)}")
    if saved_example is not None:
        print(f"saved: {os.path.basename(saved_example)}")


if __name__ == "__main__":
    main()
