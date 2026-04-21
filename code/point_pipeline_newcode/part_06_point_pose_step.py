import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import part_00_project_config as config
from part_02_orb_point_features import get_feature_3d_correspondences
from part_04_icp_registration import icp_register
from part_05_gicp_registration import gicp_register
from part_17_reconstruct_3d_with_estimated_trajectory import align_points_rigid_kabsch, load_selected_frames, make_transform


def refine_registration(source, target, init, registration_method="icp", allow_large_translation=False):
    if registration_method == "gicp":
        T_refined, fitness, rmse = gicp_register(source, target, threshold=config.ICP_THRESH, init=init)
    else:
        T_refined, fitness, rmse = icp_register(source, target, threshold=config.ICP_THRESH, init=init)

    translation = np.linalg.norm(T_refined[:3, 3])
    too_large = translation > config.MAX_TRANSLATION_JUMP and not allow_large_translation
    bad_fit = fitness < config.MIN_ICP_FITNESS or rmse > config.MAX_ICP_RMSE or too_large
    if bad_fit:
        return init, fitness, rmse, False
    return T_refined, fitness, rmse, True


def estimate_point_transform(
    frame_a,
    frame_b,
    prev_rel_T=None,
    allow_large_translation=False,
    frontend_method=None,
    registration_method="icp",
    feature_method=None,
):
    feature_result = get_feature_3d_correspondences(
        frame_a["gray"],
        frame_b["gray"],
        frame_a["depth"],
        frame_b["depth"],
        frontend_method=frontend_method or "orb",
        feature_method=feature_method or "ransac",
    )
    feature_ok = feature_result is not None
    feature_strong = feature_ok and feature_result["num_ransac"] >= config.MIN_ODOM_RANSAC_INLIERS
    T_feat = None
    if feature_ok:
        R_feat, t_feat = align_points_rigid_kabsch(feature_result["pts3d_B"], feature_result["pts3d_A"])
        T_feat = make_transform(R_feat, t_feat)

    if feature_strong:
        T_init = T_feat
        init_source = "feature"
    elif prev_rel_T is not None:
        T_init = prev_rel_T.copy()
        init_source = "prev_rel"
    else:
        T_init = np.eye(4, dtype=np.float64)
        init_source = "identity"

    T_rel, fitness, rmse, reg_accepted = refine_registration(
        frame_b["pcd_down"],
        frame_a["pcd_down"],
        init=T_init,
        registration_method=registration_method,
        allow_large_translation=allow_large_translation,
    )

    if not reg_accepted:
        if prev_rel_T is not None:
            T_rel = prev_rel_T.copy()
            if init_source != "prev_rel":
                init_source = "prev_rel_fallback"
        else:
            T_rel = np.eye(4, dtype=np.float64)
            if init_source != "identity":
                init_source = "identity_fallback"

    score = 0.0
    if feature_ok:
        score += feature_result["num_ransac"] + 0.1 * feature_result["num_ratio"]
    if reg_accepted:
        score += 20.0 * fitness - 50.0 * rmse

    return {
        "T_rel": T_rel,
        "init_source": init_source,
        "feature_ok": feature_ok,
        "feature_strong": feature_strong,
        "feature_result": feature_result,
        "fitness": fitness,
        "rmse": rmse,
        "icp_accepted": reg_accepted,
        "registration_method": registration_method,
        "score": score,
    }


def _show_case(label, frame_a, frame_b):
    result = estimate_point_transform(
        frame_a,
        frame_b,
        prev_rel_T=np.eye(4),
        frontend_method="orb",
        registration_method="icp",
        feature_method="ransac",
    )

    return {
        "label": label,
        "pair": f"{frame_a['pair_idx']} -> {frame_b['pair_idx']}",
        "init_source": result["init_source"],
        "feature_ok": result["feature_ok"],
        "accepted": result["icp_accepted"],
        "rmse": result["rmse"],
    }


def save_pose_step_summary_figure(rows, out_name):
    labels = [row["label"] for row in rows]
    values = [row["rmse"] for row in rows]
    plt.figure()
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518"][: len(rows)])
    plt.ylabel("RMSE (m)")
    plt.title("One-step pose RMSE")
    plt.grid(True, alpha=0.3)
    for bar, row in zip(bars, rows):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            row["pair"],
            ha="center",
            va="bottom",
        )
    out_path = os.path.join(config.OUTPUT_DIR, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main():
    frames, _, _ = load_selected_frames(show_info=False, frame_step=5, max_frames=17)

    print("\nPart 06: pose step")
    rows = [
        _show_case("Early pair", frames[2], frames[3]),
        _show_case("Later pair", frames[15], frames[16]),
    ]
    for row in rows:
        print(f"{row['label']}: init={row['init_source']}, accepted={row['accepted']}, rmse={row['rmse']:.6f} m")
    out_path = save_pose_step_summary_figure(rows, "part_08_point_pose_step_summary.png")
    print(f"saved: {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
