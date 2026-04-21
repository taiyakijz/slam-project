import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import part_00_project_config as config
from part_02_orb_point_features import (
    _estimate_homography,
    _get_frontend_matches,
)
from part_17_reconstruct_3d_with_estimated_trajectory import load_selected_frames


DRAW_MATCH_LIMIT = 300


def _count_geometric_inliers(good, kp_a, kp_b, method):
    if len(good) < 4:
        return []
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    _, mask, _ = _estimate_homography(pts_a, pts_b, method)
    if mask is None:
        return []
    keep = mask.ravel().astype(bool)
    return [good[i] for i in range(len(good)) if keep[i]]


def _draw_match_panel(frame_a, frame_b, kp_a, kp_b, matches, title):
    vis = cv2.drawMatches(
        frame_a["color"],
        kp_a,
        frame_b["color"],
        kp_b,
        matches[:DRAW_MATCH_LIMIT],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    plt.imshow(vis)
    plt.title(title)
    plt.axis("off")


def _save_match_panel(frame_a, frame_b, kp_a, kp_b, matches, title, out_path):
    plt.figure(figsize=(10, 4))
    _draw_match_panel(frame_a, frame_b, kp_a, kp_b, matches, title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    frames, _, _ = load_selected_frames(show_info=False, frame_step=5, max_frames=16)
    frame_a, frame_b = frames[14], frames[15]
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\nPart 03: rejection")

    frontend_rows = []
    plot_labels = []
    plot_values = []
    for frontend_method in ["orb", "shi_tomasi_lk"]:
        frontend = _get_frontend_matches(frame_a["gray"], frame_b["gray"], frontend_method=frontend_method)
        if frontend is None:
            print(f"{frontend_method}: no matches")
            continue

        kp_a = frontend["kp_a"]
        kp_b = frontend["kp_b"]
        good = frontend["good"]
        ransac_matches = _count_geometric_inliers(good, kp_a, kp_b, "ransac")
        lmeds_matches = _count_geometric_inliers(good, kp_a, kp_b, "lmeds")
        rows = [
            ("raw candidates", frontend["num_raw"]),
            (frontend["frontend_stage_label"], frontend["num_after_frontend"]),
            ("after RANSAC", len(ransac_matches)),
            ("after LMedS", len(lmeds_matches)),
        ]

        for label, value in rows:
            frontend_rows.append((frontend_method, label, value))
            plot_labels.append(f"{frontend_method}\n{label}")
            plot_values.append(value)

        _save_match_panel(
            frame_a,
            frame_b,
            kp_a,
            kp_b,
            good,
            "frontend matches",
            os.path.join(config.OUTPUT_DIR, f"part_03_{frontend_method}_frontend_matches.png"),
        )
        _save_match_panel(
            frame_a,
            frame_b,
            kp_a,
            kp_b,
            ransac_matches,
            "RANSAC inliers",
            os.path.join(config.OUTPUT_DIR, f"part_03_{frontend_method}_ransac_inliers.png"),
        )
        _save_match_panel(
            frame_a,
            frame_b,
            kp_a,
            kp_b,
            lmeds_matches,
            "LMedS inliers",
            os.path.join(config.OUTPUT_DIR, f"part_03_{frontend_method}_lmeds_inliers.png"),
        )

    out_csv = os.path.join(config.OUTPUT_DIR, "part_03_bad_match_rejection_counts.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("frontend,stage,count\n")
        for frontend_method, label, value in frontend_rows:
            f.write(f"{frontend_method},{label},{value}\n")

    plt.figure()
    plt.bar(plot_labels, plot_values)
    plt.ylabel("Number of matches")
    plt.title("Bad match rejection counts")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True)
    plt.tight_layout()
    out_img = os.path.join(config.OUTPUT_DIR, "part_03_bad_match_rejection_counts.png")
    plt.savefig(out_img, dpi=200)
    plt.close()

    for frontend_method in ["orb", "shi_tomasi_lk"]:
        frontend_dict = {label: value for f, label, value in frontend_rows if f == frontend_method}
        if not frontend_dict:
            continue
        print(f"{frontend_method}: RANSAC={frontend_dict.get('after RANSAC', 0)}, LMedS={frontend_dict.get('after LMedS', 0)}")
    print(f"saved: {os.path.basename(out_img)}")
    print(f"saved: {os.path.basename(out_csv)}")


if __name__ == "__main__":
    main()
