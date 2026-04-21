import os

import cv2
import numpy as np

import part_00_project_config as config
from part_17_reconstruct_3d_with_estimated_trajectory import load_selected_frames, pixel_to_3d


def make_roi_mask(gray):
    h, w = gray.shape[:2]
    return np.ones((h, w), dtype=np.uint8) * 255


def _estimate_homography(pts_a_2d, pts_b_2d, method_name):
    method_name = (method_name or "ransac").lower()
    if method_name == "lmeds":
        method_flag = cv2.LMEDS
    else:
        method_flag = cv2.RANSAC

    H, mask_h = cv2.findHomography(pts_a_2d, pts_b_2d, method_flag, config.RANSAC_THRESH)
    return H, mask_h, method_name


def _make_keypoint(x, y, size=7.0):
    return cv2.KeyPoint(float(x), float(y), float(size))


def _orb_match_features(img_a_gray, img_b_gray):
    orb = cv2.ORB_create(nfeatures=config.NFEATURES)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    kp_a, des_a = orb.detectAndCompute(img_a_gray, make_roi_mask(img_a_gray))
    kp_b, des_b = orb.detectAndCompute(img_b_gray, make_roi_mask(img_b_gray))
    if des_a is None or des_b is None:
        return None

    knn = bf.knnMatch(des_a, des_b, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < config.RATIO_THRESH * n.distance:
                good.append(m)

    return {
        "kp_a": kp_a,
        "kp_b": kp_b,
        "knn": knn,
        "good": good,
        "frontend_method": "orb",
        "num_raw": len(knn),
        "num_after_frontend": len(good),
        "frontend_stage_label": "after Lowe ratio",
    }


def _shi_tomasi_lk_match_features(img_a_gray, img_b_gray):
    pts_a = cv2.goodFeaturesToTrack(
        img_a_gray,
        maxCorners=config.LK_MAX_CORNERS,
        qualityLevel=config.LK_QUALITY_LEVEL,
        minDistance=config.LK_MIN_DISTANCE,
        blockSize=config.LK_BLOCK_SIZE,
        useHarrisDetector=False,
    )
    if pts_a is None or len(pts_a) == 0:
        return None

    pts_b, st, _ = cv2.calcOpticalFlowPyrLK(
        img_a_gray,
        img_b_gray,
        pts_a,
        None,
        winSize=(config.LK_WIN_SIZE, config.LK_WIN_SIZE),
        maxLevel=config.LK_MAX_LEVEL,
    )
    if pts_b is None or st is None:
        return None

    pts_a_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
        img_b_gray,
        img_a_gray,
        pts_b,
        None,
        winSize=(config.LK_WIN_SIZE, config.LK_WIN_SIZE),
        maxLevel=config.LK_MAX_LEVEL,
    )
    if pts_a_back is None or st_back is None:
        return None

    pts_a = pts_a.reshape(-1, 2)
    pts_b = pts_b.reshape(-1, 2)
    pts_a_back = pts_a_back.reshape(-1, 2)
    st = st.reshape(-1).astype(bool)
    st_back = st_back.reshape(-1).astype(bool)
    fb_err = np.linalg.norm(pts_a - pts_a_back, axis=1)
    keep = st & st_back & np.isfinite(fb_err) & (fb_err <= config.LK_FB_THRESH)

    pts_a_keep = pts_a[keep]
    pts_b_keep = pts_b[keep]
    if len(pts_a_keep) == 0:
        return None

    kp_a = []
    kp_b = []
    matches = []
    for idx, (pt_a, pt_b) in enumerate(zip(pts_a_keep, pts_b_keep)):
        kp_a.append(_make_keypoint(pt_a[0], pt_a[1]))
        kp_b.append(_make_keypoint(pt_b[0], pt_b[1]))
        matches.append(cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _imgIdx=0, _distance=float(fb_err[keep][idx])))

    return {
        "kp_a": kp_a,
        "kp_b": kp_b,
        "knn": matches,
        "good": matches,
        "frontend_method": "shi_tomasi_lk",
        "num_raw": int(len(pts_a)),
        "num_after_frontend": int(len(matches)),
        "frontend_stage_label": "after LK tracking",
    }


def _get_frontend_matches(img_a_gray, img_b_gray, frontend_method=None):
    frontend_method = (frontend_method or "orb").lower()
    if frontend_method == "shi_tomasi_lk":
        result = _shi_tomasi_lk_match_features(img_a_gray, img_b_gray)
    else:
        frontend_method = "orb"
        result = _orb_match_features(img_a_gray, img_b_gray)
    if result is None:
        return None
    result["frontend_method"] = frontend_method
    return result


def save_input_pair(frame_a, frame_b, out_name):
    img = np.hstack([frame_a["color"], frame_b["color"]])
    cv2.imwrite(os.path.join(config.OUTPUT_DIR, out_name), img)


def save_frontend_visual(frame_a, frame_b, frontend, out_name):
    if frontend is None:
        return
    vis = cv2.drawMatches(
        frame_a["color"],
        frontend["kp_a"],
        frame_b["color"],
        frontend["kp_b"],
        frontend["good"][:300],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(os.path.join(config.OUTPUT_DIR, out_name), vis)


def get_feature_3d_correspondences(
    img_a_gray,
    img_b_gray,
    depth_a,
    depth_b,
    frontend_method=None,
    feature_method=None,
):
    frontend = _get_frontend_matches(img_a_gray, img_b_gray, frontend_method=frontend_method)
    if frontend is None:
        return None
    kp_a = frontend["kp_a"]
    kp_b = frontend["kp_b"]
    knn = frontend["knn"]
    good = frontend["good"]
    if len(good) < 4:
        return None

    pts_a_2d = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b_2d = np.float32([kp_b[m.trainIdx].pt for m in good])
    H, mask_h, feature_method = _estimate_homography(pts_a_2d, pts_b_2d, feature_method)
    if H is None or mask_h is None:
        return None

    mask_bool = mask_h.ravel().astype(bool)
    inliers = [good[i] for i in range(len(good)) if mask_bool[i]]
    num_geometric = len(inliers)

    if len(inliers) < config.MIN_3D_MATCHES:
        return None

    pts3d_a = []
    pts3d_b = []
    for m in inliers:
        u_a, v_a = kp_a[m.queryIdx].pt
        u_b, v_b = kp_b[m.trainIdx].pt
        u_a_i, v_a_i = int(round(u_a)), int(round(v_a))
        u_b_i, v_b_i = int(round(u_b)), int(round(v_b))
        if not (0 <= u_a_i < depth_a.shape[1] and 0 <= v_a_i < depth_a.shape[0]):
            continue
        if not (0 <= u_b_i < depth_b.shape[1] and 0 <= v_b_i < depth_b.shape[0]):
            continue
        d_a = depth_a[v_a_i, u_a_i]
        d_b = depth_b[v_b_i, u_b_i]
        if d_a <= 0 or d_b <= 0:
            continue
        p_a = pixel_to_3d(u_a, v_a, d_a)
        p_b = pixel_to_3d(u_b, v_b, d_b)
        if p_a is None or p_b is None:
            continue
        pts3d_a.append(p_a)
        pts3d_b.append(p_b)

    if len(pts3d_a) < config.MIN_3D_MATCHES:
        return None

    return {
        "pts3d_A": np.array(pts3d_a, dtype=np.float64),
        "pts3d_B": np.array(pts3d_b, dtype=np.float64),
        "kpA": kp_a,
        "kpB": kp_b,
        "inliers": inliers,
        "frontend_method": frontend["frontend_method"],
        "num_raw_knn": len(knn),
        "num_ratio": len(good),
        "num_frontend": frontend["num_after_frontend"],
        "frontend_stage_label": frontend["frontend_stage_label"],
        "num_geometric": num_geometric,
        "num_ransac": len(inliers),
        "feature_method": feature_method,
    }


def save_match_visual(frame_a, frame_b, feature_result, out_name):
    if feature_result is None:
        return
    vis = cv2.drawMatches(
        frame_a["color"],
        feature_result["kpA"],
        frame_b["color"],
        feature_result["kpB"],
        feature_result["inliers"][:300],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(os.path.join(config.OUTPUT_DIR, out_name), vis)


def main():
    frames, _, _ = load_selected_frames(show_info=False, frame_step=5, max_frames=16)
    frame_a, frame_b = frames[14], frames[15]
    print("\nPart 02: frontends")
    save_input_pair(frame_a, frame_b, "part_02_input_pair.jpg")

    rows = []

    for frontend_method, out_name in [
        ("orb", "part_02_orb_point_feature_matches.jpg"),
        ("shi_tomasi_lk", "part_02_shi_tomasi_lk_matches.jpg"),
    ]:
        frontend = _get_frontend_matches(frame_a["gray"], frame_b["gray"], frontend_method=frontend_method)
        save_frontend_visual(frame_a, frame_b, frontend, out_name)
        result = get_feature_3d_correspondences(
            frame_a["gray"],
            frame_b["gray"],
            frame_a["depth"],
            frame_b["depth"],
            frontend_method=frontend_method,
            feature_method="ransac",
        )
        if result is None:
            rows.append((frontend_method, 0, 0, "none"))
            continue
        rows.append(
            (
                frontend_method,
                len(result["pts3d_A"]),
                result["num_geometric"],
                out_name,
            )
        )

    for frontend_method, num_3d, num_geo, out_name in rows:
        print(f"{frontend_method}: 3D={num_3d}, inliers={num_geo}, fig={out_name}")
    if rows:
        best = rows[0]
        for row in rows[1:]:
            if row[1] > best[1]:
                best = row
        print(f"Best frontend on this pair: {best[0]}")


if __name__ == "__main__":
    main()
