import os

import numpy as np
import open3d as o3d

import part_00_project_config as config
from part_04_icp_registration import icp_register
from part_05_gicp_registration import gicp_register


def use_global_loop_fallback(dataset_dir):
    name = os.path.basename(os.path.normpath(dataset_dir)).lower()
    return name == "tum_loop"


def _compute_fpfh(pcd, voxel_size):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )


def _refine(source, target, init, registration_method):
    if registration_method == "gicp":
        return gicp_register(source, target, threshold=config.ICP_THRESH, init=init)
    return icp_register(source, target, threshold=config.ICP_THRESH, init=init)


def search_global_loop_candidate(frames, registration_method="gicp"):
    num_frames = len(frames)
    if num_frames < 2:
        return None

    voxel_size = config.VOXEL_SIZE
    max_head = min(30, num_frames - 1)
    query_start = max(0, num_frames - 40)
    feature_cache = {}
    best = None

    def get_feature(idx):
        feat = feature_cache.get(idx)
        if feat is None:
            feat = _compute_fpfh(frames[idx]["pcd_down"], voxel_size)
            feature_cache[idx] = feat
        return feat

    for candidate_idx in range(0, max_head, 5):
        candidate = frames[candidate_idx]
        target = candidate["pcd_down"]
        target_feat = get_feature(candidate_idx)
        for query_idx in range(query_start, num_frames, 5):
            if query_idx - candidate_idx < config.MIN_LOOP_SEPARATION:
                continue
            query = frames[query_idx]
            source = query["pcd_down"]
            source_feat = get_feature(query_idx)

            reg = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source,
                target,
                source_feat,
                target_feat,
                True,
                voxel_size * 4.0,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 4.0),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(5000, 0.999),
            )

            T_refined, fitness, rmse = _refine(source, target, reg.transformation, registration_method)
            corr = len(reg.correspondence_set)
            translation = float(np.linalg.norm(T_refined[:3, 3]))

            if corr < 500:
                continue
            if fitness < 0.55:
                continue
            if rmse > 0.04:
                continue
            if translation > 2.0:
                continue

            score = 100.0 * fitness - 100.0 * rmse + 0.001 * corr - translation
            if best is None or score > best["score"]:
                best = {
                    "candidate_idx": candidate_idx,
                    "query_idx": query_idx,
                    "candidate_pair_idx": candidate["pair_idx"],
                    "query_pair_idx": query["pair_idx"],
                    "score": score,
                    "fitness": float(fitness),
                    "rmse": float(rmse),
                    "corr": int(corr),
                    "translation": translation,
                    "T_rel": T_refined,
                    "source": "global-fpfh",
                }

    return best
