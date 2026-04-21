import os

import numpy as np

from part_01_ground_truth_given_data import associate_gt_with_diffs, interpolate_gt_positions


def use_nearest_only(dataset_dir):
    name = os.path.basename(os.path.normpath(dataset_dir)).lower()
    return name == "tum_loop"


def select_gt_positions(rgb_times, gt_data, dataset_dir, max_diff):
    if use_nearest_only(dataset_dir):
        gt_xyz, valid_indices, time_diffs = associate_gt_with_diffs(
            rgb_times,
            gt_data,
            max_diff=max_diff,
        )
        return {
            "gt_xyz": gt_xyz,
            "valid_indices": valid_indices,
            "time_diffs": time_diffs,
            "mode": "nearest-only",
        }

    gt_xyz, valid_indices = interpolate_gt_positions(rgb_times, gt_data)
    return {
        "gt_xyz": gt_xyz,
        "valid_indices": valid_indices,
        "time_diffs": np.empty((0,), dtype=np.float64),
        "mode": "interpolated",
    }
