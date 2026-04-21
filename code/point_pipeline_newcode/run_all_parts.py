import importlib
import sys


EARLY_PARTS = [
    "part_01_ground_truth_given_data",
    "part_02_orb_point_features",
    "part_03_bad_match_rejection_compare",
    "part_04_icp_registration",
    "part_05_gicp_registration",
    "part_06_point_pose_step",
]

FULL_PARTS = [
    "part_08_run_baseline_icp_ransac",
    "part_09_run_lmeds_rejection",
    "part_10_run_gicp_registration",
    "part_11_run_gicp_lmeds",
]

LATE_PARTS = [
    "part_14_compare_methods_table",
    "part_12_loop_compare",
    "part_13_estimated_trajectory_calculated",
    "part_16_compare_filters_table",
    "part_17_reconstruct_3d_with_estimated_trajectory",
]

STRESS_PARTS = [
    "part_18_run_full_sequence_stress_test",
]


def run_module(module_name):
    print(f"\n[{module_name}]")
    module = importlib.import_module(module_name)
    if hasattr(module, "run"):
        module.run()
    elif hasattr(module, "main"):
        module.main()
    else:
        raise AttributeError(f"{module_name} has no run() or main() function")


def main():
    include_full = "--include-full" in sys.argv or "--full" in sys.argv
    include_stress = "--include-stress" in sys.argv or "--stress" in sys.argv

    parts = []
    for name in EARLY_PARTS:
        parts.append(name)
    if include_full:
        for name in FULL_PARTS:
            parts.append(name)
        for name in LATE_PARTS:
            parts.append(name)
    if include_stress:
        for name in STRESS_PARTS:
            parts.append(name)

    for module_name in parts:
        run_module(module_name)


if __name__ == "__main__":
    main()
