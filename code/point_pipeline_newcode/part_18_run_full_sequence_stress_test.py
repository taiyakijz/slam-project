from part_07_trajectory_odometry_chain import run_full_pipeline


PREFIX = "gicp_ransac_full"


def run():
    print("\nPart 18")
    print("stress")
    result = run_full_pipeline(
        registration_method="gicp",
        feature_method="ransac",
        prefix=PREFIX,
        frame_step=1,
        max_frames=None,
    )
    print(f"loop: {result['loop_detected']}")
    if result["raw_rms"] is not None and result["corrected_rms"] is not None:
        print(f"raw: {result['raw_rms']:.6f} m")
        print(f"corrected: {result['corrected_rms']:.6f} m")
    print(f"saved: {PREFIX}_summary.txt")
    return result


if __name__ == "__main__":
    run()
