from part_07_trajectory_odometry_chain import run_full_pipeline


PREFIX = "gicp_lmeds"


def run():
    print("\nPart 11")
    out = run_full_pipeline(registration_method="gicp", feature_method="lmeds", prefix=PREFIX)
    print(f"loop: {out['loop_detected']}")
    if out["raw_rms"] is not None:
        print(f"raw: {out['raw_rms']:.6f} m")
        print(f"corrected: {out['corrected_rms']:.6f} m")
    print(f"saved: {PREFIX}_summary.txt")
    return out


if __name__ == "__main__":
    run()
