import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import part_00_project_config as config
from best_prefix import get_best_prefix


def rms(values):
    return float(np.sqrt(np.mean(values ** 2)))


def save_xy(raw_traj, corrected_traj, prefix):
    out1 = os.path.join(config.OUTPUT_DIR, "part_15_loop_closure_xy.png")
    out2 = os.path.join(config.OUTPUT_DIR, f"part_15_loop_closure_xy__{prefix}.png")
    plt.figure()
    plt.plot(raw_traj[:, 0], raw_traj[:, 1], label="raw")
    plt.plot(corrected_traj[:, 0], corrected_traj[:, 1], label="corrected")
    plt.plot(corrected_traj[0, 0], corrected_traj[0, 1], "o")
    plt.plot(corrected_traj[-1, 0], corrected_traj[-1, 1], "x")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Raw vs loop-corrected trajectory")
    plt.grid(True)
    plt.legend()
    plt.savefig(out1)
    plt.savefig(out2)
    plt.close()
    old_3d = os.path.join(config.OUTPUT_DIR, "part_15_loop_closure_3d.png")
    if os.path.exists(old_3d):
        os.remove(old_3d)
    print(f"saved: {os.path.basename(out1)}, {os.path.basename(out2)}")


def main():
    prefix = get_best_prefix("SLAM_RESULT_PREFIX", "gicp_ransac")
    base = config.OUTPUT_DIR
    raw_err_path = os.path.join(base, f"{prefix}_raw_position_error.npy")
    corrected_err_path = os.path.join(base, f"{prefix}_corrected_position_error.npy")
    raw_traj_path = os.path.join(base, f"{prefix}_raw_trajectory.npy")
    corrected_traj_path = os.path.join(base, f"{prefix}_corrected_trajectory.npy")
    loop_info_path = os.path.join(base, f"{prefix}_closure_info.npy")

    if not (os.path.exists(raw_err_path) and os.path.exists(corrected_err_path)):
        print("\nPart 12: loop")
        print("result not found")
        print("run full first")
        return

    raw_err = np.load(raw_err_path)
    corrected_err = np.load(corrected_err_path)
    raw_rms = rms(raw_err)
    corrected_rms = rms(corrected_err)
    improve = 100.0 * (raw_rms - corrected_rms) / raw_rms

    print("\nPart 12: loop")
    print(f"raw: {raw_rms:.6f} m")
    print(f"corrected: {corrected_rms:.6f} m")
    print(f"improve: {improve:.2f}%")

    if os.path.exists(raw_traj_path) and os.path.exists(corrected_traj_path):
        raw_traj = np.load(raw_traj_path)
        corrected_traj = np.load(corrected_traj_path)
        save_xy(raw_traj, corrected_traj, prefix)

    if os.path.exists(loop_info_path):
        loop_info = np.load(loop_info_path)
        print("loop file: yes")
        if len(loop_info) >= 4:
            print(f"candidate: {int(loop_info[2])}, query: {int(loop_info[3])}")
    else:
        print("loop file: no")


if __name__ == "__main__":
    main()
