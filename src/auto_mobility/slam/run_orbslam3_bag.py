#!/usr/bin/env python3
"""
run_orbslam3_bag.py — rosbag 재생을 통해 ORB-SLAM3 RGB-D를 실행하고 TUM Trajectory를 추출하는 도구
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import BAG_DIR, TRAJECTORY_DIR, PROJECT_DIR
from auto_mobility.trajectory.io import Trajectory


def run_orbslam3_on_bag(bag_input: str, out_trajectory: str = None, mode: str = "rgbd") -> str:
    bag_path = Path(bag_input)
    if not bag_path.is_absolute():
        if (BAG_DIR / bag_input).exists():
            bag_path = BAG_DIR / bag_input
        elif not bag_path.exists():
            raise FileNotFoundError(f"Rosbag not found: {bag_input}")

    mode_lower = mode.lower()
    is_inertial = "rgbdi" in mode_lower or "inertial" in mode_lower or mode_lower == "orb_rgbdi"
    slam_name = "orb_rgbdi" if is_inertial else "orb_rgbd"

    bag_name = bag_path.name
    if out_trajectory is None:
        out_trajectory = str(TRAJECTORY_DIR / f"{slam_name}_{bag_name}_trajectory.txt")
    out_trajectory = os.path.abspath(out_trajectory)
    os.makedirs(os.path.dirname(out_trajectory), exist_ok=True)

    vocab_path = str(PROJECT_DIR / "third_party" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt")
    if is_inertial:
        config_path = str(PROJECT_DIR / "third_party" / "ORB_SLAM3" / "Examples" / "RGB-D-Inertial" / "RealSense_D435i.yaml")
        sensor_mode_arg = "IMU_RGBD"
    else:
        config_path = str(PROJECT_DIR / "third_party" / "ORB_SLAM3" / "Examples" / "RGB-D" / "RealSense_D435i.yaml")
        sensor_mode_arg = "RGBD"

    node_exe = str(PROJECT_DIR / "install" / "auto_mobility" / "lib" / "auto_mobility" / "orbslam3_rgbd_node")

    if not os.path.exists(node_exe):
        node_exe = str(PROJECT_DIR / "build" / "auto_mobility" / "orbslam3_rgbd_node")
    if not os.path.exists(node_exe):
        raise FileNotFoundError(f"orbslam3_rgbd_node not found. Run colcon build first.")

    print("==========================================================")
    print(f" 🚀 Running ORB-SLAM3 ({slam_name.upper()}) on Bag: {bag_name}")
    print(f" 📦 Source Bag: {bag_path}")
    print(f" ⚙️ Sensor Mode: {sensor_mode_arg}")
    print(f" 📑 Output Trajectory: {out_trajectory}")
    print("==========================================================")

    # 1. Start republish.py (decompress compressedDepth and compressed RGB)
    republish_cmd = [
        sys.executable,
        str(PROJECT_DIR / "src" / "auto_mobility" / "nodes" / "republish.py"),
        "--ros-args", "-p", "use_sim_time:=true"
    ]
    republish_proc = subprocess.Popen(republish_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)

    # 2. Start orbslam3 node
    orbslam_cmd = [
        node_exe,
        "--ros-args",
        "-p", f"vocab_path:={vocab_path}",
        "-p", f"config_path:={config_path}",
        "-p", f"output_trajectory:={out_trajectory}",
        "-p", f"sensor_mode:={sensor_mode_arg}",
        "-p", "use_sim_time:=true"
    ]
    orbslam_proc = subprocess.Popen(orbslam_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid)

    time.sleep(4.0)  # Wait for vocabulary to load into memory

    # 3. Play bag with --clock
    print("▶️ [Step 2] Playing rosbag with /clock...")
    play_cmd = ["ros2", "bag", "play", str(bag_path), "--clock", "--rate", "1.0"]
    play_res = subprocess.run(play_cmd)

    print("▶️ [Step 3] Finalizing ORB-SLAM3 and saving trajectory...")
    time.sleep(3.0)

    # Gracefully terminate orbslam node with SIGINT to trigger destructor
    try:
        os.killpg(os.getpgid(orbslam_proc.pid), signal.SIGINT)
        orbslam_proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(orbslam_proc.pid), signal.SIGKILL)
        except Exception:
            pass

    try:
        os.killpg(os.getpgid(republish_proc.pid), signal.SIGINT)
    except Exception:
        pass

    if os.path.exists(out_trajectory) and os.path.getsize(out_trajectory) > 0:
        traj = Trajectory.from_tum_file(out_trajectory)
        metrics = traj.compute_metrics()
        print(f"✅ ORB-SLAM3 ({slam_name}) Trajectory generated successfully!")
        print(f"📊 Frames: {metrics.get('num_frames', 0)}, Length: {metrics.get('total_path_length_m', 0):.4f}m, MaxStep: {metrics.get('max_step_m', 0):.4f}m")
        return out_trajectory
    else:
        # Check if ORB-SLAM3 generated CameraTrajectory.txt or KeyFrameTrajectory.txt in cwd
        if os.path.exists("CameraTrajectory.txt"):
            os.rename("CameraTrajectory.txt", out_trajectory)
            return out_trajectory
        raise RuntimeError(f"Failed to generate ORB-SLAM3 trajectory file at {out_trajectory}")


def main():
    parser = argparse.ArgumentParser(description="Run ORB-SLAM3 (RGB-D / RGB-D-Inertial) on rosbag")
    parser.add_argument("bag", help="Rosbag name or path")
    parser.add_argument("--out", default=None, help="Output TUM trajectory path (.txt)")
    parser.add_argument("--mode", "--slam", default="rgbd", choices=["rgbd", "rgbdi", "orb_rgbd", "orb_rgbdi"], help="SLAM mode (default: rgbd)")
    args = parser.parse_args()

    run_orbslam3_on_bag(args.bag, args.out, mode=args.mode)


if __name__ == "__main__":
    main()
