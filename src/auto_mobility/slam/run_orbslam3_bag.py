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
from auto_mobility.config import BAG_DIR, TRAJECTORY_DIR, PROJECT_DIR, FRAME_DIR
from auto_mobility.dataset.frame_dataset import FrameDataset, CameraIntrinsics
from auto_mobility.trajectory.io import Trajectory


def generate_orbslam3_config(
    intrinsics: CameraIntrinsics = None,
    is_inertial: bool = False,
    output_path: str = None
) -> str:
    """bag 실제 Intrinsics를 반영한 ORB-SLAM3 YAML 설정을 동적 생성한다."""
    fx = intrinsics.fx if intrinsics else 606.5387
    fy = intrinsics.fy if intrinsics else 606.4935
    cx = intrinsics.cx if intrinsics else 324.4991
    cy = intrinsics.cy if intrinsics else 241.7047
    w = intrinsics.width if intrinsics else 640
    h = intrinsics.height if intrinsics else 480

    lines = [
        "%YAML:1.0",
        "",
        'File.version: "1.0"',
        'Camera.type: "PinHole"',
        f"Camera1.fx: {fx:.6f}",
        f"Camera1.fy: {fy:.6f}",
        f"Camera1.cx: {cx:.6f}",
        f"Camera1.cy: {cy:.6f}",
        "Camera1.k1: 0.0",
        "Camera1.k2: 0.0",
        "Camera1.p1: 0.0",
        "Camera1.p2: 0.0",
        f"Camera.width: {w}",
        f"Camera.height: {h}",
        "Camera.fps: 30",
        "Camera.RGB: 1",
        "Stereo.ThDepth: 40.0",
        "Stereo.b: 0.0745",
        "RGBD.DepthMapFactor: 1000.0",
        "",
    ]
    if is_inertial:
        lines.extend([
            "# Transformation from body-frame (imu) to left camera",
            "IMU.T_b_c1: !!opencv-matrix",
            "   rows: 4",
            "   cols: 4",
            "   dt: f",
            "   data: [0.999903, -0.0138036, -0.00208099, -0.0202141,",
            "         0.0137985, 0.999902, -0.00243498, 0.00505961,",
            "         0.0021144, 0.00240603, 0.999995, 0.0114047,",
            "         0.0, 0.0, 0.0, 1.0]",
            "",
            "IMU.InsertKFsWhenLost: 0",
            "IMU.fastInit: 1",
            "IMU.NoiseGyro: 1e-2",
            "IMU.NoiseAcc: 1e-1",
            "IMU.GyroWalk: 1e-6",
            "IMU.AccWalk: 1e-4",
            "IMU.Frequency: 200.0",
            "",
        ])

    lines.extend([
        "ORBextractor.nFeatures: 2000",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        "ORBextractor.iniThFAST: 15",
        "ORBextractor.minThFAST: 5",
        "",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 0.9",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -3.5",
        "Viewer.ViewpointF: 500.0",
        ""
    ])

    if output_path is None:
        fname = "orbslam3_rgbdi_custom.yaml" if is_inertial else "orbslam3_rgbd_custom.yaml"
        output_path = str(PROJECT_DIR / "config" / fname)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


def run_orbslam3_on_bag(bag_input: str, out_trajectory: str = None, mode: str = "rgbd", rate: float = 1.0) -> str:
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

    # Load intrinsics from canonical dataset if present
    dataset_path = FRAME_DIR / bag_name
    intrinsics = None
    if dataset_path.exists() and (dataset_path / "frames.csv").exists():
        ds = FrameDataset(dataset_path)
        intrinsics = ds.intrinsics
    if intrinsics is None:
        intrinsics = CameraIntrinsics(fx=606.5387, fy=606.4935, cx=324.4991, cy=241.7047, width=640, height=480)

    vocab_path = str(PROJECT_DIR / "third_party" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt")
    config_path = generate_orbslam3_config(intrinsics=intrinsics, is_inertial=is_inertial)
    sensor_mode_arg = "IMU_RGBD" if is_inertial else "RGBD"

    offline_exe = str(PROJECT_DIR / "install" / "auto_mobility" / "lib" / "auto_mobility" / "orbslam3_offline")
    if not os.path.exists(offline_exe):
        offline_exe = str(PROJECT_DIR / "build" / "auto_mobility" / "orbslam3_offline")

    # Fast & Lossless Direct Offline Runner
    if os.path.exists(offline_exe) and dataset_path.exists() and (dataset_path / "frames.csv").exists():
        print("==========================================================")
        print(f" 🚀 Running ORB-SLAM3 ({slam_name.upper()}) DIRECT OFFLINE (Zero Frame Drop)")
        print(f" 📦 Dataset Path: {dataset_path}")
        print(f" ⚙️ Sensor Mode:  {sensor_mode_arg}")
        print(f" 📑 Output Traj:  {out_trajectory}")
        print("==========================================================")
        cmd = [
            offline_exe,
            "--dataset", str(dataset_path),
            "--vocab", vocab_path,
            "--config", config_path,
            "--out", out_trajectory,
            "--mode", "rgbdi" if is_inertial else "rgbd"
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0 and os.path.exists(out_trajectory) and os.path.getsize(out_trajectory) > 0:
            traj = Trajectory.from_tum_file(out_trajectory)
            metrics = traj.compute_metrics()
            print(f"✅ ORB-SLAM3 ({slam_name}) Trajectory generated successfully!")
            print(f"📊 Frames: {metrics.get('num_frames', 0)}, Length: {metrics.get('total_path_length_m', 0):.4f}m, MaxStep: {metrics.get('max_step_m', 0):.4f}m")
            return out_trajectory
        else:
            print("⚠️ Direct offline runner returned non-zero code, falling back to ROS bag playback...")

    node_exe = str(PROJECT_DIR / "install" / "auto_mobility" / "lib" / "auto_mobility" / "orbslam3_rgbd_node")
    if not os.path.exists(node_exe):
        node_exe = str(PROJECT_DIR / "build" / "auto_mobility" / "orbslam3_rgbd_node")
    if not os.path.exists(node_exe):
        raise FileNotFoundError(f"orbslam3_rgbd_node not found. Run colcon build first.")

    print("==========================================================")
    print(f" 🚀 Running ORB-SLAM3 ({slam_name.upper()}) on Bag: {bag_name}")
    print(f" 📦 Source Bag: {bag_path}")
    print(f" ⚙️ Sensor Mode: {sensor_mode_arg}")
    print(f" ⏩ Play Rate: {rate}x")
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
    log_dir = PROJECT_DIR / "ros2_data" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    orbslam_log_path = log_dir / f"orbslam3_{bag_name}_{slam_name}_rate{rate}.log"
    orbslam_log_file = open(orbslam_log_path, "w")

    orbslam_proc = subprocess.Popen(
        orbslam_cmd,
        stdout=orbslam_log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )

    time.sleep(4.0)  # Wait for vocabulary to load into memory

    # 3. Play bag with --clock
    print(f"▶️ [Step 2] Playing rosbag with /clock (Rate: {rate}x)...")
    play_cmd = ["ros2", "bag", "play", str(bag_path), "--clock", "--rate", str(rate)]
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

    try:
        orbslam_log_file.close()
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
        log_snippet = ""
        if os.path.exists(orbslam_log_path):
            with open(orbslam_log_path, "r") as f:
                lines = f.readlines()
                log_snippet = "".join(lines[-25:])
        raise RuntimeError(f"Failed to generate ORB-SLAM3 trajectory file at {out_trajectory}\n--- Last log lines ---\n{log_snippet}")


def main():
    parser = argparse.ArgumentParser(description="Run ORB-SLAM3 (RGB-D / RGB-D-Inertial) on rosbag")
    parser.add_argument("bag", help="Rosbag name or path")
    parser.add_argument("--out", default=None, help="Output TUM trajectory path (.txt)")
    parser.add_argument("--mode", "--slam", default="rgbd", choices=["rgbd", "rgbdi", "orb_rgbd", "orb_rgbdi"], help="SLAM mode (default: rgbd)")
    parser.add_argument("--rate", type=float, default=1.0, help="Bag playback rate (default: 1.0)")
    args = parser.parse_args()

    run_orbslam3_on_bag(args.bag, args.out, mode=args.mode, rate=args.rate)


if __name__ == "__main__":
    main()
