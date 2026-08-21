#!/usr/bin/env python3
"""
run_stella_bag.py — rosbag 재생을 통해 stella_vslam RGB-D를 실행하고 TUM Trajectory를 추출하는 도구
"""

import os
import sys
import time
import yaml
import signal
import argparse
import subprocess
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import BAG_DIR, TRAJECTORY_DIR, PROJECT_DIR, FRAME_DIR
from auto_mobility.dataset.frame_dataset import FrameDataset, CameraIntrinsics
from auto_mobility.trajectory.io import Trajectory


def convert_stella_trajectory_to_tum(input_path: str, output_path: str) -> str:
    """Convert stella_vslam raw output format to standard TUM trajectory format."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            parts = line_str.split()
            if len(parts) >= 8:
                f_out.write(f"{float(parts[0]):.6f} {float(parts[1]):.6f} {float(parts[2]):.6f} {float(parts[3]):.6f} {float(parts[4]):.6f} {float(parts[5]):.6f} {float(parts[6]):.6f} {float(parts[7]):.6f}\n")
    return output_path


def generate_stella_config(
    intrinsics: Optional[CameraIntrinsics] = None,
    output_path: Optional[str] = None,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    depth_factor: float = 1000.0,
    depth_threshold: float = 40.0
) -> str:
    """stella_vslam 호환 YAML 설정을 생성하고 저장한다."""
    fx = intrinsics.fx if intrinsics else 385.0
    fy = intrinsics.fy if intrinsics else 385.0
    cx = intrinsics.cx if intrinsics else 320.0
    cy = intrinsics.cy if intrinsics else 240.0

    cfg = {
        "Camera": {
            "name": "RealSense D435i RGB-D",
            "setup": "RGBD",
            "model": "perspective",
            "color_order": "BGR",
            "cols": width,
            "rows": height,
            "fps": fps,
            "focal_x_baseline": fx * 0.05,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
            "depth_threshold": depth_threshold,
            "depth_factor": depth_factor
        },
        "Feature": {
            "max_num_keypoints": 1200,
            "scale_factor": 1.2,
            "num_levels": 8,
            "ini_fast_threshold": 20,
            "min_fast_threshold": 7
        },
        "Mapping": {
            "baseline_dist_thr_ratio": 0.02
        },
        "Tracking": {
            "min_num_tracked_keypoints": 15
        }
    }

    if output_path is None:
        output_path = str(PROJECT_DIR / "config" / "stella_vslam_d435i.yaml")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return output_path


def run_stella_vslam_on_bag(bag_input: str, out_trajectory: Optional[str] = None, rate: float = 1.0) -> str:
    bag_path = Path(bag_input)
    if not bag_path.is_absolute():
        if (BAG_DIR / bag_input).exists():
            bag_path = BAG_DIR / bag_input
        elif not bag_path.exists():
            raise FileNotFoundError(f"Rosbag not found: {bag_input}")

    bag_name = bag_path.name
    if out_trajectory is None:
        out_trajectory = str(TRAJECTORY_DIR / f"stella_{bag_name}_trajectory.txt")
    out_trajectory = os.path.abspath(out_trajectory)
    os.makedirs(os.path.dirname(out_trajectory), exist_ok=True)

    vocab_path = str(PROJECT_DIR / "third_party" / "stella_vslam" / "vocab" / "orb_vocab.fbow")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"stella_vslam vocabulary not found at {vocab_path}")

    # Generate config
    dataset_path = FRAME_DIR / bag_name
    intrinsics = None
    if dataset_path.exists() and (dataset_path / "frames.csv").exists():
        ds = FrameDataset(dataset_path)
        intrinsics = ds.intrinsics
    if intrinsics is None:
        intrinsics = CameraIntrinsics(fx=606.5387, fy=606.4935, cx=324.4991, cy=241.7047, width=640, height=480)

    config_path = str(PROJECT_DIR / "config" / "stella_vslam_d435i.yaml")
    generate_stella_config(intrinsics=intrinsics, output_path=config_path)

    node_exe = str(PROJECT_DIR / "install" / "auto_mobility" / "lib" / "auto_mobility" / "stella_rgbd_node")
    if not os.path.exists(node_exe):
        node_exe = str(PROJECT_DIR / "build" / "auto_mobility" / "stella_rgbd_node")
    if not os.path.exists(node_exe):
        raise FileNotFoundError("stella_rgbd_node not found. Run colcon build first.")

    print("==========================================================")
    print(f" 🚀 Running stella_vslam (RGB-D) on Bag: {bag_name}")
    print(f" 📦 Source Bag: {bag_path}")
    print(f" ⏩ Play Rate: {rate}x")
    print(f" 📑 Output Trajectory: {out_trajectory}")
    print("==========================================================")

    env = os.environ.copy()
    stella_lib_dir = str(PROJECT_DIR / "third_party" / "installed" / "lib")
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{stella_lib_dir}:{existing_ld}" if existing_ld else stella_lib_dir

    # 1. Start republish.py (decompress compressedDepth and compressed RGB)
    republish_cmd = [
        sys.executable,
        str(PROJECT_DIR / "src" / "auto_mobility" / "nodes" / "republish.py"),
        "--ros-args", "-p", "use_sim_time:=true"
    ]
    republish_proc = subprocess.Popen(republish_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid, env=env)

    # 2. Start stella_rgbd_node
    stella_cmd = [
        node_exe,
        "--ros-args",
        "-p", f"vocab_path:={vocab_path}",
        "-p", f"config_path:={config_path}",
        "-p", f"output_trajectory:={out_trajectory}",
        "-p", "use_sim_time:=true"
    ]
    log_dir = PROJECT_DIR / "ros2_data" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    stella_log_path = log_dir / f"stella_{bag_name}.log"
    stella_log_file = open(stella_log_path, "w")

    stella_proc = subprocess.Popen(
        stella_cmd,
        stdout=stella_log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env
    )

    time.sleep(3.0)  # Wait for vocabulary to load into memory

    # 3. Play bag with --clock
    print(f"▶️ [Step 2] Playing rosbag with /clock (Rate: {rate}x)...")
    play_cmd = ["ros2", "bag", "play", str(bag_path), "--clock", "--rate", str(rate)]
    play_res = subprocess.run(play_cmd, env=env)

    print("▶️ [Step 3] Finalizing stella_vslam and saving trajectory...")
    time.sleep(3.0)

    # Gracefully terminate stella node with SIGINT to trigger destructor
    try:
        os.killpg(os.getpgid(stella_proc.pid), signal.SIGINT)
        stella_proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(stella_proc.pid), signal.SIGKILL)
        except Exception:
            pass

    try:
        os.killpg(os.getpgid(republish_proc.pid), signal.SIGINT)
    except Exception:
        pass

    try:
        stella_log_file.close()
    except Exception:
        pass

    if os.path.exists(out_trajectory) and os.path.getsize(out_trajectory) > 0:
        traj = Trajectory.from_tum_file(out_trajectory)
        metrics = traj.compute_metrics()
        print(f"✅ stella_vslam Trajectory generated successfully!")
        print(f"📊 Frames: {metrics.get('num_frames', 0)}, Length: {metrics.get('total_path_length_m', 0):.4f}m, MaxStep: {metrics.get('max_step_m', 0):.4f}m")
        return out_trajectory
    else:
        log_snippet = ""
        if os.path.exists(stella_log_path):
            with open(stella_log_path, "r") as f:
                lines = f.readlines()
                log_snippet = "".join(lines[-25:])
        raise RuntimeError(f"Failed to generate stella_vslam trajectory file at {out_trajectory}\n--- Last log lines ---\n{log_snippet}")


def main():
    parser = argparse.ArgumentParser(description="Run stella_vslam RGB-D on rosbag")
    parser.add_argument("bag", help="Rosbag name or path")
    parser.add_argument("--out", default=None, help="Output TUM trajectory path (.txt)")
    parser.add_argument("--rate", type=float, default=1.0, help="Bag playback rate (default: 1.0)")
    args = parser.parse_args()

    run_stella_vslam_on_bag(args.bag, args.out, rate=args.rate)


if __name__ == "__main__":
    main()
