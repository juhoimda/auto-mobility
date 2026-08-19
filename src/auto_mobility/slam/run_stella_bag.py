#!/usr/bin/env python3
"""
run_stella_bag.py — stella_vslam RGB-D SLAM 실행 및 TUM Trajectory 변환 어댑터

기능:
  1. CameraInfo / Canonical Dataset 기반 stella_vslam 호환 YAML 설정 파일 자동 생성
  2. stella_vslam 프로세스 실행 (stella_vslam / run_slam / stella_vslam_ros)
  3. stella_vslam 궤적 결과(Keyframe / Frame 궤적)를 표준 TUM format으로 변환
  4. 프로세스 반환 코드 및 환경 미설치 시 명확한 에러 핸들링
"""

import os
import sys
import time
import yaml
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import BAG_DIR, TRAJECTORY_DIR, PROJECT_DIR, FRAME_DIR
from auto_mobility.dataset.frame_dataset import FrameDataset, CameraIntrinsics
from auto_mobility.trajectory.io import Trajectory


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
            "color_order": "RGB",
            "cols": width,
            "rows": height,
            "fps": fps,
            "focal_x_baseline": fx * 0.05,  # virtual stereo baseline
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
        output_path = "/tmp/stella_vslam_d435i.yaml"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return output_path


def convert_stella_trajectory_to_tum(stella_traj_file: str, tum_out_file: str) -> str:
    """stella_vslam 출력 궤적 파일(timestamp x y z qx qy qz qw 또는 4x4 matrix)을 표준 TUM 형식으로 변환."""
    if not os.path.exists(stella_traj_file) or os.path.getsize(stella_traj_file) == 0:
        raise FileNotFoundError(f"stella_vslam trajectory file not found or empty: {stella_traj_file}")

    lines_out = []
    with open(stella_traj_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                # Format: timestamp tx ty tz qx qy qz qw
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                lines_out.append(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
            elif len(parts) == 12 or len(parts) == 16:
                # 3x4 or 4x4 matrix row format: timestamp r00 r01 ...
                ts = float(parts[0])
                m_vals = [float(p) for p in parts[1:13]]
                from scipy.spatial.transform import Rotation
                R_mat = np.array([
                    [m_vals[0], m_vals[1], m_vals[2]],
                    [m_vals[4], m_vals[5], m_vals[6]],
                    [m_vals[8], m_vals[9], m_vals[10]]
                ])
                tx, ty, tz = m_vals[3], m_vals[7], m_vals[11]
                q = Rotation.from_matrix(R_mat).as_quat() # x, y, z, w
                lines_out.append(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    os.makedirs(os.path.dirname(os.path.abspath(tum_out_file)), exist_ok=True)
    with open(tum_out_file, "w", encoding="utf-8") as f:
        f.writelines(lines_out)

    return tum_out_file


def run_stella_vslam_on_bag(bag_input: str, out_trajectory: Optional[str] = None) -> str:
    """stella_vslam을 실행하고 표준 TUM trajectory를 생성한다."""
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

    # Check for executable
    stella_exec = shutil.which("stella_vslam") or shutil.which("run_slam")
    if stella_exec is None:
        candidate_paths = [
            PROJECT_DIR / "third_party" / "stella_vslam" / "build" / "run_slam",
            PROJECT_DIR / "third_party" / "installed" / "bin" / "run_slam",
            Path("/usr/local/bin/run_slam"),
        ]
        for c in candidate_paths:
            if c.exists() and os.access(str(c), os.X_OK):
                stella_exec = str(c)
                break

    if stella_exec is None:
        raise RuntimeError(
            "stella_vslam executable not found in system PATH or third_party. "
            "Please build stella_vslam in third_party/stella_vslam or install via official repo."
        )

    # Generate config
    dataset_path = FRAME_DIR / bag_name
    intrinsics = None
    if dataset_path.exists() and (dataset_path / "frames.csv").exists():
        ds = FrameDataset(dataset_path)
        intrinsics = ds.intrinsics

    config_file = str(PROJECT_DIR / "config" / "stella_vslam_d435i.yaml")
    generate_stella_config(intrinsics=intrinsics, output_path=config_file)

    raw_traj_out = f"/tmp/stella_{bag_name}_raw_traj.txt"
    cmd = [
        stella_exec,
        "-c", config_file,
        "--eval-log-dir", "/tmp",
        "--ros-args", "-p", "use_sim_time:=true"
    ]

    print(f"🚀 Running stella_vslam on {bag_name}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"stella_vslam failed with exit code {res.returncode}:\n{res.stderr}\n{res.stdout}")

    if os.path.exists(raw_traj_out):
        convert_stella_trajectory_to_tum(raw_traj_out, out_trajectory)
    else:
        raise RuntimeError(f"stella_vslam trajectory output missing at {raw_traj_out}")

    return out_trajectory


def main():
    parser = argparse.ArgumentParser(description="Run stella_vslam RGB-D on rosbag")
    parser.add_argument("bag", help="Rosbag name or path")
    parser.add_argument("--out", default=None, help="Output TUM trajectory path (.txt)")
    args = parser.parse_args()

    run_stella_vslam_on_bag(args.bag, args.out)


if __name__ == "__main__":
    main()
