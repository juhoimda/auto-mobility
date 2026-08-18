#!/usr/bin/env python3
"""
export_trajectory.py — RTAB-Map DB 또는 Odom 데이터에서 표준 TUM Trajectory 파일 추출기

사용법:
  python3 export_trajectory.py <session.db> [--out trajectory.txt] [--opt 0|2]

출력:
  # timestamp tx ty tz qx qy qz qw id
"""

import sys
import os
import argparse
import subprocess
import re
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import TRAJECTORY_DIR
from auto_mobility.trajectory.io import Trajectory


def export_from_db(db_path: str, out_path: str, opt: int = 0) -> str:
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    base = os.path.splitext(os.path.basename(db_path))[0]
    if out_path is None:
        out_path = os.path.join(TRAJECTORY_DIR, f"{base}_opt{opt}_trajectory.txt")
    out_path = os.path.abspath(out_path)

    cmd = ["rtabmap-export", "--poses_camera", f"--opt={opt}",
           f"--output={base}_export_poses", db_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr

    m = re.search(r'"([^"]*_camera_poses\.txt)"', out)
    if m:
        temp_poses_file = m.group(1)
    else:
        temp_poses_file = os.path.join(os.path.dirname(db_path), f"{base}_export_poses_camera_poses.txt")

    if not os.path.exists(temp_poses_file):
        raise RuntimeError(f"rtabmap-export failed:\n{out}")

    traj = Trajectory.from_tum_file(temp_poses_file)
    traj.to_tum_file(out_path, comment=f"Source: {db_path}, opt={opt}")

    # 임시 파일 정리
    try:
        os.unlink(temp_poses_file)
    except OSError:
        pass

    metrics = traj.compute_metrics()
    print(f"✅ Trajectory exported -> {out_path}")
    print(f"📊 Frames: {metrics['num_frames']}, Length: {metrics.get('total_path_length_m', 0)}m, MaxStep: {metrics.get('max_step_m', 0)}m")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export TUM Trajectory from RTAB-Map DB")
    parser.add_argument("db", help="Path to RTAB-Map DB (.db)")
    parser.add_argument("--out", default=None, help="Output TUM trajectory filepath (.txt)")
    parser.add_argument("--opt", type=int, default=0, choices=[0, 2], help="0: Global SLAM optimized, 2: Raw DB pose")
    args = parser.parse_args()

    export_from_db(args.db, args.out, args.opt)


if __name__ == "__main__":
    main()
