#!/usr/bin/env python3
"""
run_rtabmap_bag.py — Canonical RGB-D Frames 직접 구동 기반 RTAB-Map Standalone Runner
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import BAG_DIR, TRAJECTORY_DIR, DB_DIR, PROJECT_DIR, FRAME_DIR
from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.benchmark.artifacts import save_trajectory_metadata
from auto_mobility.benchmark.candidate import SlamProfileSpec


def run_rtabmap_on_bag(
    bag_input: str,
    out_trajectory: str = None,
    out_db: str = None,
    profile: str = "normal",
    rate: float = 1.0
) -> str:
    bag_path = Path(bag_input)
    if not bag_path.is_absolute():
        if (BAG_DIR / bag_input).exists():
            bag_path = BAG_DIR / bag_input
        elif not bag_path.exists():
            raise FileNotFoundError(f"Rosbag not found: {bag_input}")

    bag_name = bag_path.name
    key = f"rtab_{profile}_rate{rate:g}"
    if out_trajectory is None:
        out_trajectory = str(TRAJECTORY_DIR / f"{key}_{bag_name}_trajectory.txt")
    out_trajectory = os.path.abspath(out_trajectory)
    os.makedirs(os.path.dirname(out_trajectory), exist_ok=True)

    if out_db is None:
        out_db = str(DB_DIR / f"{bag_name}_{key}.db")
    out_db = os.path.abspath(out_db)
    os.makedirs(os.path.dirname(out_db), exist_ok=True)

    dataset_path = FRAME_DIR / bag_name
    if not dataset_path.exists() or not (dataset_path / "frames.csv").exists():
        print(f"⚙️ Canonical Frame Dataset이 없습니다. {bag_name}에서 프레임을 사전 추출합니다...")
        ros_setup = "/opt/ros/humble/setup.bash"
        extract_cmd = f"source {ros_setup} && PYTHONPATH=\"{PROJECT_DIR}/src:$PYTHONPATH\" python3 \"{PROJECT_DIR}/src/auto_mobility/dataset/extract_frames.py\" \"{bag_path}\""
        res_ext = subprocess.run(["bash", "-c", extract_cmd])
        if res_ext.returncode != 0:
            raise RuntimeError(f"Frame extraction failed for {bag_path}")

    offline_exe = str(PROJECT_DIR / "install" / "auto_mobility" / "lib" / "auto_mobility" / "rtabmap_offline")
    if not os.path.exists(offline_exe):
        offline_exe = str(PROJECT_DIR / "build" / "auto_mobility" / "rtabmap_offline")

    if not os.path.exists(offline_exe):
        raise FileNotFoundError(f"rtabmap_offline executable not found: {offline_exe}")

    print("==========================================================")
    print(f" 🚀 Running RTAB-Map ({profile.upper()}) DIRECT OFFLINE (Zero Frame Drop)")
    print(f" 📦 Dataset Path: {dataset_path}")
    print(f" 📑 Output Traj : {out_trajectory}")
    print(f" 🗄️ Output DB   : {out_db}")
    print("==========================================================")

    cmd = [
        offline_exe,
        "--dataset", str(dataset_path),
        "--out", out_trajectory,
        "--db", out_db,
        "--profile", profile
    ]

    t0 = time.time()
    res = subprocess.run(cmd)
    runtime = time.time() - t0

    if res.returncode == 0 and os.path.exists(out_trajectory) and os.path.getsize(out_trajectory) > 0:
        traj = Trajectory.from_tum_file(out_trajectory)
        metrics = traj.compute_metrics()
        print(f"\n✅ RTAB-Map ({profile}) Trajectory generated successfully in {runtime:.2f}s!")
        print(f"📊 Frames: {metrics.get('num_frames', 0)}, Length: {metrics.get('total_path_length_m', 0):.4f}m, MaxStep: {metrics.get('max_step_m', 0):.4f}m")
        
        # Save trajectory metadata
        spec = SlamProfileSpec(candidate_key=key, backend="rtab", profile=profile, replay_rate=float(rate))
        save_trajectory_metadata(out_trajectory, spec)
        
        # Also copy legacy aliases if normal profile
        if profile == "normal" and rate == 1.0:
            alt_traj = str(TRAJECTORY_DIR / f"rtab_{bag_name}_trajectory.txt")
            alt_db = str(DB_DIR / f"{bag_name}_rtab.db")
            try:
                import shutil
                shutil.copyfile(out_trajectory, alt_traj)
                shutil.copyfile(out_db, alt_db)
            except Exception:
                pass

        return out_trajectory
    else:
        raise RuntimeError(f"rtabmap_offline failed with return code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run RTAB-Map on Canonical Frame Dataset (Zero Frame Drop)")
    parser.add_argument("bag", help="Rosbag name or path")
    parser.add_argument("--out", default=None, help="Output TUM trajectory path (.txt)")
    parser.add_argument("--db", default=None, help="Output RTAB-Map database path (.db)")
    parser.add_argument("--profile", default="normal", choices=["normal", "dense"], help="Profile (default: normal)")
    parser.add_argument("--rate", type=float, default=1.0, help="Replay rate tag (default: 1.0)")
    args = parser.parse_args()

    run_rtabmap_on_bag(args.bag, args.out, out_db=args.db, profile=args.profile, rate=args.rate)


if __name__ == "__main__":
    main()
