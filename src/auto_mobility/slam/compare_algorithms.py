#!/usr/bin/env python3
"""
compare_algorithms.py — 동일 Dataset 기반 Multi-SLAM 및 Reconstruction 기하 품질 종합 비교 벤치마크

구조화된 실험 계층 (Separation of Concerns):
  Experiment
  ├── Dataset (Canonical Frame Dataset)
  ├── SLAM Backend (RTAB-Map / ORB-SLAM3)
  ├── Trajectory
  ├── Reconstruction Backend (Open3D TSDF / Poisson / BPA)
  ├── Mesh Generation (Voxel resolution variation)
  └── Quantitative Geometry Quality Evaluation (Hold-out Sensor Consistency)
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import BAG_DIR, DB_DIR, MESH_DIR, TRAJECTORY_DIR, BENCHMARK_DIR, FRAME_DIR, PROJECT_DIR
from auto_mobility.dataset.extract_frames import extract_dataset_from_bag
from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.trajectory.export_trajectory import export_from_db
from auto_mobility.mesh.reconstruct_tsdf import reconstruct
from auto_mobility.mesh.mesh_open3d import generate_mesh
from auto_mobility.evaluation.evaluator import evaluate_reconstruction
from auto_mobility.evaluation.compare_results import rank_candidates
from auto_mobility.evaluation.split import create_holdout_split, save_split_json

# 체계적으로 분리된 후보 실험 정의
EXPERIMENT_PRESETS = {
    # 1. SLAM Backend Variations (동일 10mm TSDF 복원)
    "slam_rtab_global_opt": {
        "desc": "RTAB-Map Visual SLAM (Global Loop Closure Optimization) + 10mm TSDF",
        "slam": {"type": "rtabmap", "opt": 0},
        "reconstruction": {"method": "tsdf", "voxel_size": 0.01},
        "mesh": {"type": "tsdf_iso"}
    },
    "slam_rtab_raw_odom": {
        "desc": "RTAB-Map Raw Odometry (No Loop Closure Optimization) + 10mm TSDF",
        "slam": {"type": "rtabmap", "opt": 2},
        "reconstruction": {"method": "tsdf", "voxel_size": 0.01},
        "mesh": {"type": "tsdf_iso"}
    },
    "slam_orbslam3_rgbd": {
        "desc": "ORB-SLAM3 RGB-D Visual SLAM + 10mm TSDF",
        "slam": {"type": "orbslam3", "opt": 0},
        "reconstruction": {"method": "tsdf", "voxel_size": 0.01},
        "mesh": {"type": "tsdf_iso"}
    },
    # 2. Reconstruction Parameter Variations (동일 RTAB SLAM 기준)
    "recon_tsdf_5mm_fine": {
        "desc": "RTAB-Map SLAM + 5mm High-Resolution TSDF",
        "slam": {"type": "rtabmap", "opt": 0},
        "reconstruction": {"method": "tsdf", "voxel_size": 0.005},
        "mesh": {"type": "tsdf_iso"}
    },
    "recon_tsdf_20mm_fast": {
        "desc": "RTAB-Map SLAM + 20mm Fast TSDF",
        "slam": {"type": "rtabmap", "opt": 0},
        "reconstruction": {"method": "tsdf", "voxel_size": 0.02},
        "mesh": {"type": "tsdf_iso"}
    },
    # 3. Mesh Backend Variations (Point Cloud -> Poisson / BPA)
    "mesh_poisson_depth8": {
        "desc": "RTAB-Map Point Cloud + Poisson Surface Reconstruction (Depth=8)",
        "slam": {"type": "rtabmap", "opt": 0},
        "reconstruction": {"method": "poisson", "depth": 8, "voxel_size": 0.02},
        "mesh": {"type": "poisson"}
    },
    "mesh_bpa": {
        "desc": "RTAB-Map Point Cloud + Ball Pivoting Algorithm (BPA)",
        "slam": {"type": "rtabmap", "opt": 0},
        "reconstruction": {"method": "bpa", "voxel_size": 0.02},
        "mesh": {"type": "bpa"}
    }
}


def run_benchmark(bag_input: str, out_dir: Optional[str] = None, quick: bool = False) -> dict:
    bag_path = Path(bag_input)
    if not bag_path.is_absolute():
        if (BAG_DIR / bag_input).exists():
            bag_path = BAG_DIR / bag_input
        elif not bag_path.exists():
            print(f"❌ Rosbag not found: {bag_input}")
            sys.exit(1)

    bag_name = bag_path.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_id = f"bench_{bag_name}_{timestamp}"
    out_path = Path(out_dir) if out_dir else (BENCHMARK_DIR / benchmark_id)
    out_path.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print(f" 🧪 Auto-Mobility Multi-Layer Benchmark & Comparison")
    print(f" 📦 Source Rosbag: {bag_path}")
    print(f" 📁 Benchmark Out: {out_path}")
    print("==========================================================")

    # 1. Canonical Dataset 준비 (모든 알고리즘의 공통 Source of Truth)
    dataset_dir = FRAME_DIR / bag_name
    if not dataset_dir.exists() or not (dataset_dir / "frames.csv").exists():
        print(f"▶️ [Step 1] Rosbag -> Canonical Frame Dataset 자동 추출...")
        extract_dataset_from_bag(str(bag_path), str(dataset_dir))
    else:
        print(f"▶️ [Step 1] 기존 Canonical Frame Dataset 사용: {dataset_dir}")

    dataset = FrameDataset(dataset_dir)

    # 공통 Hold-out Split 생성 (동일 데이터셋 내 모든 실험에서 100% 동일한 테스트셋 공유)
    split_data = create_holdout_split(total_frames=len(dataset), policy="every_nth", nth=5)
    split_file = out_path / "shared_holdout_split.json"
    save_split_json(split_data, split_file)
    print(f"✂️ 공통 Hold-out Split 설정 완료: {split_data['train_count']} train / {split_data['holdout_count']} hold-out frames")

    # 2. SLAM Backend 실행 및 Trajectory 준비
    trajectories: Dict[str, str] = {}

    # RTAB-Map
    rtab_db = DB_DIR / f"{bag_name}.db"
    rtab_opt_traj = out_path / "rtab_opt_trajectory.txt"
    rtab_raw_traj = out_path / "rtab_raw_trajectory.txt"

    if not rtab_db.exists():
        print(f"⚙️ RTAB-Map SLAM 실행 중 (DB 생성: {rtab_db})...")
        run_slam_script = PROJECT_DIR / "scripts" / "pipeline" / "run_slam.sh"
        subprocess.run(["bash", str(run_slam_script), bag_name, "--slam=rtab"], capture_output=True, text=True)

    if rtab_db.exists():
        export_from_db(str(rtab_db), str(rtab_opt_traj), opt=0)
        export_from_db(str(rtab_db), str(rtab_raw_traj), opt=2)
        trajectories["rtab_opt"] = str(rtab_opt_traj)
        trajectories["rtab_raw"] = str(rtab_raw_traj)

    # ORB-SLAM3
    orb_traj = out_path / "orbslam3_trajectory.txt"
    if not quick:
        try:
            from auto_mobility.slam.run_orbslam3_bag import run_orbslam3_on_bag
            run_orbslam3_on_bag(str(bag_path), str(orb_traj))
            trajectories["orbslam3"] = str(orb_traj)
        except Exception as e:
            print(f"⚠️ ORB-SLAM3 실행 생략: {e}")

    # 3. Preset 실행 및 통합 평가
    results_list = []
    eval_summaries = []

    presets_to_run = EXPERIMENT_PRESETS.copy()
    if quick:
        presets_to_run = {
            "slam_rtab_global_opt": EXPERIMENT_PRESETS["slam_rtab_global_opt"],
            "recon_tsdf_20mm_fast": EXPERIMENT_PRESETS["recon_tsdf_20mm_fast"]
        }

    for key, spec in presets_to_run.items():
        print(f"\n▶️ [Step 2] 후보 실행 & 평가: {key} ({spec['desc']})")
        slam_cfg = spec["slam"]
        recon_cfg = spec["reconstruction"]

        # Determine trajectory to use
        if slam_cfg["type"] == "orbslam3":
            t_file = trajectories.get("orbslam3")
        elif slam_cfg.get("opt") == 2:
            t_file = trajectories.get("rtab_raw")
        else:
            t_file = trajectories.get("rtab_opt")

        if not t_file or not os.path.exists(t_file):
            print(f"   ⚠️ Trajectory 누락으로 {key} 건너뜀")
            continue

        mesh_file = out_path / f"{key}_mesh.obj"
        pcd_file = out_path / f"{key}_cloud.ply"

        t0 = time.time()
        # Reconstruction execution
        if recon_cfg["method"] == "tsdf":
            reconstruct(
                dataset=dataset,
                trajectory=t_file,
                voxel_size=recon_cfg["voxel_size"],
                output_mesh=str(mesh_file),
                output_pcd=str(pcd_file),
                train_indices=split_data["train_indices"],
                no_gpu=quick
            )
        elif recon_cfg["method"] in ("poisson", "bpa"):
            # PointCloud -> Poisson/BPA
            base_pcd = out_path / "slam_rtab_global_opt_cloud.ply"
            if not base_pcd.exists():
                reconstruct(
                    dataset=dataset, trajectory=t_file, voxel_size=0.02,
                    output_pcd=str(base_pcd), train_indices=split_data["train_indices"]
                )
            generate_mesh(
                input_ply=str(base_pcd),
                output_mesh=str(mesh_file),
                method=recon_cfg["method"],
                depth=recon_cfg.get("depth", 8),
                voxel_size=recon_cfg.get("voxel_size", 0.02)
            )
        recon_time = time.time() - t0

        # Quantitative Geometry Evaluation
        if mesh_file.exists():
            eval_dir = out_path / "evaluations" / key
            summary = evaluate_reconstruction(
                dataset_input=dataset,
                trajectory_input=t_file,
                mesh_input=str(mesh_file),
                output_dir=eval_dir,
                candidate_name=key,
                split_json=str(split_file),
                runtime_sec=recon_time
            )
            summary["experiment_spec"] = spec
            eval_summaries.append(summary)

    # 4. Multi-Candidate Automatic Ranking
    ranked_candidates = rank_candidates(eval_summaries)

    benchmark_report = {
        "benchmark_id": benchmark_id,
        "bag_name": bag_name,
        "evaluated_at": timestamp,
        "ranked_results": ranked_candidates
    }

    # Save summary JSON
    sum_json = out_path / "benchmark_summary.json"
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

    # Markdown Report Generation
    md_file = out_path / "benchmark_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# 🏆 Multi-Layer Benchmark Report: `{bag_name}`\n\n")
        f.write(f"- **Benchmark ID**: `{benchmark_id}`\n")
        f.write(f"- **Date**: {timestamp}\n")
        f.write(f"- **Source Rosbag**: `{bag_path}`\n\n")
        f.write("## 🥇 Overall Ranked Candidates\n\n")
        f.write("| Rank | Candidate Preset | Composite Score | Depth MAE | Depth P95 | Sensor Coverage | Within 20mm | Runtime |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in ranked_candidates:
            raw = r["raw_metrics"]
            mae = f"{raw['depth_mae_mm']} mm" if raw['depth_mae_mm'] is not None else "N/A"
            p95 = f"{raw['depth_p95_mm']} mm" if raw['depth_p95_mm'] is not None else "N/A"
            cov = f"{raw['depth_coverage_ratio']*100:.1f}%" if raw['depth_coverage_ratio'] is not None else "N/A"
            w20 = f"{raw['within_20mm_ratio']*100:.1f}%" if raw['within_20mm_ratio'] is not None else "N/A"
            f.write(f"| **#{r['rank']}** | **{r['candidate_name']}** | **{r['composite_score']:.1f}** | {mae} | {p95} | {cov} | {w20} | {raw['runtime_sec']}s |\n")

        f.write("\n## 🔬 Detailed Experiment Specifications\n\n")
        for s in eval_summaries:
            spec = s.get("experiment_spec", {})
            f.write(f"### `{s['candidate_name']}`\n")
            f.write(f"- **Description**: {spec.get('desc')}\n")
            f.write(f"- **SLAM Backend**: `{spec.get('slam')}`\n")
            f.write(f"- **Reconstruction**: `{spec.get('reconstruction')}`\n")
            f.write(f"- **Mesh Representation**: `{spec.get('mesh')}`\n\n")

    print("\n==========================================================")
    print(f" 🎉 Benchmark All Done!")
    print(f" 📄 Summary JSON: {sum_json}")
    print(f" 📑 Markdown    : {md_file}")
    print("==========================================================")
    return benchmark_report


def main():
    parser = argparse.ArgumentParser(description="Multi-SLAM and Reconstruction Benchmark Tool")
    parser.add_argument("bag", help="Rosbag name or path")
    parser.add_argument("--out-dir", default=None, help="Benchmark output directory")
    parser.add_argument("--quick", action="store_true", help="Run fast comparison")
    args = parser.parse_args()

    run_benchmark(args.bag, args.out_dir, args.quick)


if __name__ == "__main__":
    main()
