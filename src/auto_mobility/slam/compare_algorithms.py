#!/usr/bin/env python3
"""
compare_algorithms.py — Multi-Axis SLAM / Fusion / Surface Reconstruction Benchmarking Tool

설계 원칙 (feedback.md 준수):
  PHASE A: SLAM Trajectory 비교 (동일 10mm TSDF 복원 및 Held-out Depth 일치도 평가)
    - rtab_rgbd
    - orb_rgbd
    - orb_rgbdi
    - stella_rgbd
  PHASE B: TSDF Fusion Parameter 비교 (동일 SLAM 궤적 고정)
    - 5mm (0.005m)
    - 10mm (0.010m)
    - 20mm (0.020m)
  PHASE C: Surface Reconstruction 비교 (동일 Point Cloud 고정)
    - tsdf_direct
    - poisson
    - bpa
    - alpha_shape
    - cgal_polygonal
  PHASE D: Final Combination & Separate Rankings

Experiment Manifest:
  - Git Commit SHA, Hardware (GPU, VRAM, RAM), Software (Open3D, ROS, CUDA) 기록
"""

import os
import sys
import time
import json
import psutil
import hashlib
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import (
    BAG_DIR, DB_DIR, MESH_DIR, TRAJECTORY_DIR, BENCHMARK_DIR, FRAME_DIR, POINTCLOUD_DIR, PROJECT_DIR
)
from auto_mobility.dataset.extract_frames import extract_dataset_from_bag
from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.trajectory.export_trajectory import export_from_db
from auto_mobility.mesh.reconstruct_tsdf import reconstruct
from auto_mobility.mesh.mesh_open3d import generate_mesh
from auto_mobility.evaluation.evaluator import evaluate_reconstruction
from auto_mobility.evaluation.split import create_holdout_split, save_split_json


def get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(PROJECT_DIR))
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_system_hardware_info() -> dict:
    mem = psutil.virtual_memory()
    gpu_name = "N/A"
    vram_mb = 0
    try:
        import open3d.core as o3c
        if o3c.cuda.is_available() and o3c.cuda.device_count() > 0:
            gpu_name = "NVIDIA CUDA GPU"
    except Exception:
        pass

    smi_paths = ["/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi"]
    for sp in smi_paths:
        try:
            res = subprocess.run([sp, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().splitlines()
                parts = lines[0].split(",")
                gpu_name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                break
        except Exception:
            pass

    return {
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_mb": round(mem.total / (1024 * 1024), 1),
        "ram_available_mb": round(mem.available / (1024 * 1024), 1),
        "gpu_name": gpu_name,
        "vram_total_mb": vram_mb
    }


def get_software_info() -> dict:
    import open3d as o3d
    return {
        "python": sys.version.split()[0],
        "open3d": o3d.__version__,
        "ros_distro": os.getenv("ROS_DISTRO", "humble"),
        "git_commit": get_git_commit()
    }


def run_benchmark(
    bag_input: str,
    out_dir: Optional[str] = None,
    phase: str = "all",
    quick: bool = False
) -> dict:
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
    print(f" 🧪 Auto-Mobility Multi-Axis Modular Benchmark")
    print(f" 📦 Source Rosbag: {bag_path}")
    print(f" 🎯 Target Phase: {phase.upper()}")
    print(f" 📁 Benchmark Out: {out_path}")
    print("==========================================================")

    # 1. Canonical Dataset 준비
    dataset_dir = FRAME_DIR / bag_name
    if not dataset_dir.exists() or not (dataset_dir / "frames.csv").exists():
        print(f"▶️ [Step 1] Rosbag -> Canonical Frame Dataset 자동 추출...")
        extract_dataset_from_bag(str(bag_path), str(dataset_dir))
    else:
        print(f"▶️ [Step 1] Canonical Frame Dataset 확인: {dataset_dir}")

    dataset = FrameDataset(dataset_dir)

    # 공통 Hold-out Split 생성
    split_data = create_holdout_split(total_frames=len(dataset), policy="every_nth", nth=5)
    split_file = out_path / "shared_holdout_split.json"
    save_split_json(split_data, split_file)
    print(f"✂️ 공통 Hold-out Split: {split_data['train_count']} train / {split_data['holdout_count']} hold-out frames")

    # 2. Trajectories 준비
    trajectories: Dict[str, str] = {}
    traj_metrics: Dict[str, dict] = {}

    # RTAB-Map
    rtab_db = DB_DIR / f"{bag_name}.db"
    rtab_opt_traj = out_path / "rtab_opt_trajectory.txt"
    if not rtab_db.exists():
        print(f"⚙️ RTAB-Map SLAM 실행 중 (DB 생성: {rtab_db})...")
        run_slam_script = PROJECT_DIR / "scripts" / "pipeline" / "run_slam.sh"
        subprocess.run(["bash", str(run_slam_script), bag_name, "--slam=rtab"], capture_output=True, text=True)

    if rtab_db.exists():
        export_from_db(str(rtab_db), str(rtab_opt_traj), opt=0)
        trajectories["rtab_rgbd"] = str(rtab_opt_traj)
        t_obj = Trajectory.from_tum_file(str(rtab_opt_traj))
        traj_metrics["rtab_rgbd"] = t_obj.compute_metrics()

    # ORB-SLAM3 RGB-D
    orb_rgbd_traj = out_path / "orbslam3_rgbd_trajectory.txt"
    if not quick:
        try:
            from auto_mobility.slam.run_orbslam3_bag import run_orbslam3_on_bag
            run_orbslam3_on_bag(str(bag_path), str(orb_rgbd_traj), mode="rgbd")
            trajectories["orb_rgbd"] = str(orb_rgbd_traj)
            traj_metrics["orb_rgbd"] = Trajectory.from_tum_file(str(orb_rgbd_traj)).compute_metrics()
        except Exception as e:
            print(f"⚠️ ORB-SLAM3 RGB-D 실행 건너뜀: {e}")

    # ORB-SLAM3 RGB-D-I
    orb_rgbdi_traj = out_path / "orbslam3_rgbdi_trajectory.txt"
    if not quick:
        try:
            from auto_mobility.slam.run_orbslam3_bag import run_orbslam3_on_bag
            run_orbslam3_on_bag(str(bag_path), str(orb_rgbdi_traj), mode="rgbdi")
            trajectories["orb_rgbdi"] = str(orb_rgbdi_traj)
            traj_metrics["orb_rgbdi"] = Trajectory.from_tum_file(str(orb_rgbdi_traj)).compute_metrics()
        except Exception as e:
            print(f"⚠️ ORB-SLAM3 RGB-D-I 실행 건너뜀 (IMU 누락 또는 빌드): {e}")

    # stella_vslam
    stella_traj = out_path / "stella_rgbd_trajectory.txt"
    if not quick:
        try:
            from auto_mobility.slam.run_stella_bag import run_stella_vslam_on_bag
            run_stella_vslam_on_bag(str(bag_path), str(stella_traj))
            trajectories["stella_rgbd"] = str(stella_traj)
            traj_metrics["stella_rgbd"] = Trajectory.from_tum_file(str(stella_traj)).compute_metrics()
        except Exception as e:
            print(f"⚠️ stella_vslam RGB-D 실행 건너뜀 (독립 바이너리 미설치): {e}")

    # Fallback trajectory if others failed
    primary_slam = "rtab_rgbd" if "rtab_rgbd" in trajectories else list(trajectories.keys())[0] if trajectories else None
    if not primary_slam:
        raise RuntimeError("No valid SLAM trajectory generated!")

    slam_eval_results = []
    tsdf_eval_results = []
    surface_eval_results = []

    # ───────────────────────────────────────────────────────────
    # PHASE A: SLAM Comparison (Fixed: TSDF 10mm, TSDF direct mesh)
    # ───────────────────────────────────────────────────────────
    if phase.lower() in ("all", "a", "slam"):
        print("\n==========================================================")
        print(" 🚀 [PHASE A] SLAM Backend Comparison (Fixed 10mm TSDF)")
        print("==========================================================")
        for slam_k, traj_file in trajectories.items():
            print(f"▶️ Evaluating SLAM candidate: {slam_k}")
            mesh_out = out_path / f"phase_a_{slam_k}_mesh.obj"
            pcd_out = out_path / f"phase_a_{slam_k}_cloud.ply"
            t0 = time.time()
            reconstruct(
                dataset=dataset,
                trajectory=traj_file,
                voxel_size=0.010,
                output_mesh=str(mesh_out),
                output_pcd=str(pcd_out),
                train_indices=split_data["train_indices"],
                no_gpu=quick
            )
            recon_t = time.time() - t0

            eval_dir = out_path / "evaluations" / "phase_a" / slam_k
            summary = evaluate_reconstruction(
                dataset_input=dataset,
                trajectory_input=traj_file,
                mesh_input=str(mesh_out),
                output_dir=eval_dir,
                candidate_name=slam_k,
                split_json=str(split_file),
                runtime_sec=recon_t
            )
            summary["trajectory_metrics"] = traj_metrics.get(slam_k, {})
            slam_eval_results.append(summary)

    # ───────────────────────────────────────────────────────────
    # PHASE B: TSDF Parameter Comparison (Fixed: Best SLAM trajectory)
    # ───────────────────────────────────────────────────────────
    if phase.lower() in ("all", "b", "tsdf", "fusion"):
        print("\n==========================================================")
        print(f" 🚀 [PHASE B] TSDF Parameter Comparison (Fixed SLAM: {primary_slam})")
        print("==========================================================")
        tsdf_voxels = [0.005, 0.010, 0.020]
        if quick:
            tsdf_voxels = [0.010, 0.020]

        fixed_traj = trajectories[primary_slam]
        for v in tsdf_voxels:
            v_tag = f"tsdf_{int(v*1000)}mm"
            print(f"▶️ Evaluating TSDF Voxel: {v_tag} ({v*1000:.1f}mm)")
            mesh_out = out_path / f"phase_b_{v_tag}_mesh.obj"
            pcd_out = out_path / f"phase_b_{v_tag}_cloud.ply"
            t0 = time.time()
            try:
                reconstruct(
                    dataset=dataset,
                    trajectory=fixed_traj,
                    voxel_size=v,
                    output_mesh=str(mesh_out),
                    output_pcd=str(pcd_out),
                    train_indices=split_data["train_indices"],
                    no_gpu=quick
                )
                recon_t = time.time() - t0
                eval_dir = out_path / "evaluations" / "phase_b" / v_tag
                summary = evaluate_reconstruction(
                    dataset_input=dataset,
                    trajectory_input=fixed_traj,
                    mesh_input=str(mesh_out),
                    output_dir=eval_dir,
                    candidate_name=v_tag,
                    split_json=str(split_file),
                    runtime_sec=recon_t
                )
                summary["voxel_size_m"] = v
                tsdf_eval_results.append(summary)
            except Exception as e:
                print(f"❌ TSDF Voxel {v*1000:.1f}mm 실패 (OOM/메모리 한계): {e}")
                tsdf_eval_results.append({
                    "candidate_name": v_tag,
                    "voxel_size_m": v,
                    "status": "FAIL_OOM",
                    "error": str(e),
                    "geometry": {},
                    "mesh": {}
                })

    # ───────────────────────────────────────────────────────────
    # PHASE C: Surface Reconstruction Comparison (Fixed Point Cloud)
    # ───────────────────────────────────────────────────────────
    if phase.lower() in ("all", "c", "surface", "mesh"):
        print("\n==========================================================")
        print(" 🚀 [PHASE C] Surface Reconstruction Comparison (Fixed Geometry)")
        print("==========================================================")
        # Prepare canonical baseline point cloud
        base_pcd = out_path / "phase_c_base_cloud.ply"
        if not base_pcd.exists():
            reconstruct(
                dataset=dataset,
                trajectory=trajectories[primary_slam],
                voxel_size=0.015,
                output_pcd=str(base_pcd),
                train_indices=split_data["train_indices"]
            )

        surface_methods = ["tsdf_direct", "poisson", "bpa", "alpha_shape", "cgal_polygonal"]
        if quick:
            surface_methods = ["tsdf_direct", "poisson", "alpha_shape"]

        for sm in surface_methods:
            print(f"▶️ Evaluating Surface Backend: {sm}")
            mesh_out = out_path / f"phase_c_{sm}_mesh.obj"
            t0 = time.time()
            if sm == "tsdf_direct":
                # Already generated via TSDF
                reconstruct(
                    dataset=dataset,
                    trajectory=trajectories[primary_slam],
                    voxel_size=0.015,
                    output_mesh=str(mesh_out),
                    train_indices=split_data["train_indices"],
                    no_gpu=quick
                )
            else:
                generate_mesh(
                    input_ply=str(base_pcd),
                    output_mesh=str(mesh_out),
                    method=sm,
                    depth=8 if sm == "poisson" else 7,
                    voxel_size=0.015
                )
            mesh_t = time.time() - t0

            eval_dir = out_path / "evaluations" / "phase_c" / sm
            summary = evaluate_reconstruction(
                dataset_input=dataset,
                trajectory_input=trajectories[primary_slam],
                mesh_input=str(mesh_out),
                output_dir=eval_dir,
                candidate_name=sm,
                split_json=str(split_file),
                runtime_sec=mesh_t
            )
            summary["surface_method"] = sm
            surface_eval_results.append(summary)

    # ───────────────────────────────────────────────────────────
    # PHASE D: Manifest, Rankings, and Markdown Report Generation
    # ───────────────────────────────────────────────────────────
    manifest = {
        "benchmark_id": benchmark_id,
        "bag_name": bag_name,
        "evaluated_at": timestamp,
        "hardware": get_system_hardware_info(),
        "software": get_software_info(),
        "phase_a_slam_results": slam_eval_results,
        "phase_b_tsdf_results": tsdf_eval_results,
        "phase_c_surface_results": surface_eval_results
    }

    manifest_file = out_path / "experiment_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    md_report_file = out_path / "benchmark_report.md"
    _generate_modular_markdown_report(manifest, md_report_file)

    print("\n==========================================================")
    print(f" 🎉 Multi-Axis Benchmark Complete!")
    print(f" 📄 Manifest JSON: {manifest_file}")
    print(f" 📑 Report MD    : {md_report_file}")
    print("==========================================================")
    return manifest


def _generate_modular_markdown_report(manifest: dict, report_path: Path):
    bag_name = manifest["bag_name"]
    ts = manifest["evaluated_at"]
    hw = manifest["hardware"]
    sw = manifest["software"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 📊 Multi-Axis Robotics SLAM & Reconstruction Benchmark Report\n\n")
        f.write(f"- **Dataset**: `{bag_name}`\n")
        f.write(f"- **Timestamp**: `{ts}`\n")
        f.write(f"- **Git Commit**: `{sw.get('git_commit')}`\n")
        f.write(f"- **Hardware**: CPU {hw.get('cpu_count')} cores, RAM {hw.get('ram_total_mb')} MB | GPU `{hw.get('gpu_name')}` (VRAM: {hw.get('vram_total_mb')} MB)\n")
        f.write(f"- **Software**: ROS2 `{sw.get('ros_distro')}`, Open3D `{sw.get('open3d')}`, Python `{sw.get('python')}`\n\n")

        # 1. SLAM Ranking
        f.write("## 1. [SLAM Ranking] Trajectory & Downstream Consistency\n\n")
        f.write("| Backend | Tracking Frames | Coverage | Depth MAE | Depth P95 | Within 20mm | Runtime |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for s in manifest.get("phase_a_slam_results", []):
            tm = s.get("trajectory_metrics", {})
            gm = s.get("geometry", {})
            mae = f"{gm.get('depth_mae_mm', 0):.2f} mm" if gm.get('depth_mae_mm') is not None else "N/A"
            p95 = f"{gm.get('depth_p95_mm', 0):.2f} mm" if gm.get('depth_p95_mm') is not None else "N/A"
            cov = f"{gm.get('depth_coverage_ratio', 0)*100:.1f}%" if gm.get('depth_coverage_ratio') is not None else "N/A"
            w20 = f"{gm.get('within_20mm_ratio', 0)*100:.1f}%" if gm.get('within_20mm_ratio') is not None else "N/A"
            f.write(f"| **{s['candidate_name']}** | {tm.get('num_frames', 'N/A')} | {cov} | {mae} | {p95} | {w20} | {s.get('runtime_sec', 0):.2f}s |\n")

        # 2. TSDF Ranking
        f.write("\n## 2. [TSDF Ranking] Fusion Resolution & Memory\n\n")
        f.write("| Voxel Size | Depth MAE | Depth P95 | Coverage | Triangles | Runtime | Status |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for s in manifest.get("phase_b_tsdf_results", []):
            if s.get("status") == "FAIL_OOM":
                f.write(f"| **{s['candidate_name']}** | FAIL | FAIL | FAIL | 0 | 0.0s | ❌ OOM ({s.get('error')}) |\n")
                continue
            gm = s.get("geometry", {})
            mm = s.get("mesh", {})
            mae = f"{gm.get('depth_mae_mm', 0):.2f} mm" if gm.get('depth_mae_mm') is not None else "N/A"
            p95 = f"{gm.get('depth_p95_mm', 0):.2f} mm" if gm.get('depth_p95_mm') is not None else "N/A"
            cov = f"{gm.get('depth_coverage_ratio', 0)*100:.1f}%" if gm.get('depth_coverage_ratio') is not None else "N/A"
            tri = f"{mm.get('num_triangles', 0):,}"
            f.write(f"| **{s['candidate_name']}** | {mae} | {p95} | {cov} | {tri} | {s.get('runtime_sec', 0):.2f}s | ✅ PASS |\n")

        # 3. Surface Ranking
        f.write("\n## 3. [Surface Ranking] Surface Representation & Topology\n\n")
        f.write("| Method | Depth MAE | Point-Mesh P95 | Sensor Coverage | Non-Manifold | Triangles | Runtime |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for s in manifest.get("phase_c_surface_results", []):
            gm = s.get("geometry", {})
            mm = s.get("mesh", {})
            mae = f"{gm.get('depth_mae_mm', 0):.2f} mm" if gm.get('depth_mae_mm') is not None else "N/A"
            p95 = f"{gm.get('point_to_mesh_p95_mm', 0):.2f} mm" if gm.get('point_to_mesh_p95_mm') is not None else "N/A"
            cov = f"{gm.get('depth_coverage_ratio', 0)*100:.1f}%" if gm.get('depth_coverage_ratio') is not None else "N/A"
            nm = f"{mm.get('non_manifold_edges', 0):,}"
            tri = f"{mm.get('num_triangles', 0):,}"
            f.write(f"| **{s['candidate_name']}** | {mae} | {p95} | {cov} | {nm} | {tri} | {s.get('runtime_sec', 0):.2f}s |\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Axis Modular SLAM & Reconstruction Benchmarking Tool")
    parser.add_argument("bag", help="Rosbag name or path")
    parser.add_argument("--out-dir", default=None, help="Benchmark output directory")
    parser.add_argument("--phase", choices=["all", "a", "b", "c", "slam", "tsdf", "surface"], default="all", help="Target benchmark phase (default: all)")
    parser.add_argument("--quick", action="store_true", help="Run quick comparison")
    args = parser.parse_args()

    run_benchmark(args.bag, args.out_dir, phase=args.phase, quick=args.quick)


if __name__ == "__main__":
    main()
