#!/usr/bin/env python3
"""
compare_algorithms.py — 동일 rosbag 기반 SLAM / Odometry 및 Mesh 재구성 비교 벤치마크 도구

기능:
  1) 동일 rosbag에 대해 여러 Odometry / SLAM 파라미터 백엔드 실행
  2) 각 실행 결과로부터 표준 TUM Trajectory (.txt) 및 Mesh (.obj) 생성
  3) Trajectory 지표 (프레임 수, 총 경로 길이, 최대 스텝, 속도 점프 등) 및
     Mesh 품질 지표 (Vertices, Triangles, 표면적, 밀도, Bounding Box 크기 등) 비교 요약 리포트(JSON/Markdown) 출력

사용법:
  python3 compare_algorithms.py <bag_name_or_path> [--out-dir ros2_data/benchmarks] [--quick]
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import BAG_DIR, DB_DIR, MESH_DIR, TRAJECTORY_DIR, BENCHMARK_DIR
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.trajectory.export_trajectory import export_from_db

try:
    import open3d as o3d
except ImportError:
    o3d = None


# 비교할 SLAM/Odom 파라미터 프리셋 정의
ALGO_PRESETS = {
    "rtab_f2m_opt": {
        "desc": "RTAB-Map Frame-to-Map (Default, OdomF2M 1000, Global Opt)",
        "opt": 0,
        "tsdf_voxel": 0.01,
    },
    "rtab_f2m_raw": {
        "desc": "RTAB-Map Frame-to-Map (Raw Odometry Pose, No Loop Closure Opt)",
        "opt": 2,
        "tsdf_voxel": 0.01,
    },
    "rtab_tsdf_fine": {
        "desc": "RTAB-Map Global Opt + Fine TSDF (Voxel 5mm)",
        "opt": 0,
        "tsdf_voxel": 0.005,
    },
}


def evaluate_mesh(mesh_path: str) -> dict:
    if not os.path.exists(mesh_path) or o3d is None:
        return {"exists": False}

    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        num_v = len(mesh.vertices)
        num_t = len(mesh.triangles)
        if num_v == 0 or num_t == 0:
            return {"exists": True, "valid": False, "num_vertices": 0, "num_triangles": 0}

        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent().tolist()
        try:
            area = float(mesh.get_surface_area())
        except Exception:
            area = 0.0

        try:
            watertight = bool(mesh.is_watertight())
        except Exception:
            watertight = False

        return {
            "exists": True,
            "valid": True,
            "num_vertices": num_v,
            "num_triangles": num_t,
            "surface_area_m2": round(area, 4),
            "density_tri_per_m2": round(num_t / max(area, 1e-5), 1),
            "bbox_extent_m": [round(x, 3) for x in extent],
            "is_watertight": watertight,
        }
    except Exception as e:
        return {"exists": True, "valid": False, "error": str(e)}


def run_comparison(bag_input: str, out_dir: str = None, quick: bool = False):
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
    out_dir = Path(out_dir) if out_dir else (BENCHMARK_DIR / benchmark_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"==========================================================")
    print(f" 🧪 Auto-Mobility Algorithm Benchmark & Comparison")
    print(f" 📦 Source Rosbag: {bag_path}")
    print(f" 📁 Benchmark Out: {out_dir}")
    print(f"==========================================================")

    # 1. Base SLAM Run (rosbag replay -> rtabmap.db)
    base_db_name = f"bench_{bag_name}.db"
    base_db_path = DB_DIR / base_db_name
    print(f"\n▶️ [Step 1] Baseline SLAM Replay Run...")
    if not base_db_path.exists():
        print(f"   Generating DB: {base_db_path} from rosbag...")
        run_bag_script = PROJECT_DIR / "scripts" / "pipeline" / "run_bag.sh"
        # run_bag.sh bag_name
        r = subprocess.run(["bash", str(run_bag_script), bag_name, "--compressed"], capture_output=True, text=True)
        if not base_db_path.exists():
            # 혹시 기본 생성된 db가 있는지 확인
            print("   Warning: Checking if custom DB path was used...")
    else:
        print(f"   ✅ Using existing Base DB: {base_db_path}")

    results = {
        "benchmark_id": benchmark_id,
        "bag_name": bag_name,
        "timestamp": timestamp,
        "algorithms": {}
    }

    # 2. Iterate Presets and reconstruct meshes & evaluate
    for key, cfg in ALGO_PRESETS.items():
        if quick and key == "rtab_tsdf_fine":
            continue

        print(f"\n▶️ [Step 2] Testing Algorithm/Preset: {key} ({cfg['desc']})")
        traj_file = out_dir / f"{key}_trajectory.txt"
        mesh_file = out_dir / f"{key}_mesh.obj"

        # Trajectory 추출
        if base_db_path.exists():
            try:
                export_from_db(str(base_db_path), str(traj_file), opt=cfg["opt"])
                traj = Trajectory.from_tum_file(str(traj_file))
                t_metrics = traj.compute_metrics()
            except Exception as e:
                print(f"   ⚠️ Trajectory export failed for {key}: {e}")
                t_metrics = {"error": str(e)}
        else:
            t_metrics = {"error": "DB not found"}

        # TSDF Reconstruction
        if base_db_path.exists():
            print(f"   🔨 Reconstructing Mesh via TSDF (voxel={cfg['tsdf_voxel']})...")
            reconstruct_script = PROJECT_DIR / "src" / "auto_mobility" / "mesh" / "reconstruct_tsdf.py"
            cmd = [
                sys.executable, str(reconstruct_script),
                str(base_db_path), str(mesh_file),
                f"--voxel={cfg['tsdf_voxel']}",
                f"--poses-opt={cfg['opt']}",
                "--no-gpu" if quick else "--voxel=" + str(cfg['tsdf_voxel'])
            ]
            if traj_file.exists():
                cmd.append(f"--trajectory={str(traj_file)}")

            t0 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            recon_time = time.time() - t0
            m_metrics = evaluate_mesh(str(mesh_file))
            m_metrics["reconstruction_time_sec"] = round(recon_time, 2)
        else:
            m_metrics = {"error": "DB not found"}

        results["algorithms"][key] = {
            "description": cfg["desc"],
            "trajectory_metrics": t_metrics,
            "mesh_metrics": m_metrics,
            "artifacts": {
                "trajectory": str(traj_file) if traj_file.exists() else None,
                "mesh": str(mesh_file) if mesh_file.exists() else None,
            }
        }

    # 3. Save Summary JSON & Markdown
    json_path = out_dir / "benchmark_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_path = out_dir / "benchmark_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 📊 Algorithm Benchmark Report: {bag_name}\n\n")
        f.write(f"- **Benchmark ID**: `{benchmark_id}`\n")
        f.write(f"- **Date**: {timestamp}\n")
        f.write(f"- **Source Bag**: `{bag_path}`\n\n")
        f.write("## 📈 Comparison Summary Table\n\n")
        f.write("| Algorithm Preset | Frames | Path Len (m) | Max Jump (m) | Vertices | Triangles | Area (m²) | Tri Density | Watertight |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for k, v in results["algorithms"].items():
            tm = v.get("trajectory_metrics", {})
            mm = v.get("mesh_metrics", {})
            f.write(f"| **{k}** | {tm.get('num_frames', 'N/A')} | {tm.get('total_path_length_m', 'N/A')} | "
                    f"{tm.get('max_step_m', 'N/A')} | {mm.get('num_vertices', 'N/A'):,} | {mm.get('num_triangles', 'N/A'):,} | "
                    f"{mm.get('surface_area_m2', 'N/A')} | {mm.get('density_tri_per_m2', 'N/A')} | "
                    f"{'✅' if mm.get('is_watertight') else '❌'} |\n")

    print("\n==========================================================")
    print(f" 🎉 Benchmark Completed!")
    print(f" 📄 Summary JSON: {json_path}")
    print(f" 📑 Markdown Report: {md_path}")
    print("==========================================================")
    print("\n" + md_path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description="Compare SLAM & Mesh algorithms on same rosbag")
    parser.add_argument("bag", help="Rosbag name or path")
    parser.add_argument("--out-dir", default=None, help="Output directory for benchmark artifacts")
    parser.add_argument("--quick", action="store_true", help="Run fast lightweight comparison")
    args = parser.parse_args()

    run_comparison(args.bag, args.out_dir, args.quick)


if __name__ == "__main__":
    main()
