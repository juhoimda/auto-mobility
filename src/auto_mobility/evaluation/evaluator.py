#!/usr/bin/env python3
"""
evaluator.py — 3D Reconstruction 정량 형상 품질(Geometry Quality) 통합 평가 도구

주요 절차:
  1) Canonical Frame Dataset + TUM Trajectory 로드
  2) Trajectory Timestamp Association (SLERP 보간)
  3) Train / Hold-out Frame Split 적용 (기본: 매 5번째 프레임, 20% Holdout)
  4) Open3D RaycastingScene 기반 Held-out Depth Reprojection 오차 측정
  5) Point-to-Mesh 3D 최단거리 및 잔차(Residual) 측정
  6) Mesh Topology (Degenerate, Manifold, Component) 및 실내 주 평면(Plane) 분석
  7) Rule-based PASS / WARN / FAIL 판정 및 QualityProfile(JSON/MD) 출력
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import FRAME_DIR, TRAJECTORY_DIR, MESH_DIR, EVALUATION_DIR, get_evaluation_config
from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.trajectory.association import (
    associate_trajectory_to_frames,
    save_association_csv,
    AssociationSummary
)
from auto_mobility.evaluation.split import create_holdout_split, save_split_json, load_split_json
from auto_mobility.evaluation.render_depth import create_raycasting_scene, render_depth_map
from auto_mobility.evaluation.geometry_metrics import (
    compute_depth_metrics,
    backproject_depth_to_world_points,
    compute_point_to_mesh_metrics,
    generate_error_visualization
)
from auto_mobility.evaluation.mesh_metrics import compute_mesh_quality_metrics, compute_plane_quality_metrics
from auto_mobility.evaluation.trajectory_metrics import compute_trajectory_quality
from auto_mobility.evaluation.report import generate_markdown_report


def evaluate_rules(summary: dict, cfg: dict) -> Tuple[str, dict]:
    """QualityThresholds 설정에 기반하여 PASS / WARN / FAIL 판정 수행."""
    thresholds = cfg.get("quality_thresholds", {})
    rules = {}
    verdicts = []

    # 1. Pose Association Rule
    pose_summary = summary.get("pose_association", {})
    pose_thr = thresholds.get("pose_association", {})
    cov = pose_summary.get("pose_coverage_ratio", 0.0)
    cov_pass = pose_thr.get("coverage_ratio", {}).get("pass", 0.85)
    cov_warn = pose_thr.get("coverage_ratio", {}).get("warn", 0.65)

    if cov >= cov_pass:
        p_status = "PASS"
    elif cov >= cov_warn:
        p_status = "WARN"
    else:
        p_status = "FAIL"
    rules["pose_association"] = {"status": p_status, "coverage": cov}
    verdicts.append(p_status)

    # 2. Geometry Accuracy Rule (Depth MAE & P95)
    geom = summary.get("geometry", {})
    geom_thr = thresholds.get("geometry", {})
    mae = geom.get("depth_mae_mm")
    p95 = geom.get("depth_p95_mm")
    mae_pass = geom_thr.get("depth_mae_mm", {}).get("pass", 25.0)
    mae_warn = geom_thr.get("depth_mae_mm", {}).get("warn", 50.0)
    p95_pass = geom_thr.get("depth_p95_mm", {}).get("pass", 60.0)
    p95_warn = geom_thr.get("depth_p95_mm", {}).get("warn", 120.0)

    if mae is not None and p95 is not None:
        if mae <= mae_pass and p95 <= p95_pass:
            g_status = "PASS"
        elif mae <= mae_warn and p95 <= p95_warn:
            g_status = "WARN"
        else:
            g_status = "FAIL"
    else:
        g_status = "FAIL"
    rules["geometry_accuracy"] = {"status": g_status, "mae": mae, "p95": p95}
    verdicts.append(g_status)

    # 3. Geometry Coverage Rule
    depth_cov = geom.get("depth_coverage_ratio", 0.0)
    cov_pass_thr = geom_thr.get("depth_coverage_ratio", {}).get("pass", 0.75)
    cov_warn_thr = geom_thr.get("depth_coverage_ratio", {}).get("warn", 0.50)
    if depth_cov >= cov_pass_thr:
        c_status = "PASS"
    elif depth_cov >= cov_warn_thr:
        c_status = "WARN"
    else:
        c_status = "FAIL"
    rules["geometry_coverage"] = {"status": c_status, "coverage": depth_cov}
    verdicts.append(c_status)

    # 4. Mesh Topology Rule
    mesh = summary.get("mesh", {})
    mesh_thr = thresholds.get("mesh_topology", {})
    deg_ratio = mesh.get("degenerate_triangle_ratio", 0.0)
    non_man_ratio = mesh.get("non_manifold_edge_ratio", 0.0)
    deg_pass = mesh_thr.get("degenerate_ratio", {}).get("pass", 0.001)
    deg_warn = mesh_thr.get("degenerate_ratio", {}).get("warn", 0.01)

    if deg_ratio <= deg_pass and non_man_ratio <= deg_pass:
        m_status = "PASS"
    elif deg_ratio <= deg_warn and non_man_ratio <= deg_warn:
        m_status = "WARN"
    else:
        m_status = "FAIL"
    rules["mesh_topology"] = {"status": m_status, "degenerate_ratio": deg_ratio}
    verdicts.append(m_status)

    # Overall Status
    if "FAIL" in verdicts:
        overall = "FAIL"
    elif "WARN" in verdicts:
        overall = "WARN"
    else:
        overall = "PASS"

    return overall, rules


def evaluate_reconstruction(
    dataset_input: Union[str, Path, FrameDataset],
    trajectory_input: Union[str, Path, Trajectory],
    mesh_input: Union[str, Path, o3d.geometry.TriangleMesh],
    output_dir: Optional[Union[str, Path]] = None,
    candidate_name: Optional[str] = None,
    split_json: Optional[Union[str, Path]] = None,
    render_samples: int = 10,
    runtime_sec: Optional[float] = None,
    peak_rss_mb: Optional[float] = None,
    peak_gpu_memory_mb: Optional[float] = None,
    cheap: bool = False,
    mode: Optional[str] = None,
    max_holdout_samples: Optional[int] = None
) -> dict:
    """Reconstruction 결과물에 대해 Held-out Depth Reprojection 및 기하 정밀도 통합 평가 수행.
    
    cheap=True 또는 mode="cheap" 시:
      - holdout frame 부분 샘플링 (최대 12장)
      - 오차 시각화 이미지 디스크 저장 생략 (render_samples=0)
      - Point-to-Mesh 거리 연산 샘플링 축소 (5k points)
      - Dominant plane RANSAC 축소 (200회, 2개 평면)
    """
    t_start = time.time()
    cfg = get_evaluation_config()
    eval_cfg = cfg.get("evaluation", {})
    is_cheap = cheap or (mode == "cheap")

    # 1. Load dataset
    if isinstance(dataset_input, FrameDataset):
        dataset = dataset_input
    else:
        d_path = Path(dataset_input)
        if not d_path.is_absolute() and (FRAME_DIR / dataset_input).exists():
            d_path = FRAME_DIR / dataset_input
        dataset = FrameDataset(d_path)

    dataset_name = dataset.dataset_dir.name

    # 2. Load trajectory
    if isinstance(trajectory_input, Trajectory):
        traj = trajectory_input
        traj_path_str = "in-memory"
    else:
        t_path = Path(trajectory_input)
        if not t_path.is_absolute() and (TRAJECTORY_DIR / trajectory_input).exists():
            t_path = TRAJECTORY_DIR / trajectory_input
        traj = Trajectory.from_tum_file(str(t_path))
        traj_path_str = str(t_path)

    # 3. Load mesh
    if isinstance(mesh_input, o3d.geometry.TriangleMesh):
        mesh = mesh_input
        mesh_path_str = "in-memory"
    else:
        m_path = Path(mesh_input)
        if not m_path.is_absolute() and (MESH_DIR / mesh_input).exists():
            m_path = MESH_DIR / mesh_input
        mesh = o3d.io.read_triangle_mesh(str(m_path))
        mesh_path_str = str(m_path)

    cand_name = candidate_name or (Path(mesh_path_str).stem if mesh_path_str != "in-memory" else "candidate")
    out_dir = Path(output_dir) if output_dir else (EVALUATION_DIR / dataset_name / cand_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = out_dir / "renders"
    if not is_cheap:
        renders_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    eval_mode_tag = "⚡ [CHEAP SCREENING]" if is_cheap else "🔬 [FULL FIDELITY]"
    print(f" 📐 3D Reconstruction 정량 품질 평가 시작 {eval_mode_tag}")
    print(f" 📦 Dataset    : {dataset_name} ({len(dataset)} frames)")
    print(f" 📍 Trajectory : {traj_path_str} ({len(traj)} poses)")
    print(f" 🔺 Mesh       : {mesh_path_str} ({len(mesh.vertices):,} vertices)")
    print(f" 📁 Output Dir : {out_dir}")
    print("==========================================================")

    warnings = []

    # 4. Trajectory Timestamp Association
    max_pose_gap = float(eval_cfg.get("max_pose_gap_ms", 50.0))
    timestamps = dataset.get_timestamps(use_rgb=True)
    poses_dict, assoc_records, assoc_summary = associate_trajectory_to_frames(
        timestamps, traj, max_pose_gap_ms=max_pose_gap, enable_interpolation=True
    )
    if assoc_summary.warning:
        warnings.append(assoc_summary.warning)

    save_association_csv(assoc_records, str(out_dir / "pose_association.csv"))

    # 5. Train / Hold-out Split
    valid_pose_indices = [r.frame_id for r in assoc_records if r.valid]
    if split_json and os.path.exists(split_json):
        split_data = load_split_json(split_json)
    else:
        holdout_cfg = eval_cfg.get("holdout", {})
        split_data = create_holdout_split(
            total_frames=len(dataset),
            policy=holdout_cfg.get("policy", "every_nth"),
            nth=holdout_cfg.get("nth", 5),
            ratio=holdout_cfg.get("ratio", 0.20),
            valid_indices=valid_pose_indices
        )
    save_split_json(split_data, str(out_dir / "split.json"))

    raw_holdout_indices = split_data.get("holdout_indices", [])
    
    # Adaptive Hold-out frame sampling:
    # cheap screening -> max 12 frames, full fidelity -> max 40 frames (statistically representative, ~99% CLT confidence)
    max_eval_frames = max_holdout_samples if max_holdout_samples is not None else (12 if is_cheap else 40)
    if len(raw_holdout_indices) > max_eval_frames:
        step = max(1, len(raw_holdout_indices) // max_eval_frames)
        holdout_indices = raw_holdout_indices[::step][:max_eval_frames]
    else:
        holdout_indices = raw_holdout_indices

    print(f"🎯 Hold-out 평가 프레임 수: {len(holdout_indices)}장 (전체 {len(raw_holdout_indices)}장 중 {'샘플링' if len(raw_holdout_indices) > len(holdout_indices) else '전체'})")

    # 6. Raycasting Depth Reprojection on Hold-out frames
    scene = create_raycasting_scene(mesh)
    depth_min_m = float(eval_cfg.get("raycasting", {}).get("depth_min_m", 0.3))
    depth_max_m = float(eval_cfg.get("raycasting", {}).get("depth_max_m", 5.0))

    frame_metrics_list = []
    all_world_points = []

    # Visualization sample indices (disabled in cheap mode)
    actual_render_samples = 0 if is_cheap else render_samples
    render_sample_indices = set(np.linspace(0, max(0, len(holdout_indices)-1), min(len(holdout_indices), actual_render_samples), dtype=int)) if actual_render_samples > 0 else set()
    render_records = []

    for h_idx, f_idx in enumerate(holdout_indices):
        if f_idx not in poses_dict:
            continue
        T_world_cam = poses_dict[f_idx]
        real_depth = dataset.get_depth(f_idx)
        if real_depth is None:
            continue

        rend_depth = render_depth_map(
            scene, T_world_cam, dataset.intrinsics,
            depth_min_m=depth_min_m, depth_max_m=depth_max_m
        )

        f_metrics = compute_depth_metrics(real_depth, rend_depth, depth_min_m*1000.0, depth_max_m*1000.0)
        f_metrics["frame_id"] = f_idx
        f_metrics["timestamp"] = dataset.frames[f_idx].rgb_timestamp
        frame_metrics_list.append(f_metrics)

        # Collect for visualization if enabled
        if h_idx in render_sample_indices:
            real_p = renders_dir / f"{f_idx:06d}_real.png"
            rend_p = renders_dir / f"{f_idx:06d}_rendered.png"
            heat_p = renders_dir / f"{f_idx:06d}_error.png"
            generate_error_visualization(
                real_depth, rend_depth, real_p, rend_p, heat_p,
                depth_min_mm=depth_min_m*1000.0, depth_max_mm=depth_max_m*1000.0
            )
            render_records.append({
                "frame_id": f_idx,
                "real_path": str(real_p.relative_to(out_dir)),
                "rendered_path": str(rend_p.relative_to(out_dir)),
                "heatmap_path": str(heat_p.relative_to(out_dir))
            })

        # Backproject points for 3D point-to-mesh distance
        pts_w = backproject_depth_to_world_points(
            real_depth, T_world_cam, dataset.intrinsics,
            depth_min_mm=depth_min_m*1000.0, depth_max_mm=depth_max_m*1000.0,
            stride=6 if is_cheap else 4
        )
        if len(pts_w) > 0:
            all_world_points.append(pts_w)

    # Save per-frame metrics CSV
    if frame_metrics_list:
        with open(out_dir / "frame_metrics.csv", "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=list(frame_metrics_list[0].keys()))
            writer.writeheader()
            for fm in frame_metrics_list:
                writer.writerow(fm)

    # Aggregate Geometry Metrics across all holdout frames
    valid_maes = [m["depth_mae_mm"] for m in frame_metrics_list if m["depth_mae_mm"] is not None]
    valid_p95s = [m["depth_p95_mm"] for m in frame_metrics_list if m["depth_p95_mm"] is not None]
    valid_covs = [m["depth_coverage_ratio"] for m in frame_metrics_list]
    valid_w20s = [m["within_20mm_ratio"] for m in frame_metrics_list]
    valid_w10s = [m["within_10mm_ratio"] for m in frame_metrics_list]
    valid_w50s = [m["within_50mm_ratio"] for m in frame_metrics_list]
    valid_compl = [m.get("observed_surface_completeness", 0.0) for m in frame_metrics_list]
    valid_fs_viol = [m.get("free_space_violation_ratio", 0.0) for m in frame_metrics_list]
    valid_fs_corr = [m.get("free_space_correctness_ratio", 1.0) for m in frame_metrics_list]

    # Point-to-Mesh Distance
    max_pts = 5000 if is_cheap else 50000
    if all_world_points:
        combined_pts = np.concatenate(all_world_points, axis=0)
        p2m_metrics = compute_point_to_mesh_metrics(scene, combined_pts, max_sample_points=max_pts)
    else:
        p2m_metrics = compute_point_to_mesh_metrics(scene, np.empty((0, 3)))

    geometry_summary = {
        "depth_mae_mm": round(float(np.mean(valid_maes)), 2) if valid_maes else None,
        "depth_rmse_mm": round(float(np.sqrt(np.mean(np.array(valid_maes)**2))), 2) if valid_maes else None,
        "depth_median_error_mm": round(float(np.median(valid_maes)), 2) if valid_maes else None,
        "depth_p90_mm": round(float(np.percentile(valid_p95s, 90)), 2) if valid_p95s else None,
        "depth_p95_mm": round(float(np.mean(valid_p95s)), 2) if valid_p95s else None,
        "depth_coverage_ratio": round(float(np.mean(valid_covs)), 4) if valid_covs else 0.0,
        "observed_surface_completeness": round(float(np.mean(valid_compl)), 4) if valid_compl else 0.0,
        "free_space_violation_ratio": round(float(np.mean(valid_fs_viol)), 4) if valid_fs_viol else 0.0,
        "free_space_correctness_ratio": round(float(np.mean(valid_fs_corr)), 4) if valid_fs_corr else 1.0,
        "within_10mm_ratio": round(float(np.mean(valid_w10s)), 4) if valid_w10s else 0.0,
        "within_20mm_ratio": round(float(np.mean(valid_w20s)), 4) if valid_w20s else 0.0,
        "within_50mm_ratio": round(float(np.mean(valid_w50s)), 4) if valid_w50s else 0.0,
        **p2m_metrics
    }

    # 7. Mesh Topology & Quality
    mesh_summary = compute_mesh_quality_metrics(mesh)

    # 8. Dominant Plane Analysis
    plane_summary = compute_plane_quality_metrics(
        mesh,
        num_iterations=200 if is_cheap else 1000,
        max_planes=2 if is_cheap else 5
    )

    # 9. Trajectory Quality
    traj_summary = compute_trajectory_quality(traj)

    # 10. Performance / Cost
    perf_summary = {
        "runtime_sec": round(runtime_sec or (time.time() - t_start), 2),
        "peak_rss_mb": peak_rss_mb,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "evaluation_duration_sec": round(time.time() - t_start, 2)
    }

    # 11. Compile QualityProfile
    quality_profile = {
        "candidate_name": cand_name,
        "dataset_name": dataset_name,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mesh_path": mesh_path_str,
        "trajectory_path": traj_path_str,
        "pose_association": {
            "num_frames": assoc_summary.num_frames,
            "pose_match_count": assoc_summary.pose_match_count,
            "pose_missing_count": assoc_summary.pose_missing_count,
            "pose_coverage_ratio": assoc_summary.pose_coverage_ratio,
            "pose_dt_mean_ms": assoc_summary.pose_dt_mean_ms,
            "pose_dt_p95_ms": assoc_summary.pose_dt_p95_ms
        },
        "geometry": geometry_summary,
        "mesh": mesh_summary,
        "plane_analysis": plane_summary,
        "trajectory": traj_summary,
        "performance": perf_summary,
        "warnings": warnings,
        "render_samples": render_records
    }

    # 12. Apply Rule-based Evaluation
    overall_status, rule_details = evaluate_rules(quality_profile, cfg)
    quality_profile["overall_status"] = overall_status
    quality_profile["rule_evaluations"] = rule_details

    # Save summary JSON
    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f_json:
        json.dump(quality_profile, f_json, indent=2, ensure_ascii=False)

    # Generate & Save Markdown report
    md_content = generate_markdown_report(quality_profile)
    with open(out_dir / "evaluation_report.md", "w", encoding="utf-8") as f_md:
        f_md.write(md_content)

    print("\n==========================================================")
    print(f" 🎉 Evaluation Completed! Status: {overall_status}")
    print(f" 📄 Summary JSON : {out_dir / 'evaluation_summary.json'}")
    print(f" 📑 Markdown     : {out_dir / 'evaluation_report.md'}")
    print(f" 🔍 Depth MAE    : {geometry_summary.get('depth_mae_mm')} mm | P95: {geometry_summary.get('depth_p95_mm')} mm | Coverage: {geometry_summary.get('depth_coverage_ratio', 0.0)*100:.1f}%")
    print("==========================================================")

    return quality_profile


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D Reconstruction Quality using Held-out Sensor Data")
    parser.add_argument("dataset", help="Dataset name or path (e.g. ros2_data/frames/room01)")
    parser.add_argument("mesh", help="Reconstructed mesh file (.obj)")
    parser.add_argument("trajectory", help="TUM trajectory file (.txt)")
    parser.add_argument("--out-dir", default=None, help="Output evaluation directory")
    parser.add_argument("--name", default=None, help="Candidate name (e.g. rtab_tsdf_10mm)")
    parser.add_argument("--split", default=None, help="Pre-computed split.json path")
    parser.add_argument("--render-samples", type=int, default=10, help="Number of holdout visualization frames to render")
    args = parser.parse_args()

    evaluate_reconstruction(
        dataset_input=args.dataset,
        trajectory_input=args.trajectory,
        mesh_input=args.mesh,
        output_dir=args.out_dir,
        candidate_name=args.name,
        split_json=args.split,
        render_samples=args.render_samples
    )


if __name__ == "__main__":
    main()
