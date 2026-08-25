"""
manifest.py — Experiment Manifest, Multi-Metric Rankings, Markdown Report, and Final Deliverables.

Produces standard reproducible deliverables:
  ├── experiment_manifest.json (Atomic)
  ├── benchmark_report.md (Atomic, Explainable Rationale, Full Raw Metrics, Decision Trace)
  ├── rankings.json (Atomic)
  ├── final/
  │   ├── best.obj (SHA256 verified)
  │   ├── best_config.json (Single source of truth from winner spec)
  │   └── quality_report.json
  └── review/
      ├── rank_01.obj (SHA256 verified)
      ├── rank_02.obj
      └── rank_03.obj
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import hashlib
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from auto_mobility.config import PROJECT_DIR
from auto_mobility.benchmark.artifacts import atomic_write_json, atomic_write_text, compute_file_sha256
from auto_mobility.benchmark.scoring import explain_winner_decision


def get_git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(PROJECT_DIR))
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_git_dirty() -> bool:
    try:
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(PROJECT_DIR))
        return bool(r.stdout.strip()) if r.returncode == 0 else False
    except Exception:
        return False


def compute_dataset_fingerprint(dataset_dir: Path) -> str:
    """Computes a SHA-256 fingerprint from frames.csv and camera_info.json in the dataset dir."""
    if not dataset_dir.exists():
        return "missing"
    sha = hashlib.sha256()
    for fname in ["frames.csv", "camera_info.json"]:
        fp = dataset_dir / fname
        if fp.exists():
            sha.update(fp.read_bytes())
    return sha.hexdigest()[:16]


def get_system_hardware_info() -> dict:
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gpu_name = "N/A"
    vram_mb = 0.0
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
                parts = res.stdout.strip().splitlines()[0].split(",")
                gpu_name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                break
        except Exception:
            pass

    return {
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_mb": round(mem.total / (1024 * 1024), 1),
        "ram_available_mb": round(mem.available / (1024 * 1024), 1),
        "swap_total_mb": round(swap.total / (1024 * 1024), 1),
        "swap_used_mb": round(swap.used / (1024 * 1024), 1),
        "gpu_name": gpu_name,
        "vram_total_mb": vram_mb
    }


def get_software_info() -> dict:
    try:
        import open3d as o3d
        o3d_ver = o3d.__version__
    except Exception:
        o3d_ver = "unknown"

    return {
        "python": sys.version.split()[0],
        "open3d": o3d_ver,
        "ros_distro": os.getenv("ROS_DISTRO", "humble"),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty()
    }


class BenchmarkManifestExporter:
    """Exports manifest, reports, and final deliverables for a benchmark run."""

    @staticmethod
    def generate_markdown_report(
        manifest: dict,
        overall_rankings: List[dict],
        report_path: Union[str, Path]
    ) -> None:
        report_path = Path(report_path)
        bag_name = manifest["bag_name"]
        ts = manifest["evaluated_at"]
        mode = manifest.get("mode", "standard").upper()
        hw = manifest.get("hardware", {})
        sw = manifest.get("software", {})
        stats = manifest.get("summary_stats", {})

        lines = []
        lines.append("# 📊 Multi-Axis Robotics SLAM & Reconstruction Benchmark Report\n")
        lines.append("> [!NOTE]\n> Results are relative to this dataset acquisition and benchmark configuration.\n")
        lines.append("## ⚙️ Benchmark Overview\n")
        lines.append(f"- **Dataset**: `{bag_name}` (Dataset Fingerprint: `{manifest.get('dataset_fingerprint', 'N/A')}`)")
        lines.append(f"- **Execution Mode**: `{mode}`")
        lines.append(f"- **Timestamp**: `{ts}`")
        lines.append(f"- **Git Commit**: `{sw.get('git_commit')}` (Dirty Tree: `{sw.get('git_dirty', False)}`)")
        lines.append(f"- **Hardware**: CPU {hw.get('cpu_count')} cores, RAM {hw.get('ram_total_mb')} MB (Available: {hw.get('ram_available_mb')} MB) | Swap {hw.get('swap_total_mb')} MB | GPU `{hw.get('gpu_name')}` (VRAM: {hw.get('vram_total_mb')} MB)")
        lines.append(f"- **Software**: ROS2 `{sw.get('ros_distro')}`, Open3D `{sw.get('open3d')}`, Python `{sw.get('python')}`")
        if stats:
            lines.append(f"- **Execution Summary**: Total Candidates: `{stats.get('total_candidates', len(overall_rankings))}`, Evaluated: `{stats.get('evaluated_count', 0)}`, Cached/Reused: `{stats.get('cached_count', 0)}`, Pruned: `{stats.get('pruned_count', 0)}`, Rebuilt: `{stats.get('rebuilt_count', 0)}`, Failed: `{stats.get('failed_count', 0)}`")
            lines.append(f"- **Total Benchmark Runtime**: `{stats.get('total_runtime_sec', 0):.1f}s`\n")
        else:
            lines.append("\n")

        # Winner summary & Explainable Decision
        if overall_rankings:
            valid_winners = [r for r in overall_rankings if r.get("hard_gate_pass", False)]
            winner = valid_winners[0] if valid_winners else None
            runner_up = valid_winners[1] if len(valid_winners) > 1 else None

            if winner:
                lines.append("## 🏆 Benchmark Winner\n")
                lines.append(f"- **Best Candidate**: `{winner.get('candidate_name')}`")
                lines.append(f"- **Quality Score**: **{winner.get('quality_score', 0):.2f} / 100**")
                lines.append(f"- **Cost Score**: `{winner.get('cost_score', 0):.2f} / 100`")
                lines.append(f"- **Composite Score**: **{winner.get('composite_score', 0):.2f} / 100**")
                raw = winner.get("raw_metrics", {})
                mae = f"{raw.get('depth_mae_mm', 0):.2f} mm" if raw.get('depth_mae_mm') is not None else "N/A"
                p95 = f"{raw.get('depth_p95_mm', 0):.2f} mm" if raw.get('depth_p95_mm') is not None else "N/A"
                cov = f"{raw.get('depth_coverage_ratio', 0)*100:.1f}%" if raw.get('depth_coverage_ratio') is not None else "N/A"
                compl = f"{raw.get('observed_surface_completeness', 0)*100:.1f}%" if raw.get('observed_surface_completeness') is not None else "N/A"
                fs_corr = f"{raw.get('free_space_correctness_ratio', 1.0)*100:.1f}%" if raw.get('free_space_correctness_ratio') is not None else "100.0%"
                lines.append(f"- **Key Raw Metrics**: Depth MAE `{mae}`, P95 `{p95}`, Coverage `{cov}`, Completeness `{compl}`, Free-Space Correctness `{fs_corr}`")
                lines.append(f"- **Deliverable**: `final/best.obj` (Reproducible via `final/best_config.json`)\n")

                # Decision Rationale
                lines.append("### 🧠 Winner Selection Rationale")
                for r_line in explain_winner_decision(winner, runner_up):
                    lines.append(r_line)
                lines.append("\n")
        else:
            lines.append("## ⚠️ Benchmark Result: No Passing Candidates\n")
            lines.append("- All candidates failed Hard Gate quality criteria; no final winner selected.\n")

        # Root-Cause Diagnosis
        diagnosis = manifest.get("pipeline_diagnosis")
        if diagnosis:
            lines.append("## 🧭 Root-Cause Pipeline Diagnosis\n")
            lines.append(f"- **Primary cause**: `{diagnosis.get('primary_cause', 'INCONCLUSIVE')}`")
            lines.append(f"- **Confidence**: `{diagnosis.get('confidence', 'low')}`")
            lines.append(f"- **Rationale**: {diagnosis.get('rationale', '')}\n")
            lines.append("| Stage | Status | Cause |")
            lines.append("| :--- | :---: | :--- |")
            for stage, detail in diagnosis.get("stages", {}).items():
                lines.append(f"| {stage} | **{detail.get('status', 'UNKNOWN')}** | {detail.get('cause', 'NONE')} |")
            lines.append("")

        # 1. Trajectory Health
        traj_health = manifest.get("trajectory_health_diagnostics", {})
        if traj_health:
            lines.append("## 1. [Trajectory Health Gate: Phase A0]\n")
            lines.append("| Trajectory | Status | Cause | Poses | Finite % | Path Len (m) | BBox Diag (m) | Max Step (m) | Max Vel (m/s) |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for t_k, t_h in traj_health.items():
                st = t_h.get("status", "UNKNOWN")
                st_badge = "✅ PASS" if st == "PASS" else ("⚠️ WARN" if st == "WARN" else "❌ FAIL")
                lines.append(f"| **{t_k}** | {st_badge} | {t_h.get('cause', 'NONE')} | {t_h.get('pose_count', 0)} | {t_h.get('finite_pose_ratio', 1.0)*100:.1f}% | {t_h.get('total_path_length_m', 0):.2f} | {t_h.get('bbox_diagonal_m', 0):.2f} | {t_h.get('translation_step_max_m', 0):.2f} | {t_h.get('linear_velocity_max_mps', 0):.2f} |")
            lines.append("")

        # 2. SLAM Screening
        lines.append("## 2. [SLAM Profile & Backend Screening: Phase A]\n")
        lines.append("| Backend / Profile | Status | Quality Score | Tracking Frames | Coverage | Depth MAE | Depth P95 | Free-Space | Runtime |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in manifest.get("phase_a_slam_results", []):
            status = s.get("status") or s.get("overall_status", "UNKNOWN")
            if status in ("FAIL", "FAIL_CRASH", "FAIL_SEGFAULT", "FAIL_OOM", "FAIL_TIMEOUT", "FAIL_EXCEPTION", "FAIL_TRAJECTORY", "FAIL_ALIGNMENT", "SKIPPED_UNAVAILABLE", "PRUNED", "BLOCKED", "NOT_EVALUATED"):
                lines.append(f"| **{s['candidate_name']}** | ❌ {status} | - | - | - | - | - | - | - |")
                continue
            tm = s.get("trajectory_metrics", {})
            gm = s.get("geometry", {})
            mae = f"{gm.get('depth_mae_mm', 0):.2f} mm" if gm.get('depth_mae_mm') is not None else "N/A"
            p95 = f"{gm.get('depth_p95_mm', 0):.2f} mm" if gm.get('depth_p95_mm') is not None else "N/A"
            cov = f"{gm.get('depth_coverage_ratio', 0)*100:.1f}%" if gm.get('depth_coverage_ratio') is not None else "N/A"
            fs = f"{gm.get('free_space_correctness_ratio', 1.0)*100:.1f}%" if gm.get('free_space_correctness_ratio') is not None else "100%"
            runtime = s.get("runtime_sec") or s.get("performance", {}).get("runtime_sec", 0)
            q_score = s.get("quality_score")
            if q_score is None:
                from auto_mobility.benchmark.scoring import compute_absolute_scores
                q_score = compute_absolute_scores(s)["quality_score"]
            lines.append(f"| **{s['candidate_name']}** | ✅ | {q_score:.1f} | {tm.get('num_frames', 'N/A')} | {cov} | {mae} | {p95} | {fs} | {runtime:.2f}s |")

        # 3. Fusion Screening
        lines.append("\n## 3. [Fusion Screening: Phase B (Common Surface Adapter)]\n")
        lines.append("| Candidate | Fusion Backend | Voxel Size | Status | Quality Score | Depth MAE | Coverage | Completeness | Triangles | Runtime |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in manifest.get("phase_b_tsdf_results", []):
            status = s.get("status") or s.get("overall_status", "UNKNOWN")
            fusion_backend = s.get("fusion_method", "tsdf").upper()
            v_m = s.get("voxel_size_m", 0.010)
            v_str = f"{v_m*1000:.1f}mm" if v_m else "10.0mm"
            if status in ("FAIL", "FAIL_CRASH", "FAIL_SEGFAULT", "FAIL_OOM", "FAIL_TIMEOUT", "FAIL_EXCEPTION", "FAIL_TRAJECTORY", "SKIPPED_UNAVAILABLE", "PRUNED", "BLOCKED", "NOT_EVALUATED"):
                lines.append(f"| **{s['candidate_name']}** | {fusion_backend} | {v_str} | ❌ {status} | - | FAIL | FAIL | FAIL | 0 | 0.0s |")
                continue
            gm = s.get("geometry", {})
            mm = s.get("mesh", {})
            mae = f"{gm.get('depth_mae_mm', 0):.2f} mm" if gm.get('depth_mae_mm') is not None else "N/A"
            cov = f"{gm.get('depth_coverage_ratio', 0)*100:.1f}%" if gm.get('depth_coverage_ratio') is not None else "N/A"
            compl = f"{gm.get('observed_surface_completeness', 0)*100:.1f}%" if gm.get('observed_surface_completeness') is not None else "N/A"
            tri = f"{mm.get('num_triangles', 0):,}"
            runtime = s.get("runtime_sec") or s.get("performance", {}).get("runtime_sec", 0)
            q_score = s.get("quality_score")
            if q_score is None:
                from auto_mobility.benchmark.scoring import compute_absolute_scores
                q_score = compute_absolute_scores(s)["quality_score"]
            lines.append(f"| **{s['candidate_name']}** | {fusion_backend} | {v_str} | ✅ | {q_score:.1f} | {mae} | {cov} | {compl} | {tri} | {runtime:.2f}s |")

        # 4. Surface Screening
        lines.append("\n## 4. [Surface Screening: Phase C (Fair No-Simplification Baseline)]\n")
        lines.append("| Candidate | Surface Method | Status | Quality Score | Depth MAE | Point-Mesh P95 | Free-Space | Non-Manifold | Triangles | Runtime |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in manifest.get("phase_c_surface_results", []):
            status = s.get("status") or s.get("overall_status", "UNKNOWN")
            surf_method = s.get("surface_method", "tsdf_direct")
            if status in ("FAIL", "FAIL_CRASH", "FAIL_SEGFAULT", "FAIL_OOM", "FAIL_TIMEOUT", "FAIL_EXCEPTION", "SKIPPED_UNAVAILABLE", "PRUNED", "BLOCKED", "NOT_EVALUATED"):
                lines.append(f"| **{s['candidate_name']}** | {surf_method} | ❌ {status} | - | - | - | - | - | - | - |")
                continue
            gm = s.get("geometry", {})
            mm = s.get("mesh", {})
            mae = f"{gm.get('depth_mae_mm', 0):.2f} mm" if gm.get('depth_mae_mm') is not None else "N/A"
            p95 = f"{gm.get('point_to_mesh_p95_mm', 0):.2f} mm" if gm.get('point_to_mesh_p95_mm') is not None else "N/A"
            fs = f"{gm.get('free_space_correctness_ratio', 1.0)*100:.1f}%" if gm.get('free_space_correctness_ratio') is not None else "100%"
            nm = f"{mm.get('non_manifold_edges', 0):,}"
            tri = f"{mm.get('num_triangles', 0):,}"
            runtime = s.get("runtime_sec") or s.get("performance", {}).get("runtime_sec", 0)
            q_score = s.get("quality_score")
            if q_score is None:
                from auto_mobility.benchmark.scoring import compute_absolute_scores
                q_score = compute_absolute_scores(s)["quality_score"]
            lines.append(f"| **{s['candidate_name']}** | {surf_method} | ✅ | {q_score:.1f} | {mae} | {p95} | {fs} | {nm} | {tri} | {runtime:.2f}s |")

        # 5. Overall Full Rebuilt Multi-Metric Joint Ranking
        if overall_rankings:
            lines.append("\n## 5. Overall Final Multi-Metric Joint Ranking (stride=1, ALL TRAIN FRAMES)\n")
            lines.append("| Rank | Candidate | Quality | Cost | Composite | Status | Depth MAE | Depth RMSE | Depth P95 | Coverage | Within 20mm | Free Space | Non-Manifold | Triangles | Runtime | Full Rebuild |")
            lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for r in overall_rankings:
                raw = r.get("raw_metrics", {})
                mae_str = f"{raw.get('depth_mae_mm', 0):.2f} mm" if raw.get('depth_mae_mm') is not None else "N/A"
                rmse_str = f"{raw.get('depth_rmse_mm', 0):.2f} mm" if raw.get('depth_rmse_mm') is not None else "N/A"
                p95_str = f"{raw.get('depth_p95_mm', 0):.2f} mm" if raw.get('depth_p95_mm') is not None else "N/A"
                cov_str = f"{raw.get('depth_coverage_ratio', 0)*100:.1f}%" if raw.get('depth_coverage_ratio') is not None else "N/A"
                w20_str = f"{raw.get('within_20mm_ratio', 0)*100:.1f}%" if raw.get('within_20mm_ratio') is not None else "N/A"
                fs_str = f"{raw.get('free_space_correctness_ratio', 1.0)*100:.1f}%" if raw.get('free_space_correctness_ratio') is not None else "100%"
                nm_str = f"{raw.get('non_manifold_ratio', 0)*100:.2f}%" if raw.get('non_manifold_ratio') is not None else "0%"
                tri_str = f"{r.get('summary_data', {}).get('mesh', {}).get('num_triangles', 'N/A')}"
                rt_str = f"{raw.get('runtime_sec', 0):.2f}s" if raw.get('runtime_sec') is not None else "N/A"
                is_rb = "✅ Full (stride=1)" if r.get('summary_data', {}).get('is_full_rebuild') else "Screening"
                lines.append(f"| {r.get('rank')} | **{r.get('candidate_name')}** | {r.get('quality_score', 0):.1f} | {r.get('cost_score', 0):.1f} | {r.get('composite_score', 0):.1f} | {r.get('status')} | {mae_str} | {rmse_str} | {p95_str} | {cov_str} | {w20_str} | {fs_str} | {nm_str} | {tri_str} | {rt_str} | {is_rb} |")

        # 6. Search Decision Trace
        trace = manifest.get("decision_trace", [])
        if trace:
            lines.append("\n## 6. Search & Pruning Decision Trace\n")
            lines.append("| Phase | Candidate / Action | Decision | Reason / Evidence |")
            lines.append("| :---: | :--- | :---: | :--- |")
            for entry in trace:
                lines.append(f"| {entry.get('phase')} | `{entry.get('candidate')}` | **{entry.get('decision')}** | {entry.get('reason')} |")

        # 7. Visual Inspection Guide
        lines.append("\n## 7. Visual Inspection & 3D Viewer Commands\n")
        lines.append("Inspect Top Reconstruction candidates using the interactive 3D viewer:\n")
        lines.append("```bash")
        top_candidates = [r for r in overall_rankings if r.get("hard_gate_pass", False)][:3]
        if not top_candidates:
            top_candidates = overall_rankings[:3]
        for idx, rank_item in enumerate(top_candidates, 1):
            lines.append(f"# Rank {idx}: {rank_item.get('candidate_name')} (Quality: {rank_item.get('quality_score', 0):.1f}, Composite: {rank_item.get('composite_score', 0):.1f})")
            lines.append(f"./scripts/utils/view_mesh.sh {report_path.parent / 'review' / f'rank_{idx:02d}.obj'}\n")
        lines.append(f"# Final Winner Mesh")
        lines.append(f"./scripts/utils/view_mesh.sh {report_path.parent / 'final' / 'best.obj'}")
        lines.append("```\n")

        atomic_write_text(report_path, "\n".join(lines))

    @staticmethod
    def export_final_artifacts(
        report_dir: Path,
        manifest_data: dict,
        overall_rankings: List[dict],
        winner_candidate: Optional[dict] = None,
        top_k: int = 3
    ) -> None:
        """Create final deliverables: experiment_manifest.json, benchmark_report.md, rankings.json, final/best.obj, best_config.json, quality_report.json, and review/ rank meshes."""
        report_dir.mkdir(parents=True, exist_ok=True)
        final_dir = report_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        review_dir = report_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export Review Artifacts (Top-K candidates)
        valid_candidates = [r for r in overall_rankings if r.get("hard_gate_pass", False)]
        top_review = valid_candidates[:top_k] if valid_candidates else overall_rankings[:top_k]
        review_sha_map = {}
        for idx, item in enumerate(top_review, 1):
            src_m = item.get("summary_data", {}).get("mesh_path")
            if src_m and Path(src_m).exists():
                dst_m = review_dir / f"rank_{idx:02d}.obj"
                shutil.copy2(Path(src_m), dst_m)
                review_sha_map[f"rank_{idx:02d}"] = {
                    "candidate_name": item.get("candidate_name"),
                    "source_path": str(src_m),
                    "sha256": compute_file_sha256(dst_m)
                }

        manifest_data["review_artifacts"] = review_sha_map

        # 2. Save experiment_manifest.json (Atomic)
        manifest_file = report_dir / "experiment_manifest.json"
        atomic_write_json(manifest_file, manifest_data)

        # Keep diagnosis separately discoverable for tooling
        diagnosis = manifest_data.get("pipeline_diagnosis")
        if diagnosis is not None:
            atomic_write_json(report_dir / "pipeline_diagnosis.json", diagnosis)

        # 3. Save rankings.json (Atomic)
        rankings_file = report_dir / "rankings.json"
        atomic_write_json(rankings_file, overall_rankings)

        # 4. Save benchmark_report.md (Atomic)
        md_report_file = report_dir / "benchmark_report.md"
        BenchmarkManifestExporter.generate_markdown_report(manifest_data, overall_rankings, md_report_file)

        # 5. Final winner deliverables
        if winner_candidate:
            winner_summary = winner_candidate.get("summary_data", {})
            winner_mesh_path_str = winner_summary.get("mesh_path")
            
            best_obj_file = final_dir / "best.obj"
            mesh_sha = "N/A"
            if winner_mesh_path_str and Path(winner_mesh_path_str).exists():
                src_mesh = Path(winner_mesh_path_str)
                shutil.copy2(src_mesh, best_obj_file)
                mesh_sha = compute_file_sha256(best_obj_file)
            elif not best_obj_file.exists():
                with open(best_obj_file, "w", encoding="utf-8") as f:
                    f.write("# empty mesh\n")

            # Save quality_report.json (Atomic)
            quality_report_file = final_dir / "quality_report.json"
            atomic_write_json(quality_report_file, winner_summary)

            # Build best_config.json directly from winner CandidateSpec
            spec_info = winner_summary.get("spec", {}).get("requested_params", {})
            fusion_method = spec_info.get("fusion_method") or winner_summary.get("fusion_method", "tsdf")
            fusion_params = spec_info.get("fusion_params") or {
                "voxel_size_m": winner_summary.get("voxel_size_m", 0.010),
                "depth_min_m": 0.3,
                "depth_max_m": 3.0,
                "trunc_mult": 4.0
            }
            surface_method = spec_info.get("surface_method") or winner_summary.get("surface_method", "tsdf_direct")
            surface_params = spec_info.get("surface_params") or {}
            postprocess_params = spec_info.get("postprocess_params") or {}
            slam_backend = spec_info.get("slam_backend") or winner_summary.get("trajectory_metrics", {}).get("slam_backend") or winner_candidate.get("candidate_name", "").split("_")[0]
            slam_profile = spec_info.get("slam_profile", "normal")
            replay_rate = spec_info.get("replay_rate", 1.0)
            frame_stride = spec_info.get("frame_stride", 1)

            traj_p = winner_summary.get("trajectory_path", "")
            traj_sha = compute_file_sha256(Path(traj_p)) if traj_p and Path(traj_p).exists() else "N/A"

            best_config_file = final_dir / "best_config.json"
            rebuild_cmd = f"python3 -m auto_mobility.benchmark.rebuild --config {best_config_file}"

            best_config = {
                "dataset": manifest_data.get("bag_name"),
                "mode": manifest_data.get("mode", "standard"),
                "winner_candidate_name": winner_candidate.get("candidate_name"),
                "quality_score": winner_candidate.get("quality_score"),
                "cost_score": winner_candidate.get("cost_score"),
                "composite_score": winner_candidate.get("composite_score"),
                "slam": {
                    "backend": slam_backend,
                    "profile": slam_profile,
                    "replay_rate": replay_rate
                },
                "trajectory_path": traj_p,
                "fusion": {
                    "method": fusion_method,
                    "params": fusion_params
                },
                "surface": {
                    "method": surface_method,
                    "params": surface_params,
                    "postprocess": postprocess_params
                },
                "reconstruction": {
                    "frame_stride": frame_stride,
                    "is_full_rebuild": True
                },
                "quality_metrics": winner_summary.get("geometry", {}),
                "mesh_topology_metrics": winner_summary.get("mesh", {}),
                "cost_metrics": winner_summary.get("performance", {}),
                "artifact_hashes": {
                    "mesh_sha256": mesh_sha,
                    "trajectory_sha256": traj_sha
                },
                "reproducibility": {
                    "random_seed": manifest_data.get("random_seed", 42),
                    "dataset_fingerprint": manifest_data.get("dataset_fingerprint", "N/A"),
                    "git_commit": manifest_data.get("software", {}).get("git_commit", "unknown"),
                    "git_dirty": manifest_data.get("software", {}).get("git_dirty", False),
                    "rebuild_command": rebuild_cmd
                },
                "software": manifest_data.get("software", {}),
                "hardware": manifest_data.get("hardware", {}),
                "evaluated_at": manifest_data.get("evaluated_at")
            }

            best_config_file = final_dir / "best_config.json"
            atomic_write_json(best_config_file, best_config)
