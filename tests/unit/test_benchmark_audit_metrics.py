"""
test_benchmark_audit_metrics.py — Quantitative Verification Tests for Architecture Audit.

Validates:
  A. Cache reuse eliminates 100% of expensive worker calls on repeated run
  B. Adaptive search drastically reduces candidate evaluations compared to Cartesian product (60 -> 6~8)
  C. Subprocess segfault on one candidate does not crash remaining workflow
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from auto_mobility.benchmark.workers import run_tsdf_worker, run_surface_worker, WorkerStatus, WorkerResult
from auto_mobility.benchmark.search import SearchEngine
from auto_mobility.benchmark.artifacts import ArtifactManager
from auto_mobility.benchmark.scoring import rank_candidate_summaries


def test_metric_a_cache_hit_eliminates_worker_execution(tmp_path):
    """Metric A: On second run with existing cache, expensive worker calls drop to 0."""
    artifact_mgr = ArtifactManager("test_bag", tmp_path / "eval")
    engine = SearchEngine("test_bag", MagicMock(), tmp_path / "split.json", artifact_mgr)

    trajectories = {"rtab_rgbd": str(tmp_path / "traj1.txt"), "orb_rgbd": str(tmp_path / "traj2.txt")}
    traj_metrics = {"rtab_rgbd": {}, "orb_rgbd": {}}

    # Pre-populate evaluation cache for both candidates
    for cand in ["rtab_rgbd_voxel10mm", "orb_rgbd_voxel10mm"]:
        eval_dir = artifact_mgr.get_candidate_eval_dir(cand)
        eval_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "candidate_name": cand,
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.90, "within_20mm_ratio": 0.85, "within_50mm_ratio": 0.95},
            "mesh": {"num_triangles": 10000},
            "performance": {"runtime_sec": 1.0}
        }
        (eval_dir / "evaluation_summary.json").write_text(__import__("json").dumps(summary))

    with patch("auto_mobility.benchmark.search.run_tsdf_worker") as mock_tsdf:
        results, winner_slam, winner_traj = engine.run_phase_a(trajectories, traj_metrics)
        # 0 expensive TSDF workers called because both were cached!
        assert mock_tsdf.call_count == 0
        assert len(results) == 2
        assert engine.stats["cached_count"] == 2
        assert engine.stats["evaluated_count"] == 0


def test_metric_b_adaptive_search_reduces_cartesian_space(tmp_path):
    """Metric B: Adaptive search explores ~6-8 candidates vs 60 Cartesian candidates (>85% compute reduction)."""
    artifact_mgr = ArtifactManager("test_bag", tmp_path / "eval")
    engine = SearchEngine("test_bag", MagicMock(), tmp_path / "split.json", artifact_mgr, mode="standard", force=False)

    # Cartesian space: 4 SLAM x 3 TSDF x 5 Surface = 60 combinations
    cartesian_combinations = 4 * 3 * 5
    assert cartesian_combinations == 60

    worker_calls = []

    def mock_tsdf(dataset_dir, traj_file, mesh_path, pcd_path, voxel, split_file, quick):
        worker_calls.append(f"tsdf_{voxel}m")
        if mesh_path:
            Path(mesh_path).write_text("v 0 0 0\n")
        if pcd_path:
            Path(pcd_path).write_text("ply\nformat ascii 1.0\n")
        return WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)

    def mock_surf(input_ply, output_mesh, method, voxel, depth):
        worker_calls.append(f"surface_{method}")
        Path(output_mesh).write_text("v 0 0 0\n")
        return WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)

    def mock_eval(**kwargs):
        cname = kwargs.get("candidate_name", "")
        return {
            "candidate_name": cname,
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 5.0, "depth_coverage_ratio": 0.98, "depth_p95_mm": 10.0, "within_20mm_ratio": 0.95, "within_50mm_ratio": 0.98},
            "mesh": {"num_triangles": 30000},
            "performance": {"runtime_sec": 1.0}
        }

    trajectories = {
        "rtab_rgbd": str(tmp_path / "t1.txt"),
        "orb_rgbd": str(tmp_path / "t2.txt"),
        "orb_rgbdi": str(tmp_path / "t3.txt"),
        "stella_rgbd": str(tmp_path / "t4.txt")
    }

    with patch("auto_mobility.benchmark.search.run_tsdf_worker", side_effect=mock_tsdf), \
         patch("auto_mobility.benchmark.search.run_surface_worker", side_effect=mock_surf), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", side_effect=mock_eval), \
         patch.object(artifact_mgr, "get_mesh_path", side_effect=lambda s, v, method=None: tmp_path / f"m_{s}_{v}_{method}.obj"), \
         patch.object(artifact_mgr, "get_pcd_path", side_effect=lambda s, v: tmp_path / f"p_{s}_{v}.ply"):

        # Phase A: 4 SLAM workers
        res_a, best_slam, best_traj = engine.run_phase_a(trajectories, {})
        # Phase B: 1 worker (20mm; 10mm reused, 5mm pruned)
        res_b, best_vox, best_pcd, best_mesh, best_sum = engine.run_phase_b(best_slam, best_traj, res_a)
        # Phase C: 1 worker (Poisson; TSDF reused, Tier 2 pruned)
        res_c, winner_c = engine.run_phase_c(best_slam, best_traj, best_vox, best_pcd, best_mesh, best_sum)

        total_actual_workers = len(worker_calls)
        assert total_actual_workers == 6
        reduction_rate = (cartesian_combinations - total_actual_workers) / cartesian_combinations
        assert reduction_rate >= 0.85, f"Expected >= 85% compute reduction, got {reduction_rate*100:.1f}%"


def test_metric_c_segfault_isolation_allows_workflow_completion(tmp_path):
    """Metric C: Subprocess SIGSEGV on one candidate is isolated and remaining workflow finishes cleanly."""
    artifact_mgr = ArtifactManager("test_bag", tmp_path / "eval")
    engine = SearchEngine("test_bag", MagicMock(), tmp_path / "split.json", artifact_mgr, force=False)

    trajectories = {
        "buggy_slam": str(tmp_path / "buggy_slam_traj.txt"),
        "healthy_slam": str(tmp_path / "healthy_slam_traj.txt")
    }

    def mock_tsdf(dataset_dir, traj_file, mesh_path, pcd_path, voxel, split_file, quick):
        if "buggy_slam" in traj_file:
            return WorkerResult(WorkerStatus.FAIL_SEGFAULT, -11, 0.5, stderr="Segmentation fault (core dumped)")
        # For healthy candidate, create mesh file
        Path(mesh_path).write_text("v 0 0 0\n")
        return WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)

    def mock_eval(**kwargs):
        cname = kwargs.get("candidate_name", "")
        return {
            "candidate_name": cname,
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 6.0, "depth_coverage_ratio": 0.95, "depth_p95_mm": 12.0, "within_20mm_ratio": 0.92, "within_50mm_ratio": 0.96},
            "mesh": {"num_triangles": 25000},
            "performance": {"runtime_sec": 1.0}
        }

    with patch("auto_mobility.benchmark.search.run_tsdf_worker", side_effect=mock_tsdf), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", side_effect=mock_eval), \
         patch.object(artifact_mgr, "get_mesh_path", side_effect=lambda s, v, method=None: tmp_path / f"m_{s}_{v}_{method}.obj"), \
         patch.object(artifact_mgr, "get_pcd_path", side_effect=lambda s, v: tmp_path / f"p_{s}_{v}.ply"):

        res_a, best_slam, best_traj = engine.run_phase_a(trajectories, {})

        assert len(res_a) == 2
        buggy_res = next(r for r in res_a if "buggy_slam" in r["candidate_name"])
        healthy_res = next(r for r in res_a if "healthy_slam" in r["candidate_name"])

        assert buggy_res["status"] == WorkerStatus.FAIL_SEGFAULT
        assert healthy_res["overall_status"] == "PASS"
        assert best_slam == "healthy_slam"
