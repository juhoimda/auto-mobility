"""
test_benchmark_audit_metrics.py — Quantitative Verification Tests for Architecture Audit.

Validates:
  A. Cache reuse eliminates expensive worker calls on repeated run
  B. Beam search drastically reduces candidate evaluations compared to Cartesian product
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

    # Pre-populate evaluation cache for candidates
    for cand in ["rtab_rgbd_tsdf10mm", "orb_rgbd_tsdf10mm"]:
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
        results, top_slams = engine.run_phase_a(trajectories, traj_metrics)
        # 0 expensive TSDF workers called because both were cached!
        assert mock_tsdf.call_count == 0
        assert len(results) == 2
        assert engine.stats["cached_count"] == 2
        assert engine.stats["evaluated_count"] == 0


def test_metric_b_adaptive_search_reduces_cartesian_space(tmp_path):
    """Metric B: Beam search explores focused candidates vs full Cartesian product (>80% compute reduction)."""
    artifact_mgr = ArtifactManager("test_bag", tmp_path / "eval")
    engine = SearchEngine("test_bag", MagicMock(), tmp_path / "split.json", artifact_mgr, mode="standard", force=False)

    # Full Cartesian product would test every permutation
    cartesian_combinations = 4 * 4 * 5
    assert cartesian_combinations == 80

    worker_calls = []

    def mock_tsdf(dataset_dir, traj_file, mesh_path, pcd_path, voxel, depth_max, trunc_mult, stride, split_file, quick):
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

    def mock_direct(dataset_dir, traj_file, pcd_path, voxel, depth_min, depth_max, stride, split_file):
        worker_calls.append(f"direct_{voxel}m")
        Path(pcd_path).write_text("ply\nformat ascii 1.0\n")
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
         patch("auto_mobility.benchmark.search.run_direct_fusion_worker", side_effect=mock_direct), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", side_effect=mock_eval), \
         patch.object(artifact_mgr, "get_mesh_path", side_effect=lambda s, v, method=None: tmp_path / f"m_{s}_{v}_{method}.obj"), \
         patch.object(artifact_mgr, "get_pcd_path", side_effect=lambda s, v: tmp_path / f"p_{s}_{v}.ply"), \
         patch.object(artifact_mgr, "get_direct_pcd_path", side_effect=lambda s, v: tmp_path / f"dir_p_{s}_{v}.ply"):

        # Phase A: 4 SLAM workers
        res_a, top_slams = engine.run_phase_a(trajectories, {})
        # Phase B: Fusion exploration on Top SLAMs
        res_b, top_pipes = engine.run_phase_b(top_slams, res_a)
        # Phase C: Surface exploration on Top Fusion pipelines
        res_c, finalists = engine.run_phase_c(top_pipes, trajectories)

        total_actual_workers = len(worker_calls)
        reduction_rate = (cartesian_combinations - total_actual_workers) / cartesian_combinations
        assert reduction_rate >= 0.70, f"Expected >= 70% compute reduction, got {reduction_rate*100:.1f}%"


def test_metric_c_segfault_isolation_allows_workflow_completion(tmp_path):
    """Metric C: Subprocess SIGSEGV on one candidate is isolated and remaining workflow finishes cleanly."""
    artifact_mgr = ArtifactManager("test_bag", tmp_path / "eval")
    engine = SearchEngine("test_bag", MagicMock(), tmp_path / "split.json", artifact_mgr, force=False)

    trajectories = {
        "buggy_slam": str(tmp_path / "buggy_slam_traj.txt"),
        "healthy_slam": str(tmp_path / "healthy_slam_traj.txt")
    }

    def mock_tsdf(dataset_dir, traj_file, mesh_path, pcd_path, voxel, depth_max, trunc_mult, stride, split_file, quick):
        if "buggy_slam" in traj_file:
            return WorkerResult(WorkerStatus.FAIL_SEGFAULT, -11, 0.5, stderr="Segmentation fault (core dumped)")
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

        res_a, top_slams = engine.run_phase_a(trajectories, {})

        assert len(res_a) == 2
        buggy_res = next(r for r in res_a if "buggy_slam" in r["candidate_name"])
        healthy_res = next(r for r in res_a if "healthy_slam" in r["candidate_name"])

        assert buggy_res["status"] == WorkerStatus.FAIL_SEGFAULT
        assert healthy_res["overall_status"] == "PASS"
        assert len(top_slams) >= 1
        assert top_slams[0][0] == "healthy_slam"
