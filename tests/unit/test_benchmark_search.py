"""
tests/unit/test_benchmark_search.py

Unit tests for SearchEngine:
  - Beam search Top-K SLAM selection from Phase A
  - Artifact reuse (Phase A 10mm in Phase B, Phase B TSDF direct in Phase C)
  - Direct Point Cloud Fusion baseline evaluation
  - Full Rebuild (stride=1) execution on Top 3 Finalists
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_mobility.benchmark.search import SearchEngine
from auto_mobility.benchmark.artifacts import ArtifactManager
from auto_mobility.benchmark.workers import WorkerResult, WorkerStatus


@pytest.fixture
def mock_search_env(tmp_path):
    bag_name = "test_bag"
    eval_dir = tmp_path / "evaluations" / bag_name
    eval_dir.mkdir(parents=True)
    
    artifact_mgr = ArtifactManager(bag_name, base_eval_dir=eval_dir)
    
    mock_dataset = MagicMock()
    mock_dataset.dataset_dir = tmp_path / "dataset"
    split_file = eval_dir / "shared_holdout_split.json"
    split_file.write_text("{}")
    
    engine = SearchEngine(
        bag_name=bag_name,
        dataset=mock_dataset,
        split_file=split_file,
        artifact_mgr=artifact_mgr,
        quick=True,
        force=False
    )
    return engine, artifact_mgr, tmp_path


def test_phase_a_winner_selection_propagates_best_slam(mock_search_env):
    engine, artifact_mgr, tmp_path = mock_search_env
    
    trajectories = {
        "orb_rgbd": str(tmp_path / "orb.txt"),
        "rtab_rgbd": str(tmp_path / "rtab.txt")
    }
    traj_metrics = {
        "orb_rgbd": {"num_frames": 100},
        "rtab_rgbd": {"num_frames": 100}
    }

    def mock_eval(**kwargs):
        cname = kwargs.get("candidate_name", "")
        if "orb_rgbd" in cname:
            return {
                "candidate_name": cname,
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": 5.0, "depth_p95_mm": 10.0, "point_to_mesh_p95_mm": 8.0, "depth_coverage_ratio": 0.98, "within_20mm_ratio": 0.95},
                "mesh": {"num_triangles": 10000},
                "performance": {"runtime_sec": 1.0}
            }
        else:
            return {
                "candidate_name": cname,
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": 25.0, "depth_p95_mm": 50.0, "point_to_mesh_p95_mm": 40.0, "depth_coverage_ratio": 0.85, "within_20mm_ratio": 0.70},
                "mesh": {"num_triangles": 10000},
                "performance": {"runtime_sec": 2.0}
            }

    with patch("auto_mobility.benchmark.search.run_tsdf_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", side_effect=mock_eval):
        
        results, top_slams = engine.run_phase_a(trajectories, traj_metrics)
        
        assert len(top_slams) >= 1
        assert top_slams[0][0] == "orb_rgbd"
        assert top_slams[0][1] == trajectories["orb_rgbd"]


def test_phase_b_fair_common_adapter_evaluation(mock_search_env):
    engine, artifact_mgr, tmp_path = mock_search_env

    top_slams = [("orb_rgbd", str(tmp_path / "orb.txt"))]
    phase_a_results = [
        {
            "candidate_name": "orb_rgbd_tsdf10mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.95, "depth_p95_mm": 20.0, "point_to_mesh_p95_mm": 15.0, "within_20mm_ratio": 0.90},
            "mesh": {"num_triangles": 30000},
            "performance": {"runtime_sec": 1.5}
        }
    ]

    with patch("auto_mobility.benchmark.search.run_tsdf_worker") as mock_worker, \
         patch("auto_mobility.benchmark.search.run_direct_fusion_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.run_surface_worker") as mock_surf_worker, \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval, \
         patch("auto_mobility.benchmark.search.is_artifact_valid", return_value=True):
        
        mock_worker.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 2.0)
        mock_surf_worker.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        mock_eval.return_value = {
            "candidate_name": "orb_rgbd_tsdf20mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 15.0, "depth_coverage_ratio": 0.90, "depth_p95_mm": 30.0, "point_to_mesh_p95_mm": 25.0, "within_20mm_ratio": 0.80},
            "mesh": {"num_triangles": 15000},
            "performance": {"runtime_sec": 2.0}
        }

        # In quick mode: 10mm and 20mm TSDF + 10mm Direct
        results, top_pipes = engine.run_phase_b(
            top_slams=top_slams,
            phase_a_results=phase_a_results
        )

        assert len(results) >= 2
        # Poisson surface adapter MUST be invoked for fair surface baseline comparison
        assert mock_surf_worker.call_count >= 1
        for call_args in mock_surf_worker.call_args_list:
            assert call_args.kwargs.get("method") == "poisson"
            assert call_args.kwargs.get("depth") == 8
            assert call_args.kwargs.get("no_simplify") is True


def test_phase_c_reuses_phase_b_tsdf_direct(mock_search_env):
    engine, artifact_mgr, tmp_path = mock_search_env

    best_mesh = tmp_path / "best_tsdf.obj"
    best_mesh.write_text("v 0 0 0\n" * 10)
    best_pcd = tmp_path / "best_tsdf.ply"
    best_pcd.write_text("ply\nformat ascii 1.0\n" * 10)

    top_fusion_pipelines = [{
        "candidate_name": "orb_rgbd_tsdf10mm",
        "overall_status": "PASS",
        "mesh_path": str(best_mesh),
        "direct_tsdf_mesh_path": str(best_mesh),
        "pcd_path": str(best_pcd),
        "fusion_method": "tsdf",
        "voxel_size_m": 0.010,
        "spec": {
            "requested_params": {
                "slam_backend": "orb_rgbd",
                "slam_profile": "normal",
                "replay_rate": 1.0,
                "fusion_method": "tsdf",
                "fusion_params": {"voxel_size_m": 0.010}
            }
        },
        "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.95, "depth_p95_mm": 20.0, "point_to_mesh_p95_mm": 15.0, "within_20mm_ratio": 0.90},
        "mesh": {"num_triangles": 30000},
        "performance": {"runtime_sec": 1.5}
    }]

    with patch("auto_mobility.benchmark.search.run_surface_worker") as mock_surf_worker, \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval, \
         patch("auto_mobility.benchmark.search.is_artifact_valid", return_value=True):

        mock_surf_worker.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        mock_eval.return_value = {
            "candidate_name": "orb_rgbd_tsdf10mm_poisson",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 11.0, "depth_coverage_ratio": 0.94, "depth_p95_mm": 22.0, "point_to_mesh_p95_mm": 16.0, "within_20mm_ratio": 0.88},
            "mesh": {"num_triangles": 25000},
            "performance": {"runtime_sec": 1.0}
        }

        results, finalists = engine.run_phase_c(
            top_fusion_pipelines=top_fusion_pipelines,
            trajectories={"orb_rgbd": str(tmp_path / "orb.txt")}
        )

        assert len(results) >= 1
        assert len(finalists) >= 1


def test_full_rebuild_stride_one(mock_search_env):
    """Verifies that Phase D Full Rebuild executes with stride=1 (all train frames)."""
    engine, artifact_mgr, tmp_path = mock_search_env

    finalists = [{
        "candidate_name": "orb_rgbd_tsdf10mm_poisson",
        "overall_status": "PASS",
        "fusion_method": "tsdf",
        "surface_method": "poisson",
        "voxel_size_m": 0.010,
        "geometry": {"depth_mae_mm": 8.0, "depth_coverage_ratio": 0.95},
        "mesh": {"num_triangles": 20000}
    }]

    tsdf_strides = []
    def mock_tsdf(dataset_dir, traj_file, mesh_path, pcd_path, voxel, depth_max, trunc_mult, stride, split_file, quick):
        tsdf_strides.append(stride)
        return WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)

    with patch("auto_mobility.benchmark.search.run_tsdf_worker", side_effect=mock_tsdf), \
         patch("auto_mobility.benchmark.search.run_surface_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval:

        mock_eval.return_value = {
            "candidate_name": "orb_rgbd_tsdf10mm_poisson_fullrebuild",
            "overall_status": "PASS",
            "is_full_rebuild": True,
            "geometry": {"depth_mae_mm": 7.5, "depth_coverage_ratio": 0.96, "within_20mm_ratio": 0.95},
            "mesh": {"num_triangles": 25000}
        }

        ranked, winner = engine.run_full_rebuild(finalists, {"orb_rgbd": str(tmp_path / "orb.txt")})

        assert len(tsdf_strides) == 1
        assert tsdf_strides[0] == 1, "Full Rebuild MUST execute with stride=1"
        assert winner is not None
        assert "fullrebuild" in winner["candidate_name"]
