"""
tests/unit/test_benchmark_search.py

Unit tests for SearchEngine:
  - Winner propagation from Phase A to downstream Phase B & C
  - Artifact reuse (Phase A 10mm in Phase B, Phase B TSDF direct in Phase C)
  - Upstream PCD reuse in Phase C
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

    # Mock evaluate_reconstruction to return better metrics for orb_rgbd than rtab_rgbd
    def mock_eval(dataset_input, trajectory_input, mesh_input, output_dir, candidate_name, split_json, runtime_sec):
        if "orb_rgbd" in candidate_name:
            return {
                "candidate_name": candidate_name,
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": 5.0, "depth_p95_mm": 10.0, "point_to_mesh_p95_mm": 8.0, "depth_coverage_ratio": 0.98, "within_20mm_ratio": 0.95},
                "mesh": {"num_triangles": 10000},
                "performance": {"runtime_sec": 1.0}
            }
        else:
            return {
                "candidate_name": candidate_name,
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": 25.0, "depth_p95_mm": 50.0, "point_to_mesh_p95_mm": 40.0, "depth_coverage_ratio": 0.85, "within_20mm_ratio": 0.70},
                "mesh": {"num_triangles": 10000},
                "performance": {"runtime_sec": 2.0}
            }

    with patch("auto_mobility.benchmark.search.run_tsdf_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", side_effect=mock_eval):
        
        results, best_slam, best_traj = engine.run_phase_a(trajectories, traj_metrics)
        
        # Must select orb_rgbd as best SLAM because it had better geometry accuracy! (not default to rtab)
        assert best_slam == "orb_rgbd"
        assert best_traj == trajectories["orb_rgbd"]


def test_phase_b_reuses_phase_a_10mm_result(mock_search_env):
    engine, artifact_mgr, tmp_path = mock_search_env

    phase_a_results = [
        {
            "candidate_name": "orb_rgbd_voxel10mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.95, "depth_p95_mm": 20.0, "point_to_mesh_p95_mm": 15.0, "within_20mm_ratio": 0.90},
            "mesh": {"num_triangles": 30000},
            "performance": {"runtime_sec": 1.5}
        }
    ]

    with patch("auto_mobility.benchmark.search.run_tsdf_worker") as mock_worker, \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval:
        
        mock_worker.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 2.0)
        mock_eval.return_value = {
            "candidate_name": "orb_rgbd_voxel20mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 15.0, "depth_coverage_ratio": 0.90, "depth_p95_mm": 30.0, "point_to_mesh_p95_mm": 25.0, "within_20mm_ratio": 0.80},
            "mesh": {"num_triangles": 15000},
            "performance": {"runtime_sec": 2.0}
        }

        # In quick mode: voxels are 10mm and 20mm
        results, best_voxel, best_pcd, best_mesh, best_sum = engine.run_phase_b(
            best_slam="orb_rgbd",
            best_traj=str(tmp_path / "orb.txt"),
            phase_a_results=phase_a_results
        )

        # Worker should only be called once (for 20mm), because 10mm is directly reused from Phase A!
        assert mock_worker.call_count == 1
        assert len(results) == 2


def test_phase_c_reuses_phase_b_tsdf_direct(mock_search_env):
    engine, artifact_mgr, tmp_path = mock_search_env

    best_mesh = tmp_path / "best_tsdf.obj"
    best_mesh.write_text("v 0 0 0\n" * 10)
    best_pcd = tmp_path / "best_tsdf.ply"
    best_pcd.write_text("ply\nformat ascii 1.0\n" * 10)

    best_tsdf_summary = {
        "candidate_name": "orb_rgbd_voxel10mm",
        "overall_status": "PASS",
        "mesh_path": str(best_mesh),
        "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.95, "depth_p95_mm": 20.0, "point_to_mesh_p95_mm": 15.0, "within_20mm_ratio": 0.90},
        "mesh": {"num_triangles": 30000},
        "performance": {"runtime_sec": 1.5}
    }

    with patch("auto_mobility.benchmark.search.run_surface_worker") as mock_surf_worker, \
         patch("auto_mobility.benchmark.search.run_tsdf_worker") as mock_tsdf_worker, \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval:

        mock_surf_worker.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        mock_eval.return_value = {
            "candidate_name": "orb_rgbd_voxel10mm_poisson",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 11.0, "depth_coverage_ratio": 0.94, "depth_p95_mm": 22.0, "point_to_mesh_p95_mm": 16.0, "within_20mm_ratio": 0.88},
            "mesh": {"num_triangles": 25000},
            "performance": {"runtime_sec": 1.0}
        }

        results, winner_c = engine.run_phase_c(
            best_slam="orb_rgbd",
            best_traj=str(tmp_path / "orb.txt"),
            best_voxel_m=0.010,
            best_pcd=best_pcd,
            best_tsdf_mesh=best_mesh,
            best_tsdf_summary=best_tsdf_summary
        )

        # tsdf_direct should NOT call run_tsdf_worker or run_surface_worker because it was reused from Phase B!
        mock_tsdf_worker.assert_not_called()
        # Find tsdf_direct in results
        tsdf_dir_entry = [r for r in results if r.get("surface_method") == "tsdf_direct"]
        assert len(tsdf_dir_entry) == 1
        assert tsdf_dir_entry[0]["candidate_name"] == "orb_rgbd_voxel10mm_tsdf_direct"


def test_adaptive_tsdf_search_skips_5mm_when_gain_is_insignificant(mock_search_env):
    """Validates: when 10mm does not improve significantly over 20mm, 5mm worker is skipped."""
    engine, artifact_mgr, tmp_path = mock_search_env
    engine.quick = False  # Full exploration mode

    # 10mm has almost identical geometry as 20mm (no meaningful quality gain)
    phase_a_results = [
        {
            "candidate_name": "orb_rgbd_voxel10mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 15.0, "depth_coverage_ratio": 0.90, "depth_p95_mm": 30.0, "point_to_mesh_p95_mm": 25.0, "within_20mm_ratio": 0.80},
            "mesh": {"num_triangles": 20000},
            "performance": {"runtime_sec": 1.5}
        }
    ]

    with patch("auto_mobility.benchmark.search.run_tsdf_worker") as mock_worker, \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval:

        mock_worker.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        mock_eval.return_value = {
            "candidate_name": "orb_rgbd_voxel20mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 15.2, "depth_coverage_ratio": 0.90, "depth_p95_mm": 30.2, "point_to_mesh_p95_mm": 25.1, "within_20mm_ratio": 0.80},
            "mesh": {"num_triangles": 18000},
            "performance": {"runtime_sec": 1.0}
        }

        results, best_voxel, best_pcd, best_mesh, best_sum = engine.run_phase_b(
            best_slam="orb_rgbd",
            best_traj=str(tmp_path / "orb.txt"),
            phase_a_results=phase_a_results
        )

        # Worker should only be called once (for 20mm). 5mm worker should NOT be called!
        assert mock_worker.call_count == 1
        voxels_evaluated = [r.get("voxel_size_m") for r in results]
        assert 0.005 not in voxels_evaluated


def test_adaptive_tsdf_search_runs_5mm_when_gain_is_significant(mock_search_env):
    """Validates: when 10mm improves significantly over 20mm, 5mm worker is executed."""
    engine, artifact_mgr, tmp_path = mock_search_env
    engine.quick = False  # Full exploration mode

    # 10mm is much better than 20mm
    phase_a_results = [
        {
            "candidate_name": "orb_rgbd_voxel10mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 6.0, "depth_coverage_ratio": 0.98, "depth_p95_mm": 12.0, "point_to_mesh_p95_mm": 10.0, "within_20mm_ratio": 0.96},
            "mesh": {"num_triangles": 40000},
            "performance": {"runtime_sec": 2.0}
        }
    ]

    with patch("auto_mobility.benchmark.search.run_tsdf_worker") as mock_worker, \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval:

        mock_worker.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        # 20mm returns poor metrics
        mock_eval.side_effect = [
            {
                "candidate_name": "orb_rgbd_voxel20mm",
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": 28.0, "depth_coverage_ratio": 0.80, "depth_p95_mm": 60.0, "point_to_mesh_p95_mm": 50.0, "within_20mm_ratio": 0.65},
                "mesh": {"num_triangles": 10000},
                "performance": {"runtime_sec": 1.0}
            },
            {
                "candidate_name": "orb_rgbd_voxel5mm",
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": 4.0, "depth_coverage_ratio": 0.99, "depth_p95_mm": 8.0, "point_to_mesh_p95_mm": 7.0, "within_20mm_ratio": 0.98},
                "mesh": {"num_triangles": 80000},
                "performance": {"runtime_sec": 5.0}
            }
        ]

        results, best_voxel, best_pcd, best_mesh, best_sum = engine.run_phase_b(
            best_slam="orb_rgbd",
            best_traj=str(tmp_path / "orb.txt"),
            phase_a_results=phase_a_results
        )

        # Worker should be called twice (for 20mm and 5mm)
        assert mock_worker.call_count == 2
        voxels_evaluated = [r.get("voxel_size_m") for r in results]
        assert 0.005 in voxels_evaluated


def test_tiered_surface_search_skips_tier_2_when_tier_1_strong(mock_search_env):
    """Validates: when Tier 1 (TSDF Direct & Poisson) produces high quality, Tier 2 is skipped."""
    engine, artifact_mgr, tmp_path = mock_search_env

    best_mesh = tmp_path / "best_tsdf.obj"
    best_mesh.write_text("v 0 0 0\n" * 10)
    best_pcd = tmp_path / "best_tsdf.ply"
    best_pcd.write_text("ply\nformat ascii 1.0\n" * 10)

    best_tsdf_summary = {
        "candidate_name": "orb_rgbd_voxel10mm",
        "overall_status": "PASS",
        "mesh_path": str(best_mesh),
        "geometry": {"depth_mae_mm": 5.0, "depth_coverage_ratio": 0.98, "depth_p95_mm": 10.0, "point_to_mesh_p95_mm": 8.0, "within_20mm_ratio": 0.96},
        "mesh": {"num_triangles": 30000},
        "performance": {"runtime_sec": 1.5}
    }

    with patch("auto_mobility.benchmark.search.run_surface_worker") as mock_surf_worker, \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval:

        mock_surf_worker.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        mock_eval.return_value = {
            "candidate_name": "orb_rgbd_voxel10mm_poisson",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 5.5, "depth_coverage_ratio": 0.97, "depth_p95_mm": 11.0, "point_to_mesh_p95_mm": 9.0, "within_20mm_ratio": 0.95},
            "mesh": {"num_triangles": 35000},
            "performance": {"runtime_sec": 1.0}
        }

        results, winner_c = engine.run_phase_c(
            best_slam="orb_rgbd",
            best_traj=str(tmp_path / "orb.txt"),
            best_voxel_m=0.010,
            best_pcd=best_pcd,
            best_tsdf_mesh=best_mesh,
            best_tsdf_summary=best_tsdf_summary,
            all_surfaces=False
        )

        # Poisson worker called once; Tier 2 (Alpha/BPA) skipped because Tier 1 quality is high (score > 70)
        assert mock_surf_worker.call_count == 1
        methods = [r.get("surface_method") for r in results]
        assert "alpha_shape" not in methods
        assert "bpa" not in methods


def test_adaptive_tsdf_preflight_blocks_5mm_when_ram_low(mock_search_env):
    """Validates: even if 10mm improves over 20mm, 5mm is skipped if resource preflight fails (low RAM)."""
    engine, artifact_mgr, tmp_path = mock_search_env
    engine.quick = False

    phase_a_results = [
        {
            "candidate_name": "orb_rgbd_voxel10mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 6.0, "depth_coverage_ratio": 0.98, "depth_p95_mm": 12.0, "point_to_mesh_p95_mm": 10.0, "within_20mm_ratio": 0.96},
            "mesh": {"num_triangles": 40000},
            "performance": {"runtime_sec": 2.0}
        }
    ]

    mock_mem = MagicMock()
    mock_mem.available = 1.0 * (1024 ** 3)  # Only 1.0 GB available (< 3.0 GB)

    with patch("auto_mobility.benchmark.search.run_tsdf_worker") as mock_worker, \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval, \
         patch("psutil.virtual_memory", return_value=mock_mem):

        mock_worker.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        mock_eval.return_value = {
            "candidate_name": "orb_rgbd_voxel20mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 28.0, "depth_coverage_ratio": 0.80, "depth_p95_mm": 60.0, "point_to_mesh_p95_mm": 50.0, "within_20mm_ratio": 0.65},
            "mesh": {"num_triangles": 10000},
            "performance": {"runtime_sec": 1.0}
        }

        results, best_voxel, best_pcd, best_mesh, best_sum = engine.run_phase_b(
            best_slam="orb_rgbd",
            best_traj=str(tmp_path / "orb.txt"),
            phase_a_results=phase_a_results
        )

        # 5mm should be skipped due to low memory preflight
        assert mock_worker.call_count == 1
        voxels_evaluated = [r.get("voxel_size_m") for r in results]
        assert 0.005 not in voxels_evaluated


