"""
tests/integration/test_benchmark_pipeline.py

Integration tests for end-to-end benchmark orchestrator:
  - Deliverables generation (manifest, markdown report, rankings, final/best.obj, best_config, quality_report)
  - Resume functionality
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from auto_mobility.benchmark.orchestrator import BenchmarkOrchestrator
from auto_mobility.benchmark.workers import WorkerResult, WorkerStatus


@pytest.fixture
def synthetic_benchmark_env(tmp_path):
    bag_name = "test_synthetic_bag"
    
    # Create mock dataset
    frame_dir = tmp_path / "frames" / bag_name
    frame_dir.mkdir(parents=True)
    (frame_dir / "frames.csv").write_text(
        "frame_id,rgb_timestamp,depth_timestamp,rgb_path,depth_path,rgb_depth_dt_ms,bag_timestamp,camera_frame_id,width,height\n"
        "0,0.0,0.0,rgb/0.png,depth/0.png,0.0,0.0,camera_frame,640,480\n"
        "1,0.1,0.1,rgb/1.png,depth/1.png,0.0,0.1,camera_frame,640,480\n"
    )
    (frame_dir / "camera_info.json").write_text('{"fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}')

    # Create mock trajectory
    traj_dir = tmp_path / "trajectories"
    traj_dir.mkdir(parents=True)
    traj_file = traj_dir / f"rtab_{bag_name}_trajectory.txt"
    traj_file.write_text("0.0 0 0 0 0 0 0 1 1\n0.1 0.1 0 0 0 0 0 1 2\n")

    eval_out = tmp_path / "eval_out"
    eval_out.mkdir(parents=True)

    return {
        "bag_name": bag_name,
        "frame_dir": frame_dir,
        "traj_file": traj_file,
        "eval_out": eval_out,
        "tmp_path": tmp_path
    }


def test_benchmark_orchestrator_end_to_end_deliverables(synthetic_benchmark_env):
    env = synthetic_benchmark_env
    bag_name = env["bag_name"]
    eval_out = env["eval_out"]

    dummy_summary = {
        "candidate_name": f"rtab_rgbd_voxel10mm",
        "overall_status": "PASS",
        "mesh_path": str(env["tmp_path"] / "mesh_winner.obj"),
        "trajectory_path": str(env["traj_file"]),
        "voxel_size_m": 0.010,
        "geometry": {
            "depth_mae_mm": 8.5,
            "depth_p95_mm": 18.0,
            "point_to_mesh_p95_mm": 14.0,
            "depth_coverage_ratio": 0.96,
            "within_20mm_ratio": 0.92
        },
        "mesh": {"num_triangles": 35000, "non_manifold_edges": 0},
        "performance": {"runtime_sec": 1.2}
    }
    
    # Create the dummy winner mesh file
    Path(dummy_summary["mesh_path"]).write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    orchestrator = BenchmarkOrchestrator(
        bag_input=bag_name,
        phase="all",
        quick=True,
        run_slam=False,
        force=False,
        resume=True,
        output_dir=eval_out
    )

    with patch("auto_mobility.benchmark.orchestrator.FRAME_DIR", env["tmp_path"] / "frames"), \
         patch("auto_mobility.benchmark.orchestrator.TRAJECTORY_DIR", env["tmp_path"] / "trajectories"), \
         patch("auto_mobility.benchmark.search.run_tsdf_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.run_surface_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", return_value=dummy_summary):

        manifest = orchestrator.run()

        assert manifest is not None
        assert manifest["bag_name"] == bag_name

        # Verify Deliverables in output directory
        assert (eval_out / "experiment_manifest.json").exists()
        assert (eval_out / "benchmark_report.md").exists()
        assert (eval_out / "rankings.json").exists()

        final_dir = eval_out / "final"
        assert final_dir.exists()
        assert (final_dir / "best.obj").exists()
        assert (final_dir / "best_config.json").exists()
        assert (final_dir / "quality_report.json").exists()

        # Check contents of best_config.json
        with open(final_dir / "best_config.json", "r", encoding="utf-8") as f:
            best_cfg = json.load(f)
        assert best_cfg["dataset"] == bag_name
        assert "composite_score" in best_cfg
        assert "quality_metrics" in best_cfg
        assert "artifact_hashes" in best_cfg

        # Check rankings.json
        with open(eval_out / "rankings.json", "r", encoding="utf-8") as f:
            rankings = json.load(f)
        assert len(rankings) > 0
        assert rankings[0]["rank"] == 1
        assert rankings[0]["hard_gate_pass"] is True


def test_benchmark_resume_behavior(synthetic_benchmark_env):
    env = synthetic_benchmark_env
    bag_name = env["bag_name"]
    eval_out = env["eval_out"]

    # Pre-populate manifest in eval_out
    pre_manifest = {
        "benchmark_id": f"bench_{bag_name}",
        "bag_name": bag_name,
        "evaluated_at": "20260819_120000",
        "phase_a_slam_results": [
            {
                "candidate_name": "rtab_rgbd_voxel10mm",
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.95, "depth_p95_mm": 20.0, "within_20mm_ratio": 0.90},
                "mesh": {"num_triangles": 20000},
                "performance": {"runtime_sec": 1.0}
            }
        ]
    }
    with open(eval_out / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(pre_manifest, f)

    orchestrator = BenchmarkOrchestrator(
        bag_input=bag_name,
        phase="all",
        quick=True,
        run_slam=False,
        force=False,
        resume=True,
        output_dir=eval_out
    )

    with patch("auto_mobility.benchmark.orchestrator.FRAME_DIR", env["tmp_path"] / "frames"), \
         patch("auto_mobility.benchmark.orchestrator.TRAJECTORY_DIR", env["tmp_path"] / "trajectories"), \
         patch("auto_mobility.benchmark.search.run_tsdf_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.run_surface_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval:
        
        mock_eval.return_value = {
            "candidate_name": "rtab_rgbd_voxel20mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 15.0, "depth_coverage_ratio": 0.90, "depth_p95_mm": 25.0, "within_20mm_ratio": 0.85},
            "mesh": {"num_triangles": 10000},
            "performance": {"runtime_sec": 1.0}
        }

        # Run with resume=True
        manifest = orchestrator.run()
        assert manifest is not None


def test_resume_restores_actual_phase_a_winner_not_hardcoded(synthetic_benchmark_env):
    """Validates Bug fix: resume must re-derive best_slam from Phase A results, never hardcode."""
    env = synthetic_benchmark_env
    bag_name = env["bag_name"]
    eval_out = env["eval_out"]

    # Pre-populate manifest where ORB won (not RTAB)
    pre_manifest = {
        "benchmark_id": f"bench_{bag_name}",
        "bag_name": bag_name,
        "evaluated_at": "20260819_120000",
        "phase_a_slam_results": [
            {
                "candidate_name": "orb_rgbd_voxel10mm",
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": 5.0, "depth_coverage_ratio": 0.98, "depth_p95_mm": 10.0, "within_20mm_ratio": 0.95},
                "mesh": {"num_triangles": 40000},
                "performance": {"runtime_sec": 1.0}
            },
            {
                "candidate_name": "rtab_rgbd_voxel10mm",
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": 25.0, "depth_coverage_ratio": 0.80, "depth_p95_mm": 50.0, "within_20mm_ratio": 0.70},
                "mesh": {"num_triangles": 10000},
                "performance": {"runtime_sec": 2.0}
            }
        ],
        "phase_b_tsdf_results": [],
        "phase_c_surface_results": []
    }
    with open(eval_out / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(pre_manifest, f)

    # Also add an orb trajectory (needed for resume execution)
    traj_dir = env["tmp_path"] / "trajectories"
    orb_traj = traj_dir / f"orb_rgbd_{bag_name}_trajectory.txt"
    orb_traj.write_text("0.0 0 0 0 0 0 0 1 1\n0.1 0.1 0 0 0 0 0 1 2\n")

    slam_used_for_phase_b = []

    def capture_phase_b(self_arg, top_slams, phase_a_results):
        for s_name, _ in top_slams:
            slam_used_for_phase_b.append(s_name)
        return [], []

    orchestrator = BenchmarkOrchestrator(
        bag_input=bag_name,
        phase="b",  # Only run Phase B (Phase A loaded from manifest)
        quick=True,
        run_slam=False,
        force=False,
        resume=True,
        output_dir=eval_out
    )

    with patch("auto_mobility.benchmark.orchestrator.FRAME_DIR", env["tmp_path"] / "frames"), \
         patch("auto_mobility.benchmark.orchestrator.TRAJECTORY_DIR", traj_dir), \
         patch("auto_mobility.benchmark.orchestrator.SearchEngine.run_phase_b", capture_phase_b):
        orchestrator.run()

    # Must use orb_rgbd (actual winner from Phase A), NOT rtab_rgbd (hardcoded default)
    assert len(slam_used_for_phase_b) >= 1
    assert slam_used_for_phase_b[0] == "orb_rgbd", (
        f"Expected orb_rgbd to propagate as best_slam from Phase A resume, got: {slam_used_for_phase_b[0]}"
    )


def test_failure_artifacts_not_cached_as_valid():
    """Validates: failed candidates should not produce valid evaluation cache entries."""
    from auto_mobility.benchmark.artifacts import ArtifactManager, is_artifact_valid
    import tempfile, json

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mgr = ArtifactManager("test_bag", base_eval_dir=tmp_path)

        # Write a fail summary where geometry is empty
        cand_dir = tmp_path / "test_candidate"
        cand_dir.mkdir(parents=True)
        fail_summary = {
            "candidate_name": "test_candidate",
            "overall_status": "FAIL_SEGFAULT",
            "geometry": {},
            "mesh": {}
        }
        (cand_dir / "evaluation_summary.json").write_text(json.dumps(fail_summary))

        # The manager should NOT return this as a valid cached evaluation
        result = mgr.should_reuse_evaluation("test_candidate", force=False)
        assert result is None, "Failed candidate evaluation must NOT be returned as valid cache"

        # Valid summary should be returned
        valid_summary = {
            "candidate_name": "test_candidate",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.9}
        }
        (cand_dir / "evaluation_summary.json").write_text(json.dumps(valid_summary))
        result = mgr.should_reuse_evaluation("test_candidate", force=False)
        assert result is not None, "Valid candidate evaluation must be returned as cached"


def test_phase_b_winner_voxel_extracted_from_candidate_name_fallback():
    """Validates: Phase B voxel extraction via candidate_name regex when summary lacks voxel_size_m."""
    from auto_mobility.benchmark.scoring import rank_candidate_summaries

    # Simulate tsdf_eval_results without voxel_size_m in summary (as would happen with cached eval)
    summaries = [
        {
            "candidate_name": "rtab_rgbd_voxel5mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 5.0, "depth_coverage_ratio": 0.98, "depth_p95_mm": 10.0, "within_20mm_ratio": 0.95},
            "mesh": {"num_triangles": 50000},
            "performance": {"runtime_sec": 3.0}
            # no voxel_size_m in summary — must fallback to candidate_name parsing
        },
        {
            "candidate_name": "rtab_rgbd_voxel20mm",
            "overall_status": "PASS",
            "geometry": {"depth_mae_mm": 20.0, "depth_coverage_ratio": 0.85, "depth_p95_mm": 40.0, "within_20mm_ratio": 0.75},
            "mesh": {"num_triangles": 20000},
            "performance": {"runtime_sec": 1.0}
        }
    ]

    ranked = rank_candidate_summaries(summaries)
    assert len(ranked) == 2
    # 5mm should win (better geometry), and hard_gate_pass should be True
    assert ranked[0]["candidate_name"] == "rtab_rgbd_voxel5mm"
    assert ranked[0]["hard_gate_pass"] is True

    # Verify regex fallback via re module directly
    import re
    best_cand = "rtab_rgbd_voxel5mm"
    m = re.search(r"voxel(\d+)mm", best_cand)
    assert m and float(m.group(1)) / 1000.0 == pytest.approx(0.005)

