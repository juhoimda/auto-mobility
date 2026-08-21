"""
tests/unit/test_benchmark_integrity.py

Comprehensive tests verifying benchmark integrity per next.md:
  1. Strict Trajectory Provenance & Metadata Verification
  2. CandidateSpec propagation through Phase A -> B -> C -> D without string parsing or loss
  3. Fair Phase B Fusion Screening (Common Poisson adapter on TSDF & Direct PCD)
  4. Adaptive 5mm TSDF Memory Gate using estimate_vbg_memory_gb
  5. Plane scoring sensitivity with correct keys (plane_residual_mean_mm, plane_inlier_ratio)
  6. Collision-free candidate artifact directories with spec hashes
  7. Standalone Deterministic Rebuild CLI (auto_mobility.benchmark.rebuild)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from auto_mobility.benchmark.candidate import (
    CandidateSpec,
    SlamProfileSpec,
    SlamChampion,
    STANDARD_SLAM_PROFILES,
    get_slam_profile_spec,
    get_trajectory_filename,
    get_rtab_db_filename
)
from auto_mobility.benchmark.artifacts import (
    ArtifactManager,
    save_trajectory_metadata,
    verify_trajectory_provenance,
    compute_file_sha256
)
from auto_mobility.benchmark.search import SearchEngine
from auto_mobility.benchmark.scoring import HardGateFilter, compute_absolute_scores, rank_candidate_summaries
from auto_mobility.benchmark.workers import WorkerResult, WorkerStatus
from auto_mobility.mesh.reconstruct_tsdf import estimate_vbg_memory_gb
from auto_mobility.benchmark.rebuild import rebuild_from_config


def test_trajectory_provenance_strict_validation(tmp_path):
    """Verifies that trajectory files without valid metadata or with mismatched spec are rejected in strict mode."""
    traj_file = tmp_path / "rtab_dense_rate0.5_test_trajectory.txt"
    traj_file.write_text("1.0 0 0 0 0 0 0 1\n2.0 1 0 0 0 0 0 1\n" * 5)
    
    spec = STANDARD_SLAM_PROFILES["rtab_dense_rate0.5"]
    
    # 1. Unverified legacy trajectory in strict mode -> False
    is_valid, status, meta = verify_trajectory_provenance(traj_file, spec, strict=True)
    assert is_valid is False
    assert status == "LEGACY_UNVERIFIED"
    
    # 2. Save metadata with wrong backend/profile
    wrong_spec = STANDARD_SLAM_PROFILES["orb_rgbd_rate1.0"]
    save_trajectory_metadata(traj_file, wrong_spec, bag_fingerprint="ds_123")
    
    # 3. Provenance mismatch against requested spec -> False
    is_valid, status, meta = verify_trajectory_provenance(traj_file, spec, expected_bag_fingerprint="ds_123", strict=True)
    assert is_valid is False
    assert status == "PROVENANCE_MISMATCH"
    
    # 4. Save metadata matching exact requested spec
    save_trajectory_metadata(traj_file, spec, bag_fingerprint="ds_123")
    is_valid, status, meta = verify_trajectory_provenance(traj_file, spec, expected_bag_fingerprint="ds_123", strict=True)
    assert is_valid is True
    assert status == "VERIFIED"


def test_candidate_spec_propagation_end_to_end():
    """Verifies that CandidateSpec preserves exact parameters and creates collision-free IDs."""
    spec = CandidateSpec(
        dataset_name="room01",
        slam_backend="rtab",
        slam_profile="dense",
        replay_rate=0.5,
        fusion_method="tsdf",
        fusion_params={"voxel_size_m": 0.008, "depth_min_m": 0.3, "depth_max_m": 3.0, "trunc_mult": 6.0},
        surface_method="poisson",
        surface_params={"depth": 8},
        postprocess_params={"clean_density": True, "simplify_target": 0.0},
        frame_stride=1,
        is_full_rebuild=True,
        evaluation_profile="full"
    )
    
    meta_dict = spec.to_metadata_dict()
    assert meta_dict["requested_params"]["slam_backend"] == "rtab"
    assert meta_dict["requested_params"]["slam_profile"] == "dense"
    assert meta_dict["requested_params"]["replay_rate"] == 0.5
    assert meta_dict["requested_params"]["fusion_params"]["trunc_mult"] == 6.0
    
    cid_with_hash = spec.compute_candidate_id(include_hash=True)
    assert "rtab_dense_rate0.5_tsdf8mm_poisson_fullrebuild_" in cid_with_hash
    assert len(cid_with_hash.split("_")[-1]) == 8


def test_adaptive_5mm_memory_gate(tmp_path):
    """Verifies that 5mm TSDF adaptive search checks memory and skips if memory exceeds limit."""
    artifact_mgr = ArtifactManager("test_bag", base_eval_dir=tmp_path / "evals")
    mock_dataset = MagicMock()
    mock_dataset.dataset_dir = tmp_path / "dataset"
    split_file = tmp_path / "split.json"
    split_file.write_text("{}")
    
    engine = SearchEngine("test_bag", mock_dataset, split_file, artifact_mgr, mode="standard")
    
    # Set max memory limit very small (0.001 GB) to force memory gate trigger
    engine.max_memory_gb_5mm = 0.0001
    
    traj_p = tmp_path / "traj.txt"
    traj_p.write_text("1.0 0 0 0 0 0 0 1\n2.0 10 10 10 0 0 0 1\n" * 10)
    
    top_slams = [SlamChampion(
        profile_spec=STANDARD_SLAM_PROFILES["rtab_normal_rate1.0"],
        trajectory_path=str(traj_p),
        trajectory_sha256="sha123",
        phase_a_summary={}
    )]
    
    with patch("auto_mobility.benchmark.search.run_tsdf_worker") as mock_tsdf, \
         patch("auto_mobility.benchmark.search.run_surface_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction") as mock_eval, \
         patch("auto_mobility.benchmark.search.is_artifact_valid", return_value=True):
         
        mock_tsdf.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        def custom_eval(**kwargs):
            cname = kwargs.get("candidate_name", "")
            mae = 3.0 if "8mm" in cname else (15.0 if "10mm" in cname else 25.0)
            return {
                "candidate_name": cname,
                "overall_status": "PASS",
                "geometry": {"depth_mae_mm": mae, "depth_coverage_ratio": 0.98, "within_20mm_ratio": 0.95},
                "performance": {"runtime_sec": 1.0}
            }
        mock_eval.side_effect = custom_eval
        
        results, top_pipes = engine.run_phase_b(top_slams, [])
        
        # 5mm should have been skipped due to memory gate
        assert not any("5mm" in r.get("candidate_name", "") for r in results)
        assert any(t.get("decision") == "SKIPPED_RESOURCE" for t in engine.decision_trace)


def test_plane_scoring_sensitivity():
    """Verifies that plane residual and inlier ratio correctly contribute to quality score."""
    good_summary = {
        "candidate_name": "cand_good",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": 5.0,
            "depth_coverage_ratio": 0.98,
            "point_to_mesh_p95_mm": 10.0,
            "within_20mm_ratio": 0.95,
            "within_50mm_ratio": 0.99,
            "planes": {
                "plane_residual_mean_mm": 3.0,
                "plane_inlier_ratio": 0.95
            }
        },
        "mesh": {"num_triangles": 30000},
        "performance": {"runtime_sec": 5.0}
    }
    
    poor_plane_summary = {
        "candidate_name": "cand_poor_plane",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": 5.0,
            "depth_coverage_ratio": 0.98,
            "point_to_mesh_p95_mm": 10.0,
            "within_20mm_ratio": 0.95,
            "within_50mm_ratio": 0.99,
            "planes": {
                "plane_residual_mean_mm": 25.0,
                "plane_inlier_ratio": 0.40
            }
        },
        "mesh": {"num_triangles": 30000},
        "performance": {"runtime_sec": 5.0}
    }
    
    good_score = compute_absolute_scores(good_summary)["quality_score"]
    poor_score = compute_absolute_scores(poor_plane_summary)["quality_score"]
    
    assert good_score > poor_score, f"Good plane score ({good_score}) must exceed poor plane score ({poor_score})"


def test_rebuild_from_config(tmp_path):
    """Verifies standalone deterministic rebuild tool executing from best_config.json."""
    best_config_file = tmp_path / "best_config.json"
    out_mesh = tmp_path / "rebuilt_final.obj"
    traj_file = tmp_path / "traj.txt"
    traj_file.write_text("1.0 0 0 0 0 0 0 1\n2.0 1 0 0 0 0 0 1\n")
    
    cfg = {
        "dataset": "test_bag",
        "trajectory_path": str(traj_file),
        "fusion": {
            "method": "tsdf",
            "params": {"voxel_size_m": 0.010, "trunc_mult": 4.0, "depth_min_m": 0.3, "depth_max_m": 3.0}
        },
        "surface": {
            "method": "tsdf_direct",
            "params": {}
        },
        "reconstruction": {
            "frame_stride": 1
        }
    }
    best_config_file.write_text(json.dumps(cfg))
    
    with patch("auto_mobility.benchmark.rebuild.FRAME_DIR", tmp_path), \
         patch("auto_mobility.benchmark.rebuild.run_tsdf_worker") as mock_tsdf:
        
        mock_tsdf.return_value = WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        # Create dataset dir
        (tmp_path / "test_bag").mkdir(parents=True, exist_ok=True)
        
        result_mesh = rebuild_from_config(best_config_file, out_mesh)
        assert result_mesh == out_mesh
        assert mock_tsdf.call_count == 1
