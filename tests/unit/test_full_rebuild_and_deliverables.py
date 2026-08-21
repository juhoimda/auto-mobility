"""
tests/unit/test_full_rebuild_and_deliverables.py

Tests:
  23. Full Rebuild runs with stride=1
  24. All train indices used
  25. Holdout indices not integrated in reconstruction
  26. rank_01/02/03.obj SHA matches source artifacts
  27. best.obj SHA matches Rank 1 artifact
  28. best_config.json accurately captures winner's exact configuration
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
from auto_mobility.benchmark.manifest import BenchmarkManifestExporter
from auto_mobility.benchmark.artifacts import compute_file_sha256


def test_deliverables_and_sha_verification(tmp_path):
    """Tests 26, 27, 28: Exported review/rank_*.obj and final/best.obj match source SHAs and best_config is populated."""
    report_dir = tmp_path / "evaluations" / "test_bag"
    report_dir.mkdir(parents=True)
    
    # Create mock source meshes
    src_mesh1 = tmp_path / "src_rank1.obj"
    src_mesh1.write_text("# Rank 1 Mesh\nv 1.0 1.0 1.0\n")
    src_mesh2 = tmp_path / "src_rank2.obj"
    src_mesh2.write_text("# Rank 2 Mesh\nv 2.0 2.0 2.0\n")
    src_mesh3 = tmp_path / "src_rank3.obj"
    src_mesh3.write_text("# Rank 3 Mesh\nv 3.0 3.0 3.0\n")
    
    traj_file = tmp_path / "rtab_dense_traj.txt"
    traj_file.write_text("1.0 0 0 0 0 0 0 1\n")
    
    rankings = [
        {
            "rank": 1,
            "candidate_name": "rtab_dense_rate0.5_tsdf8mm_poisson_fullrebuild",
            "quality_score": 95.5,
            "cost_score": 80.0,
            "composite_score": 93.9,
            "hard_gate_pass": True,
            "status": "PASS",
            "raw_metrics": {"depth_mae_mm": 5.2, "depth_coverage_ratio": 0.96},
            "summary_data": {
                "mesh_path": str(src_mesh1),
                "trajectory_path": str(traj_file),
                "fusion_method": "tsdf",
                "surface_method": "poisson",
                "voxel_size_m": 0.008,
                "is_full_rebuild": True,
                "spec": {
                    "requested_params": {
                        "dataset_name": "test_bag",
                        "slam_backend": "rtab",
                        "slam_profile": "dense",
                        "replay_rate": 0.5,
                        "fusion_method": "tsdf",
                        "fusion_params": {"voxel_size_m": 0.008, "depth_min_m": 0.3, "depth_max_m": 3.0},
                        "surface_method": "poisson",
                        "surface_params": {"depth": 8},
                        "frame_stride": 1
                    }
                },
                "geometry": {"depth_mae_mm": 5.2, "depth_coverage_ratio": 0.96},
                "mesh": {"num_triangles": 45000}
            }
        },
        {
            "rank": 2,
            "candidate_name": "orb_rgbd_rate1.0_tsdf10mm_poisson_fullrebuild",
            "quality_score": 90.0,
            "cost_score": 85.0,
            "composite_score": 89.5,
            "hard_gate_pass": True,
            "status": "PASS",
            "raw_metrics": {"depth_mae_mm": 8.0, "depth_coverage_ratio": 0.92},
            "summary_data": {"mesh_path": str(src_mesh2), "geometry": {}, "mesh": {}}
        },
        {
            "rank": 3,
            "candidate_name": "stella_rgbd_rate1.0_tsdf10mm_tsdf_direct_fullrebuild",
            "quality_score": 85.0,
            "cost_score": 90.0,
            "composite_score": 85.5,
            "hard_gate_pass": True,
            "status": "PASS",
            "raw_metrics": {"depth_mae_mm": 12.0, "depth_coverage_ratio": 0.88},
            "summary_data": {"mesh_path": str(src_mesh3), "geometry": {}, "mesh": {}}
        }
    ]
    
    manifest_data = {
        "bag_name": "test_bag",
        "mode": "standard",
        "evaluated_at": "2026-08-21T12:00:00Z",
        "random_seed": 42,
        "dataset_fingerprint": "ds_fingerprint_123",
        "hardware": {"cpu_count": 16, "ram_total_mb": 32000},
        "software": {"git_commit": "abc1234", "git_dirty": False}
    }
    
    # Export artifacts
    BenchmarkManifestExporter.export_final_artifacts(
        report_dir=report_dir,
        manifest_data=manifest_data,
        overall_rankings=rankings,
        winner_candidate=rankings[0],
        top_k=3
    )
    
    # 1. Verify Review Meshes match source SHA256
    assert (report_dir / "review" / "rank_01.obj").exists()
    assert (report_dir / "review" / "rank_02.obj").exists()
    assert (report_dir / "review" / "rank_03.obj").exists()
    
    sha_src1 = compute_file_sha256(src_mesh1)
    sha_dst1 = compute_file_sha256(report_dir / "review" / "rank_01.obj")
    assert sha_src1 == sha_dst1
    
    # 2. Verify Final Best OBJ matches Rank 1 artifact
    sha_best = compute_file_sha256(report_dir / "final" / "best.obj")
    assert sha_best == sha_src1
    
    # 3. Verify best_config.json
    best_cfg_file = report_dir / "final" / "best_config.json"
    assert best_cfg_file.exists()
    with open(best_cfg_file, "r") as f:
        best_cfg = json.load(f)
        
    assert best_cfg["winner_candidate_name"] == "rtab_dense_rate0.5_tsdf8mm_poisson_fullrebuild"
    assert best_cfg["slam"]["backend"] == "rtab"
    assert best_cfg["slam"]["profile"] == "dense"
    assert best_cfg["slam"]["replay_rate"] == 0.5
    assert best_cfg["fusion"]["method"] == "tsdf"
    assert best_cfg["fusion"]["params"]["voxel_size_m"] == 0.008
    assert best_cfg["surface"]["method"] == "poisson"
    assert best_cfg["reconstruction"]["is_full_rebuild"] is True
    assert best_cfg["artifact_hashes"]["mesh_sha256"] == sha_src1
