"""
tests/unit/test_cache_invalidation.py

Tests:
  5. voxel change -> cache miss
  6. trunc change -> cache miss
  7. surface change -> cache miss
  8. trajectory file change -> cache miss
  9. code / cache schema version change -> cache miss
  10. legacy trajectory provenance mismatch -> reuse blocked
"""

import pytest
import json
from pathlib import Path
from auto_mobility.benchmark.candidate import CandidateSpec, SlamProfileSpec
from auto_mobility.benchmark.artifacts import (
    ArtifactManager,
    save_trajectory_metadata,
    verify_trajectory_provenance
)


@pytest.fixture
def cache_env(tmp_path):
    artifact_mgr = ArtifactManager("test_bag", base_eval_dir=tmp_path)
    mesh_p = tmp_path / "mesh.obj"
    mesh_p.write_text("v 0.0 0.0 0.0\n" * 30) # > 100 bytes
    pcd_p = tmp_path / "cloud.ply"
    pcd_p.write_text("ply\nformat ascii 1.0\nelement vertex 3\nend_header\n0 0 0\n1 0 0\n0 1 0\n" * 10)
    meta_p = tmp_path / "artifact.meta.json"
    
    spec = CandidateSpec(
        dataset_name="test_bag",
        slam_backend="rtab",
        fusion_params={"voxel_size_m": 0.010, "trunc_mult": 4.0},
        surface_method="tsdf_direct"
    )
    
    artifact_mgr.save_artifact_metadata(
        meta_path=meta_p,
        candidate_spec=spec,
        dataset_fingerprint="ds_hash_123",
        trajectory_sha256="traj_sha_abc",
        split_hash="split_hash_xyz"
    )
    
    return artifact_mgr, mesh_p, pcd_p, meta_p, spec


def test_voxel_change_cache_miss(cache_env):
    """Test 5: voxel change -> cache miss."""
    mgr, mesh_p, pcd_p, meta_p, base_spec = cache_env
    
    spec_changed = CandidateSpec(
        dataset_name="test_bag",
        slam_backend="rtab",
        fusion_params={"voxel_size_m": 0.008, "trunc_mult": 4.0}, # Changed voxel
        surface_method="tsdf_direct"
    )
    
    # Base spec should match
    assert mgr.should_reuse_reconstruction(mesh_p, pcd_p, candidate_spec=base_spec, dataset_fingerprint="ds_hash_123", trajectory_sha256="traj_sha_abc", split_hash="split_hash_xyz", meta_path=meta_p) is True
    
    # Changed voxel spec should cause cache miss
    assert mgr.should_reuse_reconstruction(mesh_p, pcd_p, candidate_spec=spec_changed, dataset_fingerprint="ds_hash_123", trajectory_sha256="traj_sha_abc", split_hash="split_hash_xyz", meta_path=meta_p) is False


def test_trunc_change_cache_miss(cache_env):
    """Test 6: trunc multiplier change -> cache miss."""
    mgr, mesh_p, pcd_p, meta_p, base_spec = cache_env
    
    spec_changed = CandidateSpec(
        dataset_name="test_bag",
        slam_backend="rtab",
        fusion_params={"voxel_size_m": 0.010, "trunc_mult": 6.0}, # Changed trunc
        surface_method="tsdf_direct"
    )
    
    assert mgr.should_reuse_reconstruction(mesh_p, pcd_p, candidate_spec=spec_changed, dataset_fingerprint="ds_hash_123", trajectory_sha256="traj_sha_abc", split_hash="split_hash_xyz", meta_path=meta_p) is False


def test_surface_change_cache_miss(cache_env):
    """Test 7: surface method change -> cache miss."""
    mgr, mesh_p, pcd_p, meta_p, base_spec = cache_env
    
    spec_changed = CandidateSpec(
        dataset_name="test_bag",
        slam_backend="rtab",
        fusion_params={"voxel_size_m": 0.010, "trunc_mult": 4.0},
        surface_method="poisson" # Changed surface
    )
    
    assert mgr.should_reuse_reconstruction(mesh_p, pcd_p, candidate_spec=spec_changed, dataset_fingerprint="ds_hash_123", trajectory_sha256="traj_sha_abc", split_hash="split_hash_xyz", meta_path=meta_p) is False


def test_trajectory_change_cache_miss(cache_env):
    """Test 8: trajectory file change -> cache miss."""
    mgr, mesh_p, pcd_p, meta_p, base_spec = cache_env
    
    # Different trajectory SHA
    assert mgr.should_reuse_reconstruction(mesh_p, pcd_p, candidate_spec=base_spec, dataset_fingerprint="ds_hash_123", trajectory_sha256="different_sha", split_hash="split_hash_xyz", meta_path=meta_p) is False


def test_schema_version_change_cache_miss(cache_env):
    """Test 9: code / cache schema version change -> cache miss."""
    mgr, mesh_p, pcd_p, meta_p, base_spec = cache_env
    
    spec_changed = CandidateSpec(
        dataset_name="test_bag",
        slam_backend="rtab",
        fusion_params={"voxel_size_m": 0.010, "trunc_mult": 4.0},
        surface_method="tsdf_direct",
        cache_schema_version="v3" # Schema version bump
    )
    
    assert mgr.should_reuse_reconstruction(mesh_p, pcd_p, candidate_spec=spec_changed, dataset_fingerprint="ds_hash_123", trajectory_sha256="traj_sha_abc", split_hash="split_hash_xyz", meta_path=meta_p) is False


def test_legacy_trajectory_provenance_mismatch_blocked(tmp_path):
    """Test 10: legacy trajectory provenance mismatch -> reuse blocked."""
    traj_p = tmp_path / "rtab_legacy_trajectory.txt"
    traj_p.write_text("1.0 0 0 0 0 0 0 1\n2.0 1 0 0 0 0 0 1\n" * 10)
    
    requested_spec = SlamProfileSpec(candidate_key="rtab_dense_rate0.5", backend="rtab", profile="dense", replay_rate=0.5)
    
    # Strict mode on unverified trajectory without metadata -> False
    is_valid, status, meta = verify_trajectory_provenance(traj_p, requested_spec, strict=True)
    assert is_valid is False
    assert status == "LEGACY_UNVERIFIED"
    
    # Now save metadata for normal profile rate 1.0
    save_trajectory_metadata(traj_p, SlamProfileSpec(candidate_key="rtab_normal_rate1.0", backend="rtab", profile="normal", replay_rate=1.0))
    
    # Check against dense rate 0.5 -> Mismatch
    is_valid_2, status_2, meta_2 = verify_trajectory_provenance(traj_p, requested_spec, strict=True)
    assert is_valid_2 is False
    assert status_2 == "PROVENANCE_MISMATCH"
