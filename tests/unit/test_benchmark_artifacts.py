"""
tests/unit/test_benchmark_artifacts.py

Unit tests for artifact caching, cache key computation, invalidation, and reuse.
"""

import tempfile
from pathlib import Path
import pytest

from auto_mobility.benchmark.artifacts import (
    compute_cache_key,
    is_artifact_valid,
    ArtifactManager
)


def test_cache_key_determinism():
    key1 = compute_cache_key(
        stage="phase_a",
        dataset_name="room01",
        upstream_files=None,
        params={"voxel_size": 0.010, "stride": 1},
        version="v1"
    )
    key2 = compute_cache_key(
        stage="phase_a",
        dataset_name="room01",
        upstream_files=None,
        params={"voxel_size": 0.010, "stride": 1},
        version="v1"
    )
    assert key1 == key2
    assert len(key1) == 16


def test_cache_key_invalidation_on_param_change():
    key_10mm = compute_cache_key(
        stage="phase_b",
        dataset_name="room01",
        params={"voxel_size": 0.010}
    )
    key_20mm = compute_cache_key(
        stage="phase_b",
        dataset_name="room01",
        params={"voxel_size": 0.020}
    )
    assert key_10mm != key_20mm


def test_cache_key_invalidation_on_upstream_change(tmp_path):
    f1 = tmp_path / "traj1.txt"
    f1.write_text("1.0 0 0 0 0 0 0 1 1\n")
    
    key1 = compute_cache_key(
        stage="phase_a",
        dataset_name="room01",
        upstream_files=[f1],
        params={"voxel_size": 0.010}
    )

    # Modify file content / size
    f1.write_text("1.0 0 0 0 0 0 0 1 1\n2.0 1 0 0 0 0 0 1 2\n")
    key2 = compute_cache_key(
        stage="phase_a",
        dataset_name="room01",
        upstream_files=[f1],
        params={"voxel_size": 0.010}
    )
    assert key1 != key2


def test_is_artifact_valid(tmp_path):
    empty_file = tmp_path / "empty.obj"
    empty_file.touch()
    assert not is_artifact_valid(empty_file, min_bytes=10)

    valid_file = tmp_path / "valid.obj"
    valid_file.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n" * 10)
    assert is_artifact_valid(valid_file, min_bytes=10)
    assert not is_artifact_valid(tmp_path / "non_existent.obj")


def test_artifact_manager_reuse_logic(tmp_path):
    mgr = ArtifactManager("test_bag", base_eval_dir=tmp_path / "evals")
    
    mesh_p = tmp_path / "mesh.obj"
    pcd_p = tmp_path / "cloud.ply"
    
    # Files don't exist yet -> should not reuse
    assert not mgr.should_reuse_reconstruction(mesh_p, pcd_p, "cand1", force=False)
    
    # Create valid files
    mesh_p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n" * 10)
    pcd_p.write_text("ply\nformat ascii 1.0\nelement vertex 3\nend_header\n0 0 0\n1 0 0\n0 1 0\n" * 10)
    
    assert mgr.should_reuse_reconstruction(mesh_p, pcd_p, "cand1", force=False)
    # If force=True, should not reuse even if files exist
    assert not mgr.should_reuse_reconstruction(mesh_p, pcd_p, "cand1", force=True)
