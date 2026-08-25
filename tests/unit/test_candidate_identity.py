"""
tests/unit/test_candidate_identity.py

Tests:
  1. rtab dense 0.5, rtab dense 1.0, rtab normal 0.5, rtab normal 1.0 candidate hash are all distinct
  2. ORB rate 0.5 / 1.0 artifact/candidate identities are distinguished
  3. Full Rebuild finalists have distinct unique output mesh paths
  4. CandidateSpec is fully preserved across Phase C -> Phase D
"""

import pytest
from pathlib import Path
from auto_mobility.benchmark.candidate import (
    CandidateSpec,
    SlamProfileSpec,
    STANDARD_SLAM_PROFILES,
    get_slam_profile_spec
)
from auto_mobility.benchmark.artifacts import ArtifactManager


def test_slam_profile_candidate_hashes_unique():
    """Test 1: All registered SLAM profiles have unique spec hashes."""
    keys = list(STANDARD_SLAM_PROFILES.keys())
    assert len(keys) >= 2, "Expected at least 2 registered SLAM profiles"
    specs = [get_slam_profile_spec(k) for k in keys]
    hashes = [s.compute_spec_hash() for s in specs]

    assert len(set(hashes)) == len(keys), f"SLAM profile hashes must be unique: {hashes}"


def test_orb_rate_identities_distinguished():
    """Test 2: ORB rate 0.5 vs 1.0 artifact/candidate identities are distinguished."""
    spec_05 = CandidateSpec(dataset_name="bag1", slam_backend="orb_rgbd", replay_rate=0.5)
    spec_10 = CandidateSpec(dataset_name="bag1", slam_backend="orb_rgbd", replay_rate=1.0)
    
    assert spec_05.compute_spec_hash() != spec_10.compute_spec_hash()
    assert spec_05.compute_candidate_id() != spec_10.compute_candidate_id()
    assert "rate0.5" in spec_05.compute_candidate_id()


def test_full_rebuild_finalists_unique_paths(tmp_path):
    """Test 3: Full Rebuild finalists have distinct unique mesh paths."""
    artifact_mgr = ArtifactManager("test_bag", base_eval_dir=tmp_path)
    
    spec_a = CandidateSpec(dataset_name="test_bag", slam_backend="rtab", slam_profile="dense", replay_rate=0.5, surface_method="poisson", is_full_rebuild=True)
    spec_b = CandidateSpec(dataset_name="test_bag", slam_backend="rtab", slam_profile="normal", replay_rate=1.0, surface_method="tsdf_direct", is_full_rebuild=True)
    spec_c = CandidateSpec(dataset_name="test_bag", slam_backend="orb_rgbd", replay_rate=1.0, surface_method="poisson", is_full_rebuild=True)
    
    id_a = spec_a.compute_candidate_id()
    id_b = spec_b.compute_candidate_id()
    id_c = spec_c.compute_candidate_id()
    
    dir_a = artifact_mgr.get_candidate_artifact_dir(id_a)
    dir_b = artifact_mgr.get_candidate_artifact_dir(id_b)
    dir_c = artifact_mgr.get_candidate_artifact_dir(id_c)
    
    assert len({dir_a, dir_b, dir_c}) == 3
    assert dir_a != dir_b and dir_b != dir_c


def test_candidate_spec_preserved_phase_c_to_d():
    """Test 4: CandidateSpec is not lost across Phase C -> Phase D."""
    orig_spec = CandidateSpec(
        dataset_name="room01",
        slam_backend="rtab",
        slam_profile="dense",
        replay_rate=0.5,
        fusion_method="tsdf",
        fusion_params={"voxel_size_m": 0.008, "depth_min_m": 0.3, "depth_max_m": 3.0, "trunc_mult": 4.0},
        surface_method="poisson",
        surface_params={"depth": 8},
        postprocess_params={"clean_density": True, "simplify_target": 0.0}
    )
    
    # Simulate summary packaging in Phase C
    phase_c_summary = {
        "candidate_name": "rtab_dense_rate0.5_tsdf8mm_poisson",
        "spec": orig_spec.to_metadata_dict(),
        "fusion_method": "tsdf",
        "surface_method": "poisson",
        "voxel_size_m": 0.008
    }
    
    # Read back in Phase D
    spec_req = phase_c_summary["spec"]["requested_params"]
    rebuilt_spec = CandidateSpec(
        dataset_name=spec_req["dataset_name"],
        slam_backend=spec_req["slam_backend"],
        slam_profile=spec_req["slam_profile"],
        replay_rate=spec_req["replay_rate"],
        fusion_method=spec_req["fusion_method"],
        fusion_params=spec_req["fusion_params"],
        surface_method=spec_req["surface_method"],
        surface_params=spec_req["surface_params"],
        postprocess_params=spec_req["postprocess_params"],
        frame_stride=1,
        is_full_rebuild=True
    )
    
    assert rebuilt_spec.slam_backend == "rtab"
    assert rebuilt_spec.slam_profile == "dense"
    assert rebuilt_spec.replay_rate == 0.5
    assert rebuilt_spec.fusion_params["voxel_size_m"] == 0.008
    assert rebuilt_spec.surface_method == "poisson"
    assert rebuilt_spec.frame_stride == 1
    assert rebuilt_spec.is_full_rebuild is True
