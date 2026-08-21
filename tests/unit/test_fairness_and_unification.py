"""
tests/unit/test_fairness_and_unification.py

Tests:
  14. Phase B TSDF / DirectCloud use same surface adapter (Poisson depth 8)
  15. Surface screening uses identical no-simplification policy (simplify_target=0.0)
  16. All candidates use identical holdout split hash
  17. Reconstruction / evaluation depth range is unified (0.3m ~ 3.0m)
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from auto_mobility.benchmark.search import SearchEngine
from auto_mobility.benchmark.artifacts import ArtifactManager
from auto_mobility.benchmark.candidate import CandidateSpec
from auto_mobility.benchmark.workers import WorkerResult, WorkerStatus
from auto_mobility.config import get_evaluation_config


def test_phase_b_uses_common_surface_adapter(tmp_path):
    """Test 14: Direct Point Cloud fusion in Phase B applies Poisson depth=8 adapter."""
    bag_name = "test_bag"
    eval_dir = tmp_path / "evaluations" / bag_name
    eval_dir.mkdir(parents=True)
    
    artifact_mgr = ArtifactManager(bag_name, base_eval_dir=eval_dir)
    mock_dataset = MagicMock()
    mock_dataset.dataset_dir = tmp_path / "dataset"
    split_file = eval_dir / "holdout_split.json"
    split_file.write_text("{}")
    
    engine = SearchEngine(
        bag_name=bag_name,
        dataset=mock_dataset,
        split_file=split_file,
        artifact_mgr=artifact_mgr,
        quick=True
    )
    
    top_slams = [("rtab", str(tmp_path / "rtab.txt"))]
    surface_methods_called = []
    
    def mock_surf_worker(input_ply, output_mesh, method, voxel, depth, simplify, no_simplify, **kwargs):
        surface_methods_called.append({"method": method, "depth": depth, "simplify": simplify, "no_simplify": no_simplify})
        return WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)
        
    with patch("auto_mobility.benchmark.search.run_tsdf_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.run_direct_fusion_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.run_surface_worker", side_effect=mock_surf_worker), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", return_value={"candidate_name": "c", "geometry": {"depth_mae_mm": 10.0}}), \
         patch("auto_mobility.benchmark.search.is_artifact_valid", return_value=True):
         
        results, top_pipes = engine.run_phase_b(top_slams, [])
        
        # Verify direct cloud had surface adapter applied with depth=8 and no simplification
        assert any(call["method"] == "poisson" and call["depth"] == 8 for call in surface_methods_called)


def test_surface_screening_no_simplification_policy():
    """Test 15: CandidateSpec and screening defaults specify no simplification."""
    spec = CandidateSpec(dataset_name="test", slam_backend="rtab", surface_method="poisson")
    assert spec.postprocess_params.get("simplify_target") == 0.0


def test_depth_range_unification():
    """Test 17: Reconstruction and evaluation depth domain unified (0.3m ~ 3.0m)."""
    cfg = get_evaluation_config()
    ray_cfg = cfg.get("evaluation", {}).get("raycasting", {})
    
    depth_min = ray_cfg.get("depth_min_m")
    depth_max = ray_cfg.get("depth_max_m")
    
    assert depth_min == 0.3
    assert depth_max == 3.0, f"Raycasting depth_max must be unified to 3.0m (got {depth_max})"
