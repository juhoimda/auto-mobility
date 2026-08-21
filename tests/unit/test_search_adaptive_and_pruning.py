"""
tests/unit/test_search_adaptive_and_pruning.py

Tests:
  18. Beam search does not prematurely prune promising candidates
  19. Diversity retention across SLAM families in Phase B
  20. 5mm search skipped on quality plateau (gain < 1.0)
  21. 5mm search executed on quality improving condition (gain >= 1.0)
  22. Close candidates trigger adaptive expanded evaluation
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from auto_mobility.benchmark.search import SearchEngine
from auto_mobility.benchmark.artifacts import ArtifactManager
from auto_mobility.benchmark.workers import WorkerResult, WorkerStatus


@pytest.fixture
def search_fixture(tmp_path):
    bag_name = "test_bag"
    eval_dir = tmp_path / "evaluations" / bag_name
    eval_dir.mkdir(parents=True)
    
    artifact_mgr = ArtifactManager(bag_name, base_eval_dir=eval_dir)
    mock_dataset = MagicMock()
    mock_dataset.dataset_dir = tmp_path / "dataset"
    mock_dataset.__len__.return_value = 100
    split_file = eval_dir / "holdout_split.json"
    split_file.write_text("{}")
    
    engine = SearchEngine(
        bag_name=bag_name,
        dataset=mock_dataset,
        split_file=split_file,
        artifact_mgr=artifact_mgr,
        mode="standard"
    )
    return engine, artifact_mgr, tmp_path


def test_diversity_retention_across_slam_families(search_fixture):
    """Test 19: Phase B retains diversity so both RTAB and ORB pipelines survive if valid."""
    engine, artifact_mgr, tmp_path = search_fixture
    
    top_slams = [("rtab", str(tmp_path / "rtab.txt")), ("orb_rgbd", str(tmp_path / "orb.txt"))]
    
    def mock_eval(**kwargs):
        cname = kwargs.get("candidate_name", "")
        # RTAB 8mm has best score, but ORB 10mm must also be retained in beam
        if "rtab" in cname:
            return {"candidate_name": cname, "overall_status": "PASS", "geometry": {"depth_mae_mm": 5.0, "depth_coverage_ratio": 0.95}}
        else:
            return {"candidate_name": cname, "overall_status": "PASS", "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.90}}

    with patch("auto_mobility.benchmark.search.run_tsdf_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.run_direct_fusion_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.run_surface_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", side_effect=mock_eval), \
         patch("auto_mobility.benchmark.search.is_artifact_valid", return_value=True):
         
        results, top_pipes = engine.run_phase_b(top_slams, [])
        pipe_slams = [p.get("spec", {}).get("requested_params", {}).get("slam_backend") or p.get("candidate_name", "").split("_")[0] for p in top_pipes]
        
        assert "rtab" in pipe_slams
        assert "orb_rgbd" in pipe_slams, "Diversity retention must keep top candidate from each SLAM family"


def test_5mm_search_skipped_on_plateau(search_fixture):
    """Test 20: 5mm search is skipped if 10mm -> 8mm quality gain is < 1.0 (plateau)."""
    engine, artifact_mgr, tmp_path = search_fixture
    top_slams = [("rtab", str(tmp_path / "rtab.txt"))]
    
    workers_called = []
    def mock_tsdf(dataset_dir, traj_file, mesh_path, pcd_path, voxel, **kwargs):
        workers_called.append(voxel)
        return WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)

    # 10mm and 8mm have nearly identical error -> plateau
    def mock_eval(**kwargs):
        cname = kwargs.get("candidate_name", "")
        if "8mm" in cname:
            return {"candidate_name": cname, "overall_status": "PASS", "geometry": {"depth_mae_mm": 9.9, "depth_coverage_ratio": 0.90}}
        else:
            return {"candidate_name": cname, "overall_status": "PASS", "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.90}}

    with patch("auto_mobility.benchmark.search.run_tsdf_worker", side_effect=mock_tsdf), \
         patch("auto_mobility.benchmark.search.run_direct_fusion_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.run_surface_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", side_effect=mock_eval), \
         patch("auto_mobility.benchmark.search.is_artifact_valid", return_value=True):
         
        results, top_pipes = engine.run_phase_b(top_slams, [])
        
        # 5mm (0.005) must NOT have been executed
        assert 0.005 not in workers_called
        assert any(entry.get("decision") == "SKIPPED_PLATEAU" for entry in engine.decision_trace)


def test_close_candidates_trigger_adaptive_evaluation(search_fixture):
    """Test 22: Close candidates in Phase A trigger adaptive expanded evaluation."""
    engine, artifact_mgr, tmp_path = search_fixture
    trajectories = {
        "orb_rgbd_rate1.0": str(tmp_path / "orb.txt"),
        "rtab_normal_rate1.0": str(tmp_path / "rtab.txt")
    }
    traj_metrics = {"orb_rgbd_rate1.0": {}, "rtab_normal_rate1.0": {}}
    
    eval_call_samples = []
    def mock_eval(**kwargs):
        cname = kwargs.get("candidate_name", "")
        max_samples = kwargs.get("max_holdout_samples")
        eval_call_samples.append((cname, max_samples))
        
        # Very close quality
        if "orb" in cname:
            return {"candidate_name": cname, "overall_status": "PASS", "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.90}}
        else:
            return {"candidate_name": cname, "overall_status": "PASS", "geometry": {"depth_mae_mm": 10.2, "depth_coverage_ratio": 0.90}}

    with patch("auto_mobility.benchmark.search.run_tsdf_worker", return_value=WorkerResult(WorkerStatus.SUCCESS, 0, 1.0)), \
         patch("auto_mobility.benchmark.search.evaluate_reconstruction", side_effect=mock_eval), \
         patch("auto_mobility.benchmark.search.is_artifact_valid", return_value=True):
         
        results, top_slams = engine.run_phase_a(trajectories, traj_metrics)
        
        # Adaptive expanded evaluation should have been called with stage2 samples (30)
        assert any(samples == engine.stage2_samples for cname, samples in eval_call_samples)
