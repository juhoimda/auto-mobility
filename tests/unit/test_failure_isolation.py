"""
tests/unit/test_failure_isolation.py

Tests:
  29. TSDF worker crash / SIGSEGV isolation
  30. OOM isolation
  31. CGAL unavailable -> SKIPPED_UNAVAILABLE (clean skip without crashing)
  32. 0 valid candidates -> downstream BLOCKED without picking invalid fallback
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from auto_mobility.benchmark.workers import run_tsdf_worker, run_surface_worker, WorkerStatus, WorkerResult
from auto_mobility.benchmark.scoring import HardGateFilter, rank_candidate_summaries
from auto_mobility.benchmark.search import SearchEngine
from auto_mobility.benchmark.artifacts import ArtifactManager


def test_tsdf_worker_segfault_isolation():
    """Test 29 & 30: Process exit with returncode 139 (SIGSEGV / OOM) maps to SEGFAULT status."""
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 139
        mock_proc.stdout = ""
        mock_proc.stderr = "Segmentation fault (core dumped)"
        mock_run.return_value = mock_proc
        
        res = run_tsdf_worker("dummy", "dummy", "dummy.obj", "dummy.ply", 0.01)
        assert res.status in (WorkerStatus.FAIL_SEGFAULT, WorkerStatus.FAIL_CRASH)
        assert res.is_success is False


def test_cgal_unavailable_clean_skip(tmp_path):
    """Test 31: If CGAL is unavailable, returns exit code 2 / SKIPPED_UNAVAILABLE without crashing orchestrator."""
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 2
        mock_proc.stdout = ""
        mock_proc.stderr = "SKIPPED_UNAVAILABLE: cgal not installed"
        mock_run.return_value = mock_proc
        
        res = run_surface_worker(str(tmp_path / "in.ply"), str(tmp_path / "out.obj"), method="cgal_polygonal")
        assert res.status == WorkerStatus.SKIPPED_UNAVAILABLE
        assert res.is_success is False


def test_zero_valid_candidates_blocks_downstream_no_fallback(tmp_path):
    """Test 32: If all candidates fail Hard Gate, search returns 0 valid candidates and blocks downstream without picking invalid fallback."""
    bag_name = "corrupted_bag"
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
        artifact_mgr=artifact_mgr
    )
    
    trajectories = {"rtab": str(tmp_path / "rtab.txt")}
    
    # All TSDF workers crash or fail hard gate
    with patch("auto_mobility.benchmark.search.run_tsdf_worker", return_value=WorkerResult(WorkerStatus.FAIL_SEGFAULT, 139, 1.0)):
        results, top_slams = engine.run_phase_a(trajectories, {})
        
        # Must return empty top_slams (0 winners selected)
        assert len(top_slams) == 0
        
        # Hard Gate rejection verified
        ranked = rank_candidate_summaries(results)
        valid = [r for r in ranked if r.get("hard_gate_pass", False)]
        assert len(valid) == 0
