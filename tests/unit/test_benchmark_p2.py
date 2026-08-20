"""
test_benchmark_p2.py — Unit Tests for P2 Production Benchmark Capabilities.

Tests:
  - Atomic manifest and report writing
  - Candidate lifecycle status & metadata provenance
  - Explainable winner selection rationale
  - Top-K review artifact generation (review/rank_*.obj)
  - Search decision trace recording
  - Execution mode differentiation (Quick / Standard / Full)
  - Deterministic random seed and reproducibility fingerprint
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from auto_mobility.benchmark.artifacts import atomic_write_json, atomic_write_text
from auto_mobility.benchmark.workers import CandidateLifecycleStatus, WorkerStatus
from auto_mobility.benchmark.scoring import explain_winner_decision, rank_candidate_summaries
from auto_mobility.benchmark.manifest import (
    BenchmarkManifestExporter,
    compute_dataset_fingerprint,
    get_git_dirty
)
from auto_mobility.benchmark.search import SearchEngine
from auto_mobility.benchmark.orchestrator import BenchmarkOrchestrator


def test_atomic_write_json_and_text(tmp_path):
    """Validates: atomic_write_json and atomic_write_text safely write without leaving tmp files."""
    target_json = tmp_path / "test_manifest.json"
    data = {"name": "test_benchmark", "status": "SUCCESS", "score": 95.5}

    atomic_write_json(target_json, data)
    assert target_json.exists()
    assert json.loads(target_json.read_text(encoding="utf-8")) == data

    # Verify no dangling tmp files
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0

    target_txt = tmp_path / "test_report.md"
    content = "# Title\n\nContent here."
    atomic_write_text(target_txt, content)
    assert target_txt.exists()
    assert target_txt.read_text(encoding="utf-8") == content


def test_candidate_lifecycle_status_definitions():
    """Validates: all required lifecycle statuses exist and are distinct."""
    required = [
        "PENDING", "RUNNING", "SUCCESS", "INVALID", "FAIL_EXCEPTION",
        "FAIL_SEGFAULT", "FAIL_OOM", "FAIL_TIMEOUT", "SKIPPED_UNAVAILABLE",
        "PRUNED", "REUSED"
    ]
    for s in required:
        assert hasattr(CandidateLifecycleStatus, s)
        assert getattr(CandidateLifecycleStatus, s) == s


def test_explain_winner_decision_quality_superior():
    """Validates: explainable winner rationale when Rank 1 has superior quality."""
    winner = {
        "candidate_name": "orb_voxel10mm_poisson",
        "quality_score": 88.5,
        "cost_score": 70.0,
        "composite_score": 86.6,
        "raw_metrics": {
            "depth_mae_mm": 8.2,
            "depth_coverage_ratio": 0.96,
            "free_space_correctness_ratio": 0.98,
            "runtime_sec": 4.5
        }
    }
    runner_up = {
        "candidate_name": "orb_voxel20mm_tsdf_direct",
        "quality_score": 75.0,
        "cost_score": 90.0,
        "composite_score": 76.5,
        "raw_metrics": {
            "depth_mae_mm": 18.0,
            "depth_coverage_ratio": 0.88,
            "free_space_correctness_ratio": 0.92,
            "runtime_sec": 1.2
        }
    }

    lines = explain_winner_decision(winner, runner_up)
    explanation_text = "\n".join(lines)
    assert "Winner Decision Rationale" in explanation_text
    assert "8.2 mm" in explanation_text and "18.0 mm" in explanation_text
    assert "+13.5" in explanation_text
    assert "Significant geometric accuracy" in explanation_text or "quality score improvement" in explanation_text


def test_explain_winner_decision_cost_tiebreak():
    """Validates: explainable winner rationale when Rank 1 wins by cost tie-break on close quality."""
    winner_fast = {
        "candidate_name": "orb_voxel10mm_tsdf_direct",
        "quality_score": 85.2,
        "cost_score": 95.0,
        "composite_score": 86.2,
        "raw_metrics": {
            "depth_mae_mm": 9.8,
            "depth_coverage_ratio": 0.95,
            "free_space_correctness_ratio": 0.97,
            "runtime_sec": 1.1
        }
    }
    runner_up_slow = {
        "candidate_name": "orb_voxel10mm_poisson",
        "quality_score": 85.0,  # 0.2 difference (within 0.5 tolerance)
        "cost_score": 60.0,
        "composite_score": 82.5,
        "raw_metrics": {
            "depth_mae_mm": 9.7,
            "depth_coverage_ratio": 0.95,
            "free_space_correctness_ratio": 0.97,
            "runtime_sec": 5.8
        }
    }

    lines = explain_winner_decision(winner_fast, runner_up_slow)
    explanation_text = "\n".join(lines)
    assert "Cost Tie-Break" in explanation_text
    assert "within noise tolerance" in explanation_text


def test_top_k_review_artifacts_export(tmp_path):
    """Validates: export_final_artifacts generates review/rank_01.obj, rank_02.obj, rank_03.obj."""
    report_dir = tmp_path / "report"
    mesh1 = tmp_path / "mesh1.obj"
    mesh1.write_text("v 0 0 0\n")
    mesh2 = tmp_path / "mesh2.obj"
    mesh2.write_text("v 1 1 1\n")
    mesh3 = tmp_path / "mesh3.obj"
    mesh3.write_text("v 2 2 2\n")

    overall_rankings = [
        {
            "rank": 1,
            "candidate_name": "cand1",
            "quality_score": 90.0,
            "cost_score": 80.0,
            "composite_score": 89.0,
            "hard_gate_pass": True,
            "status": "PASS",
            "summary_data": {"mesh_path": str(mesh1), "geometry": {"depth_mae_mm": 5.0}}
        },
        {
            "rank": 2,
            "candidate_name": "cand2",
            "quality_score": 85.0,
            "cost_score": 85.0,
            "composite_score": 85.0,
            "hard_gate_pass": True,
            "status": "PASS",
            "summary_data": {"mesh_path": str(mesh2), "geometry": {"depth_mae_mm": 8.0}}
        },
        {
            "rank": 3,
            "candidate_name": "cand3",
            "quality_score": 80.0,
            "cost_score": 75.0,
            "composite_score": 79.5,
            "hard_gate_pass": True,
            "status": "PASS",
            "summary_data": {"mesh_path": str(mesh3), "geometry": {"depth_mae_mm": 12.0}}
        }
    ]

    manifest_data = {
        "bag_name": "test_dataset",
        "mode": "standard",
        "evaluated_at": "2026-08-20 12:00:00",
        "hardware": {"cpu_count": 8, "gpu_name": "CUDA GPU"},
        "software": {"ros_distro": "humble", "open3d": "0.18.0", "python": "3.10.12"},
        "decision_trace": [
            {"phase": "Phase A", "candidate": "rtab", "decision": "WINNER", "reason": "Best trajectory"}
        ]
    }

    BenchmarkManifestExporter.export_final_artifacts(
        report_dir=report_dir,
        manifest_data=manifest_data,
        overall_rankings=overall_rankings,
        winner_candidate=overall_rankings[0],
        top_k=3
    )

    review_dir = report_dir / "review"
    assert review_dir.exists()
    assert (review_dir / "rank_01.obj").exists()
    assert (review_dir / "rank_02.obj").exists()
    assert (review_dir / "rank_03.obj").exists()

    final_dir = report_dir / "final"
    assert (final_dir / "best.obj").exists()
    assert (final_dir / "best_config.json").exists()
    assert (final_dir / "quality_report.json").exists()

    report_md = report_dir / "benchmark_report.md"
    assert report_md.exists()
    md_text = report_md.read_text(encoding="utf-8")
    assert "Search & Pruning Decision Trace" in md_text
    assert "Visual Inspection & 3D Viewer Commands" in md_text
    assert "rank_01.obj" in md_text


def test_reproducibility_dataset_fingerprint(tmp_path):
    """Validates: compute_dataset_fingerprint produces deterministic 16-character SHA-256."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "frames.csv").write_text("frame_id,timestamp\n0,1000\n1,2000\n")
    (dataset_dir / "camera_info.json").write_text('{"width": 640, "height": 480}')

    fp1 = compute_dataset_fingerprint(dataset_dir)
    fp2 = compute_dataset_fingerprint(dataset_dir)
    assert len(fp1) == 16
    assert fp1 == fp2


def test_execution_modes_quick_standard_full(tmp_path):
    """Validates: SearchEngine correctly interprets and configures quick, standard, and full modes."""
    from auto_mobility.benchmark.artifacts import ArtifactManager

    mgr = ArtifactManager("test_bag", tmp_path)
    engine_quick = SearchEngine("test_bag", MagicMock(), tmp_path / "split.json", mgr, mode="quick")
    assert engine_quick.mode == "quick"
    assert engine_quick.quick is True
    assert engine_quick.full is False

    engine_std = SearchEngine("test_bag", MagicMock(), tmp_path / "split.json", mgr, mode="standard")
    assert engine_std.mode == "standard"
    assert engine_std.quick is False
    assert engine_std.full is False

    engine_full = SearchEngine("test_bag", MagicMock(), tmp_path / "split.json", mgr, mode="full")
    assert engine_full.mode == "full"
    assert engine_full.quick is False
    assert engine_full.full is True


def test_generate_markdown_report_accepts_str_and_path(tmp_path):
    """Validates: generate_markdown_report safely accepts both str and Path instances without crashing."""
    manifest = {
        "bag_name": "test_bag",
        "evaluated_at": "2026-08-20 12:00:00",
        "mode": "standard",
        "phase_a_slam_results": [],
        "phase_b_tsdf_results": [],
        "phase_c_surface_results": []
    }
    report_file_str = str(tmp_path / "report_from_str.md")
    BenchmarkManifestExporter.generate_markdown_report(manifest, [], report_file_str)
    assert Path(report_file_str).exists()

    report_file_path = tmp_path / "report_from_path.md"
    BenchmarkManifestExporter.generate_markdown_report(manifest, [], report_file_path)
    assert report_file_path.exists()
