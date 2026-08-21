"""
tests/unit/test_benchmark_scoring.py

Unit tests for Hard Gate filtering and multi-criteria candidate ranking.
"""

import pytest
from auto_mobility.benchmark.scoring import HardGateFilter, rank_candidate_summaries


def test_hard_gate_filters_crashed_and_failed_candidates():
    # Segfault candidate
    cand_crash = {
        "candidate_name": "orb_rgbd_voxel10mm",
        "status": "FAIL_SEGFAULT",
        "error": "SIGSEGV -11"
    }
    is_valid, reason = HardGateFilter.evaluate(cand_crash)
    assert not is_valid
    assert "FAIL_SEGFAULT" in reason

    # Empty geometry candidate
    cand_empty = {
        "candidate_name": "stella_voxel10mm",
        "overall_status": "PASS",
        "geometry": {}
    }
    is_valid, reason = HardGateFilter.evaluate(cand_empty)
    assert not is_valid

    # Zero triangles candidate
    cand_zero_tri = {
        "candidate_name": "test_voxel10mm",
        "overall_status": "PASS",
        "geometry": {"depth_mae_mm": 10.0, "depth_coverage_ratio": 0.9},
        "mesh": {"num_triangles": 0}
    }
    is_valid, reason = HardGateFilter.evaluate(cand_zero_tri)
    assert not is_valid
    assert "0 triangles" in reason


def test_hard_gate_passes_healthy_candidate():
    cand_healthy = {
        "candidate_name": "rtab_rgbd_voxel10mm",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": 12.0,
            "depth_p95_mm": 25.0,
            "point_to_mesh_p95_mm": 20.0,
            "depth_coverage_ratio": 0.95,
            "within_20mm_ratio": 0.90
        },
        "mesh": {
            "num_triangles": 50000,
            "small_component_area_ratio": 0.01,
            "degenerate_triangle_ratio": 0.0001
        },
        "performance": {"runtime_sec": 5.0}
    }
    is_valid, reason = HardGateFilter.evaluate(cand_healthy)
    assert is_valid
    assert reason is None


def test_rank_candidate_summaries_orders_correctly():
    cand1 = {
        "candidate_name": "best_cand",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": 10.0,
            "depth_p95_mm": 20.0,
            "point_to_mesh_p95_mm": 18.0,
            "depth_coverage_ratio": 0.98,
            "within_20mm_ratio": 0.95
        },
        "mesh": {
            "num_triangles": 40000,
            "small_component_area_ratio": 0.01,
            "degenerate_triangle_ratio": 0.0001
        },
        "performance": {"runtime_sec": 3.0}
    }

    cand2 = {
        "candidate_name": "mediocre_cand",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": 35.0,
            "depth_p95_mm": 70.0,
            "point_to_mesh_p95_mm": 60.0,
            "depth_coverage_ratio": 0.80,
            "within_20mm_ratio": 0.65
        },
        "mesh": {
            "num_triangles": 20000,
            "small_component_area_ratio": 0.05,
            "degenerate_triangle_ratio": 0.005
        },
        "performance": {"runtime_sec": 8.0}
    }

    cand_fail = {
        "candidate_name": "crashed_cand",
        "status": "FAIL_SEGFAULT",
        "error": "Process crashed with SIGSEGV"
    }

    ranked = rank_candidate_summaries([cand2, cand_fail, cand1])
    assert len(ranked) == 3
    assert ranked[0]["candidate_name"] == "best_cand"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["hard_gate_pass"] is True

    assert ranked[1]["candidate_name"] == "mediocre_cand"
    assert ranked[1]["rank"] == 2
    assert ranked[1]["hard_gate_pass"] is True

    assert ranked[2]["candidate_name"] == "crashed_cand"
    assert ranked[2]["rank"] == 3
    assert ranked[2]["hard_gate_pass"] is False
    assert ranked[2]["composite_score"] == 0.0


def test_candidate_independent_absolute_score_determinism():
    """Validates: score of candidate A is deterministic and independent of other candidates in the set."""
    from auto_mobility.benchmark.scoring import compute_absolute_scores

    cand_a = {
        "candidate_name": "cand_a",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": 8.0,
            "depth_p95_mm": 15.0,
            "point_to_mesh_p95_mm": 12.0,
            "depth_coverage_ratio": 0.96,
            "within_20mm_ratio": 0.94,
            "observed_surface_completeness": 0.92,
            "free_space_correctness_ratio": 0.99
        },
        "mesh": {
            "num_triangles": 30000,
            "small_component_area_ratio": 0.005,
            "degenerate_triangle_ratio": 0.0001
        },
        "performance": {"runtime_sec": 4.0}
    }

    cand_b = {
        "candidate_name": "cand_b",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": 40.0,
            "depth_p95_mm": 80.0,
            "point_to_mesh_p95_mm": 70.0,
            "depth_coverage_ratio": 0.70,
            "within_20mm_ratio": 0.50,
            "observed_surface_completeness": 0.45,
            "free_space_correctness_ratio": 0.85
        },
        "mesh": {
            "num_triangles": 10000,
            "small_component_area_ratio": 0.05,
            "degenerate_triangle_ratio": 0.002
        },
        "performance": {"runtime_sec": 12.0}
    }

    # Score of A when evaluated alone
    rank_alone = rank_candidate_summaries([cand_a])
    score_a_alone = rank_alone[0]["composite_score"]
    qual_a_alone = rank_alone[0]["quality_score"]

    # Score of A when evaluated with B
    rank_together = rank_candidate_summaries([cand_a, cand_b])
    entry_a = next(r for r in rank_together if r["candidate_name"] == "cand_a")

    assert entry_a["composite_score"] == pytest.approx(score_a_alone)
    assert entry_a["quality_score"] == pytest.approx(qual_a_alone)


def test_quality_primary_and_cost_tiebreaker():
    """Validates: when quality scores are equivalent, lower cost (faster/lighter) candidate wins tie-break."""
    # Both candidates have identical geometry and topology (identical quality score)
    cand_light = {
        "candidate_name": "cand_light_fast",
        "overall_status": "PASS",
        "geometry": {"depth_mae_mm": 10.0, "depth_p95_mm": 20.0, "point_to_mesh_p95_mm": 18.0, "depth_coverage_ratio": 0.95, "within_20mm_ratio": 0.90},
        "mesh": {"num_triangles": 20000, "small_component_area_ratio": 0.01, "degenerate_triangle_ratio": 0.0001},
        "performance": {"runtime_sec": 2.0}  # Fast & light
    }

    cand_heavy = {
        "candidate_name": "cand_heavy_slow",
        "overall_status": "PASS",
        "geometry": {"depth_mae_mm": 10.0, "depth_p95_mm": 20.0, "point_to_mesh_p95_mm": 18.0, "depth_coverage_ratio": 0.95, "within_20mm_ratio": 0.90},
        "mesh": {"num_triangles": 500000, "small_component_area_ratio": 0.01, "degenerate_triangle_ratio": 0.0001},
        "performance": {"runtime_sec": 120.0}  # Slow & heavy
    }

    ranked = rank_candidate_summaries([cand_heavy, cand_light])
    assert ranked[0]["candidate_name"] == "cand_light_fast"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["candidate_name"] == "cand_heavy_slow"
    assert ranked[1]["rank"] == 2


def test_nan_and_none_metric_resilience():
    """Validates: summaries with None/NaN fields compute scores gracefully without exceptions."""
    from auto_mobility.benchmark.scoring import compute_absolute_scores
    import numpy as np

    cand_nan = {
        "candidate_name": "cand_nan",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": float("nan"),
            "depth_p95_mm": None,
            "point_to_mesh_p95_mm": None,
            "depth_coverage_ratio": float("nan"),
            "within_20mm_ratio": None,
            "free_space_correctness_ratio": None
        },
        "mesh": {"num_triangles": None},
        "performance": {"runtime_sec": None}
    }

    scores = compute_absolute_scores(cand_nan)
    assert isinstance(scores["composite_score"], float)
    assert scores["composite_score"] >= 0.0

    # HardGateFilter should catch NaN MAE
    is_valid, reason = HardGateFilter.evaluate(cand_nan)
    assert not is_valid


def test_old_schema_evaluation_dict_compatibility():
    """Validates: legacy summary without free_space or completeness fields computes valid scores."""
    from auto_mobility.benchmark.scoring import compute_absolute_scores

    legacy_summary = {
        "candidate_name": "legacy_v1_candidate",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": 12.5,
            "depth_p95_mm": 25.0,
            "point_to_mesh_p95_mm": 20.0,
            "depth_coverage_ratio": 0.90,
            "within_20mm_ratio": 0.85
            # No free_space_correctness_ratio or observed_surface_completeness
        },
        "mesh": {
            "num_triangles": 35000,
            "small_component_area_ratio": 0.02
        },
        "performance": {"runtime_sec": 4.5}
    }

    scores = compute_absolute_scores(legacy_summary)
    assert scores["quality_score"] > 60.0
    assert scores["composite_score"] > 60.0

    ranked = rank_candidate_summaries([legacy_summary])
    assert len(ranked) == 1
    assert ranked[0]["hard_gate_pass"] is True
    assert ranked[0]["rank"] == 1


def test_all_candidates_failed_returns_no_winner():
    """Validates: when all candidates fail Hard Gate, no winner is selected."""
    failed_a = [{"candidate_name": "a", "status": "FAIL_SEGFAULT", "geometry": {}}]
    failed_b = [{"candidate_name": "b", "status": "FAIL_OOM", "geometry": {}}]
    failed_c = [{"candidate_name": "c", "status": "FAIL_TIMEOUT", "geometry": {}}]

    ranked = rank_candidate_summaries(failed_a + failed_b + failed_c)
    valid = [r for r in ranked if r.get("hard_gate_pass", False)]
    winner = valid[0] if valid else None

    assert winner is None, "When all candidates fail Hard Gate, winner MUST be None"
    assert len(ranked) == 3
    for item in ranked:
        assert item["hard_gate_pass"] is False
        assert item["composite_score"] == 0.0


