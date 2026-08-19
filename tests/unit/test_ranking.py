"""
tests/unit/test_ranking.py

Unit tests for rule-based ranking metric normalization and candidate comparison.
"""

import numpy as np
import pytest
from auto_mobility.evaluation.compare_results import normalize_metrics, rank_candidates


def test_ranking_directionality():
    # Lower is better (e.g. MAE: 10mm vs 50mm)
    mae_values = [10.0, 50.0]
    normed_mae = normalize_metrics(mae_values, lower_is_better=True)
    assert normed_mae[0] == 1.0 # 10mm gets score 1.0 (Best)
    assert normed_mae[1] == 0.0 # 50mm gets score 0.0 (Worst)

    # Higher is better (e.g. Coverage: 0.95 vs 0.60)
    cov_values = [0.60, 0.95]
    normed_cov = normalize_metrics(cov_values, lower_is_better=False)
    assert normed_cov[0] == 0.0 # 0.60 gets score 0.0 (Worst)
    assert normed_cov[1] == 1.0 # 0.95 gets score 1.0 (Best)


def test_candidate_ranking_order():
    cand_a = {
        "candidate_name": "A_good",
        "overall_status": "PASS",
        "geometry": {
            "depth_mae_mm": 15.0,
            "depth_p95_mm": 40.0,
            "point_to_mesh_p95_mm": 35.0,
            "depth_coverage_ratio": 0.92,
            "within_20mm_ratio": 0.85
        },
        "mesh": {
            "small_component_area_ratio": 0.01,
            "degenerate_triangle_ratio": 0.0001
        },
        "performance": {"runtime_sec": 5.0}
    }

    cand_b = {
        "candidate_name": "B_bad",
        "overall_status": "WARN",
        "geometry": {
            "depth_mae_mm": 60.0,
            "depth_p95_mm": 150.0,
            "point_to_mesh_p95_mm": 120.0,
            "depth_coverage_ratio": 0.50,
            "within_20mm_ratio": 0.30
        },
        "mesh": {
            "small_component_area_ratio": 0.15,
            "degenerate_triangle_ratio": 0.02
        },
        "performance": {"runtime_sec": 20.0}
    }

    ranked = rank_candidates([cand_a, cand_b])
    assert len(ranked) == 2
    assert ranked[0]["candidate_name"] == "A_good"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["composite_score"] > ranked[1]["composite_score"]
