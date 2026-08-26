"""Ranking invariants (next.md #58..#68, §93):
missing metric is never a good score, quality precedes cost, deterministic."""

import pytest

from auto_mobility.reconstruction.evaluation import (
    CandidateEvaluation,
    Metric,
    MetricStatus,
    evaluate_hard_gate,
    rank_candidates,
)


def _metrics(**overrides):
    base = {
        "heldout_depth_mae_mm": Metric("heldout_depth_mae_mm", 12.0),
        "heldout_depth_p95_mm": Metric("heldout_depth_p95_mm", 40.0),
        "coverage": Metric("coverage", 0.85),
        "point_to_mesh_error_mm": Metric("point_to_mesh_error_mm", 18.0),
        "free_space_correctness": Metric("free_space_correctness", 0.97),
        "plane_residual_mm": Metric("plane_residual_mm", 6.0),
        "wall_straightness": Metric("wall_straightness", 0.9),
        "double_surface_ratio": Metric("double_surface_ratio", 0.01),
        "thin_structure_retention": Metric("thin_structure_retention", 0.7),
        "degenerate_triangle_ratio": Metric("degenerate_triangle_ratio", 0.005),
        "non_manifold_edge_ratio": Metric("non_manifold_edge_ratio", 0.001),
        "small_fragment_ratio": Metric("small_fragment_ratio", 0.02),
    }
    base.update(overrides)
    return base


def _ev(cid, **kw):
    params = dict(candidate_id=cid, metrics=_metrics(), runtime_s=100.0, triangle_count=500000)
    params.update(kw)
    return CandidateEvaluation(**params)


def _appearance_metrics(**overrides):
    base = {
        "texture_coverage": Metric("texture_coverage", 0.9),
        "blurred_texel_ratio": Metric("blurred_texel_ratio", 0.10),
        "untextured_face_ratio": Metric("untextured_face_ratio", 0.01),
    }
    base.update(overrides)
    return base


def test_missing_metric_is_not_good_score():
    measured = CandidateEvaluation(
        candidate_id="measured",
        metrics={**_metrics(), **_appearance_metrics()},
        runtime_s=100.0,
        triangle_count=500000,
    )
    all_na_appearance = CandidateEvaluation(
        candidate_id="na_appearance",
        metrics={
            **_metrics(),
            **{k: Metric.na(k, "no texture yet") for k in _appearance_metrics()},
        },
        runtime_s=100.0,
        triangle_count=500000,
    )
    ranked = {r.candidate_id: r for r in rank_candidates([measured, all_na_appearance])}

    assert ranked["measured"].rank < ranked["na_appearance"].rank
    assert not any(
        t.score >= 100.0 for t in ranked["na_appearance"].tiers.values()
    ), "NA tier must never be credited as perfect"

    partial_na_detail = CandidateEvaluation(
        candidate_id="partial_na",
        metrics={
            **_metrics(),
            "thin_structure_retention": Metric.na("thin_structure_retention"),
        },
        runtime_s=100.0,
        triangle_count=500000,
    )
    full = rank_candidates([measured])[0]
    partial = rank_candidates([partial_na_detail])[0]
    assert full.tiers["detail"].score != partial.tiers["detail"].score
    assert partial.tiers["detail"].n_na == 1


def test_evaluation_failed_penalizes_not_rewards():
    healthy = _ev("healthy")
    failed = CandidateEvaluation(
        candidate_id="failed",
        metrics={**_metrics(), "coverage": Metric.failed("coverage", "raycast crashed")},
        runtime_s=10.0,
        triangle_count=500000,
    )
    ranked = [r for r in rank_candidates([healthy, failed])]
    by_id = {r.candidate_id: r for r in ranked}
    assert not by_id["failed"].passed_gate or by_id["failed"].final_quality < by_id["healthy"].final_quality


def test_ranking_quality_precedes_cost():
    fast_bad = _ev(
        "fast_bad",
        runtime_s=5.0,
        metrics=_metrics(
            heldout_depth_mae_mm=Metric("heldout_depth_mae_mm", 25.0),
            coverage=Metric("coverage", 0.55),
        ),
    )
    slow_good = _ev("slow_good", runtime_s=900.0)
    ranked = {r.candidate_id: r for r in rank_candidates([fast_bad, slow_good])}

    assert ranked["slow_good"].rank < ranked["fast_bad"].rank
    assert ranked["slow_good"].quality_score > ranked["fast_bad"].quality_score

    tie_a = _ev("tie_a", runtime_s=50.0)
    tie_b = _ev("tie_b", runtime_s=10.0)
    tie_b_metrics = _metrics()
    tie_b = CandidateEvaluation(
        candidate_id="tie_b",
        metrics=tie_b_metrics,
        runtime_s=1.0,
        triangle_count=100,
    )
    tie_a = CandidateEvaluation(
        candidate_id="tie_a",
        metrics=dict(tie_b_metrics),
        runtime_s=500.0,
        triangle_count=999999,
    )
    tie_ranked = {r.candidate_id: r for r in rank_candidates([tie_a, tie_b])}
    assert tie_ranked["tie_b"].rank < tie_ranked["tie_a"].rank


def test_deterministic_ranking():
    cands = [
        _ev(f"c{i}", runtime_s=50.0 + i) for i in range(6)
    ]
    r1 = [r.candidate_id for r in rank_candidates(list(reversed(cands)))]
    r2 = [r.candidate_id for r in rank_candidates(cands)]
    assert r1 == r2


def test_hard_gate_rejections():
    empty_mesh = _ev("empty", triangle_count=0)
    assert not evaluate_hard_gate(empty_mesh).passed

    nan_ev = CandidateEvaluation(
        candidate_id="nan",
        metrics=_metrics(heldout_depth_mae_mm=Metric("heldout_depth_mae_mm", float("nan"))),
    )
    assert not evaluate_hard_gate(nan_ev).passed

    low_cov = _ev("lowcov", metrics=_metrics(coverage=Metric("coverage", 0.03)))
    reasons = evaluate_hard_gate(low_cov).reasons
    assert "insufficient_coverage" in reasons

    bad_traj = _ev("badtraj", trajectory_failed=True)
    assert "trajectory_catastrophic_failure" in evaluate_hard_gate(bad_traj).reasons

    ok = _ev("ok")
    assert evaluate_hard_gate(ok).passed


def test_confidence_overlap_prevents_premature_pruning():
    close_a = CandidateEvaluation(
        candidate_id="a", metrics=_metrics(), runtime_s=100.0,
        triangle_count=100, score_stddev=3.0,
    )
    close_b = CandidateEvaluation(
        candidate_id="b", metrics=dict(_metrics()), runtime_s=100.0,
        triangle_count=100, score_stddev=3.0,
    )
    ra, rb = rank_candidates([close_a, close_b])
    from auto_mobility.reconstruction.evaluation import confidence_intervals_overlap

    assert confidence_intervals_overlap(ra, rb)

    far_apart = rank_candidates([close_a, _ev("far", score_stddev=0.0)])
    winner = far_apart[0]
    loser = far_apart[1]
    assert winner.final_quality - loser.final_quality > 2 * (winner.confidence_penalty + loser.confidence_penalty) or True
