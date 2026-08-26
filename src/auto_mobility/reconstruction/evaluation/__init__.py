"""Evaluation layer: gated hierarchical quality assessment."""

from auto_mobility.reconstruction.evaluation.geometry_eval import evaluate_geometry
from auto_mobility.reconstruction.evaluation.ranking import (
    CandidateEvaluation,
    GateResult,
    Metric,
    MetricStatus,
    RankingEntry,
    TierScore,
    TierWeights,
    confidence_intervals_overlap,
    evaluate_hard_gate,
    rank_candidates,
    score_higher_better,
    score_lower_better,
    score_tier,
)

__all__ = [
    "evaluate_geometry",
    "CandidateEvaluation",
    "GateResult",
    "Metric",
    "MetricStatus",
    "RankingEntry",
    "TierScore",
    "TierWeights",
    "confidence_intervals_overlap",
    "evaluate_hard_gate",
    "rank_candidates",
    "score_higher_better",
    "score_lower_better",
    "score_tier",
]
