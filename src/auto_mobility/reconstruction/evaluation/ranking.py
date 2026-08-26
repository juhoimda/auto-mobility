"""Hierarchical candidate ranking (next.md #58..#68).

Layers are strictly ordered: HardGate -> Geometry -> Structure -> Detail ->
Appearance -> Cost(tie-break only). Runtime is never mixed into quality.

Missing metric semantics:
    NOT_APPLICABLE    -> weight renormalized over applicable metrics
    EVALUATION_FAILED -> candidate confidence penalized (never a good default)

Complexity: O(C log C) for C candidates. Memory: O(C * M).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional


class MetricStatus(str, Enum):
    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    EVALUATION_FAILED = "evaluation_failed"


@dataclass(frozen=True)
class Metric:
    name: str
    value: Optional[float]
    status: MetricStatus = MetricStatus.OK
    reason: str = ""

    @classmethod
    def na(cls, name: str, why: str = "") -> "Metric":
        return cls(name, None, MetricStatus.NOT_APPLICABLE, why)

    @classmethod
    def failed(cls, name: str, why: str = "") -> "Metric":
        return cls(name, None, MetricStatus.EVALUATION_FAILED, why)

    @property
    def ok(self) -> bool:
        return self.status == MetricStatus.OK and self.value is not None and math.isfinite(self.value)


@dataclass(frozen=True)
class TierWeights:
    geometry: float = 1.0
    structure: float = 0.6
    detail: float = 0.3
    appearance: float = 0.2


GEOMETRY_METRICS = {
    "heldout_depth_mae_mm": ("lower", 30.0),
    "heldout_depth_p95_mm": ("lower", 80.0),
    "coverage": ("higher", 1.0),
    "point_to_mesh_error_mm": ("lower", 40.0),
    "free_space_correctness": ("higher", 1.0),
}
STRUCTURE_METRICS = {
    "plane_residual_mm": ("lower", 25.0),
    "wall_straightness": ("higher", 1.0),
    "double_surface_ratio": ("lower", 0.10),
}
DETAIL_METRICS = {
    "thin_structure_retention": ("higher", 1.0),
    "degenerate_triangle_ratio": ("lower", 0.05),
    "non_manifold_edge_ratio": ("lower", 0.02),
    "small_fragment_ratio": ("lower", 0.10),
}
APPEARANCE_METRICS = {
    "texture_coverage": ("higher", 1.0),
    "blurred_texel_ratio": ("lower", 0.30),
    "untextured_face_ratio": ("lower", 0.05),
}

_TIER_TABLES = {
    "geometry": GEOMETRY_METRICS,
    "structure": STRUCTURE_METRICS,
    "detail": DETAIL_METRICS,
    "appearance": APPEARANCE_METRICS,
}


def score_lower_better(value: float, worst: float) -> float:
    return max(0.0, min(100.0, 100.0 * (1.0 - value / worst))) if worst > 0 else 0.0


def score_higher_better(value: float, best: float) -> float:
    return max(0.0, min(100.0, 100.0 * value / best)) if best > 0 else 0.0


@dataclass(frozen=True)
class TierScore:
    tier: str
    score: float
    n_ok: int
    n_na: int
    n_failed: int

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "score": round(self.score, 2),
            "n_ok": self.n_ok,
            "n_not_applicable": self.n_na,
            "n_evaluation_failed": self.n_failed,
        }


def score_tier(tier: str, metrics: dict) -> TierScore:
    """Weighted [0,100] tier score; NA renormalizes, FAILED poisons the tier."""
    table = _TIER_TABLES[tier]
    total_w = 0.0
    acc = 0.0
    n_ok = n_na = n_failed = 0
    for name, (direction, anchor) in table.items():
        m = metrics.get(name)
        w = anchor if direction == "lower" else 1.0
        base_w = max(w, 1e-6)
        if m is None or m.status == MetricStatus.NOT_APPLICABLE:
            n_na += 1
            continue
        if m.status == MetricStatus.EVALUATION_FAILED or not m.ok:
            n_failed += 1
            continue
        raw = m.value
        s = score_lower_better(raw, anchor) if direction == "lower" else score_higher_better(raw, anchor)
        acc += base_w * s
        total_w += base_w
    if n_failed > 0:
        return TierScore(tier, 0.0, n_ok, n_na, n_failed)
    if total_w <= 0:
        return TierScore(tier, 0.0, n_ok, n_na, n_failed)
    return TierScore(tier, acc / total_w, len([1 for n in table if metrics.get(n) and metrics[n].ok]), n_na, n_failed)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    metrics: dict
    runtime_s: float = 0.0
    triangle_count: int = 0
    trajectory_failed: bool = False
    score_stddev: float = 0.0

    @property
    def has_invalid_values(self) -> bool:
        for m in self.metrics.values():
            if m.status == MetricStatus.OK and (m.value is None or not math.isfinite(m.value)):
                return True
        return False


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list = field(default_factory=list)


def evaluate_hard_gate(ev: CandidateEvaluation) -> GateResult:
    """Immediate disqualification conditions (#59)."""
    reasons = []
    if ev.trajectory_failed:
        reasons.append("trajectory_catastrophic_failure")
    if ev.triangle_count <= 0:
        reasons.append("mesh_empty")
    if ev.has_invalid_values:
        reasons.append("nan_or_invalid_metric")

    def val(name):
        m = ev.metrics.get(name)
        return m.value if m is not None and m.ok else None

    coverage = val("coverage")
    if coverage is not None and coverage < 0.05:
        reasons.append("insufficient_coverage")
    fs = val("free_space_correctness")
    if fs is not None and fs < 0.20:
        reasons.append("extreme_free_space_violation")

    return GateResult(passed=len(reasons) == 0, reasons=reasons)


@dataclass(frozen=True)
class RankingEntry:
    candidate_id: str
    rank: int
    passed_gate: bool
    gate_reasons: list
    tiers: dict
    quality_score: float
    confidence_penalty: float
    final_quality: float
    cost_tiebreak: float

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "passed_gate": self.passed_gate,
            "gate_reasons": self.gate_reasons,
            "tiers": {k: v.to_dict() for k, v in self.tiers.items()},
            "quality_score": round(self.quality_score, 2),
            "final_quality": round(self.final_quality, 2),
            "cost_tiebreak": round(self.cost_tiebreak, 4),
        }


def _cost_tiebreak(ev: CandidateEvaluation, ref_runtime: float, ref_tris: int) -> float:
    rt = ev.runtime_s / ref_runtime if ref_runtime > 0 else 1.0
    tr = ev.triangle_count / ref_tris if ref_tris > 0 else 1.0
    return 0.5 * rt + 0.5 * tr


def _composite_quality(tiers: dict, weights: TierWeights) -> tuple:
    """Weighted tier average, renormalized over tiers that actually have
    measured data. Fully-NOT_APPLICABLE tiers are excluded (never credited,
    never punished); EVALUATION_FAILED tiers poison everything."""
    tier_weights = {
        "geometry": weights.geometry,
        "structure": weights.structure,
        "detail": weights.detail,
        "appearance": weights.appearance,
    }
    failed_any = any(t.n_failed > 0 for t in tiers.values())
    usable = {
        name: tiers[name]
        for name, t in tiers.items()
        if t.n_ok > 0 and name in tier_weights
    }
    if failed_any or not usable:
        return 0.0, failed_any, len(usable)
    total_w = sum(tier_weights[name] for name in usable)
    acc = sum(tier_weights[name] * tiers[name].score for name in usable)
    return acc / total_w, False, len(usable)


def rank_candidates(
    evaluations: list,
    weights: TierWeights = TierWeights(),
    evaluation_failure_penalty: float = 15.0,
    stddev_scale: float = 1.96,
) -> list:
    """Deterministic hierarchical ranking. Cost participates ONLY as tie-break."""
    scored = []
    max_runtime = max((e.runtime_s for e in evaluations), default=0.0)
    max_tris = max((e.triangle_count for e in evaluations), default=0)

    for ev in evaluations:
        gate = evaluate_hard_gate(ev)
        tiers = {}
        if gate.passed:
            for tier in _TIER_TABLES:
                tiers[tier] = score_tier(tier, ev.metrics)

        quality, any_failed, _n_usable = _composite_quality(tiers, weights)
        n_failed = 1 if any_failed else 0
        penalty = evaluation_failure_penalty * n_failed
        penalty += stddev_scale * max(0.0, ev.score_stddev)
        final_quality = quality - penalty if gate.passed else -math.inf

        cost = _cost_tiebreak(ev, max_runtime, max_tris) if gate.passed else math.inf

        scored.append((gate, tiers, quality, penalty, final_quality, cost, ev))

    def sort_key(item):
        gate, tiers, quality, penalty, final_q, cost, ev = item
        return (-final_q, cost, ev.candidate_id)

    ranked = []
    for idx, item in enumerate(sorted(scored, key=sort_key), start=1):
        gate, tiers, quality, penalty, final_q, cost, ev = item
        ranked.append(
            RankingEntry(
                candidate_id=ev.candidate_id,
                rank=idx,
                passed_gate=gate.passed,
                gate_reasons=list(gate.reasons),
                tiers=tiers,
                quality_score=quality,
                confidence_penalty=penalty,
                final_quality=final_q,
                cost_tiebreak=cost,
            )
        )
    return ranked


def confidence_intervals_overlap(entry_a: RankingEntry, entry_b: RankingEntry,
                                 margin: float = 2.0) -> bool:
    """True when the two entries' scores are statistically inseparable (#67/#68):
    do NOT prune the weaker one — escalate to higher-fidelity evaluation."""
    a_lo = entry_a.final_quality - margin * entry_a.confidence_penalty
    b_hi = entry_b.final_quality + margin * entry_b.confidence_penalty
    return a_lo <= b_hi
