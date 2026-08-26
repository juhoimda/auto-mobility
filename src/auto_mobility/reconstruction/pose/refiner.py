"""Conservative pose refinement with mandatory before/after validation.

refine -> validate -> ACCEPT / ROLLBACK. Refined trajectory is adopted ONLY
when no guarded metric worsens beyond tolerance (#35).

Complexity: O(metrics). Memory: O(metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_GUARDS = {
    "heldout_residual": 1.05,
    "loop_consistency": 1.05,
    "structural_residual": 1.10,
    "discontinuity": 1.02,
}


@dataclass(frozen=True)
class PoseQualitySnapshot:
    heldout_residual: float = 0.0
    loop_consistency: float = 0.0
    structural_residual: float = 0.0
    discontinuity: float = 0.0

    def as_dict(self) -> dict:
        return {
            "heldout_residual": self.heldout_residual,
            "loop_consistency": self.loop_consistency,
            "structural_residual": self.structural_residual,
            "discontinuity": self.discontinuity,
        }


@dataclass(frozen=True)
class RefinementDecision:
    accepted: bool
    reason: str
    worsening_ratios: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "worsening_ratios": {k: round(v, 4) for k, v in self.worsening_ratios.items()},
        }


def evaluate_refinement(
    before: PoseQualitySnapshot,
    after: PoseQualitySnapshot,
    tolerances: Optional[dict] = None,
) -> RefinementDecision:
    """ACCEPT only if every guarded metric improved or stayed within tolerance."""
    tol = {**DEFAULT_GUARDS, **(tolerances or {})}
    ratios = {}
    for name in DEFAULT_GUARDS:
        b = getattr(before, name)
        a = getattr(after, name)
        if a < 0 or b < 0 or b != b or a != a:
            return RefinementDecision(False, f"invalid metric value for {name}", {})
        ratio = (a / b) if b > 1e-12 else (float("inf") if a > 0 else 1.0)
        ratios[name] = ratio
        if ratio > tol[name]:
            return RefinementDecision(
                False,
                f"{name} worsened beyond tolerance "
                f"(x{ratio:.3f} > x{tol[name]:.3f})",
                ratios,
            )
    return RefinementDecision(True, "all guarded metrics within tolerance", ratios)
