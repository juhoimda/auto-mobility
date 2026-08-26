"""Time budget management with an untouchable rank_01 final reserve.

    total = rank01_reserve + pose_exploration + geometry_exploration
          + optional_improvement

Exploration can never eat into the rank_01 reserve: spending is capped per
phase and the reserve is only released by the controller when the winner
rebuild actually starts.

Complexity: O(phases). Memory: O(phases).
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

from auto_mobility.reconstruction.config import BudgetConfig

PHASE_POSE = "pose_exploration"
PHASE_GEOMETRY = "geometry_exploration"
PHASE_OPTIONAL = "optional_improvement"

_PHASE_KEYS = {
    PHASE_POSE: "pose_exploration_fraction",
    PHASE_GEOMETRY: "geometry_exploration_fraction",
    PHASE_OPTIONAL: "optional_improvement_fraction",
}


@dataclass(frozen=True)
class PhaseBudget:
    name: str
    allocated_s: float


class OverBudgetError(RuntimeError):
    pass


class BudgetManager:
    def __init__(self, total_seconds: float, cfg: BudgetConfig):
        if total_seconds <= 0:
            raise ValueError("total_seconds must be positive")
        self._cfg = cfg
        self._total_s = float(total_seconds)
        self.reserve_s = self._total_s * cfg.rank01_reserve_fraction
        spendable = self._total_s - self.reserve_s
        frac_sum = sum(getattr(cfg, k) for k in _PHASE_KEYS.values())
        self._allocated = {
            name: spendable * (getattr(cfg, key) / frac_sum) if frac_sum > 0 else 0.0
            for name, key in _PHASE_KEYS.items()
        }
        self._spent = {name: 0.0 for name in _PHASE_KEYS}
        self._lock = threading.Lock()

    @property
    def total_s(self) -> float:
        return self._total_s

    def phase_allocated(self, phase: str) -> float:
        return self._allocated[phase]

    def phase_remaining(self, phase: str) -> float:
        with self._lock:
            return max(0.0, self._allocated[phase] - self._spent[phase])

    def reserve_remaining(self) -> float:
        return self.reserve_s

    def can_afford(self, phase: str, seconds: float) -> bool:
        return seconds <= self.phase_remaining(phase) + 1e-9

    def spend(self, phase: str, seconds: float) -> None:
        """Record spent time; over-allocation raises instead of silently eating reserve."""
        with self._lock:
            projected = self._spent[phase] + max(0.0, seconds)
            if projected > self._allocated[phase] + 1e-6:
                raise OverBudgetError(
                    f"phase {phase} exceeded allocation: "
                    f"{projected:.1f}s > {self._allocated[phase]:.1f}s"
                )
            self._spent[phase] = projected

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "total_s": round(self._total_s, 1),
                "rank01_reserve_s": round(self.reserve_s, 1),
                "phases": {
                    name: {
                        "allocated_s": round(self._allocated[name], 1),
                        "spent_s": round(self._spent[name], 1),
                        "remaining_s": round(
                            max(0.0, self._allocated[name] - self._spent[name]), 1
                        ),
                    }
                    for name in _PHASE_KEYS
                },
            }
