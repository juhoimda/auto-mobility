"""Standard pipeline controller: search/delivery separation invariants (#56/#57).

Search runs on train subset only; FINAL DELIVERY rebuilds with ALL valid FUSE
frames including holdout. The split is recorded for unbiased score provenance
but never filters the final integration set.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FrameSplit:
    train_ids: tuple
    val_ids: tuple

    def search_frames(self) -> tuple:
        return self.train_ids

    def delivery_frames(self) -> tuple:
        return tuple(sorted(set(self.train_ids) | set(self.val_ids)))


@dataclass
class PipelineState:
    dataset_hash: str = ""
    split: FrameSplit = field(default_factory=lambda: FrameSplit((), ()))
    winner_id: str = ""
    decisions: list = field(default_factory=list)

    def record(self, stage: str, decision: str, reason: str, **evidence):
        self.decisions.append(
            {"stage": stage, "decision": decision, "reason": reason, "evidence": evidence}
        )

    def final_integration_set(self) -> tuple:
        return self.split.delivery_frames()

    def to_decision_trace(self) -> dict:
        return {"winner": self.winner_id, "decisions": self.decisions}
