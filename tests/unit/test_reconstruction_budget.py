"""Budget invariants: rank_01 reserve is untouchable by exploration (§70-72)."""

import pytest

from auto_mobility.reconstruction.config import BudgetConfig
from auto_mobility.reconstruction.runtime import BudgetManager, OverBudgetError


def test_budget_reserves_rank01_final():
    cfg = BudgetConfig(
        total_minutes=30.0,
        rank01_reserve_fraction=0.30,
        pose_exploration_fraction=0.25,
        geometry_exploration_fraction=0.25,
        optional_improvement_fraction=0.50,
    )
    mgr = BudgetManager(1800.0, cfg)

    assert mgr.reserve_remaining() == pytest.approx(540.0)
    phases = mgr.to_dict()["phases"]
    allocated_sum = sum(p["allocated_s"] for p in phases.values())
    assert allocated_sum == pytest.approx(1260.0)
    assert allocated_sum + mgr.reserve_remaining() == pytest.approx(1800.0)


def test_spend_and_overbudget_isolation():
    mgr = BudgetManager(100.0, BudgetConfig())
    pose_alloc = mgr.phase_allocated("pose_exploration")
    assert pose_alloc > 0

    mgr.spend("pose_exploration", pose_alloc * 0.7)
    assert mgr.can_afford("pose_exploration", pose_alloc * 0.4) is False
    assert mgr.can_afford("pose_exploration", pose_alloc * 0.25) is True

    with pytest.raises(OverBudgetError):
        mgr.spend("pose_exploration", pose_alloc * 10)
    assert mgr.reserve_remaining() > 0


def test_optional_phase_cannot_eat_reserve():
    cfg = BudgetConfig(rank01_reserve_fraction=0.5, optional_improvement_fraction=1.0)
    mgr = BudgetManager(100.0, cfg)
    alloc = mgr.phase_allocated("optional_improvement")

    mgr.spend("optional_improvement", alloc)
    with pytest.raises(OverBudgetError):
        mgr.spend("optional_improvement", 0.001)
    assert mgr.reserve_remaining() == pytest.approx(50.0)
