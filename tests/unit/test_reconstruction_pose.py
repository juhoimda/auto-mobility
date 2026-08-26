"""PoseRefiner rollback invariants (next.md #34..#36, §93)."""

from auto_mobility.reconstruction.pose import (
    PoseQualitySnapshot,
    evaluate_refinement,
)


def test_pose_refinement_can_rollback():
    before = PoseQualitySnapshot(
        heldout_residual=8.0,
        loop_consistency=5.0,
        structural_residual=4.0,
        discontinuity=0.01,
    )
    worse = PoseQualitySnapshot(
        heldout_residual=9.5,
        loop_consistency=5.2,
        structural_residual=4.1,
        discontinuity=0.0102,
    )

    decision = evaluate_refinement(before, worse)
    assert not decision.accepted
    assert "heldout_residual" in decision.reason
    assert decision.to_dict()["accepted"] is False


def test_bad_refinement_is_rejected_on_any_guard():
    before = PoseQualitySnapshot(6.0, 3.0, 2.0, 0.008)

    slightly_worse_loop = evaluate_refinement(
        before,
        PoseQualitySnapshot(5.5, 3.3, 1.9, 0.007),
    )
    assert not slightly_worse_loop.accepted

    slightly_worse_disc = evaluate_refinement(
        before,
        PoseQualitySnapshot(5.5, 2.9, 1.9, 0.010),
    )
    assert not slightly_worse_disc.accepted


def test_good_refinement_is_accepted():
    before = PoseQualitySnapshot(8.0, 5.0, 4.0, 0.010)
    better = PoseQualitySnapshot(6.5, 4.0, 3.5, 0.009)
    d = evaluate_refinement(before, better)
    assert d.accepted
    assert d.worsening_ratios["heldout_residual"] == round(6.5 / 8.0, 4)


def test_zero_baseline_and_invalid_inputs_rejected():
    zero_base = evaluate_refinement(
        PoseQualitySnapshot(0.0, 1.0, 1.0, 1.0),
        PoseQualitySnapshot(2.0, 1.0, 1.0, 1.0),
    )
    assert not zero_base.accepted

    nan_case = evaluate_refinement(
        PoseQualitySnapshot(float("nan"), 1.0, 1.0, 1.0),
        PoseQualitySnapshot(1.0, 1.0, 1.0, 1.0),
    )
    assert not nan_case.accepted
