"""Pose layer judging invariants (next.md #28..#33, §93 partial)."""

import numpy as np

from auto_mobility.reconstruction.pose.judge import (
    score_trajectory,
    select_top_trajectories,
)


def _walk_with_return(k=120, gap_at=None, gap_len=20, fps=30.0):
    ts = np.arange(k) / fps
    pos = np.stack([np.linspace(0, 3.0, k), np.zeros(k), np.zeros(k)], axis=1)
    back = np.linspace(3.0, 0.05, k)[::-1]
    pos = np.concatenate([pos, np.stack([back, np.zeros(k), np.zeros(k)], axis=1)])
    ts = np.concatenate([ts, ts + k / fps + 0.1])
    rot = np.full(len(ts), 0.01)
    if gap_at is not None:
        keep = np.ones(len(ts), dtype=bool)
        keep[gap_at : gap_at + gap_len] = False
        ts, pos, rot = ts[keep], pos[keep], rot[keep]
    frame_ts = np.arange(0, len(ts)) / fps
    return frame_ts[: len(ts)], ts, pos, rot


def test_good_roundtrip_trajectory_scores_well():
    fts, ts, pos, rot = _walk_with_return()
    s = score_trajectory("rtab", fts, ts, pos, rot)

    assert s.ok
    assert s.coverage_ratio > 0.95
    assert s.loop_region_residual < 0.30
    assert s.reverse_overlap_ratio > 0.5
    assert s.composite() > 50.0


def test_tracking_gap_is_flagged():
    fts, ts, pos, rot = _walk_with_return(gap_at=40, gap_len=50)
    s = score_trajectory("cuvslam", fts, ts, pos, rot)
    assert "tracking_gap" in s.failures
    assert not s.ok


def test_pose_jump_is_flagged():
    k = 60
    ts = np.arange(k) / 30.0
    pos = np.stack([np.linspace(0, 1.0, k)] * 3, axis=1)
    pos[40:, 0] += 8.0
    rot = np.full(k, 0.001)
    s = score_trajectory("orb", np.arange(k) / 30.0, ts, pos, rot)
    assert "translation_jump" in s.failures


def test_insufficient_poses_fail_fast():
    ts = np.array([0.0])
    s = score_trajectory("x", np.array([0.0]), ts, np.zeros((1, 3)), np.zeros(1))
    assert not s.ok and s.failures == ["insufficient_poses"]


def _straight_line(k=10, length=1.0):
    ts = np.arange(k) / 30.0
    pos = np.stack([np.linspace(0, length, k)] * 3, axis=1)
    return ts, pos, np.full(k, 0.001)


def test_select_top_is_quality_first_not_diversity_preserving():
    g_ts, g_pos, g_rot = _straight_line(10, 1.0)
    m_ts, m_pos, m_rot = _straight_line(10, 0.5)
    good = score_trajectory("good", g_ts, g_ts.copy(), g_pos, g_rot)
    bad = score_trajectory("bad", np.array([0.0]), np.array([0.0]), np.zeros((1, 3)), np.zeros(1))
    mediocre = score_trajectory("mediocre", m_ts, m_ts.copy(), m_pos, m_rot)

    top = select_top_trajectories([good, bad, mediocre], top_k=2)
    assert [t.backend for t in top] == ["mediocre", "good"]
    assert all(t.ok for t in top)
    assert "bad" not in [t.backend for t in top]


def test_deterministic_scoring():
    fts, ts, pos, rot = _walk_with_return()
    a = score_trajectory("a", fts, ts, pos, rot).composite()
    b = score_trajectory("a", fts, ts, pos, rot).composite()
    assert a == b
