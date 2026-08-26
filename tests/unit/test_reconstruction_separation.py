"""Regression tests for V2 audit fixes:

[P0] search/delivery separation: holdout never in search set, delivery = ALL
     valid FUSE frames (train + holdout).
[P0] pose refinement must not corrupt an already-correct trajectory, and the
     guard metric must be pose-dependent (collapse detection).
"""

import numpy as np
import pytest

from auto_mobility.reconstruction.data.frame_selector import FrameRole
from auto_mobility.reconstruction.data.split import create_pose_space_split
from auto_mobility.reconstruction.pipeline.standard import (
    compute_search_delivery_sets,
)


def _make_split(n=120):
    rng = np.random.RandomState(0)
    ids = list(range(n))
    positions = np.cumsum(rng.uniform(0.0, 0.05, size=(n, 3)), axis=0)
    rots = np.zeros(n)
    return create_pose_space_split(ids, positions, rots)


def _roles_all_fuse(ids):
    return {i: FrameRole.FUSE for i in ids}


def test_holdout_never_enters_search_set():
    split = _make_split()
    roles = _roles_all_fuse(range(120))
    search, delivery, relaxed = compute_search_delivery_sets(split, roles, range(120))
    assert not relaxed
    assert len(search) > 0 and len(delivery) > 0
    assert set(split.val_ids).isdisjoint(search), "holdout leaked into search"
    assert set(search) == set(split.train_ids)


def test_delivery_includes_all_fuse_frames_including_holdout():
    split = _make_split()
    roles = _roles_all_fuse(range(120))
    search, delivery, _ = compute_search_delivery_sets(split, roles, range(120))
    assert set(delivery) == set(split.train_ids) | set(split.val_ids)
    assert set(search) <= set(delivery)


def test_reject_frames_excluded_from_both_sets():
    split = _make_split()
    roles = {i: (FrameRole.REJECT if i % 10 == 0 else FrameRole.FUSE)
             for i in range(120)}
    search, delivery, _ = compute_search_delivery_sets(split, roles, range(120))
    rejected = {i for i, r in roles.items() if r == FrameRole.REJECT}
    assert rejected.isdisjoint(search)
    assert rejected.isdisjoint(delivery)


def test_relax_to_non_reject_when_too_few_fuse():
    split = _make_split()
    roles = {i: (FrameRole.REJECT if i % 2 else FrameRole.TRACK) for i in range(120)}
    search, delivery, relaxed = compute_search_delivery_sets(
        split, roles, range(120), min_fuse_frames=20)
    assert relaxed
    assert all(roles[i] != FrameRole.REJECT for i in delivery)


# ---------------------------------------------------------------------------
# Pose refinement guards
# ---------------------------------------------------------------------------


def _T(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def test_alignment_residual_is_pose_dependent_and_detects_collapse():
    from auto_mobility.reconstruction.pose.refine_pipeline import alignment_residual_mm

    pytest.importorskip("scipy")
    pytest.importorskip("open3d")
    import open3d as o3d

    # two adjacent keyframes observing the same wall from ~5cm apart;
    # clouds are in camera frames related by the known relative pose.
    T0 = _T(0, 0, 0)
    T1 = _T(0.05, 0, 0)

    pts = np.array([[x, y, 3.0] for x in np.linspace(-1, 1, 30)
                    for y in np.linspace(-1, 1, 30)])
    cloud0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cloud1 = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(pts - np.array([0.05, 0, 0])))

    clouds_cam = {0: cloud0, 1: cloud1}
    order = [0, 1]

    good = alignment_residual_mm(clouds_cam, {0: T0, 1: T1}, order)
    collapsed = alignment_residual_mm(clouds_cam, {0: T0, 1: T0.copy()}, order)
    assert good >= 0
    assert collapsed > good * 5, (
        "guard metric must inflate when trajectory collapses "
        f"(good={good:.1f}mm collapsed={collapsed:.1f}mm)")


def test_refine_keeps_already_correct_trajectory():
    pytest.importorskip("open3d")
    from auto_mobility.reconstruction.pose.refine_pipeline import refine_trajectory

    W, H = 160, 120
    K = np.array([[100.0, 0, 80.0], [0, 100.0, 60.0], [0, 0, 1]])

    def render(T_wc):
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        dirs = np.stack([(us - K[0, 2]) / K[0, 0],
                         (vs - K[1, 2]) / K[1, 1],
                         np.ones_like(us, dtype=np.float64)], axis=-1)
        dw = dirs @ T_wc[:3, :3].T
        t = T_wc[:3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -t[2] / dw[..., 2]
        z = s * dw[..., 2]
        mm = np.where((s > 0.3) & (z > 0.3) & (z < 8.0), z * 1000.0, 0)
        return mm.astype(np.uint16)

    poses, frames = {}, []
    F = type("F", (), {})
    for i in range(12):
        T = _T(i * 0.03, 0, 0)
        poses[i] = T
        f = F()
        f.frame_id = i
        frames.append(f)

    depth_map = {fid: render(T) for fid, T in poses.items()}

    out = refine_trajectory(frames, poses,
                            lambda fid: depth_map[fid], K,
                            width=W, height=H)
    refined = out["pose_by_frame"]
    max_dev = max(float(np.linalg.norm(refined[i][:3, 3] - poses[i][:3, 3]))
                  for i in poses)
    assert max_dev < 0.05, (
        f"refinement corrupted a correct trajectory by {max_dev*100:.1f}cm; "
        "rollback should have returned the original poses")
