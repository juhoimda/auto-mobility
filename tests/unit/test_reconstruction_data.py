"""Phase 2 data layer invariants: split distribution, TRACK/FUSE/REJECT,
streaming audit (next.md #21..#24, #54, #55, §93)."""

import cv2
import numpy as np
import pytest

from auto_mobility.reconstruction.data import (
    FrameQuality,
    FrameRole,
    FrameSelector,
    audit_dataset,
    create_pose_space_split,
    split_from_poses,
)
from auto_mobility.reconstruction.model import FrameMeta


def _synthetic_frames(n=40):
    return [
        FrameMeta(
            frame_id=i,
            rgb_timestamp=float(i) / 30.0,
            depth_timestamp=(float(i) / 30.0) + 0.004,
            rgb_path=f"/data/rgb/{i:06d}.png",
            depth_path=f"/data/depth/{i:06d}.png",
            rgb_depth_dt_ms=4.0,
        )
        for i in range(n)
    ]


def _corridor_like_poses(n=40):
    poses = []
    for i in range(n):
        p = np.eye(4)
        p[:3, 3] = [0.05 * i, 0.01 * np.sin(i / 5.0), 0]
        yaw = 0.02 * np.sin(i)
        if n - 10 <= i < n - 5:
            yaw = 0.6
        elif i >= n - 5:
            yaw = np.pi
        c, s = np.cos(yaw), np.sin(yaw)
        p[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        poses.append(p)
    return poses


class TestPoseSpaceSplit:
    def test_holdout_is_contiguous_blocks(self):
        fids = list(range(60))
        split = split_from_poses(fids, _corridor_like_poses(60), val_ratio=0.15, seed=42)

        assert len(split.val_ids) > 0 and len(split.train_ids) > 0
        assert set(split.train_ids).isdisjoint(set(split.val_ids))
        assert len(split.train_ids) + len(split.val_ids) == 60

        val_sorted = sorted(split.val_ids)
        blocks = 1
        for prev, cur in zip(val_sorted, val_sorted[1:]):
            if cur != prev + 1:
                blocks += 1
        assert blocks <= 12

    def test_covers_start_middle_end_regions(self):
        fids = list(range(80))
        poses = _corridor_like_poses(80)
        split = create_pose_space_split(
            fids,
            np.array([p[:3, 3] for p in poses]),
            np.zeros(80),
            val_ratio=0.15,
            target_segments=8,
            seed=42,
        )
        val = sorted(split.val_ids)
        first_seg_len = 80 // 8
        assert max(val) > 80 - 2 * first_seg_len
        assert len(val) >= 4

    def test_turn_region_preferred_for_validation(self):
        n = 80
        fids = list(range(n))
        poses = _corridor_like_poses(n)
        rot = np.zeros(n)
        rot[n - 10 : n] = 1.5
        split = create_pose_space_split(
            fids,
            np.array([p[:3, 3] for p in poses]),
            rot,
            val_ratio=0.15,
            target_segments=4,
            seed=42,
        )
        last_seg_lo = n * 3 // 4
        assert max(split.val_ids) >= last_seg_lo

    def test_deterministic_under_seed(self):
        def run(seed):
            return create_pose_space_split(
                list(range(50)),
                np.random.RandomState(0).rand(50, 3),
                np.abs(np.random.RandomState(1).randn(50)),
                seed=seed,
            )

        assert run(7) == run(7)

    def test_rejects_degenerate_inputs(self):
        with pytest.raises(ValueError):
            create_pose_space_split([1, 2], [[0, 0, 0]], [0.0])


class TestFrameSelector:
    def _mk_arrays(self, sharp=True, valid_depth=True):
        rng = np.random.RandomState(3 if sharp else 4)
        base = rng.randint(0, 255, size=(240, 320), dtype=np.uint8)
        gray = base if sharp else cv2.GaussianBlur(base, (31, 31), 12.0)
        bgr = np.stack([gray] * 3, axis=-1)
        rows = np.arange(240, dtype=np.float32).reshape(-1, 1)
        depth_plane = (1500.0 + 0.5 * rows).astype(np.uint16)
        depth = (
            depth_plane
            if valid_depth
            else np.zeros((240, 320), dtype=np.uint16)
        )
        return bgr, depth

    def _quality(self, fid, bgr, depth):
        sel = FrameSelector()
        q = sel.quality_from_arrays(type("F", (), {"frame_id": fid})(), bgr, depth)
        return FrameQuality(
            frame_id=fid,
            sharpness=q.sharpness,
            brightness=q.brightness,
            clipped_dark_ratio=q.clipped_dark_ratio,
            clipped_bright_ratio=q.clipped_bright_ratio,
            depth_valid_ratio=q.depth_valid_ratio,
            depth_edge_ratio=q.depth_edge_ratio,
        )

    def test_track_fuse_reject_classification(self):
        good_bgr, good_depth = self._mk_arrays(sharp=True, valid_depth=True)
        bad_bgr, bad_depth = self._mk_arrays(sharp=False, valid_depth=False)

        qualities = [self._quality(fid, good_bgr, good_depth) for fid in range(20)]
        qualities += [self._quality(fid, bad_bgr, bad_depth) for fid in range(20, 26)]

        roles = FrameSelector().classify(qualities)
        fuse_ids = {f for f, r in roles.items() if r == FrameRole.FUSE}
        reject_ids = {f for f, r in roles.items() if r == FrameRole.REJECT}

        assert len(fuse_ids) > 0
        assert set(range(20, 26)) <= reject_ids
        assert fuse_ids & set(range(20))

    def test_sync_failure_rejects_frame(self):
        q = self._quality(1, *self._mk_arrays())
        roles = FrameSelector().classify([q], sync_dt_ms_by_frame={1: 200.0})
        assert roles[1] == FrameRole.REJECT


class TestDatasetAudit:
    def test_audit_streaming_pass(self):
        frames = _synthetic_frames(30)
        selector = FrameSelector()

        def load_rgb(frame):
            return np.full((60, 80), 128, dtype=np.uint8)

        def load_depth(frame):
            return np.full((60, 80), 1500, dtype=np.uint16)

        result = audit_dataset(frames, load_rgb, load_depth, probe_count=8, selector=selector)
        d = result.to_dict()
        assert result.ok
        assert d["monotonic_rgb"] and d["monotonic_depth"]
        assert d["sync_failures"] == 0
        assert d["n_frames"] == 30

    def test_audit_flags_corruption(self):
        frames = _synthetic_frames(30)
        broken = {frames[0].frame_id, frames[7].frame_id}

        def load_rgb(frame):
            if frame.frame_id in broken:
                raise OSError("truncated png")
            return np.full((60, 80), 128, dtype=np.uint8)

        result = audit_dataset(
            frames, load_rgb, lambda f: np.full((60, 80), 1500, dtype=np.uint16), probe_count=30
        )
        codes = {i.code for i in result.issues}
        assert "CORRUPT_FRAMES" in codes
        assert sorted(result.corrupt_frames) == sorted(broken)

    def test_audit_flags_sparse_depth(self):
        frames = _synthetic_frames(10)
        result = audit_dataset(
            frames,
            lambda f: np.full((60, 80), 128, dtype=np.uint8),
            lambda f: np.zeros((60, 80), dtype=np.uint16),
            probe_count=10,
        )
        codes = {i.code for i in result.issues}
        assert "DEPTH_SPARSE" in codes or "CORRUPT_FRAMES" in codes

    def test_audit_empty_dataset_fails(self):
        result = audit_dataset([], lambda f: None, lambda f: None)
        assert not result.ok
        assert result.issues[0].code == "EMPTY_DATASET"
