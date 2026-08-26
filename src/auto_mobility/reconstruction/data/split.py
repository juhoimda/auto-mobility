"""Pose-space / block validation split.

Replaces legacy every-nth splitting (neighbor correlation too high).
Trajectory is segmented into contiguous blocks by pose-space coverage;
validation blocks are chosen so start/middle/end/high-turn regions are all
represented. Deterministic under a fixed seed.

Complexity: Time O(N log N) for sorting + O(N) segmentation.
Memory: O(N) pose metadata (no images).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import numpy as np

from auto_mobility.reconstruction.model import PoseMatrix


@dataclass(frozen=True)
class HoldoutSplit:
    train_ids: tuple = field(default_factory=tuple)
    val_ids: tuple = field(default_factory=tuple)
    segment_count: int = 0
    policy: str = "pose_space_blocks"

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "segment_count": self.segment_count,
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
            "n_train": len(self.train_ids),
            "n_val": len(self.val_ids),
        }


def _frame_activity(positions: np.ndarray, rotation_deltas: np.ndarray) -> np.ndarray:
    """Per-frame motion activity score, aligned with frame index."""
    n = len(positions)
    trans = np.zeros(n)
    if n > 1:
        trans[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return rotation_deltas + 10.0 * trans


def create_pose_space_split(
    frame_ids,
    positions_m,
    rotation_deltas_rad,
    val_ratio: float = 0.15,
    target_segments: int = 8,
    seed: int = 42,
    min_train_frames: int = 10,
    min_val_frames: int = 4,
) -> HoldoutSplit:
    """Select contiguous validation blocks spread across pose space.

    frame_ids: ordered unique ids aligned with positions_m (N,3) meters and
    rotation_deltas_rad (N,) per-frame rotation magnitude.
    """
    n = len(frame_ids)
    if n != len(positions_m) or n != len(rotation_deltas_rad):
        raise ValueError("frame_ids, positions, rotation deltas must align")
    if n < min_train_frames + min_val_frames:
        raise ValueError(f"not enough frames ({n}) for a meaningful split")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0,1)")

    order = sorted(range(n), key=lambda i: frame_ids[i])
    fids = [frame_ids[i] for i in order]
    pos = np.asarray(positions_m, dtype=np.float64)[order]
    rot = np.asarray(rotation_deltas_rad, dtype=np.float64)[order]

    segments = max(1, min(target_segments, n // max(1, min_train_frames)))
    bounds = np.linspace(0, n, segments + 1, dtype=int)

    rng = random.Random(seed)
    activity = _frame_activity(pos, rot)

    val_set = []
    for s in range(segments):
        lo, hi = int(bounds[s]), int(bounds[s + 1])
        seg_len = hi - lo
        if seg_len <= 0:
            continue
        block_len = max(1, int(round(seg_len * val_ratio)))
        block_len = min(block_len, max(1, seg_len - 1))
        window_activity = np.convolve(activity[lo:hi], np.ones(3), mode="same")
        candidates = range(lo, hi - block_len + 1)
        if s == 0:
            best_start = hi - block_len
        elif s == segments - 1:
            best_start = lo
        else:
            ranked = sorted(candidates, key=lambda st: (-window_activity[st - lo], rng.random()))
            best_start = ranked[0]
        val_set.extend(range(best_start, best_start + block_len))

    val_set = sorted(set(val_set))
    if len(val_set) < min(min_val_frames, n // 5):
        needed = min(min_val_frames, n // 5) - len(val_set)
        pool = [i for i in range(n) if i not in set(val_set)]
        val_set = sorted(val_set + rng.sample(pool, needed)) if pool else val_set

    train_set = [i for i in range(n) if i not in set(val_set)]
    if len(train_set) < min_train_frames:
        keep = min(len(val_set), len(train_set) + len(val_set) - min_train_frames)
        drop = val_set[:keep]
        val_set = [i for i in val_set if i not in set(drop)]
        train_set = [i for i in range(n) if i not in set(val_set)]

    return HoldoutSplit(
        train_ids=tuple(fids[i] for i in train_set),
        val_ids=tuple(fids[i] for i in sorted(val_set)),
        segment_count=segments,
    )


def split_from_poses(
    frame_ids,
    poses_world_camera,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> HoldoutSplit:
    """Convenience wrapper deriving motion features from T_world_camera poses."""
    poses = [np.asarray(p, dtype=np.float64) for p in poses_world_camera]
    if len(poses) != len(frame_ids):
        raise ValueError("poses must align with frame_ids")
    positions = np.array([p[:3, 3] for p in poses])
    rots = [p[:3, :3] for p in poses]
    deltas = np.zeros(len(poses))
    for i in range(1, len(poses)):
        R_rel = rots[i - 1].T @ rots[i]
        cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        deltas[i] = float(np.arccos(cos_theta))
    return create_pose_space_split(
        frame_ids, positions, deltas, val_ratio=val_ratio, seed=seed
    )
