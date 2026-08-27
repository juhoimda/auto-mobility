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


@dataclass(frozen=True)
class BenchmarkSplit:
    """3-way deterministic split from common pose set for fair benchmark.

    - train_ids: search, pose refinement, coarse/mask/fine
    - tuning_val_ids: voxel/Poisson selection only
    - benchmark_holdout_ids: excluded from ALL fusion/mask/refinement/texture; used only for final evaluate_geometry
    """
    train_ids: tuple = field(default_factory=tuple)
    tuning_val_ids: tuple = field(default_factory=tuple)
    benchmark_holdout_ids: tuple = field(default_factory=tuple)
    segment_count: int = 0
    policy: str = "pose_space_blocks_3way"
    seed: int = 42
    dataset_fingerprint: str = ""
    common_pose_count: int = 0
    # SHAs for provenance
    train_ids_sha256: str = ""
    tuning_val_ids_sha256: str = ""
    benchmark_holdout_ids_sha256: str = ""
    generation_rule: str = ""

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "segment_count": self.segment_count,
            "seed": self.seed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "common_pose_count": self.common_pose_count,
            "train_ids": list(self.train_ids),
            "tuning_val_ids": list(self.tuning_val_ids),
            "benchmark_holdout_ids": list(self.benchmark_holdout_ids),
            "n_train": len(self.train_ids),
            "n_tuning_val": len(self.tuning_val_ids),
            "n_benchmark_holdout": len(self.benchmark_holdout_ids),
            "train_ids_sha256": self.train_ids_sha256,
            "tuning_val_ids_sha256": self.tuning_val_ids_sha256,
            "benchmark_holdout_ids_sha256": self.benchmark_holdout_ids_sha256,
            "generation_rule": self.generation_rule,
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


def _sha256_ids(ids) -> str:
    import hashlib as _h

    return _h.sha256(",".join(str(x) for x in sorted(ids)).encode()).hexdigest()[:16] if ids else "empty"


def _fingerprint_to_seed(fp: str, default: int = 42) -> int:
    """Deterministic seed from dataset fingerprint hex."""
    if not fp:
        return default
    try:
        # Use first 8 hex chars as int
        return int(fp[:8], 16) % (2**31 - 1)
    except Exception:
        return default


def create_benchmark_split(
    common_frame_ids,
    positions_m,
    rotation_deltas_rad,
    dataset_fingerprint: str = "",
    benchmark_ratio: float = 0.12,
    tuning_ratio: float = 0.12,
    target_segments: int = 8,
    seed: int | None = None,
    min_benchmark: int = 20,
    min_tuning: int = 10,
    min_train: int = 20,
) -> BenchmarkSplit:
    """Deterministic 3-way split from common pose set.

    Dataset fingerprint drives seed; benchmark_holdout covers pose space via blocks.
    Returns BenchmarkSplit with train/tuning/benchmark sets.
    """
    n = len(common_frame_ids)
    if n < min_train + min_tuning + min_benchmark:
        raise ValueError(f"not enough common frames ({n}) for 3-way split: need {min_train+min_tuning+min_benchmark}")

    if seed is None:
        seed = _fingerprint_to_seed(dataset_fingerprint or "", default=42)

    order = sorted(range(n), key=lambda i: common_frame_ids[i])
    fids = [common_frame_ids[i] for i in order]
    pos = np.asarray(positions_m, dtype=np.float64)[order]
    rot = np.asarray(rotation_deltas_rad, dtype=np.float64)[order]

    segments = max(1, min(target_segments, n // max(1, min_train)))
    bounds = np.linspace(0, n, segments + 1, dtype=int)
    rng = random.Random(seed)
    activity = _frame_activity(pos, rot)

    # Phase 1: select benchmark_holdout blocks spread across pose space
    benchmark_set = []
    for s in range(segments):
        lo, hi = int(bounds[s]), int(bounds[s + 1])
        seg_len = hi - lo
        if seg_len <= 0:
            continue
        block_len = max(1, int(round(seg_len * benchmark_ratio)))
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
        benchmark_set.extend(range(best_start, best_start + block_len))

    benchmark_set = sorted(set(benchmark_set))
    # Ensure at least min_benchmark
    if len(benchmark_set) < min_benchmark:
        needed = min_benchmark - len(benchmark_set)
        pool = [i for i in range(n) if i not in set(benchmark_set)]
        if pool:
            add = rng.sample(pool, min(needed, len(pool)))
            benchmark_set = sorted(set(benchmark_set) | set(add))
    # Ensure not too many (cap ~25% of n)
    max_benchmark = max(min_benchmark, int(n * 0.25))
    if len(benchmark_set) > max_benchmark:
        # trim evenly: keep pose-space spread by sampling
        keep = sorted(rng.sample(benchmark_set, max_benchmark))
        benchmark_set = sorted(keep)

    # Phase 2: split remaining into train and tuning_val
    remaining = [i for i in range(n) if i not in set(benchmark_set)]
    # tuning ratio applied to original n, but now from remaining
    tuning_target = max(min_tuning, int(round(n * tuning_ratio)))
    tuning_target = min(tuning_target, max(1, len(remaining) - min_train))
    # Select tuning blocks from remaining using same activity-driven segmentation
    # Map remaining indices to contiguous segments for sampling
    # Simpler: random stratified sampling across segments from remaining
    tuning_set = []
    # Re-segment remaining by original segment boundaries: count how many remaining per segment -> allocate proportionally
    remaining_by_seg = []
    for s in range(segments):
        lo, hi = int(bounds[s]), int(bounds[s + 1])
        seg_remaining = [idx for idx in remaining if lo <= idx < hi]
        if seg_remaining:
            remaining_by_seg.append(seg_remaining)
    # Distribute tuning_target across segments proportionally
    if remaining_by_seg:
        per_seg = max(1, tuning_target // len(remaining_by_seg))
        for seg_list in remaining_by_seg:
            take = min(per_seg, len(seg_list))
            if take > 0:
                # prefer high activity within segment
                seg_activity = [activity[i] for i in seg_list]
                # sort seg_list by activity descending, then random tie
                sorted_seg = sorted(range(len(seg_list)), key=lambda k: (-seg_activity[k], rng.random()))
                chosen_idxs = [seg_list[sorted_seg[k]] for k in range(take)]
                tuning_set.extend(chosen_idxs)
        # Adjust to exact tuning_target
        tuning_set = sorted(set(tuning_set))
        if len(tuning_set) > tuning_target:
            tuning_set = sorted(rng.sample(tuning_set, tuning_target))
        elif len(tuning_set) < tuning_target:
            pool = [i for i in remaining if i not in set(tuning_set)]
            if pool:
                need = tuning_target - len(tuning_set)
                tuning_set = sorted(set(tuning_set) | set(rng.sample(pool, min(need, len(pool)))))

    train_set = [i for i in remaining if i not in set(tuning_set)]

    # Final validation
    if len(train_set) < min_train or len(tuning_set) < min_tuning or len(benchmark_set) < min_benchmark:
        raise ValueError(f"3-way split failed size check: train {len(train_set)} tuning {len(tuning_set)} benchmark {len(benchmark_set)}")

    train_ids = tuple(fids[i] for i in sorted(train_set))
    tuning_ids = tuple(fids[i] for i in sorted(tuning_set))
    benchmark_ids = tuple(fids[i] for i in sorted(benchmark_set))

    generation_rule = (
        f"pose_space_blocks_3way: segments={segments}, benchmark_ratio={benchmark_ratio}, "
        f"tuning_ratio={tuning_ratio}, seed={seed} (from fingerprint {dataset_fingerprint[:8]})"
    )

    return BenchmarkSplit(
        train_ids=train_ids,
        tuning_val_ids=tuning_ids,
        benchmark_holdout_ids=benchmark_ids,
        segment_count=segments,
        policy="pose_space_blocks_3way",
        seed=seed,
        dataset_fingerprint=dataset_fingerprint,
        common_pose_count=n,
        train_ids_sha256=_sha256_ids(train_ids),
        tuning_val_ids_sha256=_sha256_ids(tuning_ids),
        benchmark_holdout_ids_sha256=_sha256_ids(benchmark_ids),
        generation_rule=generation_rule,
    )


def split_from_common_poses(
    common_frame_ids,
    poses_world_camera,
    dataset_fingerprint: str = "",
    seed: int | None = None,
) -> BenchmarkSplit:
    """Convenience wrapper for 3-way benchmark split from common poses."""
    poses = [np.asarray(p, dtype=np.float64) for p in poses_world_camera]
    if len(poses) != len(common_frame_ids):
        raise ValueError("poses must align with common_frame_ids")
    positions = np.array([p[:3, 3] for p in poses])
    rots = [p[:3, :3] for p in poses]
    deltas = np.zeros(len(poses))
    for i in range(1, len(poses)):
        R_rel = rots[i - 1].T @ rots[i]
        cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        deltas[i] = float(np.arccos(cos_theta))
    return create_benchmark_split(
        common_frame_ids, positions, deltas,
        dataset_fingerprint=dataset_fingerprint, seed=seed,
    )
