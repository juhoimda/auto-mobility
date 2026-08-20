"""Diagnostics for connecting canonical RGB-D frames to SLAM trajectories.

This module deliberately runs before TSDF/mesh scoring.  A bad timestamp
association can make every downstream surface look bad, so it must be
reported as a separate cause instead of being folded into a mesh failure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from auto_mobility.trajectory.association import associate_trajectory_to_frames
from auto_mobility.trajectory.io import Trajectory


@dataclass
class PoseAlignmentSummary:
    trajectory_path: str
    frame_count: int
    pose_count: int
    frame_start_s: Optional[float]
    frame_end_s: Optional[float]
    pose_start_s: Optional[float]
    pose_end_s: Optional[float]
    time_overlap_s: float
    pose_coverage_ratio: float
    pose_dt_p95_ms: float
    max_translation_step_m: float
    max_rotation_step_deg: float
    best_time_offset_s: float
    best_offset_coverage_ratio: float
    status: str
    cause: str
    warnings: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _range(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None, None
    return float(np.min(arr)), float(np.max(arr))


def _motion_metrics(traj: Trajectory) -> Tuple[float, float]:
    if len(traj) < 2:
        return 0.0, 0.0
    from scipy.spatial.transform import Rotation

    steps = np.linalg.norm(np.diff(traj.positions, axis=0), axis=1)
    rots = Rotation.from_quat(traj.orientations)
    angles = (rots[:-1].inv() * rots[1:]).magnitude() * 180.0 / np.pi
    return float(np.max(steps)), float(np.max(angles))


def _coverage(frame_stamps: np.ndarray, traj: Trajectory, offset_s: float, max_gap_ms: float) -> float:
    """Coverage after shifting trajectory time by ``offset_s``."""
    shifted = Trajectory(
        traj.timestamps + offset_s,
        traj.positions,
        traj.orientations,
        traj.frame_ids,
    )
    _, _, summary = associate_trajectory_to_frames(
        frame_stamps, shifted, max_pose_gap_ms=max_gap_ms, enable_interpolation=True
    )
    return float(summary.pose_coverage_ratio)


def diagnose_pose_alignment(
    frame_timestamps: Iterable[float],
    trajectory: Trajectory,
    trajectory_path: str = "in-memory",
    max_pose_gap_ms: float = 50.0,
    offset_search_s: float = 2.0,
    offset_step_s: float = 0.1,
    coverage_pass: float = 0.85,
    coverage_warn: float = 0.65,
) -> Dict[str, Any]:
    """Return a deterministic diagnostic summary for one SLAM trajectory.

    The offset search is diagnostic only; it does not silently alter the
    trajectory used for reconstruction.  A large improvement indicates a
    likely timestamp-base problem and should be fixed at the exporter.
    """
    frames = np.asarray(list(frame_timestamps), dtype=np.float64)
    frame_start, frame_end = _range(frames)
    pose_start, pose_end = _range(trajectory.timestamps)
    overlap = 0.0
    if frame_start is not None and pose_start is not None:
        overlap = max(0.0, min(frame_end, pose_end) - max(frame_start, pose_start))

    _, _, assoc = associate_trajectory_to_frames(
        frames, trajectory, max_pose_gap_ms=max_pose_gap_ms, enable_interpolation=True
    )
    max_step, max_rot = _motion_metrics(trajectory)

    offsets = np.arange(-offset_search_s, offset_search_s + offset_step_s * 0.5, offset_step_s)
    coverages = [_coverage(frames, trajectory, float(off), max_pose_gap_ms) for off in offsets]
    best_idx = int(np.argmax(coverages)) if coverages else 0
    best_offset = float(offsets[best_idx]) if len(offsets) else 0.0
    best_coverage = float(coverages[best_idx]) if coverages else float(assoc.pose_coverage_ratio)

    warnings: list[str] = []
    cause = "NONE"
    status = "PASS"
    if assoc.pose_coverage_ratio < coverage_warn:
        status = "FAIL"
        cause = "SLAM_OR_TIME_ALIGNMENT"
        warnings.append(
            f"Pose coverage is {assoc.pose_coverage_ratio * 100:.1f}% "
            f"({assoc.pose_missing_count} frames missing)"
        )
    elif assoc.pose_coverage_ratio < coverage_pass:
        status = "WARN"
        cause = "SLAM_OR_TIME_ALIGNMENT"
        warnings.append(f"Pose coverage is {assoc.pose_coverage_ratio * 100:.1f}%")

    if best_coverage - assoc.pose_coverage_ratio >= 0.20:
        status = "FAIL"
        cause = "TIME_ALIGNMENT"
        warnings.append(
            f"A time offset of {best_offset:+.1f}s improves coverage to "
            f"{best_coverage * 100:.1f}%"
        )
    # At camera frame rates, a >1m translation or >90deg instantaneous
    # rotation is not a plausible motion sample.  Treat it as a hard SLAM
    # tracking failure; smaller jumps remain a warning for investigation.
    catastrophic_jump = max_step > 1.0 or max_rot > 90.0
    suspicious_jump = max_step > 0.2 or max_rot > 30.0
    if suspicious_jump:
        warnings.append(
            f"Large pose jump detected (translation {max_step:.3f}m, rotation {max_rot:.1f}deg)"
        )
        if catastrophic_jump:
            status = "FAIL"
            cause = "SLAM_TRACKING"
        elif cause == "NONE":
            cause = "SLAM_TRACKING"
            status = "WARN"

    return PoseAlignmentSummary(
        trajectory_path=str(trajectory_path),
        frame_count=int(len(frames)),
        pose_count=int(len(trajectory)),
        frame_start_s=frame_start,
        frame_end_s=frame_end,
        pose_start_s=pose_start,
        pose_end_s=pose_end,
        time_overlap_s=float(overlap),
        pose_coverage_ratio=float(assoc.pose_coverage_ratio),
        pose_dt_p95_ms=float(assoc.pose_dt_p95_ms),
        max_translation_step_m=max_step,
        max_rotation_step_deg=max_rot,
        best_time_offset_s=best_offset,
        best_offset_coverage_ratio=best_coverage,
        status=status,
        cause=cause,
        warnings=warnings,
    ).to_dict()
