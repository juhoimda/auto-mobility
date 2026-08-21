"""
trajectory_health.py — Phase A0 Trajectory Health Gate & Sanity Diagnostics.

Validates SLAM trajectories before Open3D / TSDF reconstruction to prevent
wasting computational resources and memory crashes on broken, jumping, or NaN trajectories.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.transform import Rotation

from auto_mobility.trajectory.io import Trajectory


@dataclass
class TrajectoryHealthResult:
    trajectory_path: str
    status: str                         # "PASS", "WARN", "FAIL_TRAJECTORY"
    cause: str                          # "NONE", "NAN_INF_VALUES", "INVALID_QUATERNION", "INSUFFICIENT_POSES", "NON_MONOTONIC_TIMESTAMPS", "EXTREME_JUMP", "EXTREME_VELOCITY", "EXTREME_PATH_LENGTH", "EXTREME_BBOX"
    pose_count: int
    finite_pose_ratio: float
    quat_norm_valid_ratio: float
    monotonic_timestamp_ratio: float
    total_path_length_m: float
    bbox_extent_m: List[float]          # [dx, dy, dz]
    bbox_diagonal_m: float
    translation_step_median_m: float
    translation_step_p95_m: float
    translation_step_p99_m: float
    translation_step_max_m: float
    rotation_step_median_deg: float
    rotation_step_p95_deg: float
    rotation_step_max_deg: float
    linear_velocity_median_mps: float
    linear_velocity_p95_mps: float
    linear_velocity_max_mps: float
    angular_velocity_max_degps: float
    isolated_jump_count: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_pass(self) -> bool:
        return self.status in ("PASS", "WARN")


def check_trajectory_health(
    trajectory_input: Union[Trajectory, str, Path],
    max_step_physical_limit_m: float = 5.0,
    max_velocity_physical_limit_mps: float = 30.0,
    max_bbox_diagonal_m: float = 500.0,
    max_path_length_m: float = 2000.0,
    relative_step_multiplier: float = 20.0,
    min_pose_count: int = 2,
    quat_tol: float = 1e-2
) -> TrajectoryHealthResult:
    """Diagnoses trajectory health to gate downstream reconstruction.

    Evaluates:
      - NaN / Inf finite checks
      - Quaternion unit-norm validity
      - Timestamp monotonicity
      - Translation step & velocity distributions (Median, P95, P99, Max)
      - Bounding box & total length extent
      - Relative and absolute jump anomaly detection
    """
    traj_path_str = str(trajectory_input) if isinstance(trajectory_input, (str, Path)) else "in-memory"
    if isinstance(trajectory_input, (str, Path)):
        p = Path(trajectory_input)
        if not p.exists() or p.stat().st_size == 0:
            return TrajectoryHealthResult(
                trajectory_path=traj_path_str,
                status="FAIL_TRAJECTORY",
                cause="FILE_NOT_FOUND_OR_EMPTY",
                pose_count=0,
                finite_pose_ratio=0.0,
                quat_norm_valid_ratio=0.0,
                monotonic_timestamp_ratio=0.0,
                total_path_length_m=0.0,
                bbox_extent_m=[0.0, 0.0, 0.0],
                bbox_diagonal_m=0.0,
                translation_step_median_m=0.0,
                translation_step_p95_m=0.0,
                translation_step_p99_m=0.0,
                translation_step_max_m=0.0,
                rotation_step_median_deg=0.0,
                rotation_step_p95_deg=0.0,
                rotation_step_max_deg=0.0,
                linear_velocity_median_mps=0.0,
                linear_velocity_p95_mps=0.0,
                linear_velocity_max_mps=0.0,
                angular_velocity_max_degps=0.0,
                isolated_jump_count=0,
                warnings=["Trajectory file is missing or empty"]
            )
        traj = Trajectory.from_tum_file(str(p))
    else:
        traj = trajectory_input

    n_poses = len(traj)
    warnings: List[str] = []

    if n_poses < min_pose_count:
        return TrajectoryHealthResult(
            trajectory_path=traj_path_str,
            status="FAIL_TRAJECTORY",
            cause="INSUFFICIENT_POSES",
            pose_count=n_poses,
            finite_pose_ratio=0.0,
            quat_norm_valid_ratio=0.0,
            monotonic_timestamp_ratio=0.0,
            total_path_length_m=0.0,
            bbox_extent_m=[0.0, 0.0, 0.0],
            bbox_diagonal_m=0.0,
            translation_step_median_m=0.0,
            translation_step_p95_m=0.0,
            translation_step_p99_m=0.0,
            translation_step_max_m=0.0,
            rotation_step_median_deg=0.0,
            rotation_step_p95_deg=0.0,
            rotation_step_max_deg=0.0,
            linear_velocity_median_mps=0.0,
            linear_velocity_p95_mps=0.0,
            linear_velocity_max_mps=0.0,
            angular_velocity_max_degps=0.0,
            isolated_jump_count=0,
            warnings=[f"Trajectory has only {n_poses} poses (< {min_pose_count})"]
        )

    # 1. Finite checks (NaN / Inf)
    pos = traj.positions
    quat = traj.orientations
    stamps = traj.timestamps

    pos_finite = np.all(np.isfinite(pos), axis=1)
    quat_finite = np.all(np.isfinite(quat), axis=1)
    stamps_finite = np.isfinite(stamps)
    all_finite = pos_finite & quat_finite & stamps_finite
    finite_ratio = float(np.mean(all_finite))

    if finite_ratio < 0.99:
        return TrajectoryHealthResult(
            trajectory_path=traj_path_str,
            status="FAIL_TRAJECTORY",
            cause="NAN_INF_VALUES",
            pose_count=n_poses,
            finite_pose_ratio=finite_ratio,
            quat_norm_valid_ratio=0.0,
            monotonic_timestamp_ratio=0.0,
            total_path_length_m=0.0,
            bbox_extent_m=[0.0, 0.0, 0.0],
            bbox_diagonal_m=0.0,
            translation_step_median_m=0.0,
            translation_step_p95_m=0.0,
            translation_step_p99_m=0.0,
            translation_step_max_m=0.0,
            rotation_step_median_deg=0.0,
            rotation_step_p95_deg=0.0,
            rotation_step_max_deg=0.0,
            linear_velocity_median_mps=0.0,
            linear_velocity_p95_mps=0.0,
            linear_velocity_max_mps=0.0,
            angular_velocity_max_degps=0.0,
            isolated_jump_count=0,
            warnings=[f"Trajectory contains non-finite values (finite ratio: {finite_ratio*100:.1f}%)"]
        )

    # 2. Quaternion norm check
    q_norms = np.linalg.norm(quat, axis=1)
    q_valid = np.abs(q_norms - 1.0) <= quat_tol
    quat_valid_ratio = float(np.mean(q_valid))
    if quat_valid_ratio < 0.90:
        return TrajectoryHealthResult(
            trajectory_path=traj_path_str,
            status="FAIL_TRAJECTORY",
            cause="INVALID_QUATERNION",
            pose_count=n_poses,
            finite_pose_ratio=finite_ratio,
            quat_norm_valid_ratio=quat_valid_ratio,
            monotonic_timestamp_ratio=0.0,
            total_path_length_m=0.0,
            bbox_extent_m=[0.0, 0.0, 0.0],
            bbox_diagonal_m=0.0,
            translation_step_median_m=0.0,
            translation_step_p95_m=0.0,
            translation_step_p99_m=0.0,
            translation_step_max_m=0.0,
            rotation_step_median_deg=0.0,
            rotation_step_p95_deg=0.0,
            rotation_step_max_deg=0.0,
            linear_velocity_median_mps=0.0,
            linear_velocity_p95_mps=0.0,
            linear_velocity_max_mps=0.0,
            angular_velocity_max_degps=0.0,
            isolated_jump_count=0,
            warnings=[f"Quaternions not normalized (valid ratio: {quat_valid_ratio*100:.1f}%)"]
        )

    # 3. Timestamp monotonicity
    dt = np.diff(stamps)
    monotonic_ratio = float(np.mean(dt > 0.0)) if len(dt) > 0 else 1.0
    if monotonic_ratio < 0.95:
        warnings.append(f"Timestamp monotonicity violations: {monotonic_ratio*100:.1f}% monotonic")

    # 4. Translation and Rotation Steps
    pos_diffs = np.diff(pos, axis=0)
    steps = np.linalg.norm(pos_diffs, axis=1)

    step_med = float(np.median(steps)) if len(steps) > 0 else 0.0
    step_p95 = float(np.percentile(steps, 95)) if len(steps) > 0 else 0.0
    step_p99 = float(np.percentile(steps, 99)) if len(steps) > 0 else 0.0
    step_max = float(np.max(steps)) if len(steps) > 0 else 0.0
    path_len = float(np.sum(steps))

    # Rotation angle step
    rots = Rotation.from_quat(quat)
    rot_diffs = (rots[:-1].inv() * rots[1:]).magnitude() * (180.0 / np.pi)
    rot_med = float(np.median(rot_diffs)) if len(rot_diffs) > 0 else 0.0
    rot_p95 = float(np.percentile(rot_diffs, 95)) if len(rot_diffs) > 0 else 0.0
    rot_max = float(np.max(rot_diffs)) if len(rot_diffs) > 0 else 0.0

    # 5. Velocities
    valid_dt = dt > 1e-4
    vels = steps[valid_dt] / dt[valid_dt] if np.any(valid_dt) else np.array([0.0])
    vel_med = float(np.median(vels)) if len(vels) > 0 else 0.0
    vel_p95 = float(np.percentile(vels, 95)) if len(vels) > 0 else 0.0
    vel_max = float(np.max(vels)) if len(vels) > 0 else 0.0

    ang_vels = rot_diffs[valid_dt] / dt[valid_dt] if np.any(valid_dt) else np.array([0.0])
    ang_vel_max = float(np.max(ang_vels)) if len(ang_vels) > 0 else 0.0

    # 6. Bounding box
    bbox_min = np.min(pos, axis=0)
    bbox_max = np.max(pos, axis=0)
    bbox_extent = (bbox_max - bbox_min).tolist()
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    # 7. Isolated jump detection (jumps that return or break physical limits)
    isolated_jumps = int(np.sum(steps > max(max_step_physical_limit_m, step_p95 * relative_step_multiplier)))

    status = "PASS"
    cause = "NONE"

    # Hard failures (broken / impossible acquisition for indoor handheld/robot)
    if path_len > max_path_length_m:
        status = "FAIL_TRAJECTORY"
        cause = "EXTREME_PATH_LENGTH"
        warnings.append(f"Excessive trajectory path length: {path_len:.1f}m > {max_path_length_m}m")
    elif bbox_diag > max_bbox_diagonal_m:
        status = "FAIL_TRAJECTORY"
        cause = "EXTREME_BBOX"
        warnings.append(f"Excessive bounding box diagonal: {bbox_diag:.1f}m > {max_bbox_diagonal_m}m")
    elif step_max > max(max_step_physical_limit_m, step_p95 * relative_step_multiplier) and step_max > 2.0:
        status = "FAIL_TRAJECTORY"
        cause = "EXTREME_JUMP"
        warnings.append(f"Extreme translation jump: {step_max:.2f}m (median: {step_med:.3f}m, P95: {step_p95:.3f}m)")
    elif vel_max > max_velocity_physical_limit_mps:
        status = "FAIL_TRAJECTORY"
        cause = "EXTREME_VELOCITY"
        warnings.append(f"Extreme velocity detected: {vel_max:.1f} m/s > {max_velocity_physical_limit_mps} m/s")
    elif rot_max > 120.0 and ang_vel_max > 1500.0:
        status = "FAIL_TRAJECTORY"
        cause = "EXTREME_ROTATION_JUMP"
        warnings.append(f"Instantaneous angular discontinuity: {rot_max:.1f} deg at {ang_vel_max:.1f} deg/s")
    elif step_max > 1.0 or rot_max > 60.0 or vel_max > 10.0 or isolated_jumps > 0:
        status = "WARN"
        cause = "SUSPICIOUS_MOTION"
        warnings.append(f"Suspicious motion detected (max step {step_max:.2f}m, max rot {rot_max:.1f}deg, max vel {vel_max:.1f}m/s)")

    return TrajectoryHealthResult(
        trajectory_path=traj_path_str,
        status=status,
        cause=cause,
        pose_count=n_poses,
        finite_pose_ratio=finite_ratio,
        quat_norm_valid_ratio=quat_valid_ratio,
        monotonic_timestamp_ratio=monotonic_ratio,
        total_path_length_m=round(path_len, 3),
        bbox_extent_m=[round(x, 3) for x in bbox_extent],
        bbox_diagonal_m=round(bbox_diag, 3),
        translation_step_median_m=round(step_med, 4),
        translation_step_p95_m=round(step_p95, 4),
        translation_step_p99_m=round(step_p99, 4),
        translation_step_max_m=round(step_max, 4),
        rotation_step_median_deg=round(rot_med, 3),
        rotation_step_p95_deg=round(rot_p95, 3),
        rotation_step_max_deg=round(rot_max, 3),
        linear_velocity_median_mps=round(vel_med, 3),
        linear_velocity_p95_mps=round(vel_p95, 3),
        linear_velocity_max_mps=round(vel_max, 3),
        angular_velocity_max_degps=round(ang_vel_max, 3),
        isolated_jump_count=isolated_jumps,
        warnings=warnings
    )
