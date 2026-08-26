"""
auto_mobility.evaluation.trajectory_metrics

Trajectory 품질 지표 계산 모듈 (Ground Truth 부재 시 일관성 지표 및 GT 제공 시 ATE/RPE 평가 지원).
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, Optional, Union
from auto_mobility.trajectory.io import Trajectory


def compute_trajectory_quality(
    trajectory: Union[Trajectory, str],
    gt_trajectory: Optional[Union[Trajectory, str]] = None
) -> dict:
    """Trajectory의 내재적 품질 지표 (No GT) 및 절대 정밀도 (With GT) 계산."""
    if isinstance(trajectory, str):
        traj = Trajectory.from_tum_file(trajectory)
    else:
        traj = trajectory

    n_poses = len(traj)
    if n_poses < 2:
        return {
            "num_poses": n_poses,
            "trajectory_duration_sec": 0.0,
            "path_length_m": 0.0,
            "start_to_end_distance_m": 0.0,
            "translation_step_median_m": 0.0,
            "translation_step_p95_m": 0.0,
            "translation_step_max_m": 0.0,
            "rotation_step_median_deg": 0.0,
            "rotation_step_p95_deg": 0.0,
            "rotation_step_max_deg": 0.0,
            "linear_velocity_p95_mps": 0.0,
            "angular_velocity_p95_degps": 0.0,
            "large_jump_count": 0,
            "ground_truth_available": False
        }

    positions = traj.positions
    orientations = traj.orientations
    stamps = traj.timestamps

    duration = float(stamps[-1] - stamps[0])
    diffs = np.diff(positions, axis=0)
    step_lens = np.linalg.norm(diffs, axis=1)
    total_length = float(np.sum(step_lens))
    start_end_dist = float(np.linalg.norm(positions[-1] - positions[0]))

    # Angular steps
    rotations = Rotation.from_quat(orientations)
    rel_rots = rotations[:-1].inv() * rotations[1:]
    rot_angles_deg = rel_rots.magnitude() * (180.0 / np.pi)

    dt = np.diff(stamps)
    valid_dt = dt > 1e-5
    lin_vel = step_lens[valid_dt] / dt[valid_dt] if np.any(valid_dt) else np.array([0.0])
    ang_vel = rot_angles_deg[valid_dt] / dt[valid_dt] if np.any(valid_dt) else np.array([0.0])

    # Sudden jumps (> 0.2m translation step or > 30 deg rotation step between consecutive frames)
    jump_mask = (step_lens > 0.2) | (rot_angles_deg > 30.0)
    large_jump_count = int(np.sum(jump_mask))

    result = {
        "num_poses": n_poses,
        "trajectory_duration_sec": round(duration, 3),
        "path_length_m": round(total_length, 4),
        "start_to_end_distance_m": round(start_end_dist, 4),
        "translation_step_median_m": round(float(np.median(step_lens)), 4),
        "translation_step_p95_m": round(float(np.percentile(step_lens, 95)), 4),
        "translation_step_max_m": round(float(np.max(step_lens)), 4),
        "rotation_step_median_deg": round(float(np.median(rot_angles_deg)), 2),
        "rotation_step_p95_deg": round(float(np.percentile(rot_angles_deg, 95)), 2),
        "rotation_step_max_deg": round(float(np.max(rot_angles_deg)), 2),
        "linear_velocity_p95_mps": round(float(np.percentile(lin_vel, 95)), 3) if len(lin_vel) else 0.0,
        "angular_velocity_p95_degps": round(float(np.percentile(ang_vel, 95)), 2) if len(ang_vel) else 0.0,
        "large_jump_count": large_jump_count,
        "ground_truth_available": False
    }

    # Optional Ground Truth ATE Evaluation (Umeyama Alignment)
    if gt_trajectory is not None:
        if isinstance(gt_trajectory, str):
            gt_traj = Trajectory.from_tum_file(gt_trajectory)
        else:
            gt_traj = gt_trajectory

        ate_metrics = evaluate_ate(traj, gt_traj)
        result["ground_truth_available"] = True
        result["ate"] = ate_metrics

    return result


def evaluate_ate(est_traj: Trajectory, gt_traj: Trajectory, max_time_diff: float = 0.02) -> dict:
    """Absolute Trajectory Error (ATE) 계산 with SE(3) Umeyama Alignment."""
    est_stamps = est_traj.timestamps
    gt_stamps = gt_traj.timestamps

    matched_est = []
    matched_gt = []

    for idx, t in enumerate(est_stamps):
        gt_idx = int(np.argmin(np.abs(gt_stamps - t)))
        if abs(gt_stamps[gt_idx] - t) <= max_time_diff:
            matched_est.append(est_traj.positions[idx])
            matched_gt.append(gt_traj.positions[gt_idx])

    if len(matched_est) < 3:
        return {"error": "Not enough matching timestamps for ATE evaluation"}

    P = np.array(matched_est)
    Q = np.array(matched_gt)

    # Umeyama alignment (Rigid body SE(3))
    mu_P = np.mean(P, axis=0)
    mu_Q = np.mean(Q, axis=0)
    P_c = P - mu_P
    Q_c = Q - mu_Q

    H = np.dot(P_c.T, Q_c)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = mu_Q - np.dot(R, mu_P)

    P_aligned = np.dot(P, R.T) + t
    errors = np.linalg.norm(P_aligned - Q, axis=1)

    return {
        "num_matched_poses": len(errors),
        "ate_rmse_m": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "ate_mean_m": round(float(np.mean(errors)), 4),
        "ate_median_m": round(float(np.median(errors)), 4),
        "ate_std_m": round(float(np.std(errors)), 4),
        "ate_max_m": round(float(np.max(errors)), 4)
    }
