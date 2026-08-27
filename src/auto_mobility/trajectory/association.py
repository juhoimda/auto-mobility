"""
auto_mobility.trajectory.association

Trajectory (TUM format)와 RGB-D Frame Timestamp 정밀 매칭 및 보간 모듈.

좌표계 규약 (Coordinate Convention):
  - Trajectory Pose는 World 좌표계 기준 Camera의 위치/자세(T_world_camera)를 표현한다.
  - T_world_camera: (4x4) camera_color_optical_frame -> world (map/odom) 변환 행렬.
  - T_camera_world = inv(T_world_camera): world -> camera 변환 행렬 (Open3D TSDF / Raycasting extrinsic).
"""

import os
import csv
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .io import Trajectory


@dataclass
class PoseAssociationResult:
    frame_id: int
    frame_timestamp: float
    matched_pose_timestamp: float
    pose_dt_ms: float
    association_method: str  # "exact", "slerp_interp", "nearest", "none"
    valid: bool
    T_world_camera: Optional[np.ndarray] = None


@dataclass
class AssociationSummary:
    num_frames: int
    pose_match_count: int
    pose_missing_count: int
    pose_coverage_ratio: float
    pose_dt_mean_ms: float
    pose_dt_median_ms: float
    pose_dt_p95_ms: float
    pose_dt_max_ms: float
    warning: Optional[str] = None


def associate_trajectory_to_frames(
    frame_timestamps: Union[np.ndarray, List[float]],
    trajectory: Trajectory,
    max_pose_gap_ms: float = 50.0,
    enable_interpolation: bool = True
) -> Tuple[Dict[int, np.ndarray], List[PoseAssociationResult], AssociationSummary]:
    """Frame timestamps와 Trajectory 간 timestamp 매칭 및 SLERP 보간 수행.

    동일 (timestamps, trajectory 내용, 파라미터) 조합의 반복 호출은 결과를 재사용한다.
    benchmark는 후보마다 evaluator/reconstruct가 동일 궤적에 대해 이 함수를 반복 호출하므로
    SLERP 전체 재계산(O(F log P))을 생략할 수 있다.

    주의: 반환된 dict/list는 캐시된 객체이므로 수정하지 않는다 (read-only 계약).
    """
    frame_stamps = np.ascontiguousarray(frame_timestamps, dtype=np.float64)
    h = hashlib.sha1()
    h.update(frame_stamps.tobytes())
    h.update(np.ascontiguousarray(trajectory.timestamps, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(trajectory.positions, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(trajectory.orientations, dtype=np.float64).tobytes())
    key = (h.hexdigest(), len(getattr(trajectory, "timestamps", trajectory)), float(max_pose_gap_ms), bool(enable_interpolation))

    with _ASSOC_CACHE_LOCK:
        cached = _ASSOC_CACHE.get(key)
        if cached is not None:
            _ASSOC_CACHE.move_to_end(key)
            return cached

    result = _associate_trajectory_to_frames_impl(frame_stamps, trajectory, max_pose_gap_ms, enable_interpolation)

    with _ASSOC_CACHE_LOCK:
        _ASSOC_CACHE[key] = result
        while len(_ASSOC_CACHE) > _ASSOC_CACHE_MAX:
            _ASSOC_CACHE.popitem(last=False)
    return result


_ASSOC_CACHE_MAX = 32
_ASSOC_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_ASSOC_CACHE_LOCK = threading.Lock()


def _associate_trajectory_to_frames_impl(
    frame_stamps: np.ndarray,
    trajectory: Trajectory,
    max_pose_gap_ms: float = 50.0,
    enable_interpolation: bool = True
) -> Tuple[Dict[int, np.ndarray], List[PoseAssociationResult], AssociationSummary]:
    """Frame timestamps와 Trajectory 간 timestamp 매칭 및 SLERP 보간 수행 (캐시 없는 본체).

    Args:
        frame_stamps: (N,) float64 초 단위 프레임 타임스탬프 배열
        trajectory: Trajectory 객체 (TUM 형식: timestamp, x,y,z, qx,qy,qz,qw)
        max_pose_gap_ms: 허용 최대 타임스탬프 오차 (ms)
        enable_interpolation: 두 포즈 사이 시간인 경우 선형 보간 + SLERP 수행 여부

    Returns:
        poses_by_frame_id: {frame_id: T_world_camera (4x4)}
        records: List[PoseAssociationResult]
        summary: AssociationSummary
    """
    n_frames = len(frame_stamps)
    n_traj = len(getattr(trajectory, "timestamps", trajectory))
    if n_traj == 0:
        summary = AssociationSummary(
            num_frames=n_frames,
            pose_match_count=0,
            pose_missing_count=n_frames,
            pose_coverage_ratio=0.0,
            pose_dt_mean_ms=0.0,
            pose_dt_median_ms=0.0,
            pose_dt_p95_ms=0.0,
            pose_dt_max_ms=0.0,
            warning="Trajectory is empty!"
        )
        return {}, [], summary

    traj_stamps = np.asarray(trajectory.timestamps, dtype=np.float64)
    traj_positions = np.asarray(trajectory.positions, dtype=np.float64)
    traj_orientations = np.asarray(trajectory.orientations, dtype=np.float64)

    # Ensure trajectory is sorted by timestamp
    sort_idx = np.argsort(traj_stamps)
    traj_stamps = traj_stamps[sort_idx]
    traj_positions = traj_positions[sort_idx]
    traj_orientations = traj_orientations[sort_idx]

    # Handle duplicates in trajectory timestamps for Slerp
    unique_mask = np.ones(len(traj_stamps), dtype=bool)
    unique_mask[1:] = traj_stamps[1:] > traj_stamps[:-1]
    traj_stamps = traj_stamps[unique_mask]
    traj_positions = traj_positions[unique_mask]
    traj_orientations = traj_orientations[unique_mask]

    max_gap_sec = max_pose_gap_ms / 1000.0
    rotations = Rotation.from_quat(traj_orientations)
    slerp = Slerp(traj_stamps, rotations) if (len(traj_stamps) >= 2 and enable_interpolation) else None

    # Vectorized searchsorted for all frame timestamps at once
    indices = np.searchsorted(traj_stamps, frame_stamps)

    poses_by_frame_id: Dict[int, np.ndarray] = {}
    records: List[PoseAssociationResult] = []
    dt_ms_list: List[float] = []

    # Batch SLERP evaluation for valid interpolation frames
    interp_mask = (indices > 0) & (indices < len(traj_stamps)) if (enable_interpolation and slerp is not None) else np.zeros(n_frames, dtype=bool)
    if np.any(interp_mask):
        valid_interp_times = frame_stamps[interp_mask]
        try:
            batch_interp_rots = slerp(valid_interp_times).as_matrix()
        except Exception:
            batch_interp_rots = None
    else:
        batch_interp_rots = None

    interp_cnt = 0
    for fid, f_time in enumerate(frame_stamps):
        idx = int(indices[fid])

        # 1. Exact match check
        if idx < len(traj_stamps) and abs(traj_stamps[idx] - f_time) < 1e-6:
            pos = traj_positions[idx]
            rot = rotations[idx].as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot
            T[:3, 3] = pos
            poses_by_frame_id[fid] = T
            records.append(PoseAssociationResult(
                frame_id=fid,
                frame_timestamp=f_time,
                matched_pose_timestamp=traj_stamps[idx],
                pose_dt_ms=0.0,
                association_method="exact",
                valid=True,
                T_world_camera=T
            ))
            dt_ms_list.append(0.0)
            continue

        # 2. Interpolation between idx-1 and idx
        if enable_interpolation and slerp is not None and 0 < idx < len(traj_stamps):
            t0, t1 = traj_stamps[idx - 1], traj_stamps[idx]
            dt_interval = t1 - t0
            if dt_interval <= max_gap_sec * 2.0:
                alpha = (f_time - t0) / dt_interval
                interp_pos = (1.0 - alpha) * traj_positions[idx - 1] + alpha * traj_positions[idx]
                if batch_interp_rots is not None and interp_mask[fid]:
                    interp_rot = batch_interp_rots[interp_cnt]
                    interp_cnt += 1
                else:
                    interp_rot = slerp([f_time])[0].as_matrix()

                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = interp_rot
                T[:3, 3] = interp_pos

                poses_by_frame_id[fid] = T
                records.append(PoseAssociationResult(
                    frame_id=fid,
                    frame_timestamp=f_time,
                    matched_pose_timestamp=f_time,
                    pose_dt_ms=0.0,
                    association_method="slerp_interp",
                    valid=True,
                    T_world_camera=T
                ))
                dt_ms_list.append(0.0)
                continue
            elif interp_mask[fid]:
                interp_cnt += 1

        # 3. Fallback to Nearest
        candidates = []
        if idx < len(traj_stamps):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

        best_idx = min(candidates, key=lambda i: abs(traj_stamps[i] - f_time))
        best_dt_sec = abs(traj_stamps[best_idx] - f_time)
        best_dt_ms = best_dt_sec * 1000.0

        if best_dt_sec <= max_gap_sec:
            pos = traj_positions[best_idx]
            rot = rotations[best_idx].as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot
            T[:3, 3] = pos
            poses_by_frame_id[fid] = T
            records.append(PoseAssociationResult(
                frame_id=fid,
                frame_timestamp=f_time,
                matched_pose_timestamp=traj_stamps[best_idx],
                pose_dt_ms=best_dt_ms,
                association_method="nearest",
                valid=True,
                T_world_camera=T
            ))
            dt_ms_list.append(best_dt_ms)
        else:
            records.append(PoseAssociationResult(
                frame_id=fid,
                frame_timestamp=f_time,
                matched_pose_timestamp=traj_stamps[best_idx] if len(traj_stamps) else 0.0,
                pose_dt_ms=best_dt_ms,
                association_method="none",
                valid=False,
                T_world_camera=None
            ))

    match_count = len(poses_by_frame_id)
    missing_count = n_frames - match_count
    cov_ratio = match_count / max(n_frames, 1)

    dt_arr = np.array(dt_ms_list) if dt_ms_list else np.array([0.0])
    dt_mean = float(np.mean(dt_arr)) if len(dt_arr) else 0.0
    dt_median = float(np.median(dt_arr)) if len(dt_arr) else 0.0
    dt_p95 = float(np.percentile(dt_arr, 95)) if len(dt_arr) else 0.0
    dt_max = float(np.max(dt_arr)) if len(dt_arr) else 0.0

    warning = None
    if cov_ratio < 0.70:
        warning = f"Low pose coverage: {cov_ratio:.1%} ({missing_count} frames missing pose)"
    elif dt_p95 > max_pose_gap_ms * 0.8:
        warning = f"High timestamp association delta: P95 {dt_p95:.1f}ms"

    summary = AssociationSummary(
        num_frames=n_frames,
        pose_match_count=match_count,
        pose_missing_count=missing_count,
        pose_coverage_ratio=round(cov_ratio, 4),
        pose_dt_mean_ms=round(dt_mean, 3),
        pose_dt_median_ms=round(dt_median, 3),
        pose_dt_p95_ms=round(dt_p95, 3),
        pose_dt_max_ms=round(dt_max, 3),
        warning=warning
    )
    return poses_by_frame_id, records, summary


def save_association_csv(records: List[PoseAssociationResult], csv_path: str) -> None:
    """Save association records to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "frame_timestamp", "matched_pose_timestamp",
            "pose_dt_ms", "association_method", "valid"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "frame_id": r.frame_id,
                "frame_timestamp": f"{r.frame_timestamp:.6f}",
                "matched_pose_timestamp": f"{r.matched_pose_timestamp:.6f}",
                "pose_dt_ms": f"{r.pose_dt_ms:.3f}",
                "association_method": r.association_method,
                "valid": r.valid
            })
