"""
auto_mobility.trajectory.io

표준 Trajectory (TUM format: timestamp tx ty tz qx qy qz qw) IO 및 변환 유틸리티.
다양한 Odometry / SLAM 백엔드의 출력 궤적을 단일 포맷으로 정규화하여
비교 및 Reconstruction 입력으로 공급할 수 있도록 한다.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation


class Trajectory:
    """TUM 포맷 (timestamp x y z qx qy qz qw) 궤적 표현 클래스."""

    def __init__(self, timestamps: np.ndarray, positions: np.ndarray, orientations: np.ndarray, frame_ids: Optional[List[int]] = None):
        """
        timestamps: (N,) float64 (초 단위)
        positions: (N, 3) float64 (x, y, z)
        orientations: (N, 4) float64 (qx, qy, qz, qw)
        frame_ids: (N,) int 또는 None (RTAB-Map node_id 또는 프레임 인덱스)
        """
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.positions = np.asarray(positions, dtype=np.float64)
        self.orientations = np.asarray(orientations, dtype=np.float64)
        self.frame_ids = frame_ids if frame_ids is not None else list(range(len(timestamps)))

    def __len__(self) -> int:
        return len(self.timestamps)

    def to_tum_file(self, filepath: str, comment: str = "") -> None:
        """TUM 포맷 파일로 저장."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            if comment:
                f.write(f"# {comment}\n")
            f.write("# timestamp tx ty tz qx qy qz qw id\n")
            for stamp, pos, quat, fid in zip(self.timestamps, self.positions, self.orientations, self.frame_ids):
                f.write(f"{stamp:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                        f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} {fid}\n")

    @classmethod
    def from_tum_file(cls, filepath: str) -> "Trajectory":
        """TUM 포맷 파일에서 궤적 로드."""
        stamps, poses, quats, ids = [], [], [], []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                stamps.append(float(parts[0]))
                poses.append([float(parts[1]), float(parts[2]), float(parts[3])])
                quats.append([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])
                if len(parts) >= 9:
                    try:
                        ids.append(int(parts[8]))
                    except ValueError:
                        ids.append(len(stamps) - 1)
                else:
                    ids.append(len(stamps) - 1)

        if not stamps:
            return cls(
                timestamps=np.empty((0,), dtype=np.float64),
                positions=np.empty((0, 3), dtype=np.float64),
                orientations=np.empty((0, 4), dtype=np.float64),
                frame_ids=[]
            )

        stamps_arr = np.array(stamps, dtype=np.float64)
        poses_arr = np.array(poses, dtype=np.float64)
        quats_arr = np.array(quats, dtype=np.float64)
        ids_arr = np.array(ids, dtype=int)

        # 1. Sort chronologically by timestamp (fixes unsorted rtabmap-export outputs)
        sort_order = np.argsort(stamps_arr)
        stamps_arr = stamps_arr[sort_order]
        poses_arr = poses_arr[sort_order]
        quats_arr = quats_arr[sort_order]
        ids_arr = ids_arr[sort_order]

        # 2. Filter duplicate / non-monotonic timestamps
        if len(stamps_arr) > 1:
            dt = np.diff(stamps_arr)
            valid_mask = np.insert(dt > 1e-6, 0, True)
            stamps_arr = stamps_arr[valid_mask]
            poses_arr = poses_arr[valid_mask]
            quats_arr = quats_arr[valid_mask]
            ids_arr = ids_arr[valid_mask]

        # 3. Filter isolated (0,0,0) tracking-loss reset artifacts
        if len(poses_arr) > 2:
            is_zero = np.all(poses_arr == 0.0, axis=1)
            drop_zero = np.zeros(len(poses_arr), dtype=bool)
            for i in range(1, len(poses_arr)):
                if is_zero[i] and np.linalg.norm(poses_arr[i - 1]) > 0.3:
                    drop_zero[i] = True
            if np.any(drop_zero):
                keep = ~drop_zero
                stamps_arr = stamps_arr[keep]
                poses_arr = poses_arr[keep]
                quats_arr = quats_arr[keep]
                ids_arr = ids_arr[keep]

        return cls(
            timestamps=stamps_arr,
            positions=poses_arr,
            orientations=quats_arr,
            frame_ids=ids_arr.tolist()
        )

    def to_pose_matrix_dict(self) -> Dict[int, np.ndarray]:
        """{frame_id: 4x4 T_map_cam} 딕셔너리로 변환."""
        res = {}
        for fid, pos, quat in zip(self.frame_ids, self.positions, self.orientations):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            res[fid] = T
        return res

    def compute_metrics(self) -> Dict[str, float]:
        """Ground Truth 없이 측정 가능한 궤적 품질 지표 계산."""
        if len(self.positions) < 2:
            return {
                "num_frames": len(self.positions),
                "total_path_length": 0.0,
                "max_jump": 0.0,
                "avg_velocity": 0.0,
            }

        diffs = np.diff(self.positions, axis=0)
        step_lens = np.linalg.norm(diffs, axis=1)
        total_length = float(np.sum(step_lens))
        max_step = float(np.max(step_lens)) if len(step_lens) > 0 else 0.0

        dt = np.diff(self.timestamps)
        valid_dt = dt > 1e-5
        velocities = step_lens[valid_dt] / dt[valid_dt] if np.any(valid_dt) else np.array([0.0])
        avg_vel = float(np.mean(velocities)) if len(velocities) > 0 else 0.0
        max_vel = float(np.max(velocities)) if len(velocities) > 0 else 0.0

        return {
            "num_frames": len(self.positions),
            "total_path_length_m": round(total_length, 4),
            "max_step_m": round(max_step, 4),
            "avg_velocity_mps": round(avg_vel, 4),
            "max_velocity_mps": round(max_vel, 4),
        }
