"""
tests/unit/test_trajectory_association.py

Unit tests for trajectory timestamp association, linear translation interpolation, and quaternion SLERP.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.trajectory.association import associate_trajectory_to_frames


def test_trajectory_slerp_and_linear_interpolation():
    # Trajectory with 2 poses: t=0.0 and t=1.0
    # t=0.0: pos=(0, 0, 0), rot=yaw 0 deg (identity quat)
    # t=1.0: pos=(1.0, 2.0, 3.0), rot=yaw 90 deg (z-axis rotation)
    stamps = np.array([0.0, 1.0])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0]
    ])
    q0 = Rotation.from_euler('z', 0, degrees=True).as_quat()
    q1 = Rotation.from_euler('z', 90, degrees=True).as_quat()
    orientations = np.array([q0, q1])

    traj = Trajectory(stamps, positions, orientations)

    # Query frame at t=0.5
    frame_stamps = [0.5]
    poses, records, summary = associate_trajectory_to_frames(
        frame_stamps, traj, max_pose_gap_ms=1000.0, enable_interpolation=True
    )

    assert len(poses) == 1
    assert 0 in poses
    T = poses[0]

    # Expected position: (0.5, 1.0, 1.5)
    expected_pos = np.array([0.5, 1.0, 1.5])
    np.testing.assert_allclose(T[:3, 3], expected_pos, atol=1e-5)

    # Expected orientation: yaw 45 deg
    expected_rot = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    np.testing.assert_allclose(T[:3, :3], expected_rot, atol=1e-5)

    assert records[0].association_method == "slerp_interp"
    assert records[0].valid is True
    assert summary.pose_match_count == 1
    assert summary.pose_missing_count == 0


def test_trajectory_gap_rejection():
    # Trajectory at t=0.0 and t=0.1
    traj = Trajectory(
        timestamps=np.array([0.0, 0.1]),
        positions=np.array([[0, 0, 0], [0.1, 0, 0]]),
        orientations=np.array([[0, 0, 0, 1], [0, 0, 0, 1]])
    )

    # Frame at t=10.0 (way beyond max_pose_gap_ms=50ms)
    frame_stamps = [10.0]
    poses, records, summary = associate_trajectory_to_frames(
        frame_stamps, traj, max_pose_gap_ms=50.0
    )

    assert len(poses) == 0
    assert records[0].valid is False
    assert summary.pose_missing_count == 1
    assert summary.pose_coverage_ratio == 0.0
    assert summary.warning is not None
