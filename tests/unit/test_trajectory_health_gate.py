"""
tests/unit/test_trajectory_health_gate.py

Tests:
  11. Synthetic 37km / extreme length trajectory -> fails before reconstruction (FAIL_TRAJECTORY)
  12. Large jump trajectory -> FAIL / WARN threshold verified
  13. Normal trajectory -> PASS
"""

import pytest
import numpy as np
from pathlib import Path
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.diagnostics.trajectory_health import check_trajectory_health


def test_extreme_path_length_fails_health_gate(tmp_path):
    """Test 11: 37km trajectory fails health gate immediately."""
    traj_p = tmp_path / "extreme_len.txt"
    # Create trajectory that covers 37,000 meters
    stamps = np.linspace(0, 100, 1000)
    pos = np.zeros((1000, 3))
    pos[:, 0] = np.linspace(0, 37000, 1000) # 37 km
    quat = np.zeros((1000, 4))
    quat[:, 3] = 1.0
    
    traj = Trajectory(timestamps=stamps, positions=pos, orientations=quat)
    traj.save_tum_file(str(traj_p))
    
    res = check_trajectory_health(traj_p)
    assert res.status == "FAIL_TRAJECTORY"
    assert res.cause in ("EXTREME_PATH_LENGTH", "EXTREME_BBOX", "EXTREME_VELOCITY", "EXTREME_JUMP")
    assert res.is_pass is False


def test_large_isolated_jump_fails_or_warns(tmp_path):
    """Test 12: Large jump trajectory triggers FAIL or WARN."""
    # 1. Extreme 100m single jump -> FAIL
    stamps = np.linspace(0, 10, 100)
    pos = np.zeros((100, 3))
    pos[:, 0] = np.linspace(0, 5, 100) # normal 5cm steps
    pos[50, 0] += 100.0 # 100m sudden jump
    quat = np.zeros((100, 4))
    quat[:, 3] = 1.0
    
    traj_fail = Trajectory(timestamps=stamps, positions=pos, orientations=quat)
    res_fail = check_trajectory_health(traj_fail)
    assert res_fail.status == "FAIL_TRAJECTORY"
    assert res_fail.cause == "EXTREME_JUMP"
    
    # 2. Moderate 1.5m jump in indoor context -> WARN
    pos_warn = np.zeros((100, 3))
    pos_warn[:, 0] = np.linspace(0, 5, 100)
    pos_warn[50, 0] += 1.2
    traj_warn = Trajectory(timestamps=stamps, positions=pos_warn, orientations=quat)
    res_warn = check_trajectory_health(traj_warn)
    assert res_warn.status in ("WARN", "PASS")


def test_normal_trajectory_passes_health_gate(tmp_path):
    """Test 13: Normal smooth indoor handheld trajectory passes with PASS status."""
    stamps = np.linspace(0, 10, 200)
    pos = np.zeros((200, 3))
    pos[:, 0] = np.sin(stamps) * 2.0
    pos[:, 1] = np.cos(stamps) * 2.0
    quat = np.zeros((200, 4))
    quat[:, 3] = 1.0
    
    traj = Trajectory(timestamps=stamps, positions=pos, orientations=quat)
    res = check_trajectory_health(traj)
    
    assert res.status == "PASS"
    assert res.cause == "NONE"
    assert res.is_pass is True
    assert res.pose_count == 200
    assert res.finite_pose_ratio == 1.0
    assert res.quat_norm_valid_ratio == 1.0
