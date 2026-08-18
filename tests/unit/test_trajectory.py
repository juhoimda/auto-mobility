import pytest
import numpy as np
from pathlib import Path
from auto_mobility.trajectory.io import Trajectory


class TestTrajectoryUnit:
    def test_trajectory_roundtrip(self, tmp_path):
        timestamps = np.array([0.0, 0.1, 0.2])
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        orientations = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        frame_ids = [10, 20, 30]

        traj = Trajectory(timestamps, positions, orientations, frame_ids)
        out_file = tmp_path / "test_traj.txt"
        traj.to_tum_file(str(out_file))

        loaded = Trajectory.from_tum_file(str(out_file))
        assert len(loaded) == 3
        np.testing.assert_allclose(loaded.positions, positions, atol=1e-5)
        np.testing.assert_allclose(loaded.timestamps, timestamps, atol=1e-5)
        assert loaded.frame_ids == frame_ids

    def test_trajectory_metrics(self):
        timestamps = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        orientations = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        traj = Trajectory(timestamps, positions, orientations)
        m = traj.compute_metrics()
        assert m["num_frames"] == 3
        assert m["total_path_length_m"] == 2.0
        assert m["max_step_m"] == 1.0
        assert m["avg_velocity_mps"] == 1.0
