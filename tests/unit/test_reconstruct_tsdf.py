import unittest
import os
import tempfile
import numpy as np

from auto_mobility.mesh import reconstruct_tsdf


class TestReconstructTsdfParsers(unittest.TestCase):

    def test_load_poses_parses_tum_format(self):
        content = (
            "#timestamp x y z qx qy qz qw id\n"
            "1786504901.920936 0.1 0.2 0.3 0.0 0.0 0.0 1.0 1\n"
            "1786504903.172896 0.4 0.5 0.6 0.70710678 0.0 0.0 0.70710678 2\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            poses = reconstruct_tsdf.load_poses(path)
        finally:
            os.unlink(path)
        self.assertEqual(set(poses.keys()), {1, 2})
        t1 = poses[1]
        self.assertEqual(t1.shape, (4, 4))
        np.testing.assert_allclose(t1[:3, 3], [0.1, 0.2, 0.3], atol=1e-6)
        # node 2: quat(0.707,0,0,0.707) = X축 90도 회전 행렬 검증
        t2 = poses[2]
        np.testing.assert_allclose(t2[:3, :3], [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-6)

    def test_load_poses_skips_header_and_blank(self):
        content = "#comment\n\n1.0 0 0 0 0 0 0 1 5\n"
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            poses = reconstruct_tsdf.load_poses(path)
        finally:
            os.unlink(path)
        self.assertEqual(list(poses.keys()), [5])

    def test_load_frames(self):
        content = (
            "1 1 color/000001.jpg depth/000001.png\n"
            "2 4 color/000002.jpg depth/000002.png\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            frames = reconstruct_tsdf.load_frames(path)
        finally:
            os.unlink(path)
        self.assertEqual(frames, [
            (1, 1, 0.0, "color/000001.jpg", "depth/000001.png"),
            (2, 4, 0.0, "color/000002.jpg", "depth/000002.png"),
        ])

    def test_cuda_available_no_crash(self):
        self.assertIsInstance(reconstruct_tsdf._cuda_available(), bool)


if __name__ == "__main__":
    unittest.main()
