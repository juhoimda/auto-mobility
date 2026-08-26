import unittest
import os
import tempfile
import numpy as np
from pathlib import Path

from auto_mobility.slam.run_stella_bag import generate_stella_config, convert_stella_trajectory_to_tum, run_stella_vslam_on_bag
from auto_mobility.slam.run_orbslam3_bag import run_orbslam3_on_bag
from auto_mobility.dataset.frame_dataset import CameraIntrinsics
from auto_mobility.trajectory.io import Trajectory


class TestSlamBackendUnit(unittest.TestCase):
    def test_stella_config_generation(self):
        intr = CameraIntrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_yaml = os.path.join(tmpdir, "stella.yaml")
            res_path = generate_stella_config(intrinsics=intr, output_path=out_yaml)
            self.assertTrue(os.path.exists(res_path))
            with open(res_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("fx: 615.0", content)
            self.assertIn("cols: 640", content)
            self.assertIn("setup: RGBD", content)

    def test_stella_trajectory_conversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stella_raw = os.path.join(tmpdir, "stella_raw.txt")
            tum_out = os.path.join(tmpdir, "stella_tum.txt")
            
            # Write dummy stella output: timestamp tx ty tz qx qy qz qw
            with open(stella_raw, "w") as f:
                f.write("0.000000 0.1 0.2 0.3 0.0 0.0 0.0 1.0\n")
                f.write("0.100000 0.2 0.3 0.4 0.0 0.0 0.0 1.0\n")
                
            convert_stella_trajectory_to_tum(stella_raw, tum_out)
            self.assertTrue(os.path.exists(tum_out))
            
            traj = Trajectory.from_tum_file(tum_out)
            self.assertEqual(len(traj), 2)
            self.assertAlmostEqual(traj.timestamps[0], 0.0)
            self.assertAlmostEqual(traj.positions[0, 0], 0.1)

    def test_stella_missing_binary_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_bag = os.path.join(tmpdir, "dummy.db")
            open(dummy_bag, "w").close()
            with self.assertRaises(RuntimeError):
                run_stella_vslam_on_bag(dummy_bag)

    def test_orbslam3_missing_bag_handling(self):
        with self.assertRaises(FileNotFoundError):
            run_orbslam3_on_bag("non_existent_bag_file_12345", mode="rgbdi")


if __name__ == "__main__":
    unittest.main()
