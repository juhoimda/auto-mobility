import unittest
from auto_mobility.utils.validate_bag import classify_topic, _gap_stats, _stamp_sec


class TestValidateBagUtils(unittest.TestCase):
    def test_classify_topic(self):
        self.assertEqual(classify_topic("/camera/camera/color/image_raw"), "rgb")
        self.assertEqual(classify_topic("/camera/camera/color/image_raw/compressed"), "rgb")
        self.assertEqual(classify_topic("/camera/camera/depth/image_rect_raw"), "depth")
        self.assertEqual(classify_topic("/camera/camera/aligned_depth_to_color/image_raw"), "depth")
        self.assertEqual(classify_topic("/camera/camera/color/camera_info"), "camera_info")
        self.assertEqual(classify_topic("/camera/camera/color/camera_info_windows"), "camera_info")
        self.assertEqual(classify_topic("/camera/camera/imu"), "imu")
        self.assertEqual(classify_topic("/tf_static"), "tf_static")
        self.assertEqual(classify_topic("/rtabmap/odom"), "other")

    def test_gap_stats(self):
        stats = _gap_stats([1.0, 1.033, 1.066, 1.099])
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["monotonic_violations"], 0)
        self.assertAlmostEqual(stats["hz"], 30.0, delta=0.5)
        self.assertGreater(stats["max_gap_s"], 0.03)

    def test_gap_stats_regression(self):
        stats = _gap_stats([1.0, 1.033, 1.5, 1.4])
        self.assertEqual(stats["monotonic_violations"], 1)

    def test_gap_stats_insufficient(self):
        self.assertIsNone(_gap_stats([]))
        self.assertIsNone(_gap_stats([1.0]))

    def test_stamp_sec(self):
        class S:
            sec = 100
            nanosec = 500000000
        self.assertAlmostEqual(_stamp_sec(S()), 100.5)

        class Zero:
            sec = 0
            nanosec = 0
        self.assertIsNone(_stamp_sec(Zero()))
        self.assertIsNone(_stamp_sec(None))


if __name__ == "__main__":
    unittest.main()