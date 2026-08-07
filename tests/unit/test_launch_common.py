import unittest
from auto_mobility.launch.launch_common import (
    RTABMAP_PARAMS,
    RTABMAP_ARGS,
    create_republish_node,
    create_imu_filter_node,
)
from launch_ros.actions import Node

class TestLaunchCommonUnit(unittest.TestCase):
    def test_rtabmap_params_structure(self):
        self.assertIsInstance(RTABMAP_PARAMS, dict)
        self.assertIn("Vis/MinInliers", RTABMAP_PARAMS)
        self.assertEqual(RTABMAP_PARAMS["Vis/MinInliers"], "10")
        self.assertEqual(RTABMAP_PARAMS["Grid/RayTracing"], "true")

    def test_rtabmap_args_formatting(self):
        self.assertIsInstance(RTABMAP_ARGS, str)
        self.assertIn("--Vis/MinInliers 10", RTABMAP_ARGS)
        self.assertIn("--Grid/RayTracing true", RTABMAP_ARGS)
        # 인자들이 공백으로 명확히 구분되었는지 검증
        args_list = RTABMAP_ARGS.split()
        self.assertIn("--Vis/MinInliers", args_list)
        self.assertIn("10", args_list)

    def test_create_republish_node(self):
        node = create_republish_node(use_sim_time=True, depth_compressed_topic="/custom/depth/compressed")
        self.assertIsInstance(node, Node)
        self.assertEqual(node.node_executable, "republish.py")

    def test_create_imu_filter_node(self):
        node = create_imu_filter_node(use_sim_time=False)
        self.assertIsInstance(node, Node)
        self.assertEqual(node.node_executable, "imu_filter_madgwick_node")

if __name__ == "__main__":
    unittest.main()
