import unittest
from auto_mobility.launch.launch_common import (
    RTABMAP_PARAMS,
    RTABMAP_ARGS,
    RTAB_LIVE_TOPIC_QUEUE_SIZE,
    RTAB_BAG_TOPIC_QUEUE_SIZE,
    get_rtabmap_base_args,
    create_republish_node,
    create_imu_filter_node,
)
from auto_mobility.config import CAMERA_RGB_TOPIC, CAMERA_IMU_FILTERED_TOPIC
from launch_ros.actions import Node

class TestLaunchCommonUnit(unittest.TestCase):
    def test_rtabmap_params_structure(self):
        self.assertIsInstance(RTABMAP_PARAMS, dict)
        self.assertIn("Vis/MinInliers", RTABMAP_PARAMS)
        self.assertEqual(RTABMAP_PARAMS["Vis/MinInliers"], "10")
        self.assertEqual(RTABMAP_PARAMS["Grid/RayTracing"], "false")
        # ★ 2026-08-12 euijin 실측 반영: 루프클로저 보정을 위해 전부 비활성화
        self.assertEqual(RTABMAP_PARAMS["RGBD/OptimizeFromGraphEnd"], "false")
        self.assertEqual(RTABMAP_PARAMS["RGBD/NeighborLinkRefining"], "false")
        self.assertEqual(RTABMAP_PARAMS["RGBD/ProximityBySpace"], "false")

    def test_rtabmap_args_formatting(self):
        self.assertIsInstance(RTABMAP_ARGS, str)
        self.assertIn("--Vis/MinInliers 10", RTABMAP_ARGS)
        self.assertIn("--Grid/RayTracing false", RTABMAP_ARGS)
        # 인자들이 공백으로 명확히 구분되었는지 검증
        args_list = RTABMAP_ARGS.split()
        self.assertIn("--Vis/MinInliers", args_list)
        self.assertIn("10", args_list)

    def test_rtabmap_topic_queue_constants(self):
        # live/bag 은 의도적으로 다른 큐 크기를 쓴다 (단일 소스로 명시 관리)
        self.assertEqual(RTAB_LIVE_TOPIC_QUEUE_SIZE, "50")
        self.assertEqual(RTAB_BAG_TOPIC_QUEUE_SIZE, "30")

    def test_rtabmap_base_args(self):
        base = get_rtabmap_base_args()
        self.assertEqual(base["rgb_topic"], CAMERA_RGB_TOPIC)
        self.assertEqual(base["imu_topic"], CAMERA_IMU_FILTERED_TOPIC)
        self.assertEqual(base["frame_id"], "camera_link")
        self.assertEqual(base["approx_sync"], "true")
        self.assertEqual(base["qos_image"], "2")
        self.assertEqual(base["qos_odom"], "2")
        self.assertEqual(base["approx_sync_max_interval"], "0.05")
        self.assertIn("rviz_cfg", base)

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
