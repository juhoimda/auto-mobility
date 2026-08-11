import unittest
from pathlib import Path
from auto_mobility.config import (
    get_project_root,
    load_topics_config,
    get_topic,
    TOPICS_CONFIG,
    CAMERA_RGB_TOPIC,
    CAMERA_DEPTH_TOPIC,
    CAMERA_IMU_TOPIC,
    CAMERA_IMU_FILTERED_TOPIC,
    CAMERA_ALIGNED_DEPTH_TOPIC,
    CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC,
    CAMERA_INFO_TOPIC,
    ODOM_TOPIC,
    MAP_TOPIC,
    CLOUD_MAP_TOPIC,
    CLOUD_MAP_LITE_TOPIC,
    CAMERA_PARAMS,
    CAMERA_PROFILE,
    CAMERA_RESOLUTION,
    MESH_DEFAULTS,
)

class TestConfigUnit(unittest.TestCase):
    def test_get_project_root(self):
        root = get_project_root()
        self.assertIsInstance(root, Path)
        self.assertTrue((root / "config" / "topics.yaml").exists())

    def test_load_topics_config(self):
        config = load_topics_config()
        self.assertIsInstance(config, dict)
        self.assertIn("camera", config)
        self.assertIn("rtabmap", config)
        self.assertEqual(config["camera"]["depth_topic"], "/camera/camera/depth/image_rect_raw")

    def test_singleton_topics_config(self):
        self.assertIn("camera", TOPICS_CONFIG)
        self.assertEqual(TOPICS_CONFIG["rtabmap"]["frame_id"], "camera_link")

    def test_get_topic(self):
        self.assertEqual(get_topic("camera.rgb_topic"), "/camera/camera/color/image_raw")
        self.assertEqual(get_topic("rtabmap.odom_topic"), "/rtabmap/odom")
        self.assertEqual(get_topic("missing.key", "/fallback"), "/fallback")
        with self.assertRaises(KeyError):
            get_topic("missing.key")

    def test_topic_constants_match_yaml(self):
        self.assertEqual(CAMERA_RGB_TOPIC, TOPICS_CONFIG["camera"]["rgb_topic"])
        self.assertEqual(CAMERA_DEPTH_TOPIC, TOPICS_CONFIG["camera"]["depth_topic"])
        self.assertEqual(CAMERA_IMU_TOPIC, TOPICS_CONFIG["camera"]["imu_topic"])
        self.assertEqual(CAMERA_IMU_FILTERED_TOPIC, TOPICS_CONFIG["camera"]["imu_filtered_topic"])
        self.assertEqual(CAMERA_ALIGNED_DEPTH_TOPIC, TOPICS_CONFIG["camera"]["aligned_depth_topic"])
        self.assertEqual(CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC, TOPICS_CONFIG["camera"]["aligned_depth_compressed_topic"])
        self.assertEqual(CAMERA_INFO_TOPIC, TOPICS_CONFIG["camera"]["camera_info_topic"])
        self.assertEqual(ODOM_TOPIC, TOPICS_CONFIG["rtabmap"]["odom_topic"])
        self.assertEqual(MAP_TOPIC, TOPICS_CONFIG["rtabmap"]["map_topic"])
        self.assertEqual(CLOUD_MAP_TOPIC, TOPICS_CONFIG["rtabmap"]["cloud_map_topic"])
        self.assertEqual(CLOUD_MAP_LITE_TOPIC, TOPICS_CONFIG["rtabmap"]["cloud_map_lite_topic"])

    def test_camera_params_single_source(self):
        self.assertEqual(CAMERA_PROFILE, "640x480x30")
        self.assertEqual(CAMERA_RESOLUTION, "640x480")
        self.assertEqual(CAMERA_PARAMS["depth_module.depth_profile"], CAMERA_PROFILE)
        self.assertEqual(CAMERA_PARAMS["rgb_camera.color_profile"], CAMERA_PROFILE)
        self.assertIs(CAMERA_PARAMS["align_depth.enable"], False)
        self.assertEqual(CAMERA_PARAMS["color_qos"], "SENSOR_DATA")
        # production 파라미터에 필수 키 존재
        for key in ["depth_module.depth_profile", "rgb_camera.color_profile", "enable_sync",
                    "unite_imu_method", "filters", "color_qos"]:
            self.assertIn(key, CAMERA_PARAMS)

    def test_mesh_defaults(self):
        self.assertEqual(MESH_DEFAULTS["depth"], 8)
        self.assertEqual(MESH_DEFAULTS["voxel_size"], 0.005)
        self.assertEqual(MESH_DEFAULTS["method"], "poisson")
        self.assertEqual(MESH_DEFAULTS["simplify_target"], 0.5)

if __name__ == "__main__":
    unittest.main()
