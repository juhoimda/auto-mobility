import unittest
from pathlib import Path
from auto_mobility.config import get_project_root, load_topics_config, TOPICS_CONFIG

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

if __name__ == "__main__":
    unittest.main()
