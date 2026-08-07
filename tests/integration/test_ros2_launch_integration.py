import unittest
import importlib.util
from pathlib import Path
from launch import LaunchDescription
from auto_mobility.config import get_project_root

def load_launch_module(filename: str):
    launch_path = get_project_root() / "launch" / filename
    spec = importlib.util.spec_from_file_location(filename, str(launch_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TestROS2LaunchIntegration(unittest.TestCase):
    def test_camera_launch_description(self):
        mod = load_launch_module("camera.launch.py")
        ld = mod.generate_launch_description()
        self.assertIsInstance(ld, LaunchDescription)
        self.assertGreater(len(ld.entities), 0)

    def test_rtab_live_launch_description(self):
        mod = load_launch_module("rtab_live.launch.py")
        ld = mod.generate_launch_description()
        self.assertIsInstance(ld, LaunchDescription)
        self.assertGreater(len(ld.entities), 0)

    def test_rtab_bag_launch_description(self):
        mod = load_launch_module("rtab_bag.launch.py")
        ld = mod.generate_launch_description()
        self.assertIsInstance(ld, LaunchDescription)
        self.assertGreater(len(ld.entities), 0)

if __name__ == "__main__":
    unittest.main()
