import unittest
import numpy as np
import cv2
import rclpy
from sensor_msgs.msg import CompressedImage
from auto_mobility.nodes.republish import CompressedRepublisher

class TestRepublishNodeUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = CompressedRepublisher()

    def tearDown(self):
        self.node.destroy_node()

    def test_node_initialization(self):
        self.assertEqual(self.node.get_name(), "compressed_republisher")

    def test_rgb_callback_decoding(self):
        # 10x10 dummy RGB image -> JPEG encode
        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, encoded_img = cv2.imencode('.jpg', dummy_img)
        
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = encoded_img.tobytes()

        # Mock publisher
        published_msgs = []
        self.node._pub_rgb = type('MockPub', (), {'publish': lambda self, m: published_msgs.append(m)})()
        self.node._pub_depth = type('MockPub', (), {'publish': lambda self, m: None})()
        self.node._pub_info = type('MockPub', (), {'publish': lambda self, m: None})()

        self.node._on_rgb(msg)
        # Depth dummy also required for sync publish
        dummy_depth = np.full((10, 10), 1000, dtype=np.uint16)
        _, encoded_png = cv2.imencode('.png', dummy_depth)
        depth_msg = CompressedImage()
        depth_msg.format = "png"
        depth_msg.data = b'\x00' * 12 + encoded_png.tobytes()
        self.node._on_depth(depth_msg)

        self.node._publish_sync()
        self.assertEqual(len(published_msgs), 1)
        self.assertEqual(published_msgs[0].height, 10)
        self.assertEqual(published_msgs[0].width, 10)
        self.assertEqual(published_msgs[0].encoding, "rgb8")

    def test_depth_callback_decoding(self):
        # 10x10 dummy uint16 depth image -> PNG encode
        dummy_depth = np.full((10, 10), 1000, dtype=np.uint16)
        _, encoded_png = cv2.imencode('.png', dummy_depth)
        
        # Ros compressedDepth simulated payload with PNG header magic \x89PNG
        payload = b'\x00' * 12 + encoded_png.tobytes()
        msg = CompressedImage()
        msg.format = "png"
        msg.data = payload

        published_msgs = []
        self.node._pub_rgb = type('MockPub', (), {'publish': lambda self, m: None})()
        self.node._pub_depth = type('MockPub', (), {'publish': lambda self, m: published_msgs.append(m)})()
        self.node._pub_info = type('MockPub', (), {'publish': lambda self, m: None})()

        # RGB dummy also required for sync publish
        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, encoded_img = cv2.imencode('.jpg', dummy_img)
        rgb_msg = CompressedImage()
        rgb_msg.format = "jpeg"
        rgb_msg.data = encoded_img.tobytes()
        self.node._on_rgb(rgb_msg)

        self.node._on_depth(msg)
        self.node._publish_sync()
        self.assertEqual(len(published_msgs), 1)
        self.assertEqual(published_msgs[0].height, 10)
        self.assertEqual(published_msgs[0].width, 10)
        self.assertEqual(published_msgs[0].encoding, "16UC1")

if __name__ == "__main__":
    unittest.main()
