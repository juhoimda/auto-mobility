#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class CompressedRepublisher(Node):
    def __init__(self):
        super().__init__('compressed_republisher')
        self.bridge = CvBridge()
        
        # ROS2 주제(Topic) 파라미터 선언 및 취득
        self.declare_parameter('rgb_compressed_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('depth_compressed_topic', '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth')
        self.declare_parameter('rgb_raw_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_raw_topic', '/camera/camera/aligned_depth_to_color/image_raw')

        rgb_comp = self.get_parameter('rgb_compressed_topic').get_parameter_value().string_value
        depth_comp = self.get_parameter('depth_compressed_topic').get_parameter_value().string_value
        rgb_raw = self.get_parameter('rgb_raw_topic').get_parameter_value().string_value
        depth_raw = self.get_parameter('depth_raw_topic').get_parameter_value().string_value

        self.get_logger().info(f'RGB compressed topic: {rgb_comp} -> {rgb_raw}')
        self.get_logger().info(f'Depth compressed topic: {depth_comp} -> {depth_raw}')

        # RGB 압축 해제 구독/발행
        self.sub_rgb = self.create_subscription(
            CompressedImage,
            rgb_comp,
            self.rgb_callback,
            qos_profile_sensor_data
        )
        self.pub_rgb = self.create_publisher(
            Image,
            rgb_raw,
            10
        )

        # Depth 압축 해제 구독/발행
        self.sub_depth = self.create_subscription(
            CompressedImage,
            depth_comp,
            self.depth_callback,
            qos_profile_sensor_data
        )
        self.pub_depth = self.create_publisher(
            Image,
            depth_raw,
            10
        )
        self.get_logger().info('CompressedRepublisher Node Started (Best Effort QoS & Dynamic Topic Support)')

    def rgb_callback(self, msg: CompressedImage):
        try:
            raw_bytes = bytes(msg.data)
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
                img_msg.header = msg.header
                self.pub_rgb.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'RGB decode error: {e}')

    def depth_callback(self, msg: CompressedImage):
        try:
            raw_bytes = bytes(msg.data)
            np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            
            # ROS compressedDepth PNG 헤더 위치 파악 (b'\x89PNG')
            png_idx = raw_bytes.find(b'\x89PNG')
            if png_idx != -1:
                cv_img = cv2.imdecode(np_arr[png_idx:], cv2.IMREAD_UNCHANGED)
            else:
                cv_img = cv2.imdecode(np_arr[12:], cv2.IMREAD_UNCHANGED)

            if cv_img is not None:
                img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='16UC1')
                img_msg.header = msg.header
                self.pub_depth.publish(img_msg)
            else:
                self.get_logger().error('Depth imdecode returned None')
        except Exception as e:
            self.get_logger().error(f'Depth decode error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CompressedRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
