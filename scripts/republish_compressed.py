#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

class CompressedRepublisher(Node):
    def __init__(self):
        super().__init__('compressed_republisher')
        self.bridge = CvBridge()
        
        # RGB 압축 해제 구독/발행
        self.sub_rgb = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.rgb_callback,
            qos_profile_sensor_data
        )
        self.pub_rgb = self.create_publisher(
            Image,
            '/camera/camera/color/image_raw',
            10
        )

        # Depth 압축 해제 구독/발행
        self.sub_depth = self.create_subscription(
            CompressedImage,
            '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth',
            self.depth_callback,
            qos_profile_sensor_data
        )
        self.pub_depth = self.create_publisher(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            10
        )
        self.get_logger().info('CompressedRepublisher Node Started (Best Effort QoS)')

    def rgb_callback(self, msg: CompressedImage):
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            img_msg.header = msg.header
            self.pub_rgb.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'RGB decode error: {e}')

    def depth_callback(self, msg: CompressedImage):
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='16UC1')
            img_msg.header = msg.header
            self.pub_depth.publish(img_msg)
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
