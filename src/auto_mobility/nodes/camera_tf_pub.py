#!/usr/bin/env python3
"""
camera_tf_pub.py - RealSense Optical Frame Static TF Publisher for Remote Camera Mode
WSL2 원격 카메라 모드에서 camera_color_optical_frame 기준 identity static TF를
time=0 (영구 시간 불변)으로 /tf_static에 발행하여 모든 extrapolation 경고를 원천 제거한다.
"""
import sys
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time as MsgTime
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class CameraTfPublisher(Node):
    def __init__(self):
        super().__init__('camera_tf_publisher')
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        # 정적 TF 즉시 발행 (time=0 은 모든 과거/현재/미래 타임스탬프와 영구 매칭)
        self.publish_static_transforms()
        
        # 2초 주기로 지속 리프레시 (뒤늦게 켜지는 노드 대상)
        self.create_timer(2.0, self.publish_static_transforms)
        self.get_logger().info('Camera optical static TF publisher active (time-invariant /tf_static)')

    def publish_static_transforms(self):
        zero_time = MsgTime(sec=0, nanosec=0)
        tfs = []
        pairs = [
            ('camera_link', 'camera_color_optical_frame'),
            ('camera_link', 'camera_depth_optical_frame'),
            ('camera_link', 'camera_imu_optical_frame'),
        ]
        for parent, child in pairs:
            tf = TransformStamped()
            tf.header.stamp = zero_time
            tf.header.frame_id = parent
            tf.child_frame_id = child
            # ★ 표준 optical 프레임 회전 (rpy(-π/2,0,-π/2) = Z-forward/X-right/Y-down)
            #   로컬 realsense2_camera와 동일하게 맞춰 rviz 카메라 시선 정합.
            #   (이전 identity는 optical 축이 링크 축과 같아 카메라 뷰가 90° 어긋남)
            tf.transform.rotation.x = -0.5
            tf.transform.rotation.y =  0.5
            tf.transform.rotation.z = -0.5
            tf.transform.rotation.w =  0.5
            tfs.append(tf)
        self.static_broadcaster.sendTransform(tfs)


def main(args=None):
    rclpy.init(args=args)
    node = CameraTfPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
