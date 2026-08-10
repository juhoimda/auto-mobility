#!/usr/bin/env python3
"""
cloud_throttle.py - RTAB-Map cloud_map 저주파 중계 노드

VMware 소프트웨어 렌더링 RViz의 PointCloud2 렌더링 부하를 줄이기 위해
/rtabmap/cloud_map 을 최대 max_rate(기본 2Hz)로 /rtabmap/cloud_map_lite 에 중계한다.
- SLAM 내부 처리(rtabmap 노드)에는 영향이 없다.
- 최신 프레임만 보관 후 타이머로 발행하므로 렌더링이 늦어도 지연이 누적되지 않는다.

사용법:
  ros2 run auto_mobility cloud_throttle.py [--ros-args -p max_rate:=2.0]
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class CloudThrottle(Node):
    def __init__(self):
        super().__init__('cloud_throttle')
        self.declare_parameter('input_topic', '/rtabmap/cloud_map')
        self.declare_parameter('output_topic', '/rtabmap/cloud_map_lite')
        self.declare_parameter('max_rate', 2.0)

        in_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        max_rate = max(1.0, self.get_parameter('max_rate').get_parameter_value().double_value)

        self.latest = None
        self.sub = self.create_subscription(
            PointCloud2, in_topic, self.cb, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(
            PointCloud2, out_topic, qos_profile_sensor_data
        )
        interval = 1.0 / max_rate
        self.timer = self.create_timer(interval, self.publish_latest)
        self.get_logger().info(
            f'Cloud throttle: {in_topic} -> {out_topic} @ max {max_rate:.1f} Hz'
        )

    def cb(self, msg):
        self.latest = msg

    def publish_latest(self):
        if self.latest is not None:
            self.pub.publish(self.latest)


def main(args=None):
    rclpy.init(args=args)
    node = CloudThrottle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
