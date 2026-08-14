#!/usr/bin/env python3
"""토픽 존재/수신 여부를 직접 구독으로 확인한다. (ros2 CLI 데몬 불필요)

이 환경(WSL2 + 미러링)에서 `ros2 topic list` 는 CLI 데몬(graph introspection) 때문에
hang 하는 경우가 있어, 파이프라인 토픽 체크는 이 스크립트로 대체한다.

사용법: topic_probe.py <topic> [timeout_sec]
exit 0 = 메시지 수신 성공
exit 1 = 수신 실패 / 타입 불일치
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu

SENSOR_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
)


def guess_type(topic: str):
    t = topic.lower()
    if t.endswith('/compressed') or t.endswith('/compresseddepth'):
        return CompressedImage
    if t.endswith('/camera_info'):
        return CameraInfo
    if t.endswith('/imu') or '/imu/' in t:
        return Imu
    if '/odom' in t:
        return Odometry
    return Image


def main():
    if len(sys.argv) < 2:
        print("usage: topic_probe.py <topic> [timeout_sec]")
        return 2
    topic = sys.argv[1]
    timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    try:
        rclpy.init()
    except Exception:
        pass

    node = Node('topic_probe')
    got = [False]

    def cb(msg):
        got[0] = True

    try:
        node.create_subscription(guess_type(topic), topic, cb, SENSOR_QOS)
    except Exception:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    t0 = time.time()
    try:
        while rclpy.ok() and not got[0] and (time.time() - t0 < timeout):
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, Exception):
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

    return 0 if got[0] else 1


if __name__ == '__main__':
    sys.exit(main())
