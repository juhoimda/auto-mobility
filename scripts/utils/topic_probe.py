#!/usr/bin/env python3
"""토픽 존재/수신 여부를 직접 구독으로 확인한다. (ros2 CLI 데몬 불필요)

단일 토픽 모드:
    topic_probe.py <topic> [timeout_sec]
    exit 0 = 메시지 수신 성공 / exit 1 = 실패

배치 모드:
    topic_probe.py --batch <topic1> <topic2> ... [--timeout <sec>] [--wait-all]
    출력: 각 줄에 "TOPIC:0" (성공) 또는 "TOPIC:1" (실패)
    옵션:
      --timeout <sec>: 전체 타임아웃 (기본값: 3.0초)
      --wait-all: 발견되지 않은 토픽이 있더라도 timeout까지 대기하여 최대한 수집
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
from tf2_msgs.msg import TFMessage

RELIABLE_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
)


def guess_type(topic: str):
    t = topic.lower()
    if t == "/tf_static" or t == "/tf":
        return TFMessage
    if t.endswith('/compressed') or t.endswith('/compresseddepth'):
        return CompressedImage
    # camera_info / camera_info_windows 모두 포함 (suffix 가 '_windows' 일 수 있음)
    if 'camera_info' in t:
        return CameraInfo
    if t.endswith('/imu') or '/imu/' in t:
        return Imu
    if '/odom' in t:
        return Odometry
    return Image


def run_single(topic: str, timeout: float = 3.0) -> int:
    try:
        rclpy.init()
    except Exception:
        pass

    node = Node('topic_probe')
    got = [False]

    def cb(msg):
        got[0] = True

    msg_cls = guess_type(topic)
    try:
        node.create_subscription(msg_cls, topic, cb, RELIABLE_QOS)
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
            rclpy.spin_once(node, timeout_sec=0.05)
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


def run_batch(topics: list, timeout: float = 3.0, wait_all: bool = False):
    if not topics:
        return

    try:
        rclpy.init()
    except Exception:
        pass

    node = Node('topic_probe_batch')
    status = {t: False for t in topics}

    def make_cb(target_topic):
        def _cb(msg):
            status[target_topic] = True
        return _cb

    for t in topics:
        try:
            msg_cls = guess_type(t)
            node.create_subscription(msg_cls, t, make_cb(t), RELIABLE_QOS)
        except Exception:
            pass

    t0 = time.time()
    try:
        while rclpy.ok() and (time.time() - t0 < timeout):
            rclpy.spin_once(node, timeout_sec=0.05)
            # 모든 토픽이 수신되었으면 조기 종료
            if not wait_all and all(status.values()):
                break
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

    for t in topics:
        code = 0 if status[t] else 1
        print(f"{t}:{code}")


def main():
    if len(sys.argv) < 2:
        print("usage: topic_probe.py <topic> [timeout_sec]  OR  topic_probe.py --batch <topic1> <topic2> ... [--timeout <sec>]")
        return 2

    if sys.argv[1] == "--batch":
        args = sys.argv[2:]
        topics = []
        timeout = 3.0
        wait_all = False
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--timeout" and i + 1 < len(args):
                timeout = float(args[i + 1])
                i += 2
            elif arg == "--wait-all":
                wait_all = True
                i += 1
            else:
                topics.append(arg)
                i += 1
        run_batch(topics, timeout=timeout, wait_all=wait_all)
        return 0
    else:
        topic = sys.argv[1]
        timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
        return run_single(topic, timeout=timeout)


if __name__ == '__main__':
    sys.exit(main())
