#!/usr/bin/env python3
"""
view_camera.py — 초고속 비동기 카메라 실시간 뷰어
UI 렌더링 스레드와 ROS2 수신 스레드를 분리하여 30FPS 부드러운 화면 제공
"""

import sys
import os
import time
import threading
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

from auto_mobility.config import (
    CAMERA_RGB_TOPIC,
    CAMERA_RGB_COMPRESSED_TOPIC,
    CAMERA_DEPTH_TOPIC,
    CAMERA_DEPTH_COMPRESSED_TOPIC,
)

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.rgb_cnt = 0
        self.depth_cnt = 0

        reliable_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(CompressedImage, CAMERA_RGB_COMPRESSED_TOPIC, self.cb_rgb_comp, reliable_qos)
        self.create_subscription(CompressedImage, CAMERA_DEPTH_COMPRESSED_TOPIC, self.cb_depth_comp, reliable_qos)
        self.get_logger().info("카메라 프리뷰 뷰어 시작 (종료: 창에서 'q' 또는 ESC)")

    def cb_rgb_comp(self, msg):
        try:
            arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                with self.lock:
                    self.latest_rgb = img
                    self.rgb_cnt += 1
        except Exception:
            pass

    def cb_depth_comp(self, msg):
        try:
            raw = bytes(msg.data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            idx = raw.find(b'\x89PNG')
            depth = cv2.imdecode(arr[idx:] if idx != -1 else arr[12:], cv2.IMREAD_UNCHANGED)
            if depth is not None:
                d_vis = np.clip(depth, 300, 4000)
                d_vis = ((d_vis - 300) / 3700.0 * 255.0).astype(np.uint8)
                d_colormap = cv2.applyColorMap(255 - d_vis, cv2.COLORMAP_JET)
                with self.lock:
                    self.latest_depth = d_colormap
                    self.depth_cnt += 1
        except Exception:
            pass


def _spin_worker(node):
    try:
        rclpy.spin(node)
    except Exception:
        pass


def main():
    rclpy.init()
    node = CameraViewer()

    # ROS2 수신 백그라운드 스레드
    spin_thread = threading.Thread(target=_spin_worker, args=(node,), daemon=True)
    spin_thread.start()

    # 메인 스레드: 60Hz 부드러운 GUI 렌더링
    h, w = 480, 640
    try:
        while rclpy.ok():
            with node.lock:
                rgb_img = node.latest_rgb.copy() if node.latest_rgb is not None else None
                depth_img = node.latest_depth.copy() if node.latest_depth is not None else None
                rgb_c = node.rgb_cnt
                depth_c = node.depth_cnt

            rgb_disp = rgb_img if rgb_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
            depth_disp = depth_img if depth_img is not None else np.zeros((h, w, 3), dtype=np.uint8)

            if rgb_img is None:
                cv2.putText(rgb_disp, "Waiting for RGB Stream...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if depth_img is None:
                cv2.putText(depth_disp, "Waiting for Depth Stream...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(rgb_disp, f"RGB Live ({rgb_c} f)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(depth_disp, f"Depth (0.3m-4.0m) ({depth_c} f)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            combined = np.hstack((rgb_disp, depth_disp))
            cv2.imshow("Auto-Mobility Camera Live Preview", combined)

            key = cv2.waitKey(16) & 0xFF  # ~60 FPS
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
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

