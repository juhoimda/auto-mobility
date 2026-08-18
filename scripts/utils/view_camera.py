#!/usr/bin/env python3
"""
view_camera.py — 초경량 비동기 카메라 실시간 뷰어 (Zero-Blocking)
ROS2 콜백에서는 버퍼만 저장하고, 디코딩/렌더링은 GUI 스레드에서 필요한 만큼만 비동기 처리하여
끊김(lag/stuttering)과 CPU 부하를 원천 차단한다.
"""

import sys
import os
import time
import argparse
import threading
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage

from auto_mobility.config import (
    CAMERA_RGB_COMPRESSED_TOPIC,
    CAMERA_DEPTH_COMPRESSED_TOPIC,
)


class CameraViewer(Node):
    def __init__(self, rgb_only=False):
        super().__init__('camera_viewer')
        self.rgb_only = rgb_only

        self.lock = threading.Lock()
        self.raw_rgb_data = None
        self.raw_depth_data = None
        self.rgb_seq = 0
        self.depth_seq = 0

        reliable_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            CompressedImage,
            CAMERA_RGB_COMPRESSED_TOPIC,
            self.cb_rgb,
            reliable_qos
        )

        if not self.rgb_only:
            self.create_subscription(
                CompressedImage,
                CAMERA_DEPTH_COMPRESSED_TOPIC,
                self.cb_depth,
                reliable_qos
            )

        self.get_logger().info(
            f"카메라 프리뷰 뷰어 시작 (모드: {'RGB-Only' if self.rgb_only else 'RGB + Depth'}, 종료: 'q' or ESC)"
        )

    # ── 초고속 Zero-Blocking 콜백: 수신 즉시 버퍼만 갱신 ──
    def cb_rgb(self, msg):
        with self.lock:
            self.raw_rgb_data = msg.data
            self.rgb_seq += 1

    def cb_depth(self, msg):
        with self.lock:
            self.raw_depth_data = msg.data
            self.depth_seq += 1


def _spin_worker(node):
    try:
        rclpy.spin(node)
    except Exception:
        pass


def decode_rgb(raw_data):
    if raw_data is None:
        return None
    try:
        arr = np.frombuffer(raw_data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def decode_depth(raw_data):
    if raw_data is None:
        return None
    try:
        raw = bytes(raw_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        idx = raw.find(b'\x89PNG')
        depth = cv2.imdecode(arr[idx:] if idx != -1 else arr[12:], cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        # 16-bit mm (300mm ~ 4000mm) -> 8-bit 빠른 스케일링
        d_clipped = np.clip(depth, 300, 4000)
        d_scaled = ((d_clipped - 300) * (255.0 / 3700.0)).astype(np.uint8)
        return cv2.applyColorMap(255 - d_scaled, cv2.COLORMAP_JET)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Auto-Mobility Low-Latency Camera Viewer")
    parser.add_argument("--rgb-only", action="store_true", help="RGB 뷰만 표시 (CPU 점유율 극소화)")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = CameraViewer(rgb_only=args.rgb_only)

    spin_thread = threading.Thread(target=_spin_worker, args=(node,), daemon=True)
    spin_thread.start()

    h, w = 480, 640
    last_rgb_seq = -1
    last_depth_seq = -1
    cached_rgb_img = None
    cached_depth_img = None

    fps_count = 0
    fps_val = 0.0
    fps_timer = time.time()

    win_name = "Auto-Mobility Camera Live Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            t_loop_start = time.time()

            # 최신 버퍼 스냅샷 가져오기
            with node.lock:
                cur_rgb_seq = node.rgb_seq
                cur_depth_seq = node.depth_seq
                rgb_buf = node.raw_rgb_data
                depth_buf = node.raw_depth_data

            # 변경이 있을 때만 디코딩 수행 (Skip outdated frames)
            if cur_rgb_seq != last_rgb_seq:
                new_rgb = decode_rgb(rgb_buf)
                if new_rgb is not None:
                    cached_rgb_img = new_rgb
                    last_rgb_seq = cur_rgb_seq

            if not args.rgb_only and cur_depth_seq != last_depth_seq:
                new_depth = decode_depth(depth_buf)
                if new_depth is not None:
                    cached_depth_img = new_depth
                    last_depth_seq = cur_depth_seq

            # FPS 계산
            fps_count += 1
            now_t = time.time()
            if now_t - fps_timer >= 1.0:
                fps_val = fps_count / (now_t - fps_timer)
                fps_count = 0
                fps_timer = now_t

            # 화면 구성
            if cached_rgb_img is not None:
                rgb_disp = cached_rgb_img.copy()
            else:
                rgb_disp = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(rgb_disp, "Waiting for RGB Stream...", (40, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(rgb_disp, f"RGB Live ({last_rgb_seq}f, {fps_val:.1f} FPS)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if not args.rgb_only:
                if cached_depth_img is not None:
                    depth_disp = cached_depth_img.copy()
                else:
                    depth_disp = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(depth_disp, "Waiting for Depth Stream...", (40, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.putText(depth_disp, f"Depth (0.3m-4.0m) ({last_depth_seq}f)", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                disp = np.hstack((rgb_disp, depth_disp))
            else:
                disp = rgb_disp

            cv2.imshow(win_name, disp)

            # ~30 FPS throttling (GUI 부하 방지 및 부드러운 렌더링 유지)
            elapsed = time.time() - t_loop_start
            wait_time = max(1, int((0.033 - elapsed) * 1000))
            key = cv2.waitKey(wait_time) & 0xFF
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
