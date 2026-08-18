#!/usr/bin/env python3
"""
view_camera.py — 저지연 고성능 실시간 카메라 뷰어 (GUI 렌더러)

특징:
  - RGB 전용 고속 렌더링 (CPU 점유율 극소화, 부드러운 60fps GUI)
  - 1.5배(960x720) 큼직하고 시원한 화면 기본 제공 (크기 조절 가능)
  - 백그라운드 멀티스레드 구독 (GUI 렌더링과 메시지 수신 분리)
  - 'q' 또는 ESC 키로 간편 종료
"""

import argparse
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

TRANSPORT_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
)


class CameraViewer(Node):
    def __init__(self, show_depth=False):
        super().__init__('camera_viewer')
        self.show_depth = show_depth
        self.lock = threading.Lock()

        self.raw_rgb_data = None
        self.raw_depth_data = None
        self.rgb_seq = 0
        self.depth_seq = 0

        self.sub_rgb = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self._cb_rgb,
            TRANSPORT_QOS,
        )

        if self.show_depth:
            self.sub_depth = self.create_subscription(
                CompressedImage,
                '/camera/camera/depth/image_rect_raw/compressedDepth',
                self._cb_depth,
                TRANSPORT_QOS,
            )

        mode_str = "RGB + Depth" if self.show_depth else "RGB-Only (High Performance)"
        self.get_logger().info(f"카메라 프리뷰 뷰어 시작 (모드: {mode_str}, 종료: 'q' or ESC)")

    def _cb_rgb(self, msg: CompressedImage):
        with self.lock:
            self.raw_rgb_data = bytes(msg.data)
            self.rgb_seq += 1

    def _cb_depth(self, msg: CompressedImage):
        with self.lock:
            self.raw_depth_data = bytes(msg.data)
            self.depth_seq += 1


def _spin_worker(node):
    try:
        rclpy.spin(node)
    except Exception:
        pass


def decode_rgb(raw_bytes):
    if not raw_bytes:
        return None
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def decode_depth(raw_bytes):
    if not raw_bytes:
        return None
    try:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        idx = raw_bytes.find(b'\x89PNG')
        depth = cv2.imdecode(arr[idx:] if idx != -1 else arr[12:], cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        d_clipped = np.clip(depth, 300, 4000)
        d_scaled = ((d_clipped - 300) * (255.0 / 3700.0)).astype(np.uint8)
        return cv2.applyColorMap(255 - d_scaled, cv2.COLORMAP_JET)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Auto-Mobility Low-Latency Camera Viewer")
    parser.add_argument("--depth", action="store_true", help="Depth 컬러맵 뷰도 함께 표시 (CPU 사용량 약간 증가)")
    parser.add_argument("--scale", type=float, default=1.5, help="화면 확대 배율 (기본: 1.5배 -> 960x720)")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = CameraViewer(show_depth=args.depth)

    spin_thread = threading.Thread(target=_spin_worker, args=(node,), daemon=True)
    spin_thread.start()

    h, w = 480, 640
    disp_w = int(w * args.scale)
    disp_h = int(h * args.scale)

    last_rgb_seq = -1
    last_depth_seq = -1
    cached_rgb_img = None
    cached_depth_img = None

    fps_count = 0
    fps_val = 0.0
    fps_timer = time.time()

    win_name = "Auto-Mobility Camera Live Preview (RGB 1.5x)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w if not args.depth else disp_w * 2, disp_h)

    try:
        while rclpy.ok():
            t_loop_start = time.time()

            with node.lock:
                cur_rgb_seq = node.rgb_seq
                cur_depth_seq = node.depth_seq
                rgb_buf = node.raw_rgb_data
                depth_buf = node.raw_depth_data

            if cur_rgb_seq != last_rgb_seq:
                new_rgb = decode_rgb(rgb_buf)
                if new_rgb is not None:
                    cached_rgb_img = new_rgb
                    last_rgb_seq = cur_rgb_seq

            if args.depth and cur_depth_seq != last_depth_seq:
                new_depth = decode_depth(depth_buf)
                if new_depth is not None:
                    cached_depth_img = new_depth
                    last_depth_seq = cur_depth_seq

            fps_count += 1
            now_t = time.time()
            if now_t - fps_timer >= 1.0:
                fps_val = fps_count / (now_t - fps_timer)
                fps_count = 0
                fps_timer = now_t

            if cached_rgb_img is not None:
                rgb_disp = cached_rgb_img
            else:
                rgb_disp = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(rgb_disp, "Waiting for RGB Camera...", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if args.scale != 1.0:
                rgb_disp = cv2.resize(rgb_disp, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            # OSD 오버레이 (큰 화면에 맞게 선명하게 표시)
            cv2.putText(rgb_disp, f"LIVE RGB ({last_rgb_seq}f, {fps_val:.1f} FPS)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if args.depth:
                if cached_depth_img is not None:
                    depth_disp = cached_depth_img
                else:
                    depth_disp = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(depth_disp, "Waiting for Depth...", (60, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if args.scale != 1.0:
                    depth_disp = cv2.resize(depth_disp, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

                cv2.putText(depth_disp, f"DEPTH ({last_depth_seq}f)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                disp = np.hstack((rgb_disp, depth_disp))
            else:
                disp = rgb_disp

            cv2.imshow(win_name, disp)

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
