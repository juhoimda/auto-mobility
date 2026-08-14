#!/usr/bin/env python3
"""CompressedRepublisher: Windows→WSL 원격 카메라 압축 토픽 디코딩 + 동기 재발행

★ 설계 원칙 (v3 - 완전 재설계):
  - CameraInfo 피드백 루프 완전 제거:
    Windows CameraInfo는 내부 전용 토픽(/republish/cam_info_in)으로 구독
    → 재발행은 표준 토픽(/camera/camera/color/camera_info)으로만
  - MultiThreadedExecutor 사용: 콜백 블로킹 없이 30Hz 타이머 보장
  - 모든 출력 메시지에 동일 로컬 WSL 타임스탬프 적용 → rgbd_sync 확실한 3-way 매칭
  - Windows publisher와 동일 표준 토픽에서 충돌 없도록 Windows 발행 주제를 내부 remapping
"""
import os
import sys
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from cv_bridge import CvBridge
import numpy as np
import cv2
from auto_mobility.config import (
    CAMERA_RGB_TOPIC,
    CAMERA_RGB_COMPRESSED_TOPIC,
    CAMERA_DEPTH_TOPIC,
    CAMERA_DEPTH_COMPRESSED_TOPIC,
    CAMERA_INFO_TOPIC,
)

# Windows realsense_pub.py가 조용히 수신하는 내부 전용 CameraInfo 토픽
# (표준 /camera/camera/color/camera_info와 분리 유지 → 피드백 루프 제거)
CAMERA_INFO_WINDOWS_TOPIC = '/camera/camera/color/camera_info_windows'

SENSOR_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
)


class CompressedRepublisher(Node):
    def __init__(self):
        super().__init__('compressed_republisher')
        self.bridge = CvBridge()

        # ── 파라미터 ──
        self.declare_parameter('rgb_compressed_topic', CAMERA_RGB_COMPRESSED_TOPIC)
        self.declare_parameter('depth_compressed_topic', CAMERA_DEPTH_COMPRESSED_TOPIC)
        # ★ info_in: Windows가 발행하는 내부 전용 토픽 (표준 토픽과 분리)
        self.declare_parameter('camera_info_topic', CAMERA_INFO_WINDOWS_TOPIC)
        self.declare_parameter('rgb_raw_topic', CAMERA_RGB_TOPIC)
        self.declare_parameter('depth_raw_topic', CAMERA_DEPTH_TOPIC)
        self.declare_parameter('publish_tf', False)

        rgb_comp   = self.get_parameter('rgb_compressed_topic').get_parameter_value().string_value
        depth_comp = self.get_parameter('depth_compressed_topic').get_parameter_value().string_value
        info_in    = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        info_out   = CAMERA_INFO_TOPIC  # 표준 /camera/camera/color/camera_info
        rgb_out    = self.get_parameter('rgb_raw_topic').get_parameter_value().string_value
        depth_out  = self.get_parameter('depth_raw_topic').get_parameter_value().string_value
        pub_tf     = self.get_parameter('publish_tf').get_parameter_value().bool_value

        self.get_logger().info(f'RGB:   {rgb_comp} -> {rgb_out}')
        self.get_logger().info(f'Depth: {depth_comp} -> {depth_out}')
        self.get_logger().info(f'Info:  {info_in} (Win) -> {info_out} (local timestamp, no loop)')

        # ── TF Static Broadcaster (camera_tf_pub.py가 전담하므로 기본 비활성화) ──
        self._tf_broadcaster = None
        if pub_tf:
            self._tf_broadcaster = StaticTransformBroadcaster(self)
            self._send_static_tf()

        # ── 공유 버퍼 ──
        self._lock          = threading.Lock()
        self._rgb_img       = None   # cv2 BGR image
        self._depth_img     = None   # cv2 16UC1 image
        self._cam_info      = None   # CameraInfo (캐시 전용)
        self._rgb_seq       = 0
        self._depth_seq     = 0
        self._last_pub_rgb  = -1
        self._last_pub_dep  = -1

        # ── Callback Groups (직렬 순차 실행으로 타임스탬프 역전 및 스레드 경쟁 원천 차단) ──
        self._rgb_cg   = MutuallyExclusiveCallbackGroup()
        self._depth_cg = MutuallyExclusiveCallbackGroup()
        self._info_cg  = MutuallyExclusiveCallbackGroup()
        self._timer_cg = MutuallyExclusiveCallbackGroup()

        # ── 타임스탬프 단조 증가 보장 ──
        self._last_stamp_sec = 0
        self._last_stamp_nanosec = 0

        # ── 구독자 ──
        self.create_subscription(CompressedImage, rgb_comp,   self._on_rgb,   SENSOR_QOS, callback_group=self._rgb_cg)
        self.create_subscription(CompressedImage, depth_comp, self._on_depth, SENSOR_QOS, callback_group=self._depth_cg)
        # ★ Windows 전용 내부 토픽(camera_info_windows) 구독 → 표준 토픽으로 재발행, 피드백루프 없음
        self.create_subscription(CameraInfo, info_in, self._on_info, SENSOR_QOS, callback_group=self._info_cg)
        self._info_out = info_out  # 재발행 대상: 표준 /camera/camera/color/camera_info

        # ── 발행자 ──
        self._pub_rgb   = self.create_publisher(Image,      rgb_out,  SENSOR_QOS)
        self._pub_depth = self.create_publisher(Image,      depth_out, SENSOR_QOS)
        self._pub_info  = self.create_publisher(CameraInfo, info_out,  SENSOR_QOS)

        # ── 30Hz 동기 발행 타이머 ──
        self.create_timer(0.033, self._publish_sync, callback_group=self._timer_cg)

        # ── 5초 상태 보고 타이머 ──
        self._cnt = {'rgb': 0, 'depth': 0, 'info': 0, 'pub': 0}
        self._t0  = time.monotonic()
        self.create_timer(5.0, self._report, callback_group=self._timer_cg)

        self.get_logger().info('CompressedRepublisher v3 Ready (Sequential callbacks, monotonic timestamps)')

    # ── TF ──────────────────────────────────────────────────────────────
    def _send_static_tf(self):
        if self._tf_broadcaster is None:
            return
        now = self.get_clock().now().to_msg()
        tfs = []
        for child in ['camera_color_optical_frame', 'camera_depth_optical_frame', 'camera_imu_optical_frame']:
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = 'camera_link'
            tf.child_frame_id = child
            # ★ 표준 optical 프레임 회전 (rpy(-π/2,0,-π/2), 로컬 realsense2_camera와 동일)
            tf.transform.rotation.x = -0.5
            tf.transform.rotation.y =  0.5
            tf.transform.rotation.z = -0.5
            tf.transform.rotation.w =  0.5
            tfs.append(tf)
        self._tf_broadcaster.sendTransform(tfs)

    # ── 구독 콜백 ────────────────────────────────────────────────────────
    def _on_rgb(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                with self._lock:
                    self._rgb_img = img
                    self._rgb_seq += 1
                self._cnt['rgb'] += 1
        except Exception as e:
            self.get_logger().error(f'RGB decode: {e}')

    def _on_depth(self, msg: CompressedImage):
        try:
            raw = bytes(msg.data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            idx = raw.find(b'\x89PNG')
            img = cv2.imdecode(arr[idx:] if idx != -1 else arr[12:], cv2.IMREAD_UNCHANGED)
            if img is not None:
                with self._lock:
                    self._depth_img = img
                    self._depth_seq += 1
                self._cnt['depth'] += 1
        except Exception as e:
            self.get_logger().error(f'Depth decode: {e}')

    def _on_info(self, msg: CameraInfo):
        # Windows에서 수신하는 CameraInfo: 캐시만 하고 직접 재발행 안 함
        # (피드백 루프 방지: 자신이 발행한 msg는 타임스탬프로 구분)
        with self._lock:
            if self._cam_info is None:
                self.get_logger().info(
                    f'CameraInfo cached: {msg.width}x{msg.height} '
                    f'fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}'
                )
            self._cam_info = msg
        self._cnt['info'] += 1

    # ── 동기 발행 타이머 ────────────────────────────────────────────────
    def _publish_sync(self):
        with self._lock:
            rgb_img   = self._rgb_img
            depth_img = self._depth_img
            cam_info  = self._cam_info
            rs, ds    = self._rgb_seq, self._depth_seq

        if rgb_img is None or depth_img is None:
            return
        # 새 프레임이 없으면 스킵
        if rs == self._last_pub_rgb and ds == self._last_pub_dep:
            return

        # CameraInfo 미수신 시 D435i 기본값 사용 (rgbd_sync가 동작은 하도록)
        if cam_info is None:
            cam_info = self._default_cam_info()

        # ★ 엄격한 단조 증가 타임스탬프 생성 (동일 타임스탬프 및 역순 발행 원천 차단)
        now_msg = self.get_clock().now().to_msg()
        sec = now_msg.sec
        nanosec = now_msg.nanosec
        if (sec < self._last_stamp_sec) or (sec == self._last_stamp_sec and nanosec <= self._last_stamp_nanosec):
            sec = self._last_stamp_sec
            nanosec = self._last_stamp_nanosec + 1000000  # 최소 1ms 증가
            if nanosec >= 1000000000:
                sec += 1
                nanosec -= 1000000000

        self._last_stamp_sec = sec
        self._last_stamp_nanosec = nanosec

        now = now_msg
        now.sec = sec
        now.nanosec = nanosec

        frame_id = 'camera_color_optical_frame'

        # RGB
        rgb_out = self.bridge.cv2_to_imgmsg(
            cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), encoding='rgb8')
        rgb_out.header.stamp    = now
        rgb_out.header.frame_id = frame_id
        self._pub_rgb.publish(rgb_out)

        # Depth: z16(mm, uint16) → 16UC1(mm) 유지
        dep_out = self.bridge.cv2_to_imgmsg(depth_img, encoding='16UC1')
        dep_out.header.stamp    = now
        dep_out.header.frame_id = frame_id
        self._pub_depth.publish(dep_out)

        # CameraInfo
        info_out = CameraInfo()
        info_out.header.stamp    = now
        info_out.header.frame_id = frame_id
        info_out.width  = cam_info.width
        info_out.height = cam_info.height
        info_out.distortion_model = cam_info.distortion_model
        info_out.d = list(cam_info.d)
        info_out.k = list(cam_info.k)
        info_out.r = list(cam_info.r)
        info_out.p = list(cam_info.p)
        info_out.binning_x = cam_info.binning_x
        info_out.binning_y = cam_info.binning_y
        info_out.roi       = cam_info.roi
        self._pub_info.publish(info_out)

        self._last_pub_rgb = rs
        self._last_pub_dep = ds
        self._cnt['pub'] += 1

        # TF 주기적 리프레시
        self._send_static_tf()

    # ── 상태 보고 ────────────────────────────────────────────────────────
    def _report(self):
        dt = time.monotonic() - self._t0
        if dt < 0.1:
            return
        c = self._cnt
        if c['rgb'] > 0 or c['pub'] > 0:
            self.get_logger().info(
                f'[republish] RGB:{c["rgb"]/dt:.1f}Hz Depth:{c["depth"]/dt:.1f}Hz '
                f'Info:{c["info"]/dt:.1f}Hz SyncPub:{c["pub"]/dt:.1f}Hz'
            )
        else:
            self.get_logger().warn('[republish] Windows 카메라 압축 토픽 수신 없음')
        self._cnt = {'rgb': 0, 'depth': 0, 'info': 0, 'pub': 0}
        self._t0  = time.monotonic()

    # ── 기본 카메라 모델 ──────────────────────────────────────────────────
    @staticmethod
    def _default_cam_info() -> CameraInfo:
        info = CameraInfo()
        info.header.frame_id = 'camera_color_optical_frame'
        info.width  = 640
        info.height = 480
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [385.0, 0.0, 320.0, 0.0, 385.0, 240.0, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [385.0, 0.0, 320.0, 0.0, 0.0, 385.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info


def main(args=None):
    rclpy.init(args=args)
    node = CompressedRepublisher()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        try:
            executor.remove_node(node)
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
