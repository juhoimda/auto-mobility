#!/usr/bin/env python3
"""CompressedRepublisher: Windows→WSL 원격 카메라 압축 토픽 디코딩 + 재발행

★ 설계 원칙 (v4 - timestamp 보존):
  - v3 까지: 수신 시각(WSL 시계)으로 타임스탬프를 재생성 → 캡처 시각 정보 유실
  - v4 (2026-08-14): 실시간 모드에서 Windows 원본 타임스탬프를 보존한다.
    - Windows 발행 stamp + (WSL↔Windows clock offset 추정값) 을 출력 stamp 로 사용
      → 상대 타이밍(프레임 간격)이 캡처 시점 그대로 보존되고, RTAB-Map 에 입력되는
        stamp 의 clock domain 은 WSL 로 정렬된다 (TF lookup 안전).
    - offset 은 수신 시각과 원본 stamp 의 차이의 롤링 중앙값으로 추정 (네트워크 지터 강건).
    - sim_time(오프라인 bag 재생) 모드에서는 /clock 기반 재생성 (기존 동작 유지).
  - CameraInfo: Windows 전용 토픽(/camera/camera/color/camera_info_windows) 구독 → 표준
    토픽(/camera/camera/color/camera_info)으로 재발행 (피드백 루프 없음).
  - 수신 카메라 정보가 없는 동안 기본 intrinsics 를 사용하되 명시적 WARNING 을 반복 출력.
  - MultiThreadedExecutor + CallbackGroup 분리 (v3 유지), 30Hz 동기 발행 타이머 (v3 유지).
"""
import os
import sys
import time
import threading
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from builtin_interfaces.msg import Time as RosTime
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

# Windows realsense_pub.py가 발행하는 내부 전용 CameraInfo 토픽
# (표준 /camera/camera/color/camera_info와 분리 유지 → 피드백 루프 제거)
CAMERA_INFO_WINDOWS_TOPIC = '/camera/camera/color/camera_info_windows'

SENSOR_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
)

# 2026-08-18: Windows publisher 가 RELIABLE 로 전송 → 대형 메시지 무손실 수신.
# (Windows→WSL NAT 경로에서 BE 대형 메시지 ~70% 유실 실측. RELIABLE 은 재전송으로 복구)
TRANSPORT_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
)


def _stamp_sec(msg):
    """header.stamp -> epoch 초 (0 이면 None)."""
    sec = msg.header.stamp.sec
    nsec = msg.header.stamp.nanosec
    if sec <= 0:
        return None
    return sec + nsec * 1e-9


def _sec_to_stamp(sec):
    t = RosTime()
    t.sec = int(sec)
    t.nanosec = int(round((sec - int(sec)) * 1e9))
    if t.nanosec >= 1000000000:
        t.sec += 1
        t.nanosec -= 1000000000
    return t


class ClockOffsetEstimator:
    """Windows stamp -> WSL clock domain 변환 offset 추정.

    offset = WSL_수신시각 - Windows_원본stamp 의 롤링 중앙값.
    샘플 수가 부족하면 None 을 반환 (호출자가 fallback).
    """

    def __init__(self, max_samples=64, min_samples=8):
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.samples = deque()

    def add(self, source_sec, recv_time):
        if source_sec is None or source_sec <= 0:
            return
        self.samples.append(recv_time - source_sec)
        if len(self.samples) > self.max_samples:
            self.samples.popleft()

    def estimate(self):
        if len(self.samples) < self.min_samples:
            return None
        s = sorted(self.samples)
        return s[len(s) // 2]


class MonotonicStamper:
    """스트림별 단조 증가 stamp 생성 (역전/동일 stamp 원천 차단)."""

    def __init__(self):
        self.last = 0.0

    def next(self, sec):
        if sec is not None and sec > self.last:
            self.last = sec
            return sec
        # 역전/정체 → 최소 1ms 뒤로
        self.last += 0.001
        return self.last


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
        # use_sim_time 은 rclpy Node 가 자동 선언하므로 별도 declare 하지 않음

        rgb_comp   = self.get_parameter('rgb_compressed_topic').get_parameter_value().string_value
        depth_comp = self.get_parameter('depth_compressed_topic').get_parameter_value().string_value
        info_in    = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        info_out   = CAMERA_INFO_TOPIC  # 표준 /camera/camera/color/camera_info
        rgb_out    = self.get_parameter('rgb_raw_topic').get_parameter_value().string_value
        depth_out  = self.get_parameter('depth_raw_topic').get_parameter_value().string_value
        pub_tf     = self.get_parameter('publish_tf').get_parameter_value().bool_value
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.get_logger().info(f'RGB:   {rgb_comp} -> {rgb_out}')
        self.get_logger().info(f'Depth: {depth_comp} -> {depth_out}')
        self.get_logger().info(
            f'Info:  {info_in} (Win) -> {info_out} | '
            f'timestamp: {"pass-through+offset" if not self.use_sim_time else "sim_time 재생성"}'
        )

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
        self._rgb_src_stamp = None   # 원본 stamp (epoch 초)
        self._depth_src_stamp = None
        self._info_src_stamp = None

        # ── Callback Groups (직렬 순차 실행으로 타임스탬프 역전 및 스레드 경쟁 원천 차단) ──
        self._rgb_cg   = MutuallyExclusiveCallbackGroup()
        self._depth_cg = MutuallyExclusiveCallbackGroup()
        self._info_cg  = MutuallyExclusiveCallbackGroup()
        self._timer_cg = MutuallyExclusiveCallbackGroup()

        # ── clock offset 추정 (실시간 모드) ──
        self._offset_est = ClockOffsetEstimator()
        # ── 스트림별 단조 증가 (실시간: 보정 stamp, sim: 시계 stamp 공통 적용) ──
        self._rgb_stamper = MonotonicStamper()
        self._depth_stamper = MonotonicStamper()
        self._info_stamper = MonotonicStamper()

        # ── 구독자 ──
        self.create_subscription(CompressedImage, rgb_comp,   self._on_rgb,   TRANSPORT_QOS, callback_group=self._rgb_cg)
        self.create_subscription(CompressedImage, depth_comp, self._on_depth, TRANSPORT_QOS, callback_group=self._depth_cg)
        # ★ Windows 전용 내부 토픽(camera_info_windows) 구독 → 표준 토픽으로 재발행, 피드백루프 없음
        self.create_subscription(CameraInfo, info_in, self._on_info, TRANSPORT_QOS, callback_group=self._info_cg)
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
        self._last_info_warn = 0.0
        self.create_timer(5.0, self._report, callback_group=self._timer_cg)

        self.get_logger().info('CompressedRepublisher v4 Ready (source timestamp preservation)')

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
                src = _stamp_sec(msg)
                self._offset_est.add(src, time.time())
                with self._lock:
                    self._rgb_img = img
                    self._rgb_seq += 1
                    self._rgb_src_stamp = src
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
                src = _stamp_sec(msg)
                self._offset_est.add(src, time.time())
                with self._lock:
                    self._depth_img = img
                    self._depth_seq += 1
                    self._depth_src_stamp = src
                self._cnt['depth'] += 1
        except Exception as e:
            self.get_logger().error(f'Depth decode: {e}')

    def _on_info(self, msg: CameraInfo):
        # Windows에서 수신하는 CameraInfo: 캐시만 하고 직접 재발행 안 함
        # (피드백 루프 방지: 자신이 발행한 msg는 타임스탬프로 구분)
        src = _stamp_sec(msg)
        self._offset_est.add(src, time.time())
        with self._lock:
            if self._cam_info is None:
                self.get_logger().info(
                    f'CameraInfo cached: {msg.width}x{msg.height} '
                    f'fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}'
                )
            self._cam_info = msg
            self._info_src_stamp = src
        self._cnt['info'] += 1

    # ── stamp 결정 ───────────────────────────────────────────────────────
    def _out_stamp(self, src_sec, stamper):
        """실시간 모드: 원본 stamp + clock offset (보정치 없으면 수신 시각 fallback).
        sim_time 모드: /clock 시각 (기존 v3 동작)."""
        if self.use_sim_time:
            return None  # 호출부에서 get_clock().now() 사용
        offset = self._offset_est.estimate()
        if src_sec is not None and offset is not None:
            return stamper.next(src_sec + offset)
        # offset 미수렴(초기 프레임)이거나 원본 stamp 없음 → 수신 시각 fallback
        return stamper.next(time.time())

    # ── 동기 발행 타이머 ────────────────────────────────────────────────
    def _publish_sync(self):
        with self._lock:
            rgb_img   = self._rgb_img
            depth_img = self._depth_img
            cam_info  = self._cam_info
            rs_, ds_  = self._rgb_seq, self._depth_seq
            rgb_src   = self._rgb_src_stamp
            depth_src = self._depth_src_stamp
            info_src  = self._info_src_stamp

        # 스트림별 독립 발행: RGB/Depth 각각 새 프레임이 도착하면 발행.
        # (v4-2: "둘 다 새 프레임"일 때만 발행하면 RELIABLE 재전송 지연으로
        #  발행율이 입력(30Hz)보다 떨어짐을 실측 — RGB/Depth 분리)
        new_rgb   = rgb_img is not None and rs_ != self._last_pub_rgb
        new_depth = depth_img is not None and ds_ != self._last_pub_dep
        if not new_rgb and not new_depth:
            return

        # CameraInfo 미수신 시 D435i 기본값 사용 + 명시적 경고 (30초 1회)
        if cam_info is None:
            now_t = time.time()
            if now_t - self._last_info_warn > 30.0:
                self.get_logger().warn(
                    '[republish] CameraInfo 미수신! 기본 intrinsics(fx=385) 사용 중 '
                    f'({CAMERA_INFO_WINDOWS_TOPIC} 미발행/미기록) — 기하 왜곡 위험'
                )
                self._last_info_warn = now_t
            cam_info = self._default_cam_info()

        # ── 타임스탬프 (실시간: 원본 보존+offset / sim_time: /clock) ──
        if self.use_sim_time:
            clock = self.get_clock().now().to_msg()
            rgb_stamp = clock
            depth_stamp = clock
            info_stamp = clock
        else:
            rgb_stamp = _sec_to_stamp(self._out_stamp(rgb_src, self._rgb_stamper))
            depth_stamp = _sec_to_stamp(self._out_stamp(depth_src, self._depth_stamper))
            info_stamp = _sec_to_stamp(self._out_stamp(info_src, self._info_stamper))

        frame_id = 'camera_color_optical_frame'

        # RGB
        if new_rgb:
            rgb_out = self.bridge.cv2_to_imgmsg(
                cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB), encoding='rgb8')
            rgb_out.header.stamp    = rgb_stamp
            rgb_out.header.frame_id = frame_id
            self._pub_rgb.publish(rgb_out)
            self._last_pub_rgb = rs_

        # Depth: z16(mm, uint16) → 16UC1(mm) 유지
        if new_depth:
            dep_out = self.bridge.cv2_to_imgmsg(depth_img, encoding='16UC1')
            dep_out.header.stamp    = depth_stamp
            dep_out.header.frame_id = frame_id
            self._pub_depth.publish(dep_out)
            self._last_pub_dep = ds_

        # CameraInfo
        info_out = CameraInfo()
        info_out.header.stamp    = info_stamp
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

        self._cnt['pub'] += 1

        # TF 주기적 리프레시
        self._send_static_tf()

    # ── 상태 보고 ────────────────────────────────────────────────────────
    def _report(self):
        dt = time.monotonic() - self._t0
        if dt < 0.1:
            return
        c = self._cnt
        offset = self._offset_est.estimate()
        offset_str = f'{offset * 1000.0:.0f}ms' if offset is not None else 'N/A(수렴 전)'
        if c['rgb'] > 0 or c['pub'] > 0:
            self.get_logger().info(
                f'[republish] RGB:{c["rgb"]/dt:.1f}Hz Depth:{c["depth"]/dt:.1f}Hz '
                f'Info:{c["info"]/dt:.1f}Hz SyncPub:{c["pub"]/dt:.1f}Hz clock_offset={offset_str}'
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
