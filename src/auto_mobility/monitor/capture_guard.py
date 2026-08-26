#!/usr/bin/env python3
"""
capture_guard.py - 실시간 Visual SLAM 센서 수신율 & 리소스 모니터링
직접 rclpy 구독 노드로 토픽별 수신 Hz를 초경량(CPU < 1%)으로 정밀 측정하고 보고서를 저장한다.
"""
import os
import sys
import time
import json
import signal
import threading
import subprocess
from datetime import datetime
from pathlib import Path

# repo 소스 실행 / 설치 실행 양쪽에서 auto_mobility 패키지 임포트 보장
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu
from nav_msgs.msg import Odometry
from auto_mobility.config import (
    CAMERA_RGB_TOPIC,
    CAMERA_RGB_COMPRESSED_TOPIC,
    CAMERA_DEPTH_TOPIC,
    CAMERA_DEPTH_COMPRESSED_TOPIC,
    CAMERA_INFO_TOPIC,
    CAMERA_INFO_WINDOWS_TOPIC,
    CAMERA_IMU_TOPIC,
    ODOM_TOPIC,
    LOG_DIR,
    USB_3_MIN_SPEED_MBPS,
)

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

MONITOR_TOPICS = [
    (CAMERA_RGB_TOPIC,   15.0, "RGB"),
    (CAMERA_DEPTH_TOPIC, 15.0, "Depth"),
    (CAMERA_INFO_TOPIC,  15.0, "Info"),
    (CAMERA_IMU_TOPIC,  100.0, "IMU"),
    (ODOM_TOPIC,          3.0, "Odom"),
]

# 원격(Windows 카메라) 모드에서는 raw 토픽 대신 압축 토픽을 감시한다.
# republish.py 가 없으면 raw 토픽이 발행되지 않기 때문 (capture-safe 모드).
REMOTE_MONITOR_TOPICS = [
    (CAMERA_RGB_COMPRESSED_TOPIC,   15.0, "RGB"),
    (CAMERA_DEPTH_COMPRESSED_TOPIC, 15.0, "Depth"),
    (CAMERA_INFO_WINDOWS_TOPIC,     15.0, "Info"),
    (CAMERA_IMU_TOPIC,             100.0, "IMU"),
    (ODOM_TOPIC,                     3.0, "Odom"),
]

GUARD_PROCS = [
    "realsense2_came",
    "rgbd_odometry",
    "rtabmap",
    "rviz2",
    "cloud_throttle",
    "imu_filter_madgwick",
    "camera_tf_pub",
]

SENSOR_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
)

RELIABLE_QOS = QoSProfile(
    depth=5,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
)


class TopicRateMonitor(Node):
    """모든 모니터링 토픽을 단일 노드에서 초경량 카운팅 및 IMU 모션 분석

    추가 진단 항목:
      - 토픽별 최대 프레임 gap (타임스탬프 기반): 대규모 timestamp 공백 감지
      - RGB↔Depth sync delta (같은 구간 내 최신 stamp 차이): 센서 동기화 품질 측정
      - IMU 3축 각속도(Gyro norm, deg/s): 급회전/모션 블러/SLAM Lost 위험 실시간 감지
    """
    def __init__(self, topics):
        super().__init__('capture_guard_monitor')
        self._counts = {t[0]: 0 for t in topics}
        self._lock = threading.Lock()
        self._last_stamp = {}
        self._max_gap = {t[0]: 0.0 for t in topics}
        self._rgb_topic = None
        self._depth_topic = None
        self._imu_topic = None
        self._sync_deltas = []
        self._gyro_samples = []
        self._max_gyro = 0.0

        for t, _, label in topics:
            if label == "RGB":
                self._rgb_topic = t
            elif label == "Depth":
                self._depth_topic = t
            elif label == "IMU":
                self._imu_topic = t

        for topic, _, label in topics:
            if 'camera_info' in topic:
                msg_type = CameraInfo
                qos = RELIABLE_QOS
            elif 'imu' in topic:
                msg_type = Imu
                qos = SENSOR_QOS
            elif 'odom' in topic:
                msg_type = Odometry
                qos = SENSOR_QOS
            elif 'compressed' in topic.lower() or 'compresseddepth' in topic.lower():
                msg_type = CompressedImage
                qos = RELIABLE_QOS
            else:
                msg_type = Image
                qos = SENSOR_QOS

            def make_cb(t_name):
                def _cb(msg):
                    with self._lock:
                        self._counts[t_name] += 1
                        stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
                        if stamp is not None and stamp.sec > 0:
                            sec = stamp.sec + stamp.nanosec * 1e-9
                            prev = self._last_stamp.get(t_name)
                            if prev is not None:
                                gap = sec - prev
                                if 0 < gap < 60.0:
                                    self._max_gap[t_name] = max(self._max_gap[t_name], gap)
                            self._last_stamp[t_name] = sec

                            # 프레임 단위 연속 동기화 시차 측정
                            if t_name == self._rgb_topic and self._depth_topic in self._last_stamp:
                                d_ts = self._last_stamp[self._depth_topic]
                                self._sync_deltas.append(abs(sec - d_ts))
                            elif t_name == self._depth_topic and self._rgb_topic in self._last_stamp:
                                r_ts = self._last_stamp[self._rgb_topic]
                                self._sync_deltas.append(abs(sec - r_ts))

                        # IMU 각속도(Gyro) 실시간 측정 (deg/s)
                        if t_name == self._imu_topic and hasattr(msg, 'angular_velocity'):
                            try:
                                gx = float(msg.angular_velocity.x)
                                gy = float(msg.angular_velocity.y)
                                gz = float(msg.angular_velocity.z)
                                norm_rad_s = (gx*gx + gy*gy + gz*gz) ** 0.5
                                deg_s = norm_rad_s * 57.29577951308232  # 180 / pi
                                self._gyro_samples.append(deg_s)
                                if deg_s > self._max_gyro:
                                    self._max_gyro = deg_s
                            except Exception:
                                pass
                return _cb

            self.create_subscription(msg_type, topic, make_cb(topic), qos)

    def get_and_reset(self, dt: float) -> tuple:
        with self._lock:
            rates = {}
            gaps = {}
            for t, count in self._counts.items():
                rates[t] = count / dt if dt > 0 else 0.0
                self._counts[t] = 0
                gaps[t] = self._max_gap.get(t, 0.0)
                self._max_gap[t] = 0.0

            # RGB↔Depth sync delta (연속 구간 평균치 계산)
            sync_delta = None
            if self._sync_deltas:
                sync_delta = sum(self._sync_deltas) / len(self._sync_deltas)
                self._sync_deltas = []
            elif self._rgb_topic and self._depth_topic:
                rgb_ts = self._last_stamp.get(self._rgb_topic)
                depth_ts = self._last_stamp.get(self._depth_topic)
                if rgb_ts and depth_ts:
                    sync_delta = abs(rgb_ts - depth_ts)

            # IMU 각속도 통계
            gyro_avg = (sum(self._gyro_samples) / len(self._gyro_samples)) if self._gyro_samples else 0.0
            gyro_max = self._max_gyro
            self._gyro_samples = []
            self._max_gyro = 0.0

            return rates, gaps, sync_delta, gyro_avg, gyro_max


RELQOS_NAME = "RELIABLE"


def emit(level: str, msg: str):
    """터미널 실시간 알림 출력 (즉시 flush, 비프음 지원)."""
    ts = datetime.now().strftime('%H:%M:%S')
    beep = "\a" if level == "CRITICAL" else ""
    if level == "CRITICAL":
        color = f"{RED}{BOLD}"
    elif level == "WARN":
        color = f"{YELLOW}{BOLD}"
    elif level == "OK":
        color = f"{GREEN}{BOLD}"
    else:
        color = CYAN
    print(f"{beep}{color}[capture_guard {level}] [{ts}] {msg}{RESET}", flush=True)


# 토픽별 알림 규칙: (저하 판정 비율 또는 None=중단만 감시, 선택 토픽 여부)
# 선택(Odom 등) 토픽은 한 번이라도 수신된 적이 있을 때만 중단을 감지한다.
ALERT_RULES = {
    "RGB":   (0.7, False),
    "Depth": (0.7, False),
    "Info":  (None, False),
    "IMU":   (0.5, False),
    "Odom":  (None, True),
}

GAP_ALERT_SEC = 0.4             # 프레임 타임스탬프 공백 임계 (400ms 이상 지연 시 경고)
SYNC_ALERT_SEC = 0.05           # RGB↔Depth 동기 오차 임계
WARN_ROTATION_DEG_S = 45.0      # Visual SLAM 모션 블러 / 특징점 유실 주의 임계치 (deg/s)
CRITICAL_ROTATION_DEG_S = 75.0  # 급격한 회전으로 인한 Visual Lost 위험 임계치 (deg/s)
RECOVER_ROTATION_DEG_S = 30.0   # 회전 속도 정상화 기준 (deg/s)
DEGRADED_REWARN_INTERVALS = 4   # 저하 상태 재경고 주기 (interval 배수)
DOWN_REMIND_INTERVALS = 4       # 중단 상태 리마인드 주기


class AlertManager:
    """구간별 수신율 샘플 및 IMU 각속도에서 이상 상태 전이를 감지하여 즉시 알린다.

    상태: init → ok / degraded / down / never(유예 후에도 무수신)
    - down/never 진입 및 복구 시 1회 알림, 지속 중 주기적 리마인드
    - degraded 진입 시 1회 알림, 지속 중 주기적 재경고
    - IMU 급회전(>45°/s, >75°/s) 감지 및 회전 안정화 복구 알림
    """

    def __init__(self, interval: float, grace_sec: float = 15.0,
                 down_confirm: int = 2):
        self.interval = interval
        self.grace_sec = max(grace_sec, 2 * interval)
        self.down_confirm = max(1, down_confirm)
        self.ever_received = {}
        self.state = {}
        self.zero_streak = {}
        self.degraded_streak = {}
        self.down_streak = {}
        self.down_since = {}
        self.last_gap_alert_val = {}
        self.last_gap_alert_time = {}
        self.last_sync_alert_time = -1e9
        self.rotation_state = "ok"   # "ok", "warn", "critical"
        self.last_rot_alert_time = -1e9

    def update(self, sample: dict, monitor_topics) -> list:
        msgs = []
        elapsed = sample["elapsed"]

        # ── 1. 센서 스트림 수신율 및 중단/복구 검사 ──
        for t, min_hz, label in monitor_topics:
            rate = sample["rates"].get(t, 0.0)
            if rate > 0:
                self.ever_received[t] = True
            ratio, optional = ALERT_RULES.get(label, (0.7, False))
            prev = self.state.get(t, "init")

            if rate == 0.0:
                self.zero_streak[t] = self.zero_streak.get(t, 0) + 1
                if not self.ever_received.get(t):
                    if not optional and elapsed > self.grace_sec and prev != "never":
                        self.state[t] = "never"
                        msgs.append(("CRITICAL",
                                     f"{label}: 토픽 수신 자체가 없습니다 — 카메라 연결/토픽 확인 필요!"))
                    continue
                # 연속 확인 후 down 확정
                if prev not in ("down",):
                    if self.zero_streak[t] >= self.down_confirm:
                        self.state[t] = "down"
                        self.down_since[t] = elapsed - (self.zero_streak[t] - 1) * self.interval
                        self.down_streak[t] = 0
                        msgs.append(("CRITICAL", f"🚨 [{label} 스트림 중단] 0 Hz 감지 — 카메라 연결 또는 통신 끊김! (촬영 일시정지 권장)"))
                else:
                    self.down_streak[t] = self.down_streak.get(t, 0) + 1
                    n = self.down_streak[t]
                    if n % DOWN_REMIND_INTERVALS == 0:
                        dur = elapsed - self.down_since.get(t, elapsed)
                        msgs.append(("CRITICAL", f"🚨 [{label} 중단 지속] {dur:.0f}초째 데이터 없음 — 케이블/드라이버 확인 필요"))
                continue

            # ── rate > 0 (정상 또는 복구) ──
            self.zero_streak[t] = 0
            if prev == "down" or prev == "never":
                dur = elapsed - self.down_since.get(t, elapsed) if prev == "down" else elapsed
                msgs.append(("OK", f"🟢 [{label} 정상 복구] 스트림 수신 재개 ({rate:.1f} Hz, 장애 {dur:.1f}초) — 촬영을 계속 진행하세요."))
                self.state[t] = "ok"
                self.degraded_streak[t] = 0
            elif prev == "degraded":
                if ratio is not None and rate < min_hz * ratio:
                    self.degraded_streak[t] = self.degraded_streak.get(t, 0) + 1
                    n = self.degraded_streak[t]
                    if n % DEGRADED_REWARN_INTERVALS == 0:
                        msgs.append(("WARN",
                                     f"⚠️ [{label} 수신율 저하 지속] {rate:.1f} Hz (권장 {min_hz:.0f} Hz 이상)"))
                else:
                    msgs.append(("OK", f"🟢 [{label} 수신율 정상화] {rate:.1f} Hz"))
                    self.state[t] = "ok"
                    self.degraded_streak[t] = 0
            else:
                if ratio is not None and rate < min_hz * ratio:
                    self.state[t] = "degraded"
                    self.degraded_streak[t] = 1
                    msgs.append(("WARN",
                                 f"⚠️ [{label} 수신율 저하] {rate:.1f} Hz (권장 {min_hz:.0f} Hz 이상)"))
                else:
                    self.state[t] = "ok"

        # ── 2. 프레임 타임스탬프 공백 (일시적 프레임 드롭) ──
        for t, _, label in monitor_topics:
            if ALERT_RULES.get(label, (0.7, False))[1]:
                continue
            if not self.ever_received.get(t):
                continue
            g = sample["max_gaps"].get(t, 0.0)
            last_v = self.last_gap_alert_val.get(label, 0.0)
            if g > GAP_ALERT_SEC and g > last_v * 1.5 \
                    and elapsed - self.last_gap_alert_time.get(label, -1e9) > DEGRADED_REWARN_INTERVALS * self.interval / 2:
                msgs.append(("WARN", f"⚠️ [{label} 프레임 공백] {g*1000:.0f} ms 간격 지연 발생 (일시적 프레임 유실)"))
                self.last_gap_alert_val[label] = g
                self.last_gap_alert_time[label] = elapsed

        # ── 3. RGB↔Depth 동기 오차 ──
        sd = sample.get("sync_delta")
        if sd is not None and sd > SYNC_ALERT_SEC \
                and elapsed - self.last_sync_alert_time > 60.0:
            msgs.append(("WARN", f"⚠️ [RGB↔Depth 동기 오차] {sd*1000:.0f} ms — 3D 복원 품질에 영향 가능"))
            self.last_sync_alert_time = elapsed

        # ── 4. IMU 기반 카메라 급회전 / 속도 경고 (SLAM 끊김 방지) ──
        gyro_max = sample.get("gyro_max", 0.0)
        if gyro_max >= CRITICAL_ROTATION_DEG_S:
            if self.rotation_state != "critical" or (elapsed - self.last_rot_alert_time > 4.0):
                self.rotation_state = "critical"
                self.last_rot_alert_time = elapsed
                msgs.append(("CRITICAL",
                             f"🚨 [급회전 심각] 카메라 회전이 너무 빠릅니다 ({gyro_max:.0f}°/s > {CRITICAL_ROTATION_DEG_S:.0f}°/s)! SLAM 궤적이 끊길 위험이 큽니다. 즉시 회전 속도를 늦추세요!"))
        elif gyro_max >= WARN_ROTATION_DEG_S:
            if self.rotation_state == "ok" or (elapsed - self.last_rot_alert_time > 5.0 and self.rotation_state == "warn"):
                self.rotation_state = "warn"
                self.last_rot_alert_time = elapsed
                msgs.append(("WARN",
                             f"⚠️ [급회전 주의] 카메라 회전 속도({gyro_max:.0f}°/s > {WARN_ROTATION_DEG_S:.0f}°/s)가 빠릅니다! 특징점 보존을 위해 천천히 회전하세요."))
        elif self.rotation_state in ("warn", "critical") and gyro_max <= RECOVER_ROTATION_DEG_S:
            self.rotation_state = "ok"
            msgs.append(("OK",
                         f"✅ [회전 안정화] 카메라 회전이 정상 속도로 복귀했습니다 ({gyro_max:.0f}°/s) — 원활한 궤적 추적이 유지됩니다."))

        return msgs


class BagWatcher:
    """녹화 중인 bag 디렉터리의 파일 크기 성장을 감시한다.

    - 생성 지연/미생성 경고
    - 크기 불변(정지) 감지 및 재개 통보
    - 실시간 저장 속도(MB/s) 및 누적 크기 계산
    """

    def __init__(self, name: str, parents, interval: float,
                 missing_grace_intervals: int = 4, stall_intervals: int = 2):
        self.dirs = [Path(p) / name for p in parents if p]
        self.interval = interval
        self.missing_grace = missing_grace_intervals
        self.stall_threshold = stall_intervals
        self.size = None
        self.write_rate_mb_s = 0.0
        self.current_size_mb = 0.0
        self.stall = 0
        self.miss_count = 0
        self.warned_missing = False

    def _dir_size(self, d) -> int:
        total = 0
        try:
            for f in d.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def check(self):
        target = next((d for d in self.dirs if d.exists()), None)
        if target is None:
            self.miss_count += 1
            if not self.warned_missing and self.miss_count >= self.missing_grace:
                self.warned_missing = True
                return ("WARN", "⚠️ bag 녹화 파일이 아직 생성되지 않았습니다 (저장 경로/디스크 확인)")
            return None

        size = self._dir_size(target)
        self.current_size_mb = size / 1e6

        if self.size is None:
            if size == 0:
                return None
            self.size = size
            return ("INFO", f"📦 녹화 파일 생성 확인 ({self.current_size_mb:.1f} MB): {target}")

        if size > self.size:
            resumed = self.stall >= self.stall_threshold
            delta = size - self.size
            self.write_rate_mb_s = (delta / 1e6) / self.interval if self.interval > 0 else 0.0
            self.stall = 0
            self.size = size
            if resumed:
                return ("OK", f"🟢 [녹화 재개] bag 파일이 다시 정상 기록되고 있습니다 (+{delta/1e6:.1f} MB)")
            return None

        self.stall += 1
        self.write_rate_mb_s = 0.0
        frozen = int(self.stall * self.interval)
        if self.stall == self.stall_threshold:
            return ("CRITICAL",
                    f"🚨 [녹화 정지] bag 크기가 {frozen}초간 증가하지 않습니다! (recorder/디스크 확인 필요)")
        if self.stall > self.stall_threshold and self.stall % DOWN_REMIND_INTERVALS == 0:
            return ("CRITICAL", f"🚨 [녹화 정지 지속] {frozen}초간 데이터 미저장")
        return None

    def get_summary(self) -> str:
        if self.size is None or self.current_size_mb == 0:
            return "Bag 대기 중..."
        if self.stall >= self.stall_threshold:
            return f"Bag {RED}정지됨{RESET} ({self.current_size_mb:.1f}MB)"
        return f"Bag: +{self.write_rate_mb_s:.1f}MB/s (총 {self.current_size_mb:.0f}MB)"


def get_proc_cpu_ram(proc_names):
    """프로세스별 CPU(%)/전체 RAM(MB) 을 ps 1회 실행으로 수집한다."""
    proc_cpu = {name: 0.0 for name in proc_names}
    try:
        r = subprocess.run(["ps", "-eo", "comm=,%cpu=,rss="],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        total_rss_kb = 0
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            comm, cpu, rss = parts
            try:
                total_rss_kb += int(rss)
                for name in proc_names:
                    if name in comm:
                        proc_cpu[name] = max(proc_cpu[name], float(cpu))
            except ValueError:
                pass
        ram_mb = total_rss_kb / 1024.0
        return proc_cpu, ram_mb
    except Exception:
        return proc_cpu, 0.0


def get_usb_status():
    """RealSense USB 링크 속도를 확인한다."""
    usb_dir = "/sys/bus/usb/devices"
    if not os.path.exists(usb_dir):
        return "원격 모드(Windows)"
    for dev in os.listdir(usb_dir):
        path = f"{usb_dir}/{dev}/idVendor"
        try:
            if os.path.exists(path) and open(path).read().strip() == "8086":
                speed = open(f"{usb_dir}/{dev}/speed").read().strip()
                return f"{speed} Mbps (USB {'3.x' if int(speed) >= USB_3_MIN_SPEED_MBPS else '2.x'})"
        except Exception:
            pass
    return "미감지"


def build_md(samples, usb, duration, monitor_topics):
    """Markdown 보고서 생성"""
    topic_cols = [label for _, _, label in monitor_topics]
    step = max(1, len(samples) // 20)
    rows = samples[::step]

    md = f"""# 🎥 촬영 품질 모니터링 보고서 (capture_guard)

- **측정 시각**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **총 촬영 시간**: {duration:.1f}초 (총 샘플: {len(samples)}회)
- **USB 연결 상태**: {usb}

---

## 1. 토픽별 평균 수신 속도 (Hz)

| 시간(초) | {' | '.join(topic_cols)} | CPU(%) | RAM(MB) |
|---|{'---|' * len(topic_cols)}---|---|
"""
    for s in rows:
        rates_str = ' | '.join(f"{s['rates'].get(t[0], 0.0):.1f}" for t in monitor_topics)
        md += f"| {s['elapsed']:.1f} | {rates_str} | {s['cpu_total']:.1f} | {s['ram_mb']:.0f} |\n"

    avg_rates = {}
    for t, min_hz, label in monitor_topics:
        vals = [s['rates'].get(t, 0.0) for s in samples]
        avg_rates[label] = (sum(vals) / len(vals)) if vals else 0.0

    avg_cpu = sum(s['cpu_total'] for s in samples) / len(samples) if samples else 0.0
    avg_ram = sum(s['ram_mb'] for s in samples) / len(samples) if samples else 0.0

    # 최대 프레임 gap (토픽별)
    max_gaps = {}
    for t, _, label in monitor_topics:
        max_gaps[label] = max((s['max_gaps'].get(t, 0.0) for s in samples), default=0.0)

    # RGB↔Depth sync delta 통계
    deltas = [s['sync_delta'] for s in samples if s.get('sync_delta')]
    sync_summary = "N/A (RGB/Depth 수신 없음)"
    if deltas:
        d = sorted(deltas)
        n = len(d)
        sync_summary = (f"mean {sum(d) / n * 1000:.1f}ms | "
                        f"p95 {d[int(n * 0.95) - 1] * 1000:.1f}ms | "
                        f"max {d[-1] * 1000:.1f}ms")

    # 카메라 회전 속도 (IMU Gyro) 통계
    max_gyro_session = max((s.get('gyro_max', 0.0) for s in samples), default=0.0)
    avg_gyro_session = (sum(s.get('gyro_avg', 0.0) for s in samples) / len(samples)) if samples else 0.0
    gyro_status = "✅ 안정 (<45°/s)" if max_gyro_session < 45.0 else (
        "⚠️ 급회전 주의 (>45°/s)" if max_gyro_session < 75.0 else "❌ 급회전 심각 (>75°/s)"
    )

    md += f"""
---

## 2. 세션 통계 요약

| 항목 | 측정값 | 상태 |
|---|---|---|
"""
    for t, min_hz, label in monitor_topics:
        hz = avg_rates[label]
        status = "✅ 정상" if hz >= min_hz * 0.8 else "⚠️ 저하"
        md += f"| **{label} 평균 Hz** | {hz:.1f} Hz (기준: {min_hz:.0f} Hz) | {status} |\n"

    md += f"""| **RGB↔Depth sync delta** | {sync_summary} | - |
"""
    for label, gap in max_gaps.items():
        md += f"| **{label} 최대 프레임 gap** | {gap * 1000:.0f} ms | {'✅' if gap < 0.5 else '⚠️'} |\n"

    md += f"""| **카메라 최대 회전속도** | {max_gyro_session:.0f}°/s (평균: {avg_gyro_session:.1f}°/s) | {gyro_status} |
| **평균 CPU 사용률** | {avg_cpu:.1f} % | - |
| **평균 RAM 사용량** | {avg_ram:.0f} MB | - |
"""
    return md


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=2.0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--live-status', action='store_true', default=True,
                        help='실시간 1줄 상태 대시보드(FPS, 회전속도, 녹화용량) 출력')
    parser.add_argument('--no-live-status', dest='live_status', action='store_false')
    parser.add_argument('--report', type=str, default=None)
    parser.add_argument('--json', type=str, default=None)
    parser.add_argument('--remote', action='store_true',
                        help='원격(Windows 카메라) 모드: raw 대신 압축 토픽 감시')
    parser.add_argument('--alerts', action='store_true',
                        help='터미널 실시간 이상 알림 활성화 (카메라 끊김/저하/급회전/bag 정지)')
    parser.add_argument('--grace', type=float, default=15.0,
                        help='초기 수신 유예 시간(초). 이 후에도 무수신이면 경보')
    parser.add_argument('--bag-name', type=str, default=None,
                        help='성장 감시할 bag 디렉터리 이름')
    parser.add_argument('--bag-parents', type=str, default=None,
                        help='bag 후보 부모 디렉터리 (쉼표 구분, 예: RAM디스크,SSD)')
    args = parser.parse_args()

    monitor_topics = REMOTE_MONITOR_TOPICS if args.remote else MONITOR_TOPICS

    alert_mgr = AlertManager(args.interval, grace_sec=args.grace) if args.alerts else None
    bag_watcher = None
    if args.alerts and args.bag_name and args.bag_parents:
        parents = [p.strip() for p in args.bag_parents.split(',') if p.strip()]
        bag_watcher = BagWatcher(args.bag_name, parents, args.interval)

    try:
        rclpy.init()
    except Exception:
        pass

    monitor_node = TopicRateMonitor(monitor_topics)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(monitor_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    samples = []
    t_start = time.time()
    t_last = t_start
    usb = get_usb_status()

    stop_event = threading.Event()

    def _sig_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    if args.alerts:
        emit("INFO", f"실시간 모니터링 활성화 (주기: {args.interval:.1f}초, 유예: {alert_mgr.grace_sec:.0f}초"
             + (", bag 감시: ON" if bag_watcher else "")
             + ", 급회전 감지: >45°/s)")

    rgb_t = next((t for t, _, label in monitor_topics if label == "RGB"), None)
    depth_t = next((t for t, _, label in monitor_topics if label == "Depth"), None)
    imu_t = next((t for t, _, label in monitor_topics if label == "IMU"), None)

    try:
        while not stop_event.is_set():
            if stop_event.wait(timeout=args.interval):
                break
            now = time.time()
            dt = now - t_last
            t_last = now

            rates, gaps, sync_delta, gyro_avg, gyro_max = monitor_node.get_and_reset(dt)
            proc_cpu, ram_mb = get_proc_cpu_ram(GUARD_PROCS)
            cpu_total = sum(proc_cpu.values())

            sample = {
                "elapsed": now - t_start,
                "rates": rates,
                "max_gaps": gaps,
                "sync_delta": sync_delta,
                "gyro_avg": gyro_avg,
                "gyro_max": gyro_max,
                "cpu_total": cpu_total,
                "ram_mb": ram_mb,
            }
            samples.append(sample)

            # ── 이상 감지 알림 출력 ──
            if args.alerts:
                for level, msg in alert_mgr.update(sample, monitor_topics):
                    emit(level, msg)
                if bag_watcher is not None:
                    result = bag_watcher.check()
                    if result is not None:
                        emit(*result)

            # ── 실시간 1줄 상태 대시보드 출력 ──
            if args.live_status:
                rgb_rate = rates.get(rgb_t, 0.0) if rgb_t else 0.0
                depth_rate = rates.get(depth_t, 0.0) if depth_t else 0.0
                imu_rate = rates.get(imu_t, 0.0) if imu_t else 0.0

                has_critical = (alert_mgr and any(s in ("down", "never") for s in alert_mgr.state.values())) or \
                               (bag_watcher and bag_watcher.stall >= bag_watcher.stall_threshold) or \
                               (gyro_max >= CRITICAL_ROTATION_DEG_S)
                has_warn = (gyro_max >= WARN_ROTATION_DEG_S) or \
                           (alert_mgr and any(s == "degraded" for s in alert_mgr.state.values()))

                if has_critical:
                    badge = f"{RED}{BOLD}🔴 [스트림/녹화 중단]{RESET}" if (gyro_max < CRITICAL_ROTATION_DEG_S) else f"{RED}{BOLD}🚨 [급회전 심각]{RESET}"
                elif has_warn:
                    badge = f"{YELLOW}{BOLD}🟡 [주의: 급회전]{RESET}" if (gyro_max >= WARN_ROTATION_DEG_S) else f"{YELLOW}{BOLD}🟡 [수신율 저하]{RESET}"
                else:
                    badge = f"{GREEN}{BOLD}🟢 [녹화 정상]{RESET}"

                gyro_txt = f"회전: {gyro_max:.0f}°/s"
                if gyro_max >= CRITICAL_ROTATION_DEG_S:
                    gyro_txt = f"{RED}{BOLD}{gyro_txt} (속도 낮추세요!){RESET}"
                elif gyro_max >= WARN_ROTATION_DEG_S:
                    gyro_txt = f"{YELLOW}{BOLD}{gyro_txt} ⚠️{RESET}"
                else:
                    gyro_txt = f"{gyro_txt} (안정)"

                bag_txt = f" | {bag_watcher.get_summary()}" if bag_watcher else ""
                ts = datetime.now().strftime('%H:%M:%S')
                print(f"[{ts}] {badge} RGB: {rgb_rate:.1f}fps | Depth: {depth_rate:.1f}fps | IMU: {imu_rate:.0f}Hz | {gyro_txt}{bag_txt}", flush=True)

            elif not args.headless:
                rates_disp = " | ".join(f"{label}: {rates.get(t, 0.0):.1f}Hz" for t, _, label in monitor_topics)
                print(f"[capture_guard] {rates_disp} | CPU: {cpu_total:.1f}% | RAM: {ram_mb:.0f}MB", flush=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        duration = time.time() - t_start
        try:
            executor.shutdown()
            monitor_node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

        if samples:
            if args.report:
                os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
                with open(args.report, "w", encoding="utf-8") as f:
                    f.write(build_md(samples, usb, duration, monitor_topics))
                print(f"📄 보고서 저장: {args.report}")

            if args.json:
                os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
                with open(args.json, "w", encoding="utf-8") as f:
                    json.dump({"samples": samples, "usb": usb, "duration": duration}, f, indent=2)
                print(f"📊 JSON 요약 저장: {args.json}")


if __name__ == '__main__':
    main()
