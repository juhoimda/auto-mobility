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
    """모든 모니터링 토픽을 단일 노드에서 초경량 카운팅

    추가 진단 항목:
      - 토픽별 최대 프레임 gap (타임스탬프 기반): 대규모 timestamp 공백 감지
      - RGB↔Depth sync delta (같은 구간 내 최신 stamp 차이): 센서 동기화 품질 측정
    """
    def __init__(self, topics):
        super().__init__('capture_guard_monitor')
        self._counts = {t[0]: 0 for t in topics}
        self._lock = threading.Lock()
        self._last_stamp = {}
        self._max_gap = {t[0]: 0.0 for t in topics}
        self._rgb_topic = None
        self._depth_topic = None
        for t, _, label in topics:
            if label == "RGB":
                self._rgb_topic = t
            if label == "Depth":
                self._depth_topic = t

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
                return _cb

            self.create_subscription(msg_type, topic, make_cb(topic), qos)

    def get_and_reset(self, dt: float) -> dict:
        with self._lock:
            rates = {}
            gaps = {}
            for t, count in self._counts.items():
                rates[t] = count / dt if dt > 0 else 0.0
                self._counts[t] = 0
                gaps[t] = self._max_gap.get(t, 0.0)
                self._max_gap[t] = 0.0

            # RGB↔Depth sync delta (최신 stamp 차이, 1 interval 1샘플)
            sync_delta = None
            if self._rgb_topic and self._depth_topic:
                rgb_ts = self._last_stamp.get(self._rgb_topic)
                depth_ts = self._last_stamp.get(self._depth_topic)
                if rgb_ts and depth_ts:
                    sync_delta = abs(rgb_ts - depth_ts)
            return rates, gaps, sync_delta


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

    md += f"""| **평균 CPU 사용률** | {avg_cpu:.1f} % | - |
| **평균 RAM 사용량** | {avg_ram:.0f} MB | - |
"""
    return md


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=5.0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--report', type=str, default=None)
    parser.add_argument('--json', type=str, default=None)
    parser.add_argument('--remote', action='store_true',
                        help='원격(Windows 카메라) 모드: raw 대신 압축 토픽 감시')
    args = parser.parse_args()

    monitor_topics = REMOTE_MONITOR_TOPICS if args.remote else MONITOR_TOPICS

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

    try:
        while not stop_event.is_set():
            if stop_event.wait(timeout=args.interval):
                break
            now = time.time()
            dt = now - t_last
            t_last = now

            rates, gaps, sync_delta = monitor_node.get_and_reset(dt)
            proc_cpu, ram_mb = get_proc_cpu_ram(GUARD_PROCS)
            cpu_total = sum(proc_cpu.values())

            sample = {
                "elapsed": now - t_start,
                "rates": rates,
                "max_gaps": gaps,
                "sync_delta": sync_delta,
                "cpu_total": cpu_total,
                "ram_mb": ram_mb,
            }
            samples.append(sample)

            if not args.headless:
                rates_disp = " | ".join(f"{label}: {rates.get(t, 0.0):.1f}Hz" for t, _, label in monitor_topics)
                print(f"[capture_guard] {rates_disp} | CPU: {cpu_total:.1f}% | RAM: {ram_mb:.0f}MB")
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
