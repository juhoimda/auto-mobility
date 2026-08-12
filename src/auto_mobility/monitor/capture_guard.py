#!/usr/bin/env python3
"""
capture_guard.py - RTAB-Map 실시간 촬영 품질 모니터링 가드 (v2, 2026-08-12)

촬영 중 센서 토픽 FPS / 프로세스별 CPU/RAM / 지연을 실시간 감시하고,
종료 시 Markdown 보고서(.md)와 구조화 요약(.json)을 저장한다.

v2 변경 (효율 개선):
  - 5개 토픽(색상/depth/info/imu/odom) Hz를 **병렬** 측정 → 측정 주기 단축(15s→3s)
  - 기존엔 odom만 보고서에 저장했으나, **전체 토픽 Hz 시계열**을 .md/.json에 저장
  - CPU를 3개 프로세스 평균 1개 값이 아닌 **프로세스별 시계열**로 기록
  - headless 시 콘솔 무출력 → 별도 .log 파일이 0바이트로 남던 문제 제거

사용법:
  python3 capture_guard.py [--interval 5] [--duration 0] [--headless]
                           [--report path.md] [--json path.json]
  --duration 0 = Ctrl+C 까지 무한 감시
"""

import os
import sys

# repo 소스 실행 / 설치 실행 양쪽에서 auto_mobility 패키지 임포트 보장
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import json
import signal
import subprocess
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from auto_mobility.config import (
    CAMERA_RGB_TOPIC,
    CAMERA_DEPTH_TOPIC,
    CAMERA_INFO_TOPIC,
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

# 감시 토픽: (토픽명, 최소 Hz, 설명) — 토픽명은 config.py 단일 소스
MONITOR_TOPICS = [
    (CAMERA_RGB_TOPIC,   15.0, "RGB"),
    (CAMERA_DEPTH_TOPIC, 15.0, "Depth"),
    (CAMERA_INFO_TOPIC,  15.0, "Info"),
    (CAMERA_IMU_TOPIC,  100.0, "IMU"),
    (ODOM_TOPIC,          5.0, "Odom"),
]

# CPU 감시 프로세스 (ps comm 부분 일치)
PROC_NAMES = [
    "realsense2_camera",
    "rgbd_odometry",
    "rtabmap",
    "rviz2",
    "cloud_throttle",
    "imu_filter_madgwick",
]

stop_flag = threading.Event()


def measure_topic_hz_worker(topic: str, duration: float) -> float:
    """단일 토픽의 평균 Hz 를 측정해 반환한다 (최신 'average rate' 기준).

    select 기반 논블로킹 읽기 → 토픽에 데이터가 없어도 duration 후 정상 종료.
    """
    import select
    try:
        proc = subprocess.Popen(
            ["ros2", "topic", "hz", topic, "--window", "15"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        last = 0.0
        buf = b""
        deadline = time.time() + duration
        while time.time() < deadline:
            r, _, _ = select.select([proc.stdout], [], [], 0.2)
            if not r:
                continue
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            buf += chunk
            for m in re.finditer(rb"average rate:\s*([\d.]+)", buf):
                last = float(m.group(1))
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.communicate(timeout=2)
        return last
    except Exception:
        return 0.0


def measure_topics_hz(topics, duration=3.0):
    """여러 토픽을 병렬로 측정한다 (측정 주기 5토픽 15s → ~3s 로 단축)."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(topics)) as ex:
        futures = {ex.submit(measure_topic_hz_worker, t, duration): t for t, _, _ in topics}
        for fut in futures:
            t = futures[fut]
            try:
                results[t] = fut.result()
            except Exception:
                results[t] = 0.0
    return results


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
    """RealSense USB 링크 속도를 확인한다 (5000 = USB 3.x 정상)."""
    for dev in os.listdir("/sys/bus/usb/devices/"):
        path = f"/sys/bus/usb/devices/{dev}/idVendor"
        try:
            if os.path.exists(path) and open(path).read().strip() == "8086":
                speed = open(f"/sys/bus/usb/devices/{dev}/speed").read().strip()
                return f"{speed} Mbps (USB {'3.x' if int(speed) >= USB_3_MIN_SPEED_MBPS else '2.x'})"
        except Exception:
            pass
    return "미감지"


def build_md(samples, usb, duration):
    """Markdown 보고서 생성 (전체 토픽 시계열은 다운샘플링, 집계는 전부 포함)."""
    topic_cols = [label for _, _, label in MONITOR_TOPICS]

    # 다운샘플링 (최대 20행)
    step = max(1, len(samples) // 20)
    rows = samples[::step]

    md = f"""# 🎥 촬영 품질 모니터링 보고서 (capture_guard)

- **작성 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **감시 지속**: {round(duration, 1)}s / {len(samples)} 샘플
- **USB 링크**: {usb}

## 📊 토픽 FPS 시계열 ({' / '.join(topic_cols)})

| 시점(초) | {' | '.join(topic_cols)} |
|:---:|""" + "|".join([":---:"] * len(topic_cols)) + "|\n"
    for s in rows:
        md += f"| {s['t']:.0f} | " + " | ".join(f"{s['hz'].get(t, 0.0):.1f}"
              for t, _, _ in MONITOR_TOPICS) + " |\n"

    # 프로세스별 CPU 집계
    md += f"""
## 🖥️ 프로세스별 CPU (평균 / 최대)

| 프로세스 | 평균 | 최대 |
|:---|:---:|:---:|
"""
    for name in PROC_NAMES:
        cpus = [s["cpu"].get(name, 0.0) for s in samples]
        if cpus:
            md += f"| {name} | {sum(cpus)/len(cpus):.1f}% | {max(cpus):.1f}% |\n"

    # 저하 이벤트
    md += f"""
## 📉 토픽별 저하 이벤트 (최소 Hz 미만 샘플 수)

| 토픽 | 최소 Hz | 저하 샘플 |
|:---|---:|---:|
"""
    for topic, min_hz, label in MONITOR_TOPICS:
        bad = sum(1 for s in samples if 0 < s["hz"].get(topic, 0.0) < min_hz)
        md += f"| {label} | {min_hz} | {bad} |\n"

    md += f"""
## 📋 종합 판정

- **평균 CPU**: {sum(s['cpu_total'] for s in samples)/len(samples):.1f}% / **평균 RAM**: {sum(s['ram_mb'] for s in samples)/len(samples):.0f}MB
- **Odom 시계열**: 시작 {samples[0]['hz'].get(ODOM_TOPIC, 0):.1f} Hz → 종료 {samples[-1]['hz'].get(ODOM_TOPIC, 0):.1f} Hz
"""
    return md


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RTAB-Map 촬영 품질 모니터링 가드")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="감시 지속 시간(초). 0=Ctrl+C 까지")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="측정 주기(초)")
    parser.add_argument("--report", type=str, default=None,
                        help="Markdown 보고서 경로 (기본: logs/capture_guard_<ts>.md)")
    parser.add_argument("--json", type=str, default=None,
                        help="JSON 요약 경로 (기본: --report 와 동일 basename .json)")
    parser.add_argument("--headless", action="store_true",
                        help="콘솔 출력 없이 보고서/JSON만 저장")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    if args.report is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.report = os.path.join(LOG_DIR, f"capture_guard_{ts}.md")
    if args.json is None:
        args.json = os.path.splitext(args.report)[0] + ".json"

    start_time = time.time()
    usb = get_usb_status()
    samples = []

    def _sigint(sig, frame):
        stop_flag.set()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)  # pipeline이 kill(TERM)로 종료해도 보고서 저장

    if not args.headless:
        print(f"{BOLD}{CYAN}==================================================={RESET}")
        print(f"{BOLD}{CYAN}  🎥 RTAB-Map 촬영 품질 실시간 모니터링 가드{RESET}")
        print(f"{CYAN}==================================================={RESET}")
        print(f"  USB 링크      : {usb}")
        print(f"  측정 주기     : {args.interval}s")
        print(f"  감시 토픽     : {len(MONITOR_TOPICS)}개")
        print(f"  보고서        : {args.report}")
        print(f"  JSON 요약     : {args.json}")
        if args.duration > 0:
            print(f"  지속 시간     : {args.duration}s")
        print(f"{CYAN}==================================================={RESET}\n")
        header = f"{'시간':<6}" + "".join(f"{label:<8}" for _, _, label in MONITOR_TOPICS)
        print(f"{BOLD}{header}  상태{RESET}")

    while not stop_flag.is_set():
        if args.duration > 0 and time.time() - start_time >= args.duration:
            break

        elapsed = time.time() - start_time
        hz_map = measure_topics_hz(MONITOR_TOPICS, duration=3.0)
        proc_cpu, ram_mb = get_proc_cpu_ram(PROC_NAMES)
        cpu_total = sum(proc_cpu.values())

        sample = {
            "t": round(elapsed, 1),
            "hz": {label: round(hz_map.get(t, 0.0), 2) for t, _, label in MONITOR_TOPICS},
            "cpu": {k: round(v, 1) for k, v in proc_cpu.items()},
            "cpu_total": round(cpu_total, 1),
            "ram_mb": round(ram_mb, 0),
        }
        samples.append(sample)

        if not args.headless:
            status_bits = []
            for topic, min_hz, label in MONITOR_TOPICS:
                hz = hz_map.get(topic, 0.0)
                if hz < min_hz * 0.5:
                    status_bits.append(f"{RED}{hz:.1f}{RESET}")
                elif hz < min_hz:
                    status_bits.append(f"{YELLOW}{hz:.1f}{RESET}")
                else:
                    status_bits.append(f"{GREEN}{hz:.1f}{RESET}")
            print(f"{elapsed:>6.0f}  " + "  ".join(f"{v:<8}" for v in status_bits)
                  + f"  CPU {cpu_total:>5.1f}%", flush=True)

        time.sleep(args.interval)

    # ================= 결과 저장 =================
    duration = time.time() - start_time
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({
            "session": os.path.splitext(os.path.basename(args.report))[0],
            "started": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(duration, 1),
            "usb": usb,
            "interval_s": args.interval,
            "samples": samples,
            "proc_names": PROC_NAMES,
        }, f, ensure_ascii=False, indent=1)
    print(f"\n📊 JSON 요약 저장: {args.json}")

    if samples:
        md = build_md(samples, usb, duration)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"📄 보고서 저장: {args.report}")
        print(f"  시작 odom: {samples[0]['hz'].get(ODOM_TOPIC, 0):.1f}Hz → 종료 odom: {samples[-1]['hz'].get(ODOM_TOPIC, 0):.1f}Hz")
    else:
        print("샘플 없음")


if __name__ == "__main__":
    main()
