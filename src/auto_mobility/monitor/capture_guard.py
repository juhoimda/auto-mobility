#!/usr/bin/env python3
"""
capture_guard.py - RTAB-Map 실시간 촬영 모니터링 가드 (2026-08-10 신규)

촬영 중 FPS/CPU/RAM/지연을 실시간 감시하여 품질 저하를 감지한다.
촬영 종료 시 최종 보고서를 저장한다.

핵심: 벤치마크는 시작 5초만 측정해 27-30Hz를 보여주지만,
실제 촬영은 시간이 지나며 VO odom이 ~14Hz로 저하된다 (VM 한계).
이 모니터는 이 저하를 실시간으로 포착해 기록한다.

사용법:
  python3 capture_guard.py [--duration 0] [--interval 3] [--report /path/report.md]
  --duration 0 = Ctrl+C 까지 무한 감시
"""

import os
import sys
import time
import json
import signal
import subprocess
import threading
from datetime import datetime

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# 감시 토픽: (토픽명, 최소 Hz, 설명)
MONITOR_TOPICS = [
    ("/camera/camera/color/image_raw",        15.0, "RGB 컬러"),
    ("/camera/camera/depth/image_rect_raw",   15.0, "Depth"),
    ("/camera/camera/color/camera_info",      15.0, "CameraInfo"),
    ("/camera/camera/imu",                   100.0, "IMU"),
    ("/rtabmap/odom",                          5.0, "Visual Odometry"),
]

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LOG_DIR = os.path.join(PROJECT_DIR, "ros2_data", "logs")

stop_flag = threading.Event()


def measure_topic_hz(topic, duration=4.0):
    """ros2 topic hz 로 해당 토픽의 평균 Hz 를 측정한다."""
    try:
        proc = subprocess.Popen(
            ["ros2", "topic", "hz", topic, "--window", "20"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, preexec_fn=os.setsid
        )
        time.sleep(duration)
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        out, _ = proc.communicate(timeout=3)
        for line in out.splitlines():
            if "average rate:" in line:
                return float(line.split("average rate:")[1].strip().split()[0])
    except Exception:
        pass
    return 0.0


def get_process_cpu_ram(proc_names):
    """지정 프로세스들의 CPU(%)/RAM(MB) 평균을 반환한다."""
    cpu_vals, ram_vals = [], []
    for pname in proc_names:
        try:
            r = subprocess.run(
                ["ps", "aux"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            for line in r.stdout.splitlines():
                if pname in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) >= 10:
                        try:
                            cpu_vals.append(float(parts[2]))
                            ram_vals.append(float(parts[5]) / 1024.0)
                        except ValueError:
                            pass
        except Exception:
            pass
    avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0.0
    avg_ram = sum(ram_vals) / len(ram_vals) if ram_vals else 0.0
    return round(avg_cpu, 1), round(avg_ram, 1)


def get_usb_status():
    """RealSense USB 링크 속도를 확인한다 (5000 = USB 3.x 정상)."""
    for dev in os.listdir("/sys/bus/usb/devices/"):
        path = f"/sys/bus/usb/devices/{dev}/idVendor"
        try:
            if os.path.exists(path) and open(path).read().strip() == "8086":
                speed = open(f"/sys/bus/usb/devices/{dev}/speed").read().strip()
                return f"{speed} Mbps (USB {'3.x' if int(speed) >= 5000 else '2.x ⚠'})"
        except Exception:
            pass
    return "미감지"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RTAB-Map 촬영 품질 모니터링 가드")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="감시 지속 시간(초). 0=Ctrl+C 까지")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="측정 주기(초)")
    parser.add_argument("--report", type=str, default=None,
                        help="보고서 파일 경로 (기본: logs/capture_guard_<ts>.md)")
    parser.add_argument("--headless", action="store_true",
                        help="실시간 화면 출력 없이 로그만 기록")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    if args.report is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.report = os.path.join(LOG_DIR, f"capture_guard_{ts}.md")

    start_time = time.time()
    usb = get_usb_status()
    samples = []
    violations = []

    def _sigint(sig, frame):
        stop_flag.set()

    signal.signal(signal.SIGINT, _sigint)

    print(f"{BOLD}{CYAN}==================================================={RESET}")
    print(f"{BOLD}{CYAN}  🎥 RTAB-Map 촬영 품질 실시간 모니터링 가드{RESET}")
    print(f"{BOLD}{CYAN}==================================================={RESET}")
    print(f"  USB 링크      : {usb}")
    print(f"  측정 주기     : {args.interval}s")
    print(f"  감시 토픽     : {len(MONITOR_TOPICS)}개")
    print(f"  보고서        : {args.report}")
    if args.duration > 0:
        print(f"  지속 시간     : {args.duration}s")
    print(f"{CYAN}==================================================={RESET}\n")
    if not args.headless:
        print(f"{BOLD}{'시간':<8}{'color':<8}{'depth':<8}{'info':<8}{'imu':<8}{'odom':<8}{'CPU':<7}{'RAM':<8} 상태{RESET}")

    cycle = 0
    while not stop_flag.is_set():
        if args.duration > 0 and time.time() - start_time >= args.duration:
            break

        elapsed = time.time() - start_time
        hz_map = {}
        for topic, min_hz, label in MONITOR_TOPICS:
            hz_map[topic] = measure_topic_hz(topic, duration=3.0)

        cpu, ram = get_process_cpu_ram(["realsense2_camera", "rgbd_odometry", "rtabmap"])

        # 상태 판정
        status_bits = []
        for topic, min_hz, label in MONITOR_TOPICS:
            hz = hz_map.get(topic, 0.0)
            if hz < min_hz * 0.5:
                status_bits.append(f"{RED}{label}:{hz:.1f}Hz{RESET}")
            elif hz < min_hz:
                status_bits.append(f"{YELLOW}{label}:{hz:.1f}Hz{RESET}")
            else:
                status_bits.append(f"{GREEN}{label}:{hz:.1f}Hz{RESET}")

        overall_ok = all(
            hz_map.get(t, 0.0) >= min_hz for t, min_hz, _ in MONITOR_TOPICS
        )
        odom = hz_map.get("/rtabmap/odom", 0.0)
        degraded = odom > 0 and odom < 10.0

        sample = {
            "elapsed_s": round(elapsed, 1),
            "hz": {t: round(hz_map.get(t, 0.0), 2) for t, _, _ in MONITOR_TOPICS},
            "cpu_pct": cpu,
            "ram_mb": ram,
            "overall_ok": overall_ok,
        }
        samples.append(sample)
        if degraded:
            violations.append({"elapsed_s": round(elapsed, 1), "odom_hz": odom})

        if not args.headless:
            state = f"{GREEN}OK{RESET}" if overall_ok else f"{YELLOW}⚠ 저하{RESET}"
            if degraded:
                state = f"{RED}❌ VO 저하{RESET}"
            line = (f"{elapsed:>6.0f}s  "
                    f"{hz_map.get(MONITOR_TOPICS[0][0],0):>6.1f}  "
                    f"{hz_map.get(MONITOR_TOPICS[1][0],0):>6.1f}  "
                    f"{hz_map.get(MONITOR_TOPICS[2][0],0):>6.1f}  "
                    f"{hz_map.get(MONITOR_TOPICS[3][0],0):>6.1f}  "
                    f"{odom:>6.1f}  {cpu:>5.1f}  {ram:>6.0f}  {state}")
            print(line, flush=True)

        cycle += 1
        # 다음 주기까지 대기 (Ctrl+C 로 조기 종료)
        time.sleep(args.interval)

    # ================= 최종 보고서 생성 =================
    if samples:
        avg_cpu = sum(s["cpu_pct"] for s in samples) / len(samples)
        avg_ram = sum(s["ram_mb"] for s in samples) / len(samples)
        odom_hz_series = [s["hz"].get("/rtabmap/odom", 0.0) for s in samples]
        first_odom = odom_hz_series[0] if odom_hz_series else 0.0
        last_odom = odom_hz_series[-1] if odom_hz_series else 0.0

        md = f"""# 🎥 촬영 품질 모니터링 보고서 (capture_guard)

- **작성 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **감시 지속**: {round(samples[-1]['elapsed_s'], 1)}s / {len(samples)} 샘플
- **USB 링크**: {usb}
- **평균 CPU**: {avg_cpu:.1f}%  /  **평균 RAM**: {avg_ram:.0f}MB

---

## 📊 시간 경과별 Visual Odometry (촬영 품질의 핵심 지표)

| 시점(초) | Odom Hz |
|:---:|:---:|
"""
        for s in samples[:: max(1, len(samples) // 20)]:
            md += f"| {s['elapsed_s']:.0f} | {s['hz'].get('/rtabmap/odom', 0.0):.1f} |\n"

        degrade = last_odom < first_odom * 0.7
        md += f"""
---

## 📋 종합 판정

- **시작 Odom**: {first_odom:.1f} Hz  →  **종료 Odom**: {last_odom:.1f} Hz
- **시간 경과 저하**: {'❌ 있음 (촬영 시간이 길수록 VO 저하 → 촬영 세션을 분할 권장)' if degrade else '✅ 없음'}
- **저하 구간 수**: {len(violations)} 회
"""
        if violations:
            md += "\n**저하 발생 시점**:\n"
            for v in violations:
                md += f"- {v['elapsed_s']:.0f}초 시점: odom {v['odom_hz']:.1f} Hz\n"

        md += "\n---\n\n## 💡 권장 조치\n\n"
        md += ("- 촬영 세션을 2~3분 단위로 분할 (VO 저하 방지)\n"
               "- RViz/시각화를 끄거나 최소화하면 vCPU 여유 확보\n"
               "- 640x480@30 유지 (848x480은 depth 드랍으로 품질 저하)\n")

        with open(args.report, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n📄 보고서 저장: {args.report}")
        print(f"  시작 odom: {first_odom:.1f}Hz → 종료 odom: {last_odom:.1f}Hz")
    else:
        print("샘플 없음")


if __name__ == "__main__":
    main()
