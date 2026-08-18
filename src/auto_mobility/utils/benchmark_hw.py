#!/usr/bin/env python3
"""
[1단계: 센서 데이터 수집/취급 최적화 전용]
Hardware-adaptive DDS, QoS, Resolution, FPS, and Compression Benchmark Script
for RealSense D435i Camera & ROS 2 Pipeline.

⚠️ LEGACY / LOCAL-ONLY (2026-08-14):
  - WSL 로컬 realsense2_camera 노드를 직접 구동하는 usbipd 시대 벤치마크.
  - 원격(Windows 카메라) 모드에서는 실행하지 말 것 (토픽 충돌).
  - rgb_qos/depth_qos 파라미터는 production(config.py)에서 금지된 키 —
    본 벤치마크 내부 실험용으로만 사용.
"""

import os
import sys
import time
import subprocess
import json
import argparse
import signal
from datetime import datetime

# repo 소스 실행 / 설치 실행 양쪽에서 auto_mobility 패키지 임포트 보장
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from auto_mobility.config import (
    LOG_DIR,
    CONFIG_DIR,
    FASTDDS_XML,
    CAMERA_RGB_TOPIC,
    CAMERA_RGB_COMPRESSED_TOPIC,
)

# Standard colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def check_system_environment():
    """Checks USB connection, network buffer, and available RMWs."""
    info = {
        "rmws": ["rmw_fastrtps_cpp"],
        "rmem_max": "Unknown",
        "usb_type": "USB 3.x (권장)"
    }
    
    # Check CycloneDDS availability
    try:
        if os.path.exists("/opt/ros/humble/lib/librmw_cyclonedds_cpp.so"):
            info["rmws"].append("rmw_cyclonedds_cpp")
    except Exception:
        pass

    # Check Linux rmem_max
    try:
        res = subprocess.run(["sysctl", "net.core.rmem_max"], stdout=subprocess.PIPE, text=True)
        if "rmem_max" in res.stdout:
            val = int(res.stdout.split("=")[1].strip())
            info["rmem_max"] = f"{val / (1024*1024):.1f} MB"
    except Exception:
        pass

    # Check USB Port Speed if rs-enumerate-devices is installed
    try:
        res = subprocess.run(["rs-enumerate-devices", "-s"], stdout=subprocess.PIPE, text=True)
        if "2.1" in res.stdout or "2.0" in res.stdout:
            info["usb_type"] = "USB 2.1 (주의: 720p 30fps 제한 가능성 있음)"
    except Exception:
        pass

    return info

def measure_topic_hz_and_cpu(topic, duration=5.0, env=None):
    """Measures topic hz and CPU usage during sampling."""
    cmd = f"ros2 topic hz {topic}"
    hz_val = 0.0
    cpu_usage = 0.0
    
    try:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid, env=env)
        
        start_time = time.time()
        cpu_samples = []
        
        while time.time() - start_time < duration:
            time.sleep(0.5)
            try:
                ps_res = subprocess.run("ps aux | grep realsense2_camera | grep -v grep | awk '{print $3}'", 
                                        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                val = ps_res.stdout.strip()
                if val:
                    cpu_samples.append(float(val.split('\n')[0]))
            except Exception:
                pass
                
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        out, _ = proc.communicate(timeout=3)
        
        for line in out.splitlines():
            if "average rate:" in line:
                parts = line.split("average rate:")
                hz_val = float(parts[1].strip().split()[0])
                break
                
        if cpu_samples:
            cpu_usage = sum(cpu_samples) / len(cpu_samples)
            
    except Exception:
        pass
        
    return hz_val, cpu_usage

def run_single_test(rmw, use_shm, res_w, res_h, fps, qos, is_compressed, sample_duration=6.0):
    """Runs a single test combination by spawning the camera node."""
    env = os.environ.copy()
    env["RMW_IMPLEMENTATION"] = rmw
    
    # SHM configuration for FastDDS
    shm_status = "Default"
    if rmw == "rmw_fastrtps_cpp":
        if use_shm:
            fastdds_xml = str(FASTDDS_XML)
            if os.path.exists(fastdds_xml):
                env["FASTRTPS_DEFAULT_PROFILES_FILE"] = fastdds_xml
                shm_status = "SHM 활성화"
        else:
            env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
            shm_status = "UDP Only"

    color_prof = f"{res_w}x{res_h}x{fps}"
    depth_prof = f"{res_w}x{res_h}x{fps}"
    
    node_cmd = [
        "ros2", "run", "realsense2_camera", "realsense2_camera_node",
        "--ros-args",
        "-p", f"depth_module.depth_profile:={depth_prof}",
        "-p", f"rgb_camera.color_profile:={color_prof}",
        "-p", "rgb_camera.color_format:=RGB8",
        "-p", "align_depth.enable:=true",
        "-p", "enable_accel:=true",
        "-p", "enable_gyro:=true",
        "-p", "unite_imu_method:=1",
        # rgb_camera.auto_exposure_priority 는 4.58.3 에서 선언되지 않은 무효 키 (config.py INVALID_CAMERA_PARAMS)
        "-p", f"rgb_qos:={qos}",
        "-p", f"depth_qos:={qos}",
        "-p", f"pointcloud.qos:={qos}",
        "-r", "__ns:=/camera",
        "-r", "__node:=camera"
    ]

    mode_str = f"RMW: {rmw} ({shm_status}) | {res_w}x{res_h}@{fps}fps | QoS: {qos} | 포맷: {'압축' if is_compressed else 'Raw'}"
    print(f"\n{CYAN}▶ [테스트 실행]{RESET} {mode_str}")
    
    camera_proc = subprocess.Popen(node_cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    
    time.sleep(3.5)
    
    target_topic = CAMERA_RGB_COMPRESSED_TOPIC if is_compressed else CAMERA_RGB_TOPIC
    measured_fps, cpu_pct = measure_topic_hz_and_cpu(target_topic, duration=sample_duration, env=env)
    
    try:
        os.killpg(os.getpgid(camera_proc.pid), signal.SIGINT)
        camera_proc.wait(timeout=4)
    except Exception:
        try:
            os.killpg(os.getpgid(camera_proc.pid), signal.SIGKILL)
        except Exception:
            pass

    time.sleep(1.5)
    
    drop_pct = max(0.0, ((fps - measured_fps) / fps) * 100.0) if fps > 0 else 100.0
    
    result = {
        "rmw": rmw,
        "shm_status": shm_status,
        "resolution": f"{res_w}x{res_h}",
        "target_fps": fps,
        "qos": qos,
        "format": "compressed" if is_compressed else "raw",
        "measured_fps": round(measured_fps, 2),
        "drop_pct": round(drop_pct, 1),
        "cpu_pct": round(cpu_pct, 1)
    }
    
    status_color = GREEN if drop_pct < 10 else (YELLOW if drop_pct < 25 else RED)
    print(f"   └─ {status_color}측정 FPS: {measured_fps:.1f} / {fps} fps (프레임 손실률: {drop_pct:.1f}%, CPU 점유율: {cpu_pct:.1f}%){RESET}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Sensor Processing & DDS Benchmark Tool")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with primary configurations")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    
    sys_info = check_system_environment()
    
    print(f"{BOLD}====================================================={RESET}")
    print(f"{BOLD}🎥 [1단계 전용] 센서 데이터 수집 & DDS/QoS 자동 벤치마크{RESET}")
    print(f"{BOLD}====================================================={RESET}")
    print(f"🔍 [시스템 환경 점검]")
    print(f"   • 감지된 RMW 목록: {', '.join(sys_info['rmws'])}")
    print(f"   • OS 네트워크 버퍼(rmem_max): {sys_info['rmem_max']}")
    print(f"   • USB 연결 상태: {sys_info['usb_type']}\n")

    resolutions_fps = [
        (640, 480, 15),
        (640, 480, 30),
        (1280, 720, 15),
        (1280, 720, 30)
    ] if not args.quick else [(640, 480, 15), (640, 480, 30), (1280, 720, 30)]

    qos_list = ["SENSOR_DATA", "DEFAULT"] if not args.quick else ["SENSOR_DATA"]
    compressed_options = [False, True] if not args.quick else [False]

    results = []
    
    test_configs = []
    for rmw in sys_info['rmws']:
        shm_modes = [True, False] if (rmw == "rmw_fastrtps_cpp" and not args.quick) else [True]
        for shm in shm_modes:
            for res_w, res_h, fps in resolutions_fps:
                for qos in qos_list:
                    for is_comp in compressed_options:
                        test_configs.append((rmw, shm, res_w, res_h, fps, qos, is_comp))

    print(f"📊 총 {len(test_configs)}개 센서 수집 제어 조합에 대한 정밀 측정 진행 중...\n")

    for idx, (rmw, shm, res_w, res_h, fps, qos, is_comp) in enumerate(test_configs, 1):
        print(f"[{idx}/{len(test_configs)}] 측정 진행 중...", end="")
        res = run_single_test(rmw, shm, res_w, res_h, fps, qos, is_comp)
        results.append(res)

    # Scored ranking
    scored_results = []
    for r in results:
        fps_ratio = r["measured_fps"] / r["target_fps"] if r["target_fps"] > 0 else 0
        fps_score = min(1.0, fps_ratio) * 60.0
        
        res_area = 1280*720 if "1280" in r["resolution"] else 640*480
        quality_score = (res_area / (1280*720)) * 15.0 + (r["target_fps"] / 30.0) * 15.0
        qos_score = 10.0 if r["qos"] == "SENSOR_DATA" else 5.0
        
        total_score = round(fps_score + quality_score + qos_score, 1)
        r["score"] = total_score
        scored_results.append(r)

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    best_config = scored_results[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"sensor_stage1_benchmark_{timestamp}.md"
    report_path = os.path.join(LOG_DIR, report_filename)
    
    report_md = f"""# 📊 [1단계] 센서 데이터 취득 & DDS/QoS 파이프라인 정밀 분석 보고서

- **작성 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **대상 장비**: Intel RealSense D435i Camera
- **OS 네트워크 버퍼**: `{sys_info['rmem_max']}`
- **USB 연결 상태**: `{sys_info['usb_type']}`
- **테스트 수행 조합 수**: {len(results)}개

---

## 🏆 하드웨어 최적 센서 설정 (Best Recommendation)

- **DDS 미들웨어**: `{best_config['rmw']}` (`{best_config['shm_status']}`)
- **권장 해상도 & FPS**: `{best_config['resolution']} @ {best_config['target_fps']} fps`
- **QoS 정책**: `{best_config['qos']}` (Best Effort / Sensor Data)
- **전송 포맷**: `{best_config['format'].upper()}`
- **성능 실측**: `{best_config['measured_fps']} FPS` (손실률: `{best_config['drop_pct']}%`, CPU: `{best_config['cpu_pct']}%`)

---

## 📋 1단계 전체 벤치마크 테스트 결과 (성능순 정렬)

| 순위 | DDS 미들웨어 | 통신 방식 | 해상도 & 타겟 FPS | QoS 정책 | 포맷 | 실측 FPS | 프레임 손실률 | CPU 점유율 | 종합 점수 |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""

    for i, r in enumerate(scored_results, 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"{i}"))
        report_md += f"| {medal} | `{r['rmw']}` | `{r['shm_status']}` | `{r['resolution']} @ {r['target_fps']}fps` | `{r['qos']}` | `{r['format']}` | **{r['measured_fps']}** | {r['drop_pct']}% | {r['cpu_pct']}% | **{r['score']}점** |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n{BOLD}====================================================={RESET}")
    print(f"{GREEN}🎉 1단계 센서 최적화 벤치마크가 완벽히 완료되었습니다!{RESET}")
    print(f"{BOLD}====================================================={RESET}")
    print(f"🥇 {BOLD}최적 설정{RESET}: {best_config['rmw']} ({best_config['shm_status']}) | {best_config['resolution']}@{best_config['target_fps']}fps | {best_config['qos']}")
    print(f"📄 {BOLD}상세 분석 보고서 파일{RESET}: {report_path}")
    print(f"{BOLD}====================================================={RESET}\n")

if __name__ == "__main__":
    main()
