#!/usr/bin/env python3
"""
ROS2 System & Camera Diagnostic Inspector
Checks RMW/DDS, FastDDS SHM Profile, USB Version, Topic Resolution, Encoding, QoS, and Real-time FPS.
"""

import os
import sys
import time
import subprocess
import threading

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def get_dds_info():
    rmw = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp (기본값)")
    fastdds_xml = os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", "미설정")
    
    shm_active = False
    if fastdds_xml != "미설정" and os.path.exists(fastdds_xml):
        shm_active = True

    rmem_max = "확인 불가"
    try:
        res = subprocess.run(["sysctl", "net.core.rmem_max"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "rmem_max" in res.stdout:
            val = int(res.stdout.split("=")[1].strip())
            rmem_max = f"{val / (1024 * 1024):.1f} MB ({val} bytes)"
    except Exception:
        pass

    return {
        "rmw": rmw,
        "fastdds_xml": fastdds_xml,
        "shm_active": shm_active,
        "rmem_max": rmem_max
    }

def get_usb_info():
    usb_status = "확인 불가 (기본 USB 3.x 가정)"
    try:
        res = subprocess.run(["rs-enumerate-devices", "-s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            if "2.1" in res.stdout or "2.0" in res.stdout:
                usb_status = f"{RED}USB 2.1 / 2.0 인식됨 (⚠️ 주의: 대역폭 부족으로 720p 30fps 출력 불가){RESET}"
            elif "3." in res.stdout:
                usb_status = f"{GREEN}USB 3.x SuperSpeed 정상 인식됨{RESET}"
            else:
                usb_status = res.stdout.strip().split('\n')[0]
        else:
            # Try lsusb fallback
            lsusb_res = subprocess.run(["lsusb"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "Intel" in lsusb_res.stdout or "RealSense" in lsusb_res.stdout:
                usb_status = "RealSense USB 장치 연결됨 (rs-enumerate-devices 미설치)"
    except Exception:
        pass
    return usb_status

def inspect_topics_rclpy():
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo, Imu
    except ImportError:
        return None

    rclpy.init(args=None)
    node = Node("system_inspector_node")

    topic_stats = {
        "/camera/camera/color/image_raw": {"count": 0, "res": "N/A", "encoding": "N/A", "frame_id": "N/A", "qos": "SENSOR_DATA"},
        "/camera/camera/aligned_depth_to_color/image_raw": {"count": 0, "res": "N/A", "encoding": "N/A", "frame_id": "N/A", "qos": "SENSOR_DATA"},
        "/camera/camera/color/camera_info": {"count": 0, "res": "N/A", "encoding": "N/A", "frame_id": "N/A", "qos": "SENSOR_DATA"},
        "/camera/camera/imu": {"count": 0, "res": "N/A", "encoding": "N/A", "frame_id": "N/A", "qos": "SENSOR_DATA"},
    }

    def cb_color(msg):
        t = topic_stats["/camera/camera/color/image_raw"]
        t["count"] += 1
        t["res"] = f"{msg.width}x{msg.height}"
        t["encoding"] = msg.encoding
        t["frame_id"] = msg.header.frame_id

    def cb_depth(msg):
        t = topic_stats["/camera/camera/aligned_depth_to_color/image_raw"]
        t["count"] += 1
        t["res"] = f"{msg.width}x{msg.height}"
        t["encoding"] = msg.encoding
        t["frame_id"] = msg.header.frame_id

    def cb_info(msg):
        t = topic_stats["/camera/camera/color/camera_info"]
        t["count"] += 1
        t["res"] = f"{msg.width}x{msg.height}"

    def cb_imu(msg):
        t = topic_stats["/camera/camera/imu"]
        t["count"] += 1

    node.create_subscription(Image, "/camera/camera/color/image_raw", cb_color, qos_profile_sensor_data)
    node.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", cb_depth, qos_profile_sensor_data)
    node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", cb_info, qos_profile_sensor_data)
    node.create_subscription(Imu, "/camera/camera/imu", cb_imu, qos_profile_sensor_data)

    duration = 3.0
    start_time = time.time()
    while time.time() - start_time < duration:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    results = {}
    for top, data in topic_stats.items():
        fps = data["count"] / duration
        results[top] = {
            "fps": fps,
            "res": data["res"],
            "encoding": data["encoding"],
            "qos": data["qos"]
        }
    return results

def print_summary():
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}      📊 ROS 2 DDS, 카메라 해상도 / FPS / 환경 종합 진단 보고서{RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")

    # 1. DDS & System Buffer
    dds = get_dds_info()
    print(f"\n{BOLD}[1. DDS & 통신 버퍼 설정]{RESET}")
    print(f"  • RMW Implementation : {GREEN}{dds['rmw']}{RESET}")
    if dds['shm_active']:
        print(f"  • FastDDS SHM 프로필 : {GREEN}✅ 적용됨 ({dds['fastdds_xml']}){RESET}")
    else:
        print(f"  • FastDDS SHM 프로필 : {YELLOW}⚠️ 미적용 (UDP 통신 모드 사용 중){RESET}")
    print(f"  • Linux rmem_max     : {dds['rmem_max']}")

    # 2. Hardware / USB Status
    usb = get_usb_info()
    print(f"\n{BOLD}[2. 하드웨어 및 USB 연결 상태]{RESET}")
    print(f"  • RealSense USB 모드 : {usb}")

    # 3. Topic Inspection
    print(f"\n{BOLD}[3. 카메라 실시간 스트리밍 현황 (DDS, 해상도, FPS, QoS)]{RESET}")
    results = inspect_topics_rclpy()

    if results:
        for top, info in results.items():
            fps = info["fps"]
            res = info["res"]
            enc = info["encoding"]
            qos = info["qos"]

            if fps > 0:
                if fps >= 15:
                    status_str = f"{GREEN}✅ {fps:.1f} Hz (정상){RESET}"
                else:
                    status_str = f"{RED}⚠️ {fps:.1f} Hz (저하됨! 권장 15+ Hz){RESET}"
                print(f"  • {BOLD}{top}{RESET}")
                print(f"    - 실시간 FPS : {status_str}")
                print(f"    - 해상도/포맷 : {res} (Encoding: {enc})")
                print(f"    - QoS 설정   : {qos}")
            else:
                print(f"  • {BOLD}{top}{RESET} : {RED}❌ 비활성 (데이터 수신 실패){RESET}")
    else:
        print(f"  {YELLOW}⚠️ rclpy 환경을 불러올 수 없어 기본 토픽 점검으로 대체합니다.{RESET}")

    # 4. Diagnostic & Solutions Guide for Low Hz
    print(f"\n{BOLD}[4. 💡 Hz가 갑자기 낮아졌을 때 주요 원인 및 조치 방법]{RESET}")
    print(f"  1) {BOLD}가상머신 (VM) USB Pass-through 병목{RESET}:")
    print(f"     - VM 환경에서는 1280x720@30fps (약 80MB/s+) Raw 이미지 처리 시 USB 대역폭 및 CPU 스케줄링 병목으로 프레임이 3~5Hz로 급감할 수 있습니다.")
    print(f"  2) {BOLD}USB 2.1 연결 인식 문제{RESET}:")
    print(f"     - USB 3.0 포트/케이블 접촉 불량 시 카메라가 USB 2.1 모드로 격하되어 FPS가 크게 저하됩니다. (포트 재연결 필요)")
    print(f"  3) {BOLD}Auto-exposure Priority (자동 노출 우선순위){RESET}:")
    print(f"     - 어두운 환경에서 카메라 노출 시간이 길어져 FPS가 자동 감소합니다 (`rgb_camera.auto_exposure_priority:=false` 필수).")
    print(f"  4) {BOLD}QoS & SHM 버퍼 미적용{RESET}:")
    print(f"     - RELIABLE QoS 사용 시 패킷 재전송 병목이 생깁니다 (`color_qos:=SENSOR_DATA` 사용).")
    print(f"{BOLD}{CYAN}========================================================================{RESET}\n")

if __name__ == "__main__":
    print_summary()
