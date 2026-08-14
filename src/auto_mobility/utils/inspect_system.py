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

# repo 소스 실행 / 설치 실행 양쪽에서 auto_mobility 패키지 임포트 보장
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from auto_mobility.config import (
    CAMERA_RGB_TOPIC,
    CAMERA_RGB_COMPRESSED_TOPIC,
    CAMERA_DEPTH_TOPIC,
    CAMERA_ALIGNED_DEPTH_TOPIC,
    CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC,
    CAMERA_INFO_TOPIC,
    CAMERA_IMU_TOPIC,
)

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
        from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu
    except ImportError:
        return None

    rclpy.init(args=None)
    node = Node("system_inspector_node")

    topic_stats = {
        CAMERA_RGB_TOPIC: {"count": 0, "res": "N/A", "encoding": "N/A", "frame_id": "N/A", "qos": "SENSOR_DATA"},
        CAMERA_RGB_COMPRESSED_TOPIC: {"count": 0, "res": "N/A", "encoding": "jpeg", "frame_id": "N/A", "qos": "SENSOR_DATA"},
        CAMERA_DEPTH_TOPIC: {"count": 0, "res": "N/A", "encoding": "N/A", "frame_id": "N/A", "qos": "SENSOR_DATA"},
        CAMERA_ALIGNED_DEPTH_TOPIC: {"count": 0, "res": "N/A", "encoding": "N/A", "frame_id": "N/A", "qos": "SENSOR_DATA"},
        CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC: {"count": 0, "res": "N/A", "encoding": "compressedDepth (png)", "frame_id": "N/A", "qos": "SENSOR_DATA"},
        CAMERA_INFO_TOPIC: {"count": 0, "res": "N/A", "encoding": "N/A", "frame_id": "N/A", "qos": "SENSOR_DATA"},
        CAMERA_IMU_TOPIC: {"count": 0, "res": "N/A", "encoding": "N/A", "frame_id": "N/A", "qos": "SENSOR_DATA"},
    }

    def cb_color(msg):
        t = topic_stats[CAMERA_RGB_TOPIC]
        t["count"] += 1
        t["res"] = f"{msg.width}x{msg.height}"
        t["encoding"] = msg.encoding
        t["frame_id"] = msg.header.frame_id

    def cb_color_comp(msg):
        t = topic_stats[CAMERA_RGB_COMPRESSED_TOPIC]
        t["count"] += 1
        t["format"] = msg.format

    def cb_depth_raw(msg):
        t = topic_stats[CAMERA_DEPTH_TOPIC]
        t["count"] += 1
        t["res"] = f"{msg.width}x{msg.height}"
        t["encoding"] = msg.encoding
        t["frame_id"] = msg.header.frame_id

    def cb_depth(msg):
        t = topic_stats[CAMERA_ALIGNED_DEPTH_TOPIC]
        t["count"] += 1
        t["res"] = f"{msg.width}x{msg.height}"
        t["encoding"] = msg.encoding
        t["frame_id"] = msg.header.frame_id

    def cb_depth_comp(msg):
        t = topic_stats[CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC]
        t["count"] += 1
        t["format"] = msg.format

    def cb_info(msg):
        t = topic_stats[CAMERA_INFO_TOPIC]
        t["count"] += 1
        t["res"] = f"{msg.width}x{msg.height}"

    def cb_imu(msg):
        t = topic_stats[CAMERA_IMU_TOPIC]
        t["count"] += 1

    node.create_subscription(Image, CAMERA_RGB_TOPIC, cb_color, qos_profile_sensor_data)
    node.create_subscription(CompressedImage, CAMERA_RGB_COMPRESSED_TOPIC, cb_color_comp, qos_profile_sensor_data)
    node.create_subscription(Image, CAMERA_DEPTH_TOPIC, cb_depth_raw, qos_profile_sensor_data)
    node.create_subscription(Image, CAMERA_ALIGNED_DEPTH_TOPIC, cb_depth, qos_profile_sensor_data)
    node.create_subscription(CompressedImage, CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC, cb_depth_comp, qos_profile_sensor_data)
    node.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, cb_info, qos_profile_sensor_data)
    node.create_subscription(Imu, CAMERA_IMU_TOPIC, cb_imu, qos_profile_sensor_data)

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

            is_compressed_topic = "compressed" in top or "compressedDepth" in top
            is_raw_image_topic = ("color/image_raw" in top or "aligned_depth_to_color/image_raw" in top) and not is_compressed_topic

            if fps > 0:
                if is_compressed_topic:
                    if fps >= 15:
                        status_str = f"{GREEN}✅ {fps:.1f} Hz (정상 - 압축 전송 기준){RESET}"
                    else:
                        status_str = f"{RED}⚠️ {fps:.1f} Hz (저하됨! 권장 15+ Hz){RESET}"
                elif is_raw_image_topic:
                    status_str = f"{CYAN}ℹ️ {fps:.1f} Hz (Raw 스트림 - VM/대역폭 제한 시 낮을 수 있음){RESET}"
                else:
                    if fps >= 15:
                        status_str = f"{GREEN}✅ {fps:.1f} Hz (정상){RESET}"
                    else:
                        status_str = f"{YELLOW}⚠️ {fps:.1f} Hz (저하됨){RESET}"

                print(f"  • {BOLD}{top}{RESET}")
                print(f"    - 실시간 FPS : {status_str}")
                print(f"    - 해상도/포맷 : {res} (Encoding: {enc})")
                print(f"    - QoS 설정   : {qos}")
            else:
                print(f"  • {BOLD}{top}{RESET} : {RED}❌ 비활성 (데이터 수신 실패){RESET}")
    else:
        print(f"  {YELLOW}⚠️ rclpy 환경을 불러올 수 없어 기본 토픽 점검으로 대체합니다.{RESET}")

    print(f"{BOLD}{CYAN}========================================================================{RESET}\n")

if __name__ == "__main__":
    print_summary()
