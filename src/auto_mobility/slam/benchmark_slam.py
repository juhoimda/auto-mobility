#!/usr/bin/env python3
"""
[통합 벤치마크] 카메라 최적화 → SLAM 파라미터 최적화 → 파이프라인 종합 진단
RealSense D435i + RTAB-Map (rtabmap_ros) 기반 Live SLAM 전용.

실행 단계:
  Stage 1 - 카메라 / DDS / QoS 설정 최적 조합 측정 (기존 benchmark_hw.py 확장)
  Stage 2 - SLAM 파라미터 조합 측정 (DetectionRate, CornerNbThreads, OdomF2M 등)
  Stage 3 - 전체 파이프라인 종합 진단 (camera→SLAM 지연, /odom Hz, 토픽 누락 여부)

사용법:
  python3 benchmark_slam.py                # 전 단계 풀 측정
  python3 benchmark_slam.py --stage 1      # Stage 1만
  python3 benchmark_slam.py --stage 2      # Stage 2만 (Stage 1 결과 JSON 필요)
  python3 benchmark_slam.py --stage 3      # Stage 3만
  python3 benchmark_slam.py --quick        # 핵심 조합만 빠르게 측정
"""

import os
import sys
import time
import subprocess
import json
import argparse
import signal
import threading
from datetime import datetime

# repo 소스 실행 / 설치 실행 양쪽에서 auto_mobility 패키지 임포트 보장
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from auto_mobility.config import (
    CAMERA_RGB_TOPIC as TOPIC_COLOR,
    CAMERA_DEPTH_TOPIC,
    CAMERA_ALIGNED_DEPTH_TOPIC as TOPIC_DEPTH,
    CAMERA_INFO_TOPIC as TOPIC_CAMERA_INFO,
    CAMERA_IMU_TOPIC as TOPIC_IMU,
    ODOM_TOPIC as TOPIC_ODOM,
    MAP_TOPIC as TOPIC_MAP,
    CAMERA_PARAMS,
    LOG_DIR,
    FASTDDS_XML,
)

# ──────────────────────────── 터미널 컬러 ────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ──────────────────────────── 경로 설정 ──────────────────────────────
# PROJECT_DIR / CONFIG_DIR / LOG_DIR / FASTDDS_XML 는 config.py 단일 소스 사용

# ═══════════════════════════════════════════════════════════════════════
#  공통 유틸
# ═══════════════════════════════════════════════════════════════════════

def banner(title: str, color: str = BOLD):
    line = "═" * 60
    print(f"\n{color}{line}{RESET}")
    print(f"{color}  {title}{RESET}")
    print(f"{color}{line}{RESET}")

def step(msg: str):
    print(f"{CYAN}▶ {msg}{RESET}")

def ok(msg: str):
    print(f"   {GREEN}✓ {msg}{RESET}")

def warn(msg: str):
    print(f"   {YELLOW}⚠ {msg}{RESET}")

def err(msg: str):
    print(f"   {RED}✗ {msg}{RESET}")

def kill_pgroup(proc):
    """프로세스 그룹 전체를 안전하게 종료한다."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
#  시스템 환경 검사
# ═══════════════════════════════════════════════════════════════════════

def check_system():
    info = {
        "rmws"    : ["rmw_fastrtps_cpp"],
        "rmem_max": "Unknown",
        "usb_type": "USB 3.x (권장)",
        "nproc"   : os.cpu_count() or 4,
        "fastdds_xml_exists": os.path.exists(FASTDDS_XML),
    }

    # CycloneDDS 존재 여부
    if os.path.exists("/opt/ros/humble/lib/librmw_cyclonedds_cpp.so"):
        info["rmws"].append("rmw_cyclonedds_cpp")

    # 커널 소켓 버퍼
    try:
        r = subprocess.run(["sysctl", "net.core.rmem_max"],
                           stdout=subprocess.PIPE, text=True)
        if "=" in r.stdout:
            val = int(r.stdout.split("=")[1].strip())
            info["rmem_max"] = f"{val / (1024*1024):.1f} MB"
    except Exception:
        pass

    # USB 포트 속도
    try:
        r = subprocess.run(["rs-enumerate-devices", "-s"],
                           stdout=subprocess.PIPE, text=True)
        if "2.1" in r.stdout or "2.0" in r.stdout:
            info["usb_type"] = "USB 2.x (⚠ 720p 30fps 대역폭 주의)"
    except Exception:
        pass

    return info


# ═══════════════════════════════════════════════════════════════════════
#  토픽 Hz 측정 공통 함수
# ═══════════════════════════════════════════════════════════════════════

def measure_hz(topic: str, duration: float = 6.0, env=None) -> float:
    """
    `ros2 topic hz <topic>` 를 duration 초 동안 실행해 average rate 를 반환한다.
    측정 불가 시 0.0 반환.
    """
    cmd = f"ros2 topic hz {topic}"
    hz_val = 0.0
    try:
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, preexec_fn=os.setsid, env=env
        )
        time.sleep(duration)
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        out, _ = proc.communicate(timeout=4)
        for line in out.splitlines():
            if "average rate:" in line:
                hz_val = float(line.split("average rate:")[1].strip().split()[0])
                break
    except Exception:
        pass
    return hz_val


def measure_hz_multi(topics: list, duration: float = 6.0, env=None) -> dict:
    """여러 토픽을 병렬로 Hz 측정한다."""
    results = {}
    threads = []

    def _worker(t):
        results[t] = measure_hz(t, duration=duration, env=env)

    for t in topics:
        th = threading.Thread(target=_worker, args=(t,), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    return results


def measure_resources_of(process_name: str, samples: int = 4, interval: float = 0.5) -> tuple:
    """지정 프로세스 이름의 CPU 점유율(%) 및 RAM RSS(MB) 평균을 반환 (ps aux 기반)."""
    cpu_vals = []
    ram_vals = []
    for _ in range(samples):
        time.sleep(interval)
        try:
            r = subprocess.run(
                f"ps aux | grep {process_name} | grep -v grep | awk '{{print $3, $6}}'",
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            raw = r.stdout.strip().split("\n")
            for line in raw:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        cpu_vals.append(float(parts[0]))
                        ram_vals.append(float(parts[1]) / 1024.0)  # KB -> MB
                    except ValueError:
                        pass
        except Exception:
            pass
    avg_cpu = round(sum(cpu_vals) / len(cpu_vals), 1) if cpu_vals else 0.0
    avg_ram = round(sum(ram_vals) / len(ram_vals), 1) if ram_vals else 0.0
    return avg_cpu, avg_ram


def measure_db_growth(db_path: str, duration_sec: float) -> tuple:
    """DB 파일 크기(MB) 및 분당 디스크 I/O 생성율(MB/min)을 반환한다."""
    if not os.path.exists(db_path):
        return 0.0, 0.0
    size_mb = os.path.getsize(db_path) / (1024.0 * 1024.0)
    rate_mb_per_min = size_mb / (duration_sec / 60.0) if duration_sec > 0 else 0.0
    return round(size_mb, 2), round(rate_mb_per_min, 2)


def wait_for_topic(topic: str, max_timeout: float = 8.0, env=None) -> bool:
    """토픽이 메세지를 1개라도 수신할 때까지 동적으로 대기(최대 max_timeout초). 수신 시 즉시 True 반환."""
    start_time = time.time()
    time.sleep(1.0)
    while time.time() - start_time < max_timeout:
        try:
            r = subprocess.run(
                f"ros2 topic echo {topic} --once",
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env, timeout=2.0
            )
            if r.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(0.2)
    return False


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1: 카메라 / DDS / QoS 최적 조합
# ═══════════════════════════════════════════════════════════════════════

CAMERA_CONFIGS_FULL = [
    # (width, height, fps)
    (640,  480, 15),
    (640,  480, 30),
    (848,  480, 30),
    (1280, 720, 15),
    (1280, 720, 30),
]

CAMERA_CONFIGS_QUICK = [
    (640,  480, 30),
    (848,  480, 30),
    (1280, 720, 15),
    (1280, 720, 30),
]

QOS_LIST_FULL  = ["SENSOR_DATA", "DEFAULT"]
QOS_LIST_QUICK = ["SENSOR_DATA"]


def run_camera_test(rmw, use_shm, res_w, res_h, fps, qos,
                    sample_duration=5.0) -> dict:
    """카메라 노드 하나를 단독 실행해 image_raw Hz + CPU / RAM RSS 를 측정한다."""
    env = os.environ.copy()
    env["RMW_IMPLEMENTATION"] = rmw

    shm_label = "Default"
    if rmw == "rmw_fastrtps_cpp":
        if use_shm and os.path.exists(FASTDDS_XML):
            env["FASTRTPS_DEFAULT_PROFILES_FILE"] = FASTDDS_XML
            shm_label = "SHM"
        else:
            env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
            shm_label = "UDP"

    prof = f"{res_w}x{res_h}x{fps}"
    cmd = build_camera_cmd(prof, qos)

    label = f"{rmw}({shm_label}) | {res_w}x{res_h}@{fps}fps | QoS={qos}"
    step(f"[Stage 1] {label}")

    proc = subprocess.Popen(cmd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)

    # 동적 토픽 감지 (camera_info 토픽으로 RealSense UVC 오픈 감지)
    cam_active = wait_for_topic(TOPIC_CAMERA_INFO, max_timeout=8.0, env=env)
    if not cam_active:
        warn("  카메라 스트림 동적 감지 실패 -> 조기 종료(Early Exit)")
        kill_pgroup(proc)
        time.sleep(1.0)
        return {
            "rmw": rmw, "shm": shm_label, "resolution": f"{res_w}x{res_h}", "target_fps": fps,
            "qos": qos, "color_hz": 0.0, "depth_hz": 0.0, "drop_pct": 100.0, "cpu_pct": 0.0, "ram_mb": 0.0,
            "score": 0.0
        }

    # raw image_raw + raw depth 동시 측정 (샘플링 시간 4.0초)
    hz_map = measure_hz_multi(
        [TOPIC_COLOR, CAMERA_DEPTH_TOPIC],
        duration=sample_duration, env=env
    )
    cpu, ram_mb = measure_resources_of("realsense2_camera", samples=4, interval=sample_duration / 4)

    kill_pgroup(proc)
    time.sleep(1.5)  # 센서 안전 릴리즈 쿨다운 단축

    color_hz = hz_map.get(TOPIC_COLOR, 0.0)
    depth_hz = hz_map.get(CAMERA_DEPTH_TOPIC, 0.0)
    drop_pct = max(0.0, (fps - color_hz) / fps * 100) if fps > 0 else 100.0

    # 점수: FPS 달성(60) + 품질(30) + QoS(10) - RAM 사용량 감점
    fps_score     = min(1.0, color_hz / fps) * 60.0
    quality_score = (res_w * res_h / (1280 * 720)) * 15.0 + (fps / 30.0) * 15.0
    qos_score     = 10.0 if qos == "SENSOR_DATA" else 5.0
    total_score   = round(fps_score + quality_score + qos_score, 1)

    status = GREEN if drop_pct < 10 else (YELLOW if drop_pct < 25 else RED)
    print(f"   └─ {status}color: {color_hz:.1f}Hz  depth: {depth_hz:.1f}Hz  "
          f"drop: {drop_pct:.1f}%  CPU: {cpu}%  RAM: {ram_mb}MB  score: {total_score}{RESET}")

    return {
        "rmw": rmw, "shm": shm_label,
        "resolution": f"{res_w}x{res_h}", "target_fps": fps,
        "qos": qos,
        "color_hz": round(color_hz, 2), "depth_hz": round(depth_hz, 2),
        "drop_pct": round(drop_pct, 1), "cpu_pct": cpu, "ram_mb": ram_mb,
        "score": total_score,
    }


def run_stage1(sys_info, quick: bool) -> dict:
    banner("Stage 1 — 카메라 / DDS / QoS 최적 조합 탐색")

    if quick:
        # 빠른 모드: 현재 production 해상도(848x480) 포함 핵심 검증 (카메라 USB 반복 리셋 방지)
        configs = [
            ("rmw_fastrtps_cpp", True, 640, 480, 30, "SENSOR_DATA"),
            ("rmw_fastrtps_cpp", True, 848, 480, 30, "SENSOR_DATA"),
        ]
    else:
        # 정밀 모드: 핵심 해상도 3종 안정적으로 검증
        configs = [
            ("rmw_fastrtps_cpp", True, 640, 480, 30, "SENSOR_DATA"),
            ("rmw_fastrtps_cpp", True, 848, 480, 30, "SENSOR_DATA"),
            ("rmw_fastrtps_cpp", True, 1280, 720, 15, "SENSOR_DATA"),
        ]

    print(f"총 {len(configs)}개 카메라 세션 검증 시작\n")
    results = []
    for i, (rmw, shm, w, h, fps, qos) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}]", end=" ")
        r = run_camera_test(rmw, shm, w, h, fps, qos)
        results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    ok(f"Stage 1 완료. 최적: {best['rmw']}({best['shm']}) "
       f"{best['resolution']}@{best['target_fps']}fps "
       f"QoS={best['qos']}  score={best['score']}")

    return {"best": best, "all": results}


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2: SLAM(RTAB-Map) 파라미터 조합 측정 (CPU, RAM, Disk I/O 정밀 검증)
# ═══════════════════════════════════════════════════════════════════════

def _param_to_str(value) -> str:
    """bool 은 rtabmap/camera CLI 파싱과 일치하도록 소문자 'true'/'false' 로 변환."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _build_rtabmap_args(params: dict) -> str:
    """dict → '--Key Value ...' 형태의 rtabmap_args 문자열로 변환.

    ⚠️ Odom/*, OdomF2M/* 키는 rtabmap 메인 노드가 선언하지 않아
    ParameterNotDeclaredException → SIGABRT 크래시를 유발하므로 제외한다.
    (2026-08-11 실측 확인. 해당 키는 odom_args 로 전달해야 한다)
    """
    parts = []
    for k, v in params.items():
        if k != "align_depth" and not k.startswith("Odom/"):
            parts.append(f"--{k} {_param_to_str(v)}")
    return " ".join(parts)


def _build_odom_args(params: dict) -> str:
    """Odom/*, OdomF2M/* 키만 추출해 rgbd_odometry 전용 odom_args 문자열 생성."""
    parts = []
    for k, v in params.items():
        if k.startswith("Odom/"):
            parts.append(f"--{k} {_param_to_str(v)}")
    return " ".join(parts)


def build_camera_cmd(profile: str, qos: str, align_depth: bool = False) -> list:
    """config.CAMERA_PARAMS(production 단일 소스) 기준 카메라 노드 커맨드 생성.

    벤치마크가 바꾸는 해상도 프로파일/QoS 만 동적 오버라이드한다.
    (production 설정인 필터/에미터/동기화를 포함해 실환경과 정합되도록 함)
    """
    params = dict(CAMERA_PARAMS)
    params.update({
        "depth_module.depth_profile": profile,
        "rgb_camera.color_profile": profile,
        "align_depth.enable": align_depth,
        "color_qos": qos,
        "color_info_qos": qos,
        "depth_qos": qos,
        "depth_info_qos": qos,
    })
    cmd = ["ros2", "run", "realsense2_camera", "realsense2_camera_node", "--ros-args"]
    for k, v in params.items():
        cmd += ["-p", f"{k}:={_param_to_str(v)}"]
    cmd += ["-r", "__ns:=/camera", "-r", "__node:=camera"]
    return cmd


# SLAM 파라미터 후보 집합 (Depth 정렬, I/O 키프레임, F2M 로컬 맵, MinInliers 포함)
SLAM_PARAM_MATRIX_FULL = {
    "estimation_type": [
        ("EST=PnP(3D-2D)", {"Vis/EstimationType": 1}),
        ("EST=SVD(3D-3D)", {"Vis/EstimationType": 0}),
    ],
    "align_depth": [
        ("ALIGN=off", {"align_depth": False}),
        ("ALIGN=on",  {"align_depth": True}),
    ],
    "detection_rate": [
        ("DR=2",  {"Rtabmap/DetectionRate": 2}),
        ("DR=5",  {"Rtabmap/DetectionRate": 5}),
        ("DR=10", {"Rtabmap/DetectionRate": 10}),
    ],
    "vis_features": [
        ("VF=500",  {"Vis/MaxFeatures": 500}),
        ("VF=1000", {"Vis/MaxFeatures": 1000}),
        ("VF=1500", {"Vis/MaxFeatures": 1500}),
        ("VF=2000", {"Vis/MaxFeatures": 2000}),
    ],
    # ⚠️ 2026-08-11 파라미터 검증: Vis/CornerNbThreads는 RTAB-Map 0.23.7에서 제거됨 (OpenCV 자동 스레딩).
    #   OdomF2M/MaxFrames → OdomF2M/MaxSize(로컬 맵 최대 word 수, 기본 2000) 로 리네임.
    "f2m_size": [
        ("F2M=1000", {"OdomF2M/MaxSize": 1000}),
        ("F2M=2000", {"OdomF2M/MaxSize": 2000}),
        ("F2M=4000", {"OdomF2M/MaxSize": 4000}),
        ("F2M=8000", {"OdomF2M/MaxSize": 8000}),
    ],
    "stm_size": [
        ("STM=10",  {"Mem/STMSize": 10}),
        ("STM=100", {"Mem/STMSize": 100}),      # 현재 production 값 (과부하 검증)
    ],
    "keyframe": [
        ("KF_0.1", {"RGBD/LinearUpdate": 0.1, "RGBD/AngularUpdate": 0.1}),
        ("KF_0.2", {"RGBD/LinearUpdate": 0.2, "RGBD/AngularUpdate": 0.2}),
        ("KF_0.3", {"RGBD/LinearUpdate": 0.3, "RGBD/AngularUpdate": 0.3}),
    ],
    "min_inliers": [
        ("IN=6",  {"Vis/MinInliers": 6}),
        ("IN=10", {"Vis/MinInliers": 10}),
        ("IN=12", {"Vis/MinInliers": 12}),
    ],
    "max_depth": [
        ("MD=4", {"Vis/MaxDepth": 4.0}),   # 현재 production 값
        ("MD=8", {"Vis/MaxDepth": 8.0}),   # benchmark 검증값
    ],
}

SLAM_PARAM_MATRIX_QUICK = {
    "estimation_type": [
        ("EST=PnP(3D-2D)", {"Vis/EstimationType": 1}),
        ("EST=SVD(3D-3D)", {"Vis/EstimationType": 0}),
    ],
    "f2m_size": [
        ("F2M=1000", {"OdomF2M/MaxSize": 1000}),
        ("F2M=2000", {"OdomF2M/MaxSize": 2000}),
        ("F2M=4000", {"OdomF2M/MaxSize": 4000}),
    ],
    "stm_size": [
        ("STM=10",  {"Mem/STMSize": 10}),
        ("STM=100", {"Mem/STMSize": 100}),      # 현재 production 값
    ],
    "keyframe": [
        ("KF_0.1", {"RGBD/LinearUpdate": 0.1, "RGBD/AngularUpdate": 0.1}),
        ("KF_0.2", {"RGBD/LinearUpdate": 0.2, "RGBD/AngularUpdate": 0.2}),
    ],
    "min_inliers": [
        ("IN=6",  {"Vis/MinInliers": 6}),
        ("IN=10", {"Vis/MinInliers": 10}),
    ],
    "max_depth": [
        ("MD=4", {"Vis/MaxDepth": 4.0}),
        ("MD=8", {"Vis/MaxDepth": 8.0}),
    ],
}

# SLAM 기준 파라미터: production(RTABMAP_PARAMS)을 단일 소스로 사용해 drift 를 방지한다.
# 벤치마크가 독립적으로 변형하는 축(행렬) 값이 기준값을 오버라이드한다.
def _coerce_param(value):
    """RTABMAP_PARAMS 의 문자열 값을 벤치마크 처리용 네이티브 타입으로 변환."""
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    return value


def _load_slam_base_params() -> dict:
    from auto_mobility.launch.launch_common import RTABMAP_PARAMS
    base = {k: _coerce_param(v) for k, v in RTABMAP_PARAMS.items()}
    # 벤치마크 전용 launch 인자 (RTABMAP_PARAMS 에 없는 키)
    base.update({
        "approx_sync": "true",
        "approx_sync_max_interval": 0.15,
        "topic_queue_size": 30,
    })
    return base


SLAM_BASE_PARAMS = _load_slam_base_params()


def build_slam_test_configs(matrix: dict) -> list:
    """각 축의 값을 독립적으로 변경하는 1-at-a-time 방식으로 configs 생성."""
    default = {}
    default_labels = {}
    for axis, options in matrix.items():
        label, params = options[0]
        default.update(params)
        default_labels[axis] = label

    configs = []
    seen = set()

    def _add(labels_dict, params_dict):
        key = json.dumps(params_dict, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append({
                "label": " | ".join(labels_dict.values()),
                "params": {**SLAM_BASE_PARAMS, **params_dict},
            })

    _add(default_labels, default)

    for axis, options in matrix.items():
        for label, params in options[1:]:
            cur_labels  = {**default_labels, axis: label}
            cur_params  = {**default, **params}
            _add(cur_labels, cur_params)

    # ⚠️ 2026-08-11: Vis/CornerNbThreads는 RTAB-Map 0.23.7에서 제거됨. nproc 필터는 생략.
    return configs


def run_slam_test(camera_config: dict, slam_config: dict,
                  sample_duration: float = 5.0, env=None) -> dict:
    """
    지속적으로 켜져 있는 카메라 노드 위에서 RTAB-Map 노드만 단독 기동/종료하며
    CPU, RAM(RSS MB), Disk I/O(DB MB/min), Hz를 정밀 측정한다.
    (카메라 USB 버스 재연결 및 리셋 방지)
    """
    if env is None:
        env = os.environ.copy()

    rmw = camera_config["rmw"]
    env["RMW_IMPLEMENTATION"] = rmw
    if camera_config["shm"] == "SHM" and os.path.exists(FASTDDS_XML):
        env["FASTRTPS_DEFAULT_PROFILES_FILE"] = FASTDDS_XML
    else:
        env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)

    fps = camera_config["target_fps"]
    align_depth_enabled = bool(slam_config["params"].get("align_depth", False))
    depth_topic_to_use = TOPIC_DEPTH if align_depth_enabled else CAMERA_DEPTH_TOPIC

    LAUNCH_KEYS = {"approx_sync", "approx_sync_max_interval", "topic_queue_size", "align_depth"}
    rtabmap_only = {k: v for k, v in slam_config["params"].items()
                    if k not in LAUNCH_KEYS}
    rtabmap_args = _build_rtabmap_args(rtabmap_only)
    odom_args = _build_odom_args(rtabmap_only)

    bench_db_path = os.path.join(LOG_DIR, "bench_rtabmap.db")
    if os.path.exists(bench_db_path):
        try:
            os.remove(bench_db_path)
        except Exception:
            pass

    slam_cmd = [
        "ros2", "launch", "rtabmap_launch", "rtabmap.launch.py",
        f"rgb_topic:={TOPIC_COLOR}",
        f"depth_topic:={depth_topic_to_use}",
        f"camera_info_topic:={TOPIC_CAMERA_INFO}",
        "qos_image:=2",
        "qos_depth:=2",
        "qos_camera_info:=2",
        "frame_id:=camera_link",
        f"approx_sync:={slam_config['params'].get('approx_sync', 'true')}",
        f"approx_sync_max_interval:={slam_config['params'].get('approx_sync_max_interval', 0.15)}",
        f"topic_queue_size:={slam_config['params'].get('topic_queue_size', 30)}",
        "visual_odometry:=true",
        "rviz:=false",
        "rtabmap_viz:=false",
        f"database_path:={bench_db_path}",
        f"rtabmap_args:={rtabmap_args}",
        f"odom_args:={odom_args}",
    ]

    label = slam_config["label"]
    step(f"[Stage 2] {label}")

    slam_proc = subprocess.Popen(slam_cmd, env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 preexec_fn=os.setsid)

    # 동적 토픽 감지 (SLAM /rtabmap/odom)
    slam_ok_start = wait_for_topic(TOPIC_ODOM, max_timeout=4.0, env=env)

    if not slam_ok_start:
        warn("  SLAM 노드 초기화 감지 실패 -> 조기 종료(Early Exit)")
        kill_pgroup(slam_proc)
        time.sleep(0.5)
        return {
            "label": label, "params": {k: str(v) for k, v in slam_config["params"].items()},
            "color_hz": 0.0, "odom_hz": 0.0, "map_hz": 0.0, "cpu_cam": 0.0, "ram_cam": 0.0,
            "cpu_slam": 0.0, "ram_slam": 0.0, "db_growth_mb_min": 0.0, "camera_ok": False,
            "slam_ok": False, "score": 0.0
        }

    # Hz 및 프로세스 자원 병렬 측정 (5.0초 샘플링)
    hz_map = measure_hz_multi(
        [TOPIC_COLOR, TOPIC_ODOM, TOPIC_MAP],
        duration=sample_duration, env=env
    )
    cpu_cam, ram_cam   = measure_resources_of("realsense2_camera", samples=3, interval=sample_duration / 3)
    cpu_slam, ram_slam = measure_resources_of("rtabmap",           samples=3, interval=sample_duration / 3)

    kill_pgroup(slam_proc)
    time.sleep(0.8)

    db_size_mb, db_rate_mb_min = measure_db_growth(bench_db_path, sample_duration)
    if os.path.exists(bench_db_path):
        try:
            os.remove(bench_db_path)
        except Exception:
            pass

    color_hz = hz_map.get(TOPIC_COLOR, 0.0)
    odom_hz  = hz_map.get(TOPIC_ODOM,  0.0)
    map_hz   = hz_map.get(TOPIC_MAP,   0.0)

    camera_ok = color_hz >= fps * 0.8
    slam_ok   = odom_hz  >= 1.0

    # 종합 스코어링 공식: (Odom Hz + Color Hz) - CPU/RAM/Disk I/O 감점 + Inlier 가산점
    dr = int(slam_config["params"].get("Rtabmap/DetectionRate", 5))
    odom_score   = min(1.0, odom_hz / max(dr, 1)) * 40.0
    color_score  = min(1.0, color_hz / fps) * 20.0
    map_score    = 10.0 if map_hz > 0 else 0.0
    cpu_penalty  = min(15.0, (cpu_cam + cpu_slam) * 0.1)
    io_penalty   = min(15.0, db_rate_mb_min * 0.5)
    inlier_val   = int(slam_config["params"].get("Vis/MinInliers", 10))
    inlier_bonus = 10.0 if inlier_val >= 10 else 5.0

    total_score = round(odom_score + color_score + map_score + inlier_bonus - cpu_penalty - io_penalty, 1)

    status = GREEN if (camera_ok and slam_ok) else (YELLOW if slam_ok else RED)
    print(f"   └─ {status}color:{color_hz:.1f}Hz  odom:{odom_hz:.1f}Hz  "
          f"RAM cam:{ram_cam}MB slam:{ram_slam}MB  Disk IO:{db_rate_mb_min}MB/min  "
          f"CPU cam:{cpu_cam}% slam:{cpu_slam}%  score:{total_score}{RESET}")

    return {
        "label"          : label,
        "params"         : {k: str(v) for k, v in slam_config["params"].items()},
        "color_hz"       : round(color_hz, 2),
        "odom_hz"        : round(odom_hz,  2),
        "map_hz"         : round(map_hz,   2),
        "cpu_cam"        : cpu_cam,
        "ram_cam"        : ram_cam,
        "cpu_slam"       : cpu_slam,
        "ram_slam"       : ram_slam,
        "db_growth_mb_min": db_rate_mb_min,
        "camera_ok"      : camera_ok,
        "slam_ok"        : slam_ok,
        "score"          : total_score,
    }


def run_stage2(stage1_result: dict, sys_info: dict, quick: bool) -> dict:
    banner("Stage 2 — SLAM 파라미터 최적 조합 탐색")

    best_cam = stage1_result["best"]
    matrix   = SLAM_PARAM_MATRIX_QUICK if quick else SLAM_PARAM_MATRIX_FULL
    configs  = build_slam_test_configs(matrix)

    env = os.environ.copy()
    rmw = best_cam["rmw"]
    env["RMW_IMPLEMENTATION"] = rmw
    if best_cam["shm"] == "SHM" and os.path.exists(FASTDDS_XML):
        env["FASTRTPS_DEFAULT_PROFILES_FILE"] = FASTDDS_XML

    res_w, res_h = best_cam["resolution"].split("x")
    fps   = best_cam["target_fps"]
    qos   = best_cam["qos"]
    prof  = f"{res_w}x{res_h}x{fps}"

    print(f"카메라 설정: {best_cam['rmw']}({best_cam['shm']}) "
          f"{best_cam['resolution']}@{fps}fps QoS={qos}")

    step("Stage 2 시작: 카메라 지속 세션 기동 중... (USB 리셋 방지)")
    camera_cmd = build_camera_cmd(prof, qos)

    cam_proc = subprocess.Popen(camera_cmd, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=os.setsid)

    cam_ready = wait_for_topic(TOPIC_CAMERA_INFO, max_timeout=8.0, env=env)
    if not cam_ready:
        err("Stage 2 카메라 기동 실패 (센서 연결 상태 점검 필요)")
        kill_pgroup(cam_proc)
        default_params = {**SLAM_BASE_PARAMS, "align_depth": False}
        return {"best": {"label": "N/A", "score": 0.0, "params": default_params}, "all": []}

    ok("카메라 지속 세션 연결 성공! SLAM 파라미터 조합 측정 시작\n")
    print(f"총 {len(configs)}개 SLAM 파라미터 조합 측정 시작\n")

    results = []
    for i, sc in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}]", end=" ")
        r = run_slam_test(best_cam, sc, sample_duration=5.0, env=env)
        results.append(r)

    step("Stage 2 종료: 카메라 지속 세션 정리 중...")
    kill_pgroup(cam_proc)
    time.sleep(1.0)

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0] if results else {"label": "N/A", "score": 0.0}
    ok(f"Stage 2 완료. 최적 조합: {best['label']}  score={best['score']}")

    return {"best": best, "all": results, "cam_proc": cam_proc}


# ═══════════════════════════════════════════════════════════════════════
#  Stage 3: 전체 파이프라인 종합 진단
# ═══════════════════════════════════════════════════════════════════════

PIPELINE_TOPICS = [
    (TOPIC_COLOR,       "image_raw (color)",    15),
    (TOPIC_DEPTH,       "aligned_depth",        15),
    (TOPIC_CAMERA_INFO, "camera_info",          15),
    (TOPIC_IMU,         "imu",                 100),
    (TOPIC_ODOM,        "rtabmap/odom",          2),
    (TOPIC_MAP,         "rtabmap/mapData",       1),
]


def run_stage3(stage1_result: dict, stage2_result: dict, quick: bool) -> dict:
    banner("Stage 3 — 전체 파이프라인 종합 진단")

    best_cam  = stage1_result["best"]
    best_slam = stage2_result["best"]

    env = os.environ.copy()
    env["RMW_IMPLEMENTATION"] = best_cam["rmw"]
    if best_cam["shm"] == "SHM" and os.path.exists(FASTDDS_XML):
        env["FASTRTPS_DEFAULT_PROFILES_FILE"] = FASTDDS_XML

    res_w, res_h = best_cam["resolution"].split("x")
    fps  = best_cam["target_fps"]
    qos  = best_cam["qos"]
    prof = f"{res_w}x{res_h}x{fps}"

    align_depth_enabled = bool(best_slam["params"].get("align_depth", False))
    depth_topic_to_use = TOPIC_DEPTH if align_depth_enabled else CAMERA_DEPTH_TOPIC

    print(f"최적 카메라 설정: {best_cam['rmw']}({best_cam['shm']}) "
          f"{best_cam['resolution']}@{fps}fps QoS={qos}")
    print(f"최적 SLAM 파라미터: {best_slam['label']}\n")

    cam_cmd = build_camera_cmd(prof, qos, align_depth=align_depth_enabled)

    LAUNCH_KEYS = {"approx_sync", "approx_sync_max_interval", "topic_queue_size", "align_depth"}
    rtabmap_only = {k: v for k, v in best_slam["params"].items()
                    if k not in LAUNCH_KEYS}
    rtabmap_args = _build_rtabmap_args(rtabmap_only)
    odom_args = _build_odom_args(rtabmap_only)

    slam_cmd = [
        "ros2", "launch", "rtabmap_launch", "rtabmap.launch.py",
        f"rgb_topic:={TOPIC_COLOR}",
        f"depth_topic:={depth_topic_to_use}",
        f"camera_info_topic:={TOPIC_CAMERA_INFO}",
        "qos_image:=2", "qos_depth:=2", "qos_camera_info:=2",
        "frame_id:=camera_link",
        "visual_odometry:=true",
        "rviz:=false", "rtabmap_viz:=false",
        f"rtabmap_args:={rtabmap_args}",
        f"odom_args:={odom_args}",
    ]

    cam_proc = stage2_result.get("cam_proc")
    if cam_proc is None or cam_proc.poll() is not None:
        step("카메라 노드 기동 중...")
        cam_proc = subprocess.Popen(cam_cmd, env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     preexec_fn=os.setsid)
        wait_for_topic(TOPIC_CAMERA_INFO, max_timeout=8.0, env=env)
    else:
        ok("Stage 2 지속 카메라 세션을 Stage 3로 연속 유지합니다 (USB 리셋 방지).")

    step("RTAB-Map 노드 기동 중...")
    slam_proc = subprocess.Popen(slam_cmd, env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 preexec_fn=os.setsid)
    wait_for_topic(TOPIC_ODOM, max_timeout=6.0, env=env)

    sample_dur = 8.0 if quick else 15.0
    step(f"전체 토픽 Hz 측정 중 ({sample_dur}초)...")

    pipeline_topics = [
        (TOPIC_COLOR,       "image_raw (color)",    15),
        (depth_topic_to_use,"depth (raw/aligned)",  15),
        (TOPIC_CAMERA_INFO, "camera_info",          15),
        (TOPIC_IMU,         "imu",                 100),
        (TOPIC_ODOM,        "rtabmap/odom",          2),
        (TOPIC_MAP,         "rtabmap/mapData",       1),
    ]

    topics = [t for t, _, _ in pipeline_topics]
    hz_map = measure_hz_multi(topics, duration=sample_dur, env=env)

    cpu_cam, ram_cam   = measure_resources_of("realsense2_camera", samples=5, interval=sample_dur / 5)
    cpu_slam, ram_slam = measure_resources_of("rtabmap",           samples=5, interval=sample_dur / 5)

    kill_pgroup(slam_proc)
    time.sleep(1.0)
    if cam_proc:
        kill_pgroup(cam_proc)
    time.sleep(1.5)

    print("\n  ┌─ 토픽별 상태 ──────────────────────────────────────────")
    diag = []
    for topic, label, min_hz in PIPELINE_TOPICS:
        measured = hz_map.get(topic, 0.0)
        if measured >= min_hz:
            symbol = f"{GREEN}✓{RESET}"
            status = "OK"
        elif measured > 0:
            symbol = f"{YELLOW}△{RESET}"
            status = "저하"
        else:
            symbol = f"{RED}✗{RESET}"
            status = "없음"
        print(f"  │  {symbol} [{label:30s}] {measured:6.1f} Hz  (기준: ≥{min_hz}Hz)  → {status}")
        diag.append({
            "topic": topic, "label": label,
            "min_hz": min_hz, "measured_hz": round(measured, 2),
            "status": status,
        })
    print("  └──────────────────────────────────────────────────────")

    print(f"\n  리소스 점유율 |  카메라 CPU: {cpu_cam}% RAM: {ram_cam}MB   |   SLAM CPU: {cpu_slam}% RAM: {ram_slam}MB")

    all_ok    = all(d["status"] == "OK" for d in diag)
    slam_live = diag[4]["status"] != "없음"  # TOPIC_ODOM
    map_live  = diag[5]["status"] != "없음"  # TOPIC_MAP

    if all_ok:
        print(f"\n  {GREEN}{BOLD}✅ 파이프라인 전체 정상 동작. RViz2 에서 데이터 확인 가능합니다!{RESET}")
    elif slam_live:
        print(f"\n  {YELLOW}{BOLD}⚠️  SLAM 동작 중이지만 일부 토픽 성능 저하. 보고서를 확인하세요.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}❌ SLAM이 시작되지 않았습니다. Stage 2 파라미터를 재검토하세요.{RESET}")

    return {
        "diagnostics": diag,
        "cpu_cam"    : cpu_cam,
        "ram_cam"    : ram_cam,
        "cpu_slam"   : cpu_slam,
        "ram_slam"   : ram_slam,
        "all_ok"     : all_ok,
        "slam_live"  : slam_live,
        "map_live"   : map_live,
    }


# ═══════════════════════════════════════════════════════════════════════
#  보고서 생성
# ═══════════════════════════════════════════════════════════════════════

def write_report(sys_info: dict, s1: dict, s2: dict, s3: dict,
                 timestamp: str) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    report_path = os.path.join(LOG_DIR, f"slam_benchmark_{timestamp}.md")

    best_cam  = s1["best"]
    best_slam = s2["best"]

    slam_params = best_slam["params"]
    rtabmap_args_lines = []
    for k, v in slam_params.items():
        if k not in {"approx_sync", "approx_sync_max_interval", "topic_queue_size", "align_depth"}:
            rtabmap_args_lines.append(f"    '--{k} {_param_to_str(v)} '")
    suggested_rtabmap_args = "RTABMAP_ARGS = (\n" + "\n".join(rtabmap_args_lines) + "\n)"

    md = f"""# 📊 통합 SLAM 파이프라인 벤치마크 보고서 (CPU, RAM, Disk I/O 하드웨어 가속 검증)

- **생성 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **대상 장비**: Intel RealSense D435i + RTAB-Map (rtabmap_ros)
- **OS 네트워크 버퍼**: `{sys_info['rmem_max']}`
- **USB 연결 상태**: `{sys_info['usb_type']}`
- **vCPU 수**: {sys_info['nproc']}
- **FastDDS XML 존재**: `{'✓' if sys_info['fastdds_xml_exists'] else '✗'}`

---

## 🏆 최종 권장 설정

### 카메라 / DDS / QoS
| 항목 | 값 |
|:---|:---|
| DDS 미들웨어 | `{best_cam['rmw']}` (`{best_cam['shm']}`) |
| 해상도 & FPS | `{best_cam['resolution']} @ {best_cam['target_fps']} fps` |
| QoS | `{best_cam['qos']}` |
| 실측 color Hz | **{best_cam['color_hz']} Hz** |
| 실측 depth Hz | **{best_cam['depth_hz']} Hz** |
| 프레임 손실률 | {best_cam['drop_pct']}% |
| 카메라 CPU / RAM | **{best_cam['cpu_pct']}%** / **{best_cam['ram_mb']} MB** |

### SLAM 파라미터
| 항목 | 값 |
|:---|:---|
| 조합 라벨 | `{best_slam['label']}` |
| /odom Hz | **{best_slam['odom_hz']} Hz** |
| /mapData Hz | **{best_slam['map_hz']} Hz** |
| 프로세스 CPU (Cam / SLAM) | **{best_slam['cpu_cam']}%** / **{best_slam['cpu_slam']}%** |
| 메모리 RSS (Cam / SLAM) | **{best_slam['ram_cam']} MB** / **{best_slam['ram_slam']} MB** |
| Disk I/O (DB 생성율) | **{best_slam['db_growth_mb_min']} MB/min** |
| 종합 점수 | **{best_slam['score']} 점** |

---

## 💡 `launch_common.py` 반영 제안

아래 내용을 `src/auto_mobility/launch_common.py` 의 `RTABMAP_ARGS` 에 적용하세요:

```python
{suggested_rtabmap_args}
```

---

## 🔬 Stage 3 — 전체 파이프라인 진단 결과

"""
    diag = s3["diagnostics"]
    md += "| 토픽 | 기준 (≥Hz) | 실측 Hz | 상태 |\n"
    md += "|:---|:---:|:---:|:---:|\n"
    for d in diag:
        icon = "✅" if d["status"] == "OK" else ("⚠️" if d["status"] == "저하" else "❌")
        md += f"| `{d['topic']}` | {d['min_hz']} | **{d['measured_hz']}** | {icon} {d['status']} |\n"

    overall = "✅ 전체 정상" if s3["all_ok"] else ("⚠️ 일부 저하" if s3["slam_live"] else "❌ SLAM 미동작")
    md += f"\n**종합 판정**: {overall}  |  카메라(CPU: {s3['cpu_cam']}%, RAM: {s3['ram_cam']}MB)  |  SLAM(CPU: {s3['cpu_slam']}%, RAM: {s3['ram_slam']}MB)\n"

    # ─── Stage 1 전체 결과 테이블 ─────────────────────────────────────
    md += "\n---\n\n## 📋 Stage 1 — 카메라/DDS 전체 결과 (성능순)\n\n"
    md += "| 순위 | RMW | SHM | 해상도@FPS | QoS | color Hz | depth Hz | 손실률 | CPU | RAM | 점수 |\n"
    md += "|:---:|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    for i, r in enumerate(s1["all"], 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
        md += (f"| {medal} | `{r['rmw']}` | `{r['shm']}` | "
               f"`{r['resolution']}@{r['target_fps']}fps` | `{r['qos']}` | "
               f"**{r['color_hz']}** | {r['depth_hz']} | "
               f"{r['drop_pct']}% | {r['cpu_pct']}% | {r['ram_mb']}MB | **{r['score']}** |\n")

    # ─── Stage 2 전체 결과 테이블 ─────────────────────────────────────
    md += "\n---\n\n## 📋 Stage 2 — SLAM 파라미터 전체 결과 (성능순)\n\n"
    md += "| 순위 | 조합 | odom Hz | map Hz | Cam (CPU/RAM) | SLAM (CPU/RAM) | Disk I/O | 점수 |\n"
    md += "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    for i, r in enumerate(s2["all"], 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
        cam_ok  = "✅" if r["camera_ok"] else "⚠️"
        slam_ok = "✅" if r["slam_ok"]  else "❌"
        md += (f"| {medal} | `{r['label']}` | "
               f"**{r['odom_hz']}** {slam_ok} | {r['map_hz']} | "
               f"{r['cpu_cam']}% / {r['ram_cam']}MB {cam_ok} | {r['cpu_slam']}% / {r['ram_slam']}MB | {r['db_growth_mb_min']} MB/min | **{r['score']}** |\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    return report_path


# ═══════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="통합 SLAM 파이프라인 벤치마크 (카메라 → DDS/QoS → SLAM 파라미터 → 종합 진단)"
    )
    parser.add_argument("--quick",  action="store_true",
                        help="핵심 조합만 빠르게 측정 (전체 대비 약 1/3 시간)")
    parser.add_argument("--stage",  type=int, choices=[1, 2, 3], default=0,
                        help="특정 단계만 실행 (0=전체, 1=카메라, 2=SLAM, 3=진단)")
    parser.add_argument("--stage1-json", type=str, default=None,
                        help="Stage 1 결과 JSON 파일 경로 (--stage 2/3 실행 시 사용)")
    parser.add_argument("--stage2-json", type=str, default=None,
                        help="Stage 2 결과 JSON 파일 경로 (--stage 3 실행 시 사용)")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    banner("통합 SLAM 파이프라인 벤치마크 (CPU, RAM, Disk I/O 강화판)", BOLD + BLUE)
    print(f"  모드: {'빠른 측정' if args.quick else '정밀 측정'}")
    print(f"  대상 단계: {args.stage if args.stage else '1 → 2 → 3 (전체)'}\n")

    sys_info = check_system()
    print(f"  RMW 목록: {', '.join(sys_info['rmws'])}")
    print(f"  네트워크 버퍼: {sys_info['rmem_max']}")
    print(f"  USB 상태: {sys_info['usb_type']}")
    print(f"  vCPU: {sys_info['nproc']}")
    print(f"  FastDDS XML: {'존재' if sys_info['fastdds_xml_exists'] else '없음'}\n")

    # ── Stage 1 ───────────────────────────────────────────────────────
    s1 = None
    if args.stage in (0, 1):
        s1 = run_stage1(sys_info, quick=args.quick)
        s1_path = os.path.join(LOG_DIR, f"slam_bench_s1_{timestamp}.json")
        with open(s1_path, "w", encoding="utf-8") as f:
            json.dump(s1, f, ensure_ascii=False, indent=2)
        ok(f"Stage 1 결과 저장: {s1_path}")
    elif args.stage1_json:
        with open(args.stage1_json, encoding="utf-8") as f:
            s1 = json.load(f)
        ok(f"Stage 1 결과 로드: {args.stage1_json}")
    else:
        err("--stage 2/3 실행 시 --stage1-json 이 필요합니다.")
        sys.exit(1)

    if args.stage == 1:
        print(f"\n{BOLD}Stage 1 완료. --stage 2 로 SLAM 파라미터 측정을 계속하세요.{RESET}")
        return

    # ── Stage 2 ───────────────────────────────────────────────────────
    s2 = None
    if args.stage in (0, 2):
        s2 = run_stage2(s1, sys_info, quick=args.quick)
        s2_path = os.path.join(LOG_DIR, f"slam_bench_s2_{timestamp}.json")
        s2_save = {k: v for k, v in s2.items() if k != "cam_proc"}
        with open(s2_path, "w", encoding="utf-8") as f:
            json.dump(s2_save, f, ensure_ascii=False, indent=2)
        ok(f"Stage 2 결과 저장: {s2_path}")
    elif args.stage2_json:
        with open(args.stage2_json, encoding="utf-8") as f:
            s2 = json.load(f)
        ok(f"Stage 2 결과 로드: {args.stage2_json}")
    else:
        err("--stage 3 실행 시 --stage2-json 이 필요합니다.")
        sys.exit(1)

    if args.stage == 2:
        print(f"\n{BOLD}Stage 2 완료. --stage 3 로 종합 진단을 계속하세요.{RESET}")
        return

    # ── Stage 3 ───────────────────────────────────────────────────────
    s3 = run_stage3(s1, s2, quick=args.quick)

    # ── 보고서 ────────────────────────────────────────────────────────
    report_path = write_report(sys_info, s1, s2, s3, timestamp)

    banner("벤치마크 완료", GREEN + BOLD)
    print(f"  보고서: {report_path}")
    best_cam  = s1["best"]
    best_slam = s2["best"]
    print(f"\n  {BOLD}[최적 카메라]{RESET} {best_cam['rmw']}({best_cam['shm']}) "
          f"{best_cam['resolution']}@{best_cam['target_fps']}fps  QoS={best_cam['qos']}")
    print(f"  {BOLD}[최적 SLAM  ]{RESET} {best_slam['label']}")
    print(f"\n  ※ 보고서의 'launch_common.py 반영 제안' 섹션을 참고해 파라미터를 업데이트하세요.")


if __name__ == "__main__":
    main()
