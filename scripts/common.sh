#!/bin/bash

set -e

source /opt/ros/$ROS_DISTRO/setup.bash 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/ros2_data"

BAG_DIR="$DATA_DIR/bags"
FRAME_DIR="$DATA_DIR/frames"
DB_DIR="$DATA_DIR/databases"
POINTCLOUD_DIR="$DATA_DIR/pointclouds"
MESH_DIR="$DATA_DIR/meshes"
TRAJECTORY_DIR="$DATA_DIR/trajectories"
EVALUATION_DIR="$DATA_DIR/evaluations"
BENCHMARK_DIR="$DATA_DIR/benchmarks"
ISAAC_DIR="$DATA_DIR/isaac_sim"
LOG_DIR="$DATA_DIR/logs"
SHARED_DIR="/mnt/c/ubuntu_shared"

RAM_BAG_DIR="/dev/shm/ros2_bags"
STORAGE_FORMAT="mcap"


# config/topics.yaml 를 단일 소스로 토픽명 로드 (기존 환경변수 오버라이드 유지)
if [ -f "$PROJECT_DIR/config/topics.yaml" ] && [ -z "$RGB_COMPRESSED_TOPIC" ]; then
    eval "$(python3 - "$PROJECT_DIR/config/topics.yaml" <<'PYEOF'
import sys, os, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    topics = yaml.safe_load(f) or {}
camera = topics.get("camera", {})
rtabmap = topics.get("rtabmap", {})
mapping = {
    "RGB_TOPIC": camera.get("rgb_topic"),
    "RGB_COMPRESSED_TOPIC": camera.get("rgb_compressed_topic"),
    "DEPTH_TOPIC": camera.get("depth_topic"),
    "DEPTH_COMPRESSED_TOPIC": camera.get("depth_compressed_topic"),
    "ALIGNED_DEPTH_TOPIC": camera.get("aligned_depth_topic"),
    "ALIGNED_DEPTH_COMPRESSED_TOPIC": camera.get("aligned_depth_compressed_topic"),
    "CAMERA_INFO_TOPIC": camera.get("camera_info_topic"),
    "CAMERA_INFO_WINDOWS_TOPIC": camera.get("camera_info_windows_topic"),
    "INFRA1_TOPIC": camera.get("infra1_topic"),
    "INFRA1_COMPRESSED_TOPIC": camera.get("infra1_compressed_topic"),
    "INFRA1_INFO_WINDOWS_TOPIC": camera.get("infra1_camera_info_windows_topic"),
    "INFRA2_TOPIC": camera.get("infra2_topic"),
    "INFRA2_COMPRESSED_TOPIC": camera.get("infra2_compressed_topic"),
    "INFRA2_INFO_WINDOWS_TOPIC": camera.get("infra2_camera_info_windows_topic"),
    "IMU_TOPIC": camera.get("imu_topic"),
    "IMU_FILTERED_TOPIC": camera.get("imu_filtered_topic"),
    "ODOM_TOPIC": rtabmap.get("odom_topic"),
    "MAP_TOPIC": rtabmap.get("map_topic"),
    "CLOUD_MAP_TOPIC": rtabmap.get("cloud_map_topic"),
    "CLOUD_MAP_LITE_TOPIC": rtabmap.get("cloud_map_lite_topic"),
}
for var, val in mapping.items():
    if val and var not in os.environ:
        print(f'export {var}="{val}"')
PYEOF
)"
fi

# yaml 누락/공백 시 기본값 폴백 (환경변수 오버라이드 우선)
RGB_TOPIC="${RGB_TOPIC:-/camera/camera/color/image_raw}"
RGB_COMPRESSED_TOPIC="${RGB_COMPRESSED_TOPIC:-/camera/camera/color/image_raw/compressed}"
DEPTH_TOPIC="${DEPTH_TOPIC:-/camera/camera/depth/image_rect_raw}"
DEPTH_COMPRESSED_TOPIC="${DEPTH_COMPRESSED_TOPIC:-/camera/camera/depth/image_rect_raw/compressedDepth}"
ALIGNED_DEPTH_TOPIC="${ALIGNED_DEPTH_TOPIC:-/camera/camera/aligned_depth_to_color/image_raw}"
ALIGNED_DEPTH_COMPRESSED_TOPIC="${ALIGNED_DEPTH_COMPRESSED_TOPIC:-/camera/camera/aligned_depth_to_color/image_raw/compressedDepth}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-/camera/camera/color/camera_info}"
CAMERA_INFO_WINDOWS_TOPIC="${CAMERA_INFO_WINDOWS_TOPIC:-/camera/camera/color/camera_info_windows}"
INFRA1_TOPIC="${INFRA1_TOPIC:-/camera/camera/infra1/image_rect_raw}"
INFRA1_COMPRESSED_TOPIC="${INFRA1_COMPRESSED_TOPIC:-/camera/camera/infra1/image_rect_raw/compressed}"
INFRA1_INFO_WINDOWS_TOPIC="${INFRA1_INFO_WINDOWS_TOPIC:-/camera/camera/infra1/camera_info_windows}"
INFRA2_TOPIC="${INFRA2_TOPIC:-/camera/camera/infra2/image_rect_raw}"
INFRA2_COMPRESSED_TOPIC="${INFRA2_COMPRESSED_TOPIC:-/camera/camera/infra2/image_rect_raw/compressed}"
INFRA2_INFO_WINDOWS_TOPIC="${INFRA2_INFO_WINDOWS_TOPIC:-/camera/camera/infra2/camera_info_windows}"
IMU_TOPIC="${IMU_TOPIC:-/camera/camera/imu}"
IMU_FILTERED_TOPIC="${IMU_FILTERED_TOPIC:-/camera/camera/imu/filtered}"
ODOM_TOPIC="${ODOM_TOPIC:-/rtabmap/odom}"
MAP_TOPIC="${MAP_TOPIC:-/rtabmap/mapData}"
CLOUD_MAP_TOPIC="${CLOUD_MAP_TOPIC:-/rtabmap/cloud_map}"
CLOUD_MAP_LITE_TOPIC="${CLOUD_MAP_LITE_TOPIC:-/rtabmap/cloud_map_lite}"

# 카메라 프로파일/해상도/USB 기준 단일 소스 (config.py)
if [ -z "$CAMERA_PROFILE" ]; then
    CAMERA_PROFILE="$(PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH" python3 - <<'PYEOF' 2>/dev/null || true
from auto_mobility.config import CAMERA_PROFILE
print(CAMERA_PROFILE)
PYEOF
)"
fi
CAMERA_PROFILE="${CAMERA_PROFILE:-640x480x30}"
CAMERA_RESOLUTION="${CAMERA_RESOLUTION:-${CAMERA_PROFILE%x*}}"

if [ -z "$USB_3_MIN_SPEED_MBPS" ]; then
    USB_3_MIN_SPEED_MBPS="$(PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH" python3 - <<'PYEOF' 2>/dev/null || true
from auto_mobility.config import USB_3_MIN_SPEED_MBPS
print(USB_3_MIN_SPEED_MBPS)
PYEOF
)"
fi
USB_3_MIN_SPEED_MBPS="${USB_3_MIN_SPEED_MBPS:-5000}"

# 카메라 구동 위치 (기본값: remote - Windows 네이티브 RealSense D435i 스트림 수신)
#   remote: Windows 네이티브에서 발행하는 토픽을 수신 (기본)
#   local : WSL USB 패스스루(usbipd) — WSL에서 camera.launch.py 로 직접 구동
CAMERA_MODE="${CAMERA_MODE:-remote}"
export CAMERA_MODE
# 원격 카메라 시 압축 토픽 기본 사용 (로컬은 raw 선택 가능)
USE_COMPRESSED="${USE_COMPRESSED:-true}"
export USE_COMPRESSED


# ROS 2 도메인 ID 기본값 (Windows 카메라 env_ros2.bat 의 42 와 일치)
# 미설정 시 42 로 자동 설정 → Windows/WSL 양쪽 같은 도메인에서 토픽 수신 가능
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_DOMAIN_ID

# ── DDS 미들웨어: CycloneDDS 고정 (2026-08-14) ──
# FastDDS는 WSL2에서 SHM/Data-Sharing(무복사)이 rclpy(Python) ↔ rclcpp(C++) 간
# 불안정하여 rtabmap(C++) 노드가 republish(Python)의 이미지를 수신하지 못하는 문제 발생.
# CycloneDDS(설치: ros-humble-rmw-cyclonedds-cpp)는 SHM에 의존하지 않아 안정적.
# Windows 쪽 env_ros2.bat 도 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp 로 맞춰야 양방향 통신 유지.
if [ -z "${RMW_IMPLEMENTATION:-}" ]; then
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
fi
export RMW_IMPLEMENTATION

# CycloneDDS IP 동적 동기화 (WSL ↔ Windows 간 IP 자동 매핑)
python3 "$PROJECT_DIR/scripts/utils/sync_dds_config.py" 2>/dev/null || true

# CycloneDDS 설정 파일 (Windows와의 정적 피어 포함). 없으면 기본(멀티캐스트)으로 동작.
if [ -z "${CYCLONEDDS_URI:-}" ] && [ -f "$PROJECT_DIR/config/dds/cyclonedds_camera.xml" ]; then
    export CYCLONEDDS_URI="file://$PROJECT_DIR/config/dds/cyclonedds_camera.xml"
fi

mkdir -p "$BAG_DIR" "$DB_DIR" "$POINTCLOUD_DIR" "$MESH_DIR" "$TRAJECTORY_DIR" "$BENCHMARK_DIR" "$ISAAC_DIR" "$LOG_DIR" "$RAM_BAG_DIR"


# 토픽 존재/수신 확인 (ros2 CLI 데몬 hang 문제 회피 — topic_probe.py 사용)
# 사용법: topic_exists <topic> [timeout_sec]  → 0=수신 확인 / 1=없음
topic_exists() {
    python3 "$PROJECT_DIR/scripts/utils/topic_probe.py" "$1" "${2:-1.5}" >/dev/null 2>&1
}

# 여러 토픽 동시 확인 (배치 프로브)
# 사용법: topic_probe_batch [timeout_sec] <topic1> <topic2> ...
# 출력: 각 줄에 "TOPIC:0"(성공) 또는 "TOPIC:1"(실패)
topic_probe_batch() {
    local timeout="${1:-2.0}"
    shift
    python3 "$PROJECT_DIR/scripts/utils/topic_probe.py" --batch "$@" --timeout "$timeout"
}

if [ -f "$PROJECT_DIR/install/setup.bash" ]; then
    source "$PROJECT_DIR/install/setup.bash"
elif [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
    source "$HOME/ros2_ws/install/setup.bash"
fi

# 실행 파일(republish.py 등) 미설치 시 자동 colcon build 빌드 실행
if [ ! -f "$PROJECT_DIR/install/auto_mobility/lib/auto_mobility/republish.py" ]; then
    if command -v colcon &>/dev/null; then
        echo "⚙️ [자동 빌드] 패키지 실행 파일이 install 디렉터리에 없습니다. colcon build를 실행합니다..."
        (cd "$PROJECT_DIR" && colcon build --symlink-install)
        if [ -f "$PROJECT_DIR/install/setup.bash" ]; then
            source "$PROJECT_DIR/install/setup.bash"
        fi
    fi
fi

# PYTHONPATH 추가 (src 내부 auto_mobility 패키지 모듈 탐색 보장)
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH"

# third_party 설치 라이브러리 경로 (libstella_vslam, libfbow, pangolin 등)
export LD_LIBRARY_PATH="$PROJECT_DIR/third_party/installed/lib:$LD_LIBRARY_PATH"

# 스크립트 실행 권한 자동 보장 (.py 는 python3 로 호출되므로 exec bit 불필요)
chmod +x "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*/*.sh 2>/dev/null || true

# FastDDS 프로필 자동 적용 (CycloneDDS 전환 이전 잔여 — CycloneDDS는 이 변수를 무시함)
# CycloneDDS 사용 시 무관하나, 수동으로 FastDDS로 되돌릴 때 참고용으로 유지.
if [ -f "$PROJECT_DIR/config/dds/fastdds_camera.xml" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$PROJECT_DIR/config/dds/fastdds_camera.xml"
fi

# X11 GUI Display & OpenGL 설정 자동 감지 (RViz2 창 미출력 방지)
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi
if [ -z "$XDG_RUNTIME_DIR" ]; then
    export XDG_RUNTIME_DIR="/tmp/runtime-$USER"
fi
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
export QT_X11_NO_MITSHM=1
# WSLg RViz2 검은 화면 방지 (2026-08-12 실측):
#  - glxinfo는 D3D12(Intel Arc)로 GPU 가속 동작
#  - 그러나 RViz2(OGRE/GLX)는 D3D12 경로에서 창이 검게 채워지고 무반응(사용자 실측)
#  - 따라서 RViz2만 Mesa 소프트웨어 렌더링(llvmpipe)으로 실행한다.
#  - CUDA(Open3D 등) / CPU 파이프라인 성능에는 영향 없음 (GL 미사용).
export LIBGL_ALWAYS_SOFTWARE=1

# ---------------------------------------------------------
# 파이프라인 중요 로그 자동 수집 (2026-08-10 추가, 분석용)
# 전체 출력은 콘솔에 그대로 전달하면서, 품질/성능 분석에 필요한
# 라인(WARN/ERROR/METRIC + 맥락)만 필터링해 자동 저장한다.
#   사용법: pipeline_log_start <이름>
#           → 로그 파일: $LOG_DIR/<이름>_<타임스탬프>.log
# ---------------------------------------------------------
PIPELINE_LOG_FILE=""
PIPELINE_LOG_FIFO=""
PIPELINE_COLLECTOR_PID=""

pipeline_log_start() {
    local name="${1:-pipeline}"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    PIPELINE_LOG_FILE="$LOG_DIR/${name}_${ts}.log"
    PIPELINE_LOG_FIFO="$PIPELINE_LOG_FILE.fifo"
    PIPELINE_COLLECTOR_PID=""

    rm -f "$PIPELINE_LOG_FIFO"
    mkfifo "$PIPELINE_LOG_FIFO" 2>/dev/null || {
        echo "⚠️ [로그 수집] FIFO 생성 실패 → 중요 로그 수집을 건너뜁니다."
        return 1
    }

    # 원본 콘솔 fd 저장 (3=stdout, 4=stderr)
    exec 3>&1 4>&2

    # 수집기: FIFO → 필터링 → 로그 파일 (+ 콘솔 그대로 전달)
    # --max-lines: 파일 비대화 방지용 상한 (초과분은 건수만 집계)
    python3 "$PROJECT_DIR/scripts/utils/pipeline_log_collector.py" \
        "$PIPELINE_LOG_FILE" --max-lines 50000 < "$PIPELINE_LOG_FIFO" >&3 &
    PIPELINE_COLLECTOR_PID=$!

    # stdout/stderr 전체를 FIFO 로 리다이렉트
    exec > "$PIPELINE_LOG_FIFO" 2>&1

    echo "📝 [로그 수집] 분석용 중요 로그 자동 기록 시작 → $PIPELINE_LOG_FILE"
    return 0
}

pipeline_log_stop() {
    if [ -z "$PIPELINE_LOG_FILE" ]; then
        return 0
    fi
    # 콘솔 복원 → FIFO 작성자 fd 닫힘 → 수집기가 EOF 감지 후 요약 저장
    exec >&3 2>&4
    exec 3>&- 4>&-
    rm -f "$PIPELINE_LOG_FIFO"
    if [ -n "$PIPELINE_COLLECTOR_PID" ]; then
        wait "$PIPELINE_COLLECTOR_PID" 2>/dev/null || true
    fi
    PIPELINE_LOG_FILE=""
    return 0
}

