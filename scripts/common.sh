#!/bin/bash

set -e

source /opt/ros/$ROS_DISTRO/setup.bash 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_DIR/ros2_data"

BAG_DIR="$DATA_DIR/bags"
DB_DIR="$DATA_DIR/databases"
POINTCLOUD_DIR="$DATA_DIR/pointclouds"
MESH_DIR="$DATA_DIR/meshes"
ISAAC_DIR="$DATA_DIR/isaac_sim"
LOG_DIR="$DATA_DIR/logs"
SHARED_DIR="/mnt/hgfs/ubuntu_shared"

RAM_BAG_DIR="/dev/shm/ros2_bags"
STORAGE_FORMAT="mcap"

# config/topics.yaml 를 단일 소스로 토픽명 로드 (기존 환경변수 오버라이드 유지)
if [ -f "$PROJECT_DIR/config/topics.yaml" ]; then
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

mkdir -p "$BAG_DIR" "$DB_DIR" "$POINTCLOUD_DIR" "$MESH_DIR" "$ISAAC_DIR" "$LOG_DIR" "$RAM_BAG_DIR"

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

# 스크립트 실행 권한 자동 보장 (.py 는 python3 로 호출되므로 exec bit 불필요)
chmod +x "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*/*.sh 2>/dev/null || true

# FastDDS 공유메모리(SHM) 프로필 자동 적용
if [ -f "$PROJECT_DIR/config/dds/fastdds_camera.xml" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$PROJECT_DIR/config/dds/fastdds_camera.xml"
fi

# X11 GUI Display & OpenGL 설정 자동 감지 (RViz2 창 미출력 방지)
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi
export QT_X11_NO_MITSHM=1
export LIBGL_ALWAYS_SOFTWARE=0

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
    python3 "$PROJECT_DIR/scripts/utils/pipeline_log_collector.py" \
        "$PIPELINE_LOG_FILE" < "$PIPELINE_LOG_FIFO" >&3 &
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

