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

RAM_BAG_DIR="/dev/shm/ros2_bags"
STORAGE_FORMAT="mcap"

RGB_TOPIC="/camera/camera/color/image_raw"
RGB_COMPRESSED_TOPIC="/camera/camera/color/image_raw/compressed"
DEPTH_TOPIC="/camera/camera/aligned_depth_to_color/image_raw"
DEPTH_COMPRESSED_TOPIC="/camera/camera/aligned_depth_to_color/image_raw/compressedDepth"
CAMERA_INFO_TOPIC="/camera/camera/color/camera_info"
IMU_TOPIC="/camera/camera/imu"

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

# 스크립트 및 노드 실행 권한 자동 보장
chmod +x "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*/*.sh "$PROJECT_DIR"/src/auto_mobility/nodes/*.py "$PROJECT_DIR"/src/auto_mobility/processing/*.py 2>/dev/null || true

# FastDDS 공유메모리(SHM) 프로필 자동 적용
if [ -f "$PROJECT_DIR/config/fastdds_camera.xml" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$PROJECT_DIR/config/fastdds_camera.xml"
fi
