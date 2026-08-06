#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

DB_NAME=${1:-live_$(date +%Y%m%d_%H%M%S)}
DB_PATH="$DB_DIR/$DB_NAME.db"

# 카메라 드라이버 노드가 실행 중이지 않으면 자동으로 백그라운드 구동
if ! ros2 topic list 2>/dev/null | grep -q "/camera/camera/color/image_raw"; then
    echo "📷 [자동 구동] RealSense 카메라 노드가 감지되지 않아 camera.launch.py를 백그라운드로 구동합니다..."
    ros2 launch auto_mobility camera.launch.py > /tmp/camera_launch.log 2>&1 &
    CAM_PID=$!
    trap "kill -9 $CAM_PID 2>/dev/null || true" EXIT
    sleep 3
fi

USE_COMPRESSED=${2:-false}

ros2 launch auto_mobility rtab_live.launch.py \
    database_path:="$DB_PATH" \
    use_compressed:="$USE_COMPRESSED"