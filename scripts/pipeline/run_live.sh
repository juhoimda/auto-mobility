#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

DB_NAME=${1:-live_$(date +%Y%m%d_%H%M%S)}
DB_PATH="$DB_DIR/$DB_NAME.db"

if [ "$CAMERA_MODE" = "remote" ]; then
    # Windows 네이티브 카메라: WSL에서 드라이버를 띄우지 않는다. 압축 토픽 수신이 기본.
    echo "🌐 [원격 카메라 모드] Windows에서 발행 중인 토픽을 사용합니다. camera.launch.py 자동 구동 생략."
    USE_COMPRESSED="${2:-true}"
else
    # 카메라 드라이버 노드가 실행 중이지 않으면 자동으로 백그라운드 구동
    if ! ros2 topic list 2>/dev/null | grep -q "$RGB_TOPIC"; then
        echo "📷 [자동 구동] RealSense 카메라 노드가 감지되지 않아 camera.launch.py를 백그라운드로 구동합니다..."
        ros2 launch auto_mobility camera.launch.py > /tmp/camera_launch.log 2>&1 &
        CAM_PID=$!
        trap "kill -9 $CAM_PID 2>/dev/null || true" EXIT
        sleep 3
    fi
    USE_COMPRESSED="${2:-false}"
fi

ros2 launch auto_mobility rtab_live.launch.py \
    database_path:="$DB_PATH" \
    use_compressed:="$USE_COMPRESSED"