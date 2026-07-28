#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

check_topic() {
    if ros2 topic list | grep -qx "$1"; then
        echo "[O] $1"
    else
        echo "[X] $1 (비활성)"
    fi
}

echo "=== ROS2 Topic & Storage Health Check ==="
echo "[토픽 상태 확인]"
check_topic "$RGB_TOPIC"
check_topic "$RGB_COMPRESSED_TOPIC"
check_topic "$DEPTH_TOPIC"
check_topic "$DEPTH_COMPRESSED_TOPIC"
check_topic "$CAMERA_INFO_TOPIC"
check_topic "$IMU_TOPIC"
check_topic "/tf_static"

echo ""
echo "[스토리지 포맷 확인]"
if ros2 bag record --help | grep -q "mcap"; then
    echo "[O] MCAP Storage Plugin 사용 가능"
else
    echo "[!] MCAP Storage Plugin 미설치 (ros-humble-rosbag2-storage-mcap 필요)"
fi

echo ""
echo "[RAM 디스크 영역 확인]"
if [ -d "$RAM_BAG_DIR" ] && [ -w "$RAM_BAG_DIR" ]; then
    echo "[O] RAM Disk ($RAM_BAG_DIR) 사용 가능"
else
    echo "[!] RAM Disk ($RAM_BAG_DIR) 접근 불가"
fi

