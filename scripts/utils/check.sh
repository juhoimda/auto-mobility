#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

check_topic_with_fps() {
    local topic="$1"
    local min_fps="$2"
    if ros2 topic list | grep -qx "$topic"; then
        echo -n "  [O] $topic "
        if [ -n "$min_fps" ]; then
            local hz_output=$(timeout 4.5 ros2 topic hz "$topic" 2>&1 || true)
            local rate=$(echo "$hz_output" | grep "average rate:" | tail -n 1 | awk '{print $3}')
            if [ -n "$rate" ]; then
                local rate_int=$(printf "%.0f" "$rate" 2>/dev/null || echo 0)
                if [ "$rate_int" -lt "$min_fps" ]; then
                    echo "⚠️ (현재 ${rate} Hz - 권장 ${min_fps} Hz 이상 필요)"
                else
                    echo "✅ (${rate} Hz 정상)"
                fi
            else
                echo "(데이터 수신 대기 중...)"
            fi
        else
            echo ""
        fi
    else
        echo "  [X] $topic (비활성)"
    fi
}

echo "=== ROS2 Topic & Storage Health Check ==="
echo "[토픽 및 실시간 FPS 상태 확인]"
check_topic_with_fps "$RGB_TOPIC" 15
check_topic_with_fps "$DEPTH_TOPIC" 15
check_topic_with_fps "$CAMERA_INFO_TOPIC"
check_topic_with_fps "$IMU_TOPIC"
check_topic_with_fps "/tf_static"

echo ""
echo "[스토리지 포맷 확인]"
if ros2 bag record --help | grep -q "mcap"; then
    echo "  [O] MCAP Storage Plugin 사용 가능"
else
    echo "  [!] MCAP Storage Plugin 미설치 (ros-humble-rosbag2-storage-mcap 필요)"
fi

echo ""
echo "[RAM 디스크 영역 확인]"
if [ -d "$RAM_BAG_DIR" ] && [ -w "$RAM_BAG_DIR" ]; then
    FREE_MB=$(df -m "$RAM_BAG_DIR" | tail -n 1 | awk '{print $4}')
    echo "  [O] RAM Disk ($RAM_BAG_DIR) 사용 가능 - 남은 용량: ${FREE_MB} MB"
else
    echo "  [!] RAM Disk ($RAM_BAG_DIR) 접근 불가"
fi
