#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

# 1. DDS, 해상도, QoS, 실시간 FPS 및 USB 상태 종합 점검 실행
python3 "$PROJECT_DIR/src/auto_mobility/utils/inspect_system.py" "$@"

# 2. 추가 스토리지 및 RAM 디스크 점검
echo "[스토리지 포맷 및 RAM 디스크 점검]"
if ros2 bag record --help | grep -q "mcap"; then
    echo "  [O] MCAP Storage Plugin 사용 가능"
else
    echo "  [!] MCAP Storage Plugin 미설치 (ros-humble-rosbag2-storage-mcap 필요)"
fi

if [ -d "$RAM_BAG_DIR" ] && [ -w "$RAM_BAG_DIR" ]; then
    FREE_MB=$(df -m "$RAM_BAG_DIR" | tail -n 1 | awk '{print $4}' 2>/dev/null || echo "N/A")
    echo "  [O] RAM Disk ($RAM_BAG_DIR) 사용 가능 - 남은 용량: ${FREE_MB} MB"
else
    echo "  [!] RAM Disk ($RAM_BAG_DIR) 접근 불가"
fi
echo ""

