#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

if [ -z "$1" ]; then
    echo "사용법: $0 BAG_NAME [RATE]"
    echo "예시: $0 capture_20260727 0.5"
    exit 1
fi

BAG_PATH="$BAG_DIR/$1"

if [ ! -d "$BAG_PATH" ]; then
    echo "오류: 해당 Bag 디렉토리가 존재하지 않습니다 -> $BAG_PATH"
    exit 1
fi

RATE=${2:-0.5}

echo "Playing rosbag2: $BAG_PATH (--clock -r $RATE)"
ros2 bag play "$BAG_PATH" --clock -r "$RATE"
