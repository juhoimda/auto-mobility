#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

NAME=${1:-capture_$(date +%Y%m%d_%H%M%S)}
OUTPUT="$BAG_DIR/$NAME"

echo "Bag 저장 위치: $OUTPUT"
echo "종료하려면 Ctrl+C를 누르세요."

ros2 bag record \
    "$RGB_TOPIC" \
    "$DEPTH_TOPIC" \
    "$CAMERA_INFO_TOPIC" \
    "$POINTS_TOPIC" \
    "$IMU_TOPIC" \
    /tf \
    /tf_static \
    -o "$OUTPUT"
