#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

USE_COMPRESSED=true
NAME=""

for arg in "$@"; do
    case $arg in
        --raw)
            USE_COMPRESSED=false
            ;;
        --compressed)
            USE_COMPRESSED=true
            ;;
        *)
            if [ -z "$NAME" ]; then
                NAME="$arg"
            fi
            ;;
    esac
done

NAME=${NAME:-capture_$(date +%Y%m%d_%H%M%S)}

if [ -d "$RAM_BAG_DIR" ] && [ -w "$RAM_BAG_DIR" ]; then
    TEMP_OUTPUT="$RAM_BAG_DIR/$NAME"
    FINAL_OUTPUT="$BAG_DIR/$NAME"
    USE_RAM_DISK=true
else
    TEMP_OUTPUT="$BAG_DIR/$NAME"
    FINAL_OUTPUT="$BAG_DIR/$NAME"
    USE_RAM_DISK=false
fi

cleanup() {
    if [ "$USE_RAM_DISK" = true ] && [ -d "$TEMP_OUTPUT" ]; then
        echo ""
        echo "RAM 디스크($TEMP_OUTPUT)에서 영구 저장소($FINAL_OUTPUT)로 이관 중..."
        mv "$TEMP_OUTPUT" "$FINAL_OUTPUT"
        echo "이관 완료: $FINAL_OUTPUT"
    fi
}

trap cleanup EXIT INT TERM

echo "=========================================="
echo "Bag 저장 위치 (임시): $TEMP_OUTPUT"
echo "Bag 저장 위치 (최종): $FINAL_OUTPUT"
echo "저장 포맷: $STORAGE_FORMAT (MCAP)"
echo "토픽 압축 여부: $USE_COMPRESSED"
echo "종료하려면 Ctrl+C를 누르세요."
echo "=========================================="

RECORD_TOPICS=()
if [ "$USE_COMPRESSED" = true ]; then
    RECORD_TOPICS+=(
        "$RGB_COMPRESSED_TOPIC"
        "$DEPTH_COMPRESSED_TOPIC"
        "$CAMERA_INFO_TOPIC"
        "$IMU_TOPIC"
        /tf_static
    )
else
    RECORD_TOPICS+=(
        "$RGB_TOPIC"
        "$DEPTH_TOPIC"
        "$CAMERA_INFO_TOPIC"
        "$IMU_TOPIC"
        /tf_static
    )
fi

ros2 bag record -s "$STORAGE_FORMAT" -o "$TEMP_OUTPUT" "${RECORD_TOPICS[@]}"

