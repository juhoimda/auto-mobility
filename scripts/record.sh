#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

USE_COMPRESSED=false
NAME=""
FORCE_RAW=true

for arg in "$@"; do
    case $arg in
        --compressed)
            FORCE_RAW=false
            ;;
        *)
            if [ -z "$NAME" ]; then
                NAME="$arg"
            fi
            ;;
    esac
done

NAME=${NAME:-capture_$(date +%Y%m%d_%H%M%S)}

# 1. RAM 디스크 가용성 및 용량 검사 (Raw 720p 30fps 기준 ~100MB/s)
USE_RAM_DISK=false
if [ -d "$RAM_BAG_DIR" ] && [ -w "$RAM_BAG_DIR" ]; then
    # 남은 RAM 디스크 용량 (MB 단위)
    FREE_RAM_MB=$(df -m "$RAM_BAG_DIR" | tail -n 1 | awk '{print $4}')
    
    # 최소 1GB (1024MB) 이상 남아있을 때만 RAM 디스크 사용
    if [ "$FREE_RAM_MB" -gt 1024 ]; then
        TEMP_OUTPUT="$RAM_BAG_DIR/$NAME"
        FINAL_OUTPUT="$BAG_DIR/$NAME"
        USE_RAM_DISK=true
    else
        echo "⚠️  [경고] RAM 디스크 용량이 부족합니다 (${FREE_RAM_MB}MB 남음). 일반 SSD 저장소로 녹화합니다."
        TEMP_OUTPUT="$BAG_DIR/$NAME"
        FINAL_OUTPUT="$BAG_DIR/$NAME"
    fi
else
    TEMP_OUTPUT="$BAG_DIR/$NAME"
    FINAL_OUTPUT="$BAG_DIR/$NAME"
fi

# 2. 종료 시 자동 이관 처리 (Ctrl+C 등 대응)
cleanup() {
    if [ "$USE_RAM_DISK" = true ] && [ -d "$TEMP_OUTPUT" ]; then
        echo ""
        echo "📦 RAM 디스크($TEMP_OUTPUT)에서 영구 저장소($FINAL_OUTPUT)로 이동 중..."
        mkdir -p "$BAG_DIR"
        mv "$TEMP_OUTPUT" "$FINAL_OUTPUT"
        echo "✅ 이관 완료: $FINAL_OUTPUT"
    fi
}
trap cleanup EXIT INT TERM

# 3. 녹화 토픽 설정
RECORD_TOPICS=()
if [ "$FORCE_RAW" = true ]; then
    RECORD_TOPICS=(
        "$RGB_TOPIC"
        "$DEPTH_TOPIC"
        "$CAMERA_INFO_TOPIC"
        "$IMU_TOPIC"
        /tf_static
    )
else
    RECORD_TOPICS=(
        "$RGB_COMPRESSED_TOPIC"
        "$DEPTH_COMPRESSED_TOPIC"
        "$CAMERA_INFO_TOPIC"
        "$IMU_TOPIC"
        /tf_static
    )
fi

# 4. 사전 상태 검사 (카메라 노드 실행 여부 확인)
echo "=========================================="
echo "🎥 ROS2 Bag 녹화 시작 준비"
echo " 임시 저장소 (RAM): $TEMP_OUTPUT"
echo " 최종 저장소 (SSD): $FINAL_OUTPUT"
echo " 저장 포맷        : $STORAGE_FORMAT (MCAP)"
echo " Raw 저장 여부    : $FORCE_RAW"
echo "=========================================="

MISSING_TOPIC=false
for topic in "${RECORD_TOPICS[@]}"; do
    if ! ros2 topic list | grep -qx "$topic"; then
        echo "❌ [오류] 필수 토픽이 발행되고 있지 않습니다: $topic"
        MISSING_TOPIC=true
    fi
done

if [ "$MISSING_TOPIC" = true ]; then
    echo ""
    echo "💡 카메라 노드가 실행 중인지 확인하세요! (예: ros2 launch auto_mobility camera.launch.py)"
    exit 1
fi

echo "▶️ 녹화를 시작합니다. 종료하려면 Ctrl+C를 누르세요..."
ros2 bag record -s "$STORAGE_FORMAT" -o "$TEMP_OUTPUT" "${RECORD_TOPICS[@]}"


