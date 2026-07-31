#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

USE_COMPRESSED=true
DB_NAME=""
DEPTH_COMPRESSED_TOPIC="$DEPTH_COMPRESSED_TOPIC"

for arg in "$@"; do
    case $arg in
        --raw)
            USE_COMPRESSED=false
            ;;
        --compressed)
            USE_COMPRESSED=true
            ;;
        *)
            if [ -z "$DB_NAME" ]; then
                DB_NAME="$arg"
            fi
            ;;
    esac
done

DB_NAME=${DB_NAME:-bag_$(date +%Y%m%d_%H%M%S)}
DB_PATH="$DB_DIR/$DB_NAME.db"

# 스마트 자동 감지: Bag 파일이 존재할 경우 내부 Depth 토픽 자동 검사
BAG_PATH="$BAG_DIR/$DB_NAME"
if [ -d "$BAG_PATH" ]; then
    if ros2 bag info "$BAG_PATH" 2>/dev/null | grep -q "/camera/camera/depth/image_rect_raw/compressedDepth"; then
        echo "💡 [스마트 자동 감지] 구형(Legacy) Depth 압축 토픽 적용"
        DEPTH_COMPRESSED_TOPIC="/camera/camera/depth/image_rect_raw/compressedDepth"
    fi
fi

ros2 launch auto_mobility rtab_bag.launch.py \
    database_path:="$DB_PATH" \
    use_compressed:="$USE_COMPRESSED" \
    depth_compressed_topic:="$DEPTH_COMPRESSED_TOPIC"
