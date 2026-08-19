#!/bin/bash
# prepare_dataset.sh — Rosbag에서 알고리즘 독립적인 Canonical Frame Dataset 추출

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

BAG_NAME=""
FORCE_FLAG=""

for arg in "$@"; do
    case $arg in
        --force)
            FORCE_FLAG="--force"
            ;;
        -*)
            echo "⚠️ 알 수 없는 옵션: $arg"
            ;;
        *)
            if [ -z "$BAG_NAME" ]; then
                BAG_NAME="$arg"
            fi
            ;;
    esac
done

if [ -z "$BAG_NAME" ]; then
    echo "=========================================================="
    echo " 사용법: $0 BAG_NAME [--force]"
    echo " 예시  : $0 room01"
    echo " 설명  : 불변 원본 Rosbag으로부터 알고리즘 독립적인"
    echo "         Canonical Frame Dataset (RGB-D + CameraInfo + IMU)을 생성합니다."
    echo "=========================================================="
    exit 1
fi

BAG_PATH="$BAG_DIR/$BAG_NAME"
if [ ! -d "$BAG_PATH" ] && [ ! -f "$BAG_PATH" ]; then
    if [ -d "$BAG_NAME" ] || [ -f "$BAG_NAME" ]; then
        BAG_PATH="$BAG_NAME"
        BAG_NAME="$(basename "$BAG_NAME")"
    else
        echo "❌ 오류: Rosbag을 찾을 수 없습니다 -> $BAG_PATH"
        exit 1
    fi
fi

OUT_FRAME_DIR="$FRAME_DIR/$BAG_NAME"

echo "=========================================================="
echo " 📦 Canonical Frame Dataset 생성 시작"
echo " 📂 입력 Rosbag : $BAG_PATH"
echo " 💾 출력 프레임 : $OUT_FRAME_DIR"
echo "=========================================================="

python3 "$PROJECT_DIR/src/auto_mobility/dataset/extract_frames.py" \
    "$BAG_PATH" \
    --out-dir="$OUT_FRAME_DIR" \
    $FORCE_FLAG

if [ $? -eq 0 ]; then
    echo "=========================================================="
    echo " ✅ Canonical Dataset 준비 완료 -> $OUT_FRAME_DIR"
    echo "=========================================================="
fi
