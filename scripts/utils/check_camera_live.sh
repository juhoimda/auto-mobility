#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

echo "=========================================================="
echo " 📡 카메라 수신 상태 점검"
echo "=========================================================="
for topic in "$RGB_COMPRESSED_TOPIC" "$DEPTH_COMPRESSED_TOPIC" "$CAMERA_INFO_WINDOWS_TOPIC"; do
    if topic_exists "$topic" 2; then
        echo "✅ 수신 확인: $topic"
    else
        echo "❌ 수신 실패: $topic"
    fi
done
