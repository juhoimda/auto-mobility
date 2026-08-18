#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

echo "🔄 Windows RealSense 카메라 재기동 신호 전송..."
# 1. 종료 신호
rm -f "$SHARED_DIR/camera_request.txt"
sleep 3
# 2. 시작 신호
echo "run_camera" > "$SHARED_DIR/camera_request.txt"
echo "✅ 신호 전송 완료. camera_guard.ps1 감지 대기 중..."
sleep 4
