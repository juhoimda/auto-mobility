#!/bin/bash
# ============================================================
# capture_safe.sh — 안전한 raw 데이터셋 녹화 전용 모드
#
# 파이프라인 내 역할: SENSOR → WSL INGRESS → IMMUTABLE RAW DATASET
#
#   D435i(Windows) → Windows publisher → DDS → WSL
#       ├─ rosbag recorder (압축 토픽: RGB JPEG + Depth PNG lossless)
#       └─ capture_guard (경량 진단, RViz/SLAM 없음)
#
# SLAM/TSDF/RViz 부하와 완전히 분리하여 immutable raw dataset을 확보한다.
# SLAM이 크래시되어도 rosbag은 정상 기록된다.
#
# 사용법:
#   ./scripts/pipeline/capture_safe.sh [BAG_NAME] [--raw | --compressed]
#     --raw:        무압축 토픽 기록 (대역폭 큼, RGB/Depth bit-exact)
#     --compressed: 압축 토픽 기록 (기본값)
# ============================================================


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

# common.sh 가 source 시 SCRIPT_DIR 을 scripts/ 로 덮어쓰므로 이후 경로는 PROJECT_DIR 사용
PIPELINE_DIR="$PROJECT_DIR/scripts/pipeline"

NAME=""
EXTRA_ARGS=()
for arg in "$@"; do
    case $arg in
        --raw|--compressed|--no-validate|--duration=*)
            EXTRA_ARGS+=("$arg")
            ;;
        *)
            if [ -z "$NAME" ]; then
                NAME="$arg"
            fi
            ;;
    esac
done
NAME=${NAME:-capture_safe_$(date +%Y%m%d_%H%M%S)}

# ── Windows 카메라 자동 시작 (원격 모드) ─────────────────────────
start_windows_camera() {
    [ "$CAMERA_MODE" = "remote" ] || return 0
    WIN_SIGNAL_FILE="$SHARED_DIR/camera_request.txt"
    echo "🪟 Windows 카메라 시작 신호 전송 → $WIN_SIGNAL_FILE"
    echo "run_camera" > "$WIN_SIGNAL_FILE" 2>/dev/null || {
        echo "❌ 신호 파일 기록 실패 — /mnt/c/ubuntu_shared 쓰기 권한 확인"
        return 1
    }
    for i in $(seq 1 20); do
        if topic_exists "$RGB_COMPRESSED_TOPIC" 4; then
            echo "✅ 카메라 토픽 확인됨 (${i}회 프로브)"
            return 0
        fi
        echo "  ... 카메라 토픽 대기 중 (${i}/20)"
        sleep 1
    done
    echo "❌ 카메라 토픽이 20초 내 감지되지 않았습니다. camera_guard.ps1(watcher) 확인."
    return 1
}

stop_windows_camera() {
    [ "$CAMERA_MODE" = "remote" ] || return 0
    rm -f "$SHARED_DIR/camera_request.txt" 2>/dev/null || true
    echo "🪟 Windows 카메라 종료 신호 전송 완료"
}

cleanup() {
    kill $GUARD_PID 2>/dev/null || true
    wait $GUARD_PID 2>/dev/null || true
    stop_windows_camera
}
trap cleanup EXIT INT TERM

# ── 카메라 존재 확인 (없으면 원격 모드 자동 시작) ───────────────
echo "=========================================================="
echo " 🛡️  CAPTURE-SAFE 모드 (raw dataset 확보 전용)"
echo " 📦 Bag 이름: $NAME"
echo "=========================================================="

if ! topic_exists "$RGB_COMPRESSED_TOPIC" 3 && ! topic_exists "$RGB_TOPIC" 3; then
    if [ "$CAMERA_MODE" = "remote" ]; then
        start_windows_camera || exit 1
    else
        echo "❌ 카메라 토픽이 감지되지 않습니다."
        echo "   [원격] CAMERA_MODE=remote ./scripts/pipeline/capture_safe.sh"
        echo "   [로컬] ros2 launch auto_mobility camera.launch.py"
        exit 1
    fi
fi

# ── 경량 진단 (capture_guard) 백그라운드 시작 ───────────────────
GUARD_BASE="$LOG_DIR/capture_safe_${NAME}"
GUARD_ARGS="--interval 5 --headless --report ${GUARD_BASE}.md --json ${GUARD_BASE}.json"
[ "$CAMERA_MODE" = "remote" ] && GUARD_ARGS="$GUARD_ARGS --remote"
echo "📊 capture_guard 모니터링 시작 → ${GUARD_BASE}.md"
python3 "$PROJECT_DIR/src/auto_mobility/monitor/capture_guard.py" $GUARD_ARGS &
GUARD_PID=$!

# ── 녹화 (압축 기본: RGB JPEG + Depth PNG lossless) ────────────
echo ""
echo "▶️ 녹화를 시작합니다. 종료하려면 Ctrl+C..."
"$PIPELINE_DIR/record.sh" "$NAME" "${EXTRA_ARGS[@]}"

# ── 정리 및 요약 ────────────────────────────────────────────────
kill $GUARD_PID 2>/dev/null || true
wait $GUARD_PID 2>/dev/null || true
stop_windows_camera
echo ""
echo "=========================================================="
echo " ✅ CAPTURE-SAFE 완료"
echo " 📦 Bag: $BAG_DIR/$NAME"
echo " 📊 진단 보고서: ${GUARD_BASE}.md"
echo " 📋 매니페스트: $BAG_DIR/$NAME/dataset_manifest.json"
echo "=========================================================="
