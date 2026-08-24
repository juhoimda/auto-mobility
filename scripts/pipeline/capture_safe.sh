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
SHOW_PREVIEW=false
VIEW_ARGS=()
EXTRA_ARGS=()
for arg in "$@"; do
    case $arg in
        --raw|--compressed|--no-validate|--duration=*)
            EXTRA_ARGS+=("$arg")
            ;;
        --view|--preview)
            SHOW_PREVIEW=true
            ;;
        --rgb-only)
            SHOW_PREVIEW=true
            VIEW_ARGS+=("--rgb-only")
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
    mkdir -p "$SHARED_DIR" 2>/dev/null || true
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
    echo "❌ 카메라 토픽이 20초 내 감지되지 않았습니다."
    echo "💡 해결 방법 (아래 중 하나 실행):"
    echo "   1) Windows 바탕화면의 [카메라_가드_시작.bat] 또는 C:\\ros2_humble\\start_camera_guard.vbs 실행"
    echo "   2) 또는 Windows 바탕화면의 [카메라_직접_실행.bat] 실행"
    return 1
}


stop_windows_camera() {
    [ "$CAMERA_MODE" = "remote" ] || return 0
    rm -f "$SHARED_DIR/camera_request.txt" 2>/dev/null || true
    echo "🪟 Windows 카메라 종료 신호 전송 완료"
}

PREVIEW_PID=""
cleanup() {
    kill $GUARD_PID 2>/dev/null || true
    wait $GUARD_PID 2>/dev/null || true
    # Windows 프리뷰 신호 파일 제거 (realsense_pub.exe가 감지하여 창 자동 닫힘)
    rm -f "$SHARED_DIR/camera_preview.txt" 2>/dev/null || true
    stop_windows_camera
}
trap cleanup EXIT INT TERM

# ── 카메라 존재 확인 (없으면 원격 모드 자동 시작) ───────────────
echo "=========================================================="
echo " 🛡️  CAPTURE-SAFE 모드 (raw dataset 확보 전용)"
echo " 📦 Bag 이름: $NAME"
echo "=========================================================="

if ! topic_exists "$RGB_COMPRESSED_TOPIC" 1.0 && ! topic_exists "$RGB_TOPIC" 1.0; then
    if [ "$CAMERA_MODE" = "remote" ]; then
        start_windows_camera || exit 1
    else
        echo "❌ 카메라 토픽이 감지되지 않습니다."
        echo "   [원격] CAMERA_MODE=remote ./scripts/pipeline/capture_safe.sh"
        echo "   [로컬] ros2 launch auto_mobility camera.launch.py"
        exit 1
    fi
fi

# ── 사전 상태 점검 (Preflight Sanity Check, feedback.md Section 25) ──
echo "🔍 녹화 전 사전 센서 무결성(Preflight) 점검 중..."
python3 - << 'PREFLIGHT_EOF'
import sys, time
from collections import defaultdict
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, CameraInfo, Imu

QOS = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)
IMU_QOS = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE)

class PreflightProbe(Node):
    def __init__(self):
        super().__init__('preflight_probe')
        self.stamps = defaultdict(list)
        self.create_subscription(CompressedImage, '/camera/camera/color/image_raw/compressed', lambda m: self.cb('rgb', m), QOS)
        self.create_subscription(CompressedImage, '/camera/camera/depth/image_rect_raw/compressedDepth', lambda m: self.cb('depth', m), QOS)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info_windows', lambda m: self.cb('info', m), QOS)
        self.create_subscription(Imu, '/camera/camera/imu', lambda m: self.cb('imu', m), IMU_QOS)

    def cb(self, name, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.stamps[name].append(ts)

rclpy.init()
probe = PreflightProbe()
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(probe, timeout_sec=0.05)

rgb_n = len(probe.stamps['rgb'])
dep_n = len(probe.stamps['depth'])
imu_n = len(probe.stamps['imu'])
info_n = len(probe.stamps['info'])

print(f"  [Preflight] RGB: {rgb_n} frames, Depth: {dep_n} frames, IMU: {imu_n} msgs, CameraInfo: {info_n} msgs (3s probe)")

if rgb_n < 20 or dep_n < 20:
    print("⚠️  [경고] 초기 카메라 프레임 수신율이 낮습니다 (RGB < 20fps 또는 Depth < 20fps)!")
else:
    print("✅ [정상] 센서 스트림 초기 수신율 양호 (≥ 20 FPS)")

rclpy.shutdown()
PREFLIGHT_EOF

# ── 경량 진단 (capture_guard) 백그라운드 시작 ───────────────────
GUARD_BASE="$LOG_DIR/capture_safe_${NAME}"
GUARD_ARGS="--interval 5 --headless --report ${GUARD_BASE}.md --json ${GUARD_BASE}.json"
[ "$CAMERA_MODE" = "remote" ] && GUARD_ARGS="$GUARD_ARGS --remote"
echo "📊 capture_guard 모니터링 시작 → ${GUARD_BASE}.md"
python3 "$PROJECT_DIR/src/auto_mobility/monitor/capture_guard.py" $GUARD_ARGS &
GUARD_PID=$!

# ── 실시간 카메라 프리뷰 (옵션: --view) ────────────────────────
# WSL 내부 Python 뷰어 대신 Windows 신호 파일 생성 →
# realsense_pub.exe 자체적으로 OpenCV 창 표시 (녹화 성능 영향 없음)
if [ "$SHOW_PREVIEW" = true ]; then
    echo "👁️  Windows 프리뷰 창 신호 전송 → $SHARED_DIR/camera_preview.txt"
    echo "preview" > "$SHARED_DIR/camera_preview.txt" 2>/dev/null || \
        echo "⚠️  신호 파일 쓰기 실패 (WSL→Windows 공유 폴더 확인)"
fi

# ── 녹화 (압축 기본: RGB JPEG + Depth PNG lossless) ────────────
"$PIPELINE_DIR/record.sh" "$NAME" "${EXTRA_ARGS[@]}"

# ── 정리 및 요약 ────────────────────────────────────────────────
trap - EXIT INT TERM
cleanup

echo ""
echo "=========================================================="
echo " 📊 센서 무결성 자동 정밀 진단 실행 중 (feedback.md Section 26)..."
echo "=========================================================="
DIAG_OUT="$DATA_DIR/diagnostics/$NAME"
python3 -m auto_mobility.diagnostics.sensor_integrity "$NAME" --out-dir "$DIAG_OUT" || true

echo ""
echo "=========================================================="
echo " ✅ CAPTURE-SAFE 완료"
echo " 📦 Bag        : $BAG_DIR/$NAME"
echo " 📊 진단 보고서: $DIAG_OUT/sensor_integrity.md"
echo " 📋 매니페스트 : $BAG_DIR/$NAME/dataset_manifest.json"
echo "=========================================================="
