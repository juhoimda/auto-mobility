#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

NAME=""
FORCE_RAW=false
SKIP_VALIDATE=false
DURATION=""

for arg in "$@"; do
    case $arg in
        --raw)
            FORCE_RAW=true
            ;;
        --compressed)
            FORCE_RAW=false
            ;;
        --no-validate)
            SKIP_VALIDATE=true
            ;;
        --duration=*)
            DURATION="${arg#*=}"
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
    if [ -n "$TF_PUB_PID" ]; then
        kill "$TF_PUB_PID" 2>/dev/null || true
        TF_PUB_PID=""
    fi
    if [ "$USE_RAM_DISK" = true ] && [ -d "$TEMP_OUTPUT" ]; then
        echo ""
        echo "📦 RAM 디스크($TEMP_OUTPUT)에서 영구 저장소($FINAL_OUTPUT)로 이동 중..."
        mkdir -p "$BAG_DIR"
        mv "$TEMP_OUTPUT" "$FINAL_OUTPUT"
        echo "✅ 이관 완료: $FINAL_OUTPUT"
    fi
}
trap cleanup EXIT INT TERM

# 3. 토픽 존재 확인 (topic_probe: ros2 CLI 데몬 hang 회피)
#    원격(Windows 카메라) / 로컬(WSL usbipd) 모드 자동 감지
MODE="local"
if ! topic_exists "$RGB_TOPIC" 3 && topic_exists "$RGB_COMPRESSED_TOPIC" 3; then
    MODE="remote"
fi
echo "📡 카메라 모드: $MODE"

# 압축 토픽이 없으면(예: 로컬 raw 전송 전용) 자동으로 raw 녹화로 폴백
if [ "$FORCE_RAW" = false ] && ! topic_exists "$RGB_COMPRESSED_TOPIC" 3; then
    echo "⚠️  [폴백] 압축 RGB 토픽 미발행 → raw 토픽으로 녹화합니다."
    FORCE_RAW=true
fi

DETECTED_DEPTH=""
DETECTED_DEPTH_COMPRESSED=""
if [ "$FORCE_RAW" = true ]; then
    if topic_exists "$DEPTH_TOPIC" 3; then
        DETECTED_DEPTH="$DEPTH_TOPIC"
    elif topic_exists "$ALIGNED_DEPTH_TOPIC" 3; then
        DETECTED_DEPTH="$ALIGNED_DEPTH_TOPIC"
    fi
else
    if topic_exists "$DEPTH_COMPRESSED_TOPIC" 3; then
        DETECTED_DEPTH_COMPRESSED="$DEPTH_COMPRESSED_TOPIC"
    elif topic_exists "$ALIGNED_DEPTH_COMPRESSED_TOPIC" 3; then
        DETECTED_DEPTH_COMPRESSED="$ALIGNED_DEPTH_COMPRESSED_TOPIC"
    fi
fi

# CameraInfo: 표준 토픽 + Windows 원본 토픽 모두 존재 시 함께 기록
# (오프라인 재생 시 republish.py 가 camera_info_windows 를 구독하므로 replay 정합성 확보)
INFO_TOPICS=()
topic_exists "$CAMERA_INFO_TOPIC" 3 && INFO_TOPICS+=("$CAMERA_INFO_TOPIC")
topic_exists "$CAMERA_INFO_WINDOWS_TOPIC" 3 && INFO_TOPICS+=("$CAMERA_INFO_WINDOWS_TOPIC")

# /tf_static: 로컬(realsense2_camera)은 자체 발행하지만, 원격(Windows)은 2026-08-18부터
# Windows realsense_pub.py 가 직접 static TF 를 발행한다 (TRANSIENT_LOCAL, time=0).
# topic_probe(rclpy 신규 구독)는 TRANSIENT_LOCAL 스냅샷을 받지 못하는 경우가 있어,
# /tf_static 은 ros2 topic list(파일캐시/가장 가까운)로 대체 확인한다.
TF_PUB_PID=""
has_tf_static() {
    if topic_exists /tf_static 3; then
        return 0
    fi
    timeout 10 ros2 topic list 2>/dev/null | grep -qx /tf_static
}
if ! has_tf_static; then
    echo "⚠️  [경고] /tf_static 을 확보할 수 없습니다. 기록에서 제외합니다."
    HAS_TF_STATIC=false
else
    HAS_TF_STATIC=true
fi

RECORD_TOPICS=()
if [ "$FORCE_RAW" = true ]; then
    RECORD_TOPICS+=("$RGB_TOPIC" "$DETECTED_DEPTH")
else
    RECORD_TOPICS+=("$RGB_COMPRESSED_TOPIC" "$DETECTED_DEPTH_COMPRESSED")
fi
RECORD_TOPICS+=("${INFO_TOPICS[@]}" "$IMU_TOPIC")
[ "$HAS_TF_STATIC" = true ] && RECORD_TOPICS+=(/tf_static)

# 4. 사전 상태 및 FPS 검사 (제안서 핵심: RGB, Depth, IMU, CameraInfo, TF)
echo "=========================================="
echo "🎥 ROS2 Bag 녹화 사전 상태 점검"
echo " 임시 저장소 (RAM): $TEMP_OUTPUT"
echo " 최종 저장소 (SSD): $FINAL_OUTPUT"
echo " 저장 포맷        : $STORAGE_FORMAT (MCAP)"
echo " Raw 저장 여부    : $FORCE_RAW"
echo "=========================================="

MISSING_TOPIC=false
for topic in "${RECORD_TOPICS[@]}"; do
    # /tf_static: WSL 로컬 DDS 전달 결함으로 topic_probe(rclpy) 미감지.
    #   has_tf_static() 가 감지했으므로 bag recorder(C++)에 위임.
    [ "$topic" = "/tf_static" ] && continue
    if ! topic_exists "$topic" 3; then
        echo "❌ [오류] 필수 토픽 미발행: $topic"
        MISSING_TOPIC=true
    else
        echo "✅ [정상] 토픽 확인: $topic"
    fi
done

if [ "$MISSING_TOPIC" = true ]; then
    echo ""
    if [ "$MODE" = "remote" ]; then
        echo "💡 Windows 카메라(realsense_pub.py)가 실행 중인지 확인하세요!"
        echo "   (C:\\ros2_humble\\run_camera.bat 또는 신호파일 방식 /mnt/c/ubuntu_shared/camera_request.txt)"
    else
        echo "💡 카메라 노드가 실행 중인지 확인하세요! (예: ros2 launch auto_mobility camera.launch.py)"
    fi
    exit 1
fi

# 5. 핵심 센서 토픽 FPS(주파수) 실시간 검증 (최소 10 FPS 권장)
check_fps() {
    local topic="$1"
    local min_fps="$2"
    echo -n "🔍 [$topic] FPS 측정 중 (약 2초 소요)... "

    # ros2 topic hz 출력에서 average rate 추출 (DDS 구독 연결 시간 고려 4.5초 설정)
    local hz_output=$(timeout 4.5 ros2 topic hz "$topic" 2>&1 || true)
    local rate=$(echo "$hz_output" | grep "average rate:" | tail -n 1 | awk '{print $3}')

    if [ -z "$rate" ]; then
        echo "⚠️  [경고] FPS 측정 불가 (데이터 흐름 없음)"
        return 1
    fi

    # 소수점 제거 후 정수 비교
    local rate_int=$(printf "%.0f" "$rate" 2>/dev/null || echo 0)
    if [ "$rate_int" -lt "$min_fps" ]; then
        echo "⚠️  [경고] 현재 $rate Hz (권장: ${min_fps} Hz 이상) - 프레임수가 낮습니다!"
    else
        echo "✅ [정상] $rate Hz (안정)"
    fi
}

echo ""
# FPS 체크: 로컬 모드에서만 유효 (원격 모드는 WSL→WSL 전달 결함으로 ros2 topic hz 실패)
if [ "$MODE" != "remote" ]; then
    echo "📊 핵심 센서 데이터 FPS 점검 중..."
    check_fps "${RECORD_TOPICS[0]}" 15 || true
    check_fps "${RECORD_TOPICS[1]}" 15 || true
else
    echo "📊 원격 모드: FPS 체크는 bag recorder 검증으로 대체 (WSL 로컬 전달 결함)"
fi

# 6. RELIABLE QoS 오버라이드 생성 (Windows publisher 가 RELIABLE 로 전송)
#    Windows→WSL NAT 경로 대형 메시지 유실을 재전송으로 보완 (2026-08-18 실측:
#    BE 5.7Hz → RELIABLE 29.7Hz). IMU/tf_static/raw 토픽은 BE 발행이므로 기본값 유지.
QOS_OVERRIDE_YAML=""
if [ "$FORCE_RAW" = false ]; then
    QOS_OVERRIDE_YAML="/tmp/qos_overrides_$$.yaml"
    {
        for t in "${RECORD_TOPICS[@]}"; do
            case "$t" in
                *compressed*|*compressedDepth*|*camera_info_windows*)
                    echo "\"$t\":"
                    echo "  reliability: RELIABLE"
                    echo "  history: KEEP_LAST"
                    echo "  depth: 5"
                    echo "  durability: VOLATILE"
                    ;;
            esac
        done
    } > "$QOS_OVERRIDE_YAML"
    echo "  ⚙️  RELIABLE QoS 적용 토픽: $(grep -c '\"' "$QOS_OVERRIDE_YAML")"
fi

echo ""
echo "▶️ 녹화를 시작합니다. 종료하려면 Ctrl+C를 누르세요..."
RECORD_CMD=(ros2 bag record -s "$STORAGE_FORMAT" -o "$TEMP_OUTPUT")
[ -n "$QOS_OVERRIDE_YAML" ] && RECORD_CMD+=(--qos-profile-overrides-path "$QOS_OVERRIDE_YAML")
RECORD_CMD+=("${RECORD_TOPICS[@]}")
if [ -n "$DURATION" ]; then
    echo "⏱️  자동 종료: ${DURATION}초 후 녹화를 종료합니다..."
    timeout -s INT "${DURATION}" "${RECORD_CMD[@]}"
else
    "${RECORD_CMD[@]}"
fi
rm -f "$QOS_OVERRIDE_YAML"

# 6. 녹화 완료 후 자동 데이터셋 검증 + 매니페스트 생성
if [ "$SKIP_VALIDATE" = false ]; then
    echo ""
    echo "🔍 [데이터셋 검증] ${TEMP_OUTPUT} 검사 중..."
    python3 "$PROJECT_DIR/src/auto_mobility/utils/validate_bag.py" "$TEMP_OUTPUT" \
        --out "$TEMP_OUTPUT/dataset_manifest.json" || true
fi
