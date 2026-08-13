#!/bin/bash
# End-to-End Real-to-Sim Pipeline Orchestrator with Integrity Barriers
# Step 1: Real-time Camera Capture + RTAB-Map SLAM -> rtabmap.db & PointCloud
# Step 2: Open3D Mesh Reconstruction (.obj) -> Isaac Sim Digital Twin Ingestion

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export PYTHONWARNINGS="ignore"
DB_NAME="session_${TIMESTAMP}.db"
MESH_NAME="session_${TIMESTAMP}_tsdf.obj"
SKIP_CAPTURE=false
SKIP_ISAAC=false

# CLI 옵션 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --db=*)
            DB_NAME="${1#*=}"
            MESH_NAME="${DB_NAME%.db}_tsdf.obj"
            shift
            ;;
        --skip-capture)
            SKIP_CAPTURE=true
            shift
            ;;
        --skip-isaac)
            SKIP_ISAAC=true
            shift
            ;;
        --remote-camera)
            CAMERA_MODE=remote
            shift
            ;;
        -h|--help)
            echo "=========================================================="
            echo " 사용법: $0 [--db=DB_NAME.db] [--skip-capture] [--skip-isaac] [--remote-camera]"
            echo " 예시  : $0 (실시간 캡처부터 Mesh(TSDF) 및 Isaac Sim 검증까지 전체 실행)"
            echo " 예시  : $0 --db=my_room.db --skip-capture (기존 DB로 TSDF Mesh 및 Isaac Sim 실행)"
            echo " 예시  : $0 --skip-isaac (촬영 후 TSDF Mesh만 생성, 시뮬레이션 제외)"
            echo " 예시  : $0 --remote-camera (Windows 네이티브 카메라 토픽 수신, 압축 토픽 기본 사용)"
            echo " 💡 Mesh 재구성은 TSDF가 기본(원본 RGB-D + pose → Open3D Tensor TSDF, GPU)"
            echo "    Poisson(PLY) 기반 복원을 원하면 scripts/pipeline/mesh.sh 를 직접 사용"
            echo "=========================================================="
            exit 0
            ;;
        *)
            echo "⚠️ 알 수 없는 옵션: $1"
            shift
            ;;
    esac
done

# 원격 카메라 모드: WSL에선 드라이버 미구동, 압축 토픽 수신이 기본
if [ "$CAMERA_MODE" = "remote" ]; then
    USE_COMPRESSED="${USE_COMPRESSED:-true}"
fi

TARGET_DB_PATH="$DB_DIR/$DB_NAME"
TARGET_PLY_PATH="$POINTCLOUD_DIR/${DB_NAME%.db}_cloud.ply"
TARGET_MESH_PATH="$MESH_DIR/$MESH_NAME"

# 📝 파이프라인 전체 출력에서 중요 로그(WARN/ERROR/METRIC) 자동 수집 시작
pipeline_log_start "pipeline_${DB_NAME%.db}" || true

echo "=========================================================="
echo " 🌐 Real-to-Sim End-to-End Pipeline Execution (Strict Barriers)"
echo " 📂 Target Database : $TARGET_DB_PATH"
echo " 📂 Output Mesh     : $TARGET_MESH_PATH"
echo "=========================================================="

# ---------------------------------------------------------
# [PRE-FLIGHT] 하드웨어 & 환경 사전 검증 (2026-08-10 추가)
# ---------------------------------------------------------
pre_flight() {
    echo ""
    echo "=========================================================="
    echo " 🛠️ [PRE-FLIGHT] 하드웨어 환경 사전 검증"
    echo "=========================================================="

    # 1. USB 링크 속도 확인 (USB_3_MIN_SPEED_MBPS = USB 3.x 기준)
    #    ⚠️ 원격 카메라 모드(Windows 네이티브)에서는 USB가 WSL에 없으므로 검사 생략
    if [ "$CAMERA_MODE" = "local" ]; then
        USB_OK=false
        for dev in /sys/bus/usb/devices/*; do
            if [ -f "$dev/idVendor" ] && [ "$(cat "$dev/idVendor" 2>/dev/null)" = "8086" ]; then
                SPEED=$(cat "$dev/speed" 2>/dev/null)
                if [ "$SPEED" -ge "$USB_3_MIN_SPEED_MBPS" ] 2>/dev/null; then
                    echo "✅ RealSense USB 3.x 정상 (${SPEED} Mbps)"
                    USB_OK=true
                else
                    echo "⚠️  RealSense USB ${SPEED} Mbps — USB 2.x로 낮게 연결됨 (성능 저하 위험)"
                fi
            fi
        done
        if [ "$USB_OK" = false ]; then
            echo "⚠️  RealSense USB 장치를 감지하지 못했습니다."
        fi
    else
        echo "🌐 [원격 카메라] Windows 네이티브 모드 — USB 로컬 검사를 건너뜁니다."
    fi

    # 2. /dev/shm 용량 확인
    SHM_FREE=$(df -m /dev/shm 2>/dev/null | tail -n 1 | awk '{print $4}')
    echo "✅ RAM 디스크(/dev/shm) 여유: ${SHM_FREE}MB"

    # 3. 네트워크 버퍼 확인
    RMEM=$(sysctl net.core.rmem_max 2>/dev/null | awk '{print $3}')
    echo "✅ rmem_max: $((RMEM / 1024 / 1024))MB"

    # 4. 카메라 해상도/해상도 적합성 확인 (config.py 단일 소스)
    RES_STR="$CAMERA_RESOLUTION"
    echo "✅ 카메라 설정 해상도: ${RES_STR} (640x480이 VM 최적)"
    if [ "$RES_STR" != "640x480" ]; then
        echo "⚠️  [경고] 현재 해상도가 640x480이 아닙니다! VM USB 대역폭 초과로 depth 드랍 위험."
    fi
}
if [ "$SKIP_CAPTURE" = false ]; then
    pre_flight
fi

# ---------------------------------------------------------
# STEP 1: Real-Time Sensor Processing & Visual SLAM
# ---------------------------------------------------------
if [ "$SKIP_CAPTURE" = true ]; then
    echo "⏩ [STEP 1] --skip-capture 옵션 지정됨. 실시간 캡처를 건너끕니다."
else
    echo ""
    echo "=========================================================="
    echo " 🎥 [STEP 1] RTAB-Map SLAM 실시간 데이터 수집 시작"
    echo " 💡 촬영을 마치려면 터미널에서 Ctrl+C 를 누르세요."
    echo " 📊 촬영 품질 모니터링(capture_guard)이 병렬 실행됩니다."
    echo "=========================================================="
    
    # 카메라 센서 토픽 발행 여부 사전 체크
    if ! ros2 topic list | grep -q "$RGB_TOPIC"; then
        echo "❌ [오류] RealSense 카메라 토픽이 감지되지 않습니다!"
        if [ "$CAMERA_MODE" = "remote" ]; then
            echo "👉 [Windows]에서 카메라 노드(네임스페이스 /camera)와 압축 republish를 먼저 실행해주세요."
        else
            echo "👉 [터미널 1]에서 먼저 카메라를 구동해주세요:"
            echo "   ros2 launch auto_mobility camera.launch.py"
        fi
        exit 1
    fi

    # 촬영 품질 모니터링 가드를 백그라운드로 기동 (종료 시 .md 보고서 + .json 요약 저장)
    CAPTURE_GUARD_BASE="$LOG_DIR/capture_guard_${DB_NAME%.db}"
    echo "📊 capture_guard 모니터링 시작 → ${CAPTURE_GUARD_BASE}.md / .json"
    python3 "$PROJECT_DIR/src/auto_mobility/monitor/capture_guard.py" \
        --interval 5 --headless \
        --report "${CAPTURE_GUARD_BASE}.md" \
        --json "${CAPTURE_GUARD_BASE}.json" &
    GUARD_PID=$!
    trap "kill $GUARD_PID 2>/dev/null || true" EXIT INT TERM

    # RTAB-Map Visual SLAM 실시간 데이터 수집 (종료 시 바로 DB 생성)
    RTAB_LIVE_ARGS=("database_path:=$TARGET_DB_PATH")
    if [ "$CAMERA_MODE" = "remote" ]; then
        # Windows 원격 카메라: 네트워크 대역폭 절감을 위해 압축 토픽을 로컬에서 복원
        echo "🌐 원격 카메라 모드 → use_compressed:=$USE_COMPRESSED"
        RTAB_LIVE_ARGS+=("use_compressed:=$USE_COMPRESSED")
    fi
    ros2 launch auto_mobility rtab_live.launch.py "${RTAB_LIVE_ARGS[@]}" || true

    # 촬영 종료 → 모니터 종료 & 보고서 확인
    kill $GUARD_PID 2>/dev/null || true
    wait $GUARD_PID 2>/dev/null || true
    trap - EXIT INT TERM
    echo ""
    echo "📄 촬영 품질 보고서: ${CAPTURE_GUARD_BASE}.md"
    if [ -f "${CAPTURE_GUARD_BASE}.md" ]; then
        grep -E "시작|종료|Odom 시계열|평균 CPU" "${CAPTURE_GUARD_BASE}.md" | head -5
    fi
fi

# 🛡️ BARRIER 1: DB File Integrity Check
echo ""
echo "🛡️ [GATEWAY 1] RTAB-Map DB 파일 무결성 및 구조 검증..."
if ! python3 "$PROJECT_DIR/src/auto_mobility/utils/validate.py" --db "$TARGET_DB_PATH"; then
    echo "❌ [파이프라인 차단] DB 무결성 검증 실패로 인해 이후 단계를 진행할 수 없습니다."
    exit 1
fi

# ---------------------------------------------------------
# STEP 2-1: TSDF Mesh Reconstruction (RGB-D + pose → .obj)
# ---------------------------------------------------------
echo ""
echo "=========================================================="
echo " 🛠️ [STEP 2-1] Open3D Tensor TSDF Mesh 복원 파이프라인 구동"
echo "=========================================================="

# Open3D Tensor TSDF 재구성 (기본): 누적 Point Cloud 대신 원본 RGB-D + 최적화 pose를
# TSDF로 직접 적분 (GPU). voxel 1cm + weight-thr 1.5(2026-08-12 조정, 구멍 최소화).
# Poisson(PLY) 기반 복원 경로는 제거됨 (2026-08-12) — 원하면 mesh.sh 직접 사용.
MESH_METHOD_ARGS="--method=tsdf"
MESH_ARGS="--force --voxel=0.01"
echo "🧊 [STEP 2-1] TSDF 재구성: 원본 RGB-D + pose → Open3D Tensor TSDF (voxel 1cm, weight-thr 1.5)"
"$PIPELINE_DIR/mesh.sh" "$DB_NAME" "$MESH_NAME" $MESH_METHOD_ARGS $MESH_ARGS

# 🛡️ BARRIER 2: Mesh File Integrity Check
echo ""
echo "🛡️ [GATEWAY 2] 생성된 3D Mesh 무결성 검증..."
if ! python3 "$PROJECT_DIR/src/auto_mobility/utils/validate.py" --mesh "$TARGET_MESH_PATH"; then
    echo "❌ [파이프라인 차단] Mesh 무결성 검증 실패로 인해 Isaac Sim 로드가 중단됩니다."
    exit 1
fi

# ---------------------------------------------------------
# STEP 2-2: NVIDIA Isaac Sim Digital Twin Ingestion & Verification
# ---------------------------------------------------------
if [ "$SKIP_ISAAC" = true ]; then
    echo "⏩ [STEP 2-2] --skip-isaac 옵션 지정됨. Isaac Sim 로더를 건너끁니다."
else
    echo ""
    echo "=========================================================="
    echo " 🚀 [STEP 2-2] Isaac Sim 디지털 트윈 로드 및 물리 충돌 검증"
    echo "=========================================================="

    "$PIPELINE_DIR/isaac.sh" "$TARGET_MESH_PATH"
fi

# 📂 Windows 공유 폴더(/mnt/c/ubuntu_shared)로 Mesh 자동 복사 (WSL2)
SHARED_TARGET="/mnt/c/ubuntu_shared"
COPIED_TO_SHARED=false

if [ -d "$SHARED_TARGET" ] || mkdir -p "$SHARED_TARGET" 2>/dev/null; then
    cp "$TARGET_MESH_PATH" "$SHARED_TARGET/" 2>/dev/null || true
    COPIED_TO_SHARED=true
fi

echo ""
echo "=========================================================="
echo " 🎉 Real-to-Sim 파이프라인 완료!"
echo " 📁 생성된 Mesh (.obj): $TARGET_MESH_PATH"

if [ "$COPIED_TO_SHARED" = true ]; then
    echo " 📂 공유 폴더 복사 완료 : $SHARED_TARGET/"
fi

echo "=========================================================="
echo "💡 [Windows Isaac Sim 사용 방법]"
echo "   Windows 공유 폴더 (ubuntu_shared) 안의"
echo "   '${MESH_NAME}' 파일을 Isaac Sim에서 Import하면 됩니다."
echo "=========================================================="

# 📝 중요 로그 수집 종료 (요약 저장) — 종료 전 로그 파일 경로 보존
PLOG_FILE="$PIPELINE_LOG_FILE"
pipeline_log_stop || true

# 📊 세션 통합 분석 요약 생성 (구조화 JSON, 파일 비대화 없이 핵심 지표만)
SUMMARY_JSON="$LOG_DIR/session_${DB_NAME%.db}.summary.json"
if [ -n "$PLOG_FILE" ] && [ -f "$PLOG_FILE" ]; then
    echo ""
    echo "📊 세션 분석 요약 생성 → $SUMMARY_JSON"
    python3 "$PROJECT_DIR/src/auto_mobility/monitor/analyze_session.py" \
        --db "$TARGET_DB_PATH" \
        --log "$PLOG_FILE" \
        --guard-json "$LOG_DIR/capture_guard_${DB_NAME%.db}.json" \
        --out "$SUMMARY_JSON" || true
else
    echo "⏩ 세션 분석 요약 생략 (파이프라인 로그 없음)"
fi