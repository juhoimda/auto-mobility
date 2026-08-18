#!/bin/bash
# ============================================================
# run_pipeline_all.sh — End-to-End Real-to-Sim 파이프라인 오케스트레이터
#
# 파이프라인 흐름:
#   STEP 1: D435i → Windows publisher → DDS → WSL
#           → RTAB-Map Visual SLAM (rtab_live.launch.py)
#           → databases/<session>.db
#
#   GATEWAY 1: DB 무결성 검증 (validate.py)
#
#   STEP 2-1: databases/<session>.db
#             → Open3D Tensor TSDF reconstruction (reconstruct_tsdf.py)
#             → meshes/<session>_tsdf.obj
#
#   GATEWAY 2: Mesh 무결성 검증 (validate.py)
#
#   STEP 2-2: meshes/<session>_tsdf.obj
#             → Isaac Sim 디지털 트윈 로드 (isaac.sh → load_isaac_mesh.py)
#
# 주의: RTAB-Map DB는 derived artifact이며 raw dataset(bags/)이 source of truth다.
#   Mesh reconstruction에는 /rtabmap/cloud_map이 아닌 DB 내 원본 RGB-D + 최적화 pose를 사용.
# ============================================================

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
            echo " 예시  : $0 --remote-camera (Windows 카메라 자동 실행 + 압축 토픽 수신, 종료 시 정리)"
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

# 원격 카메라 모드: 압축 토픽(JPEG/PNG) 수신 후 WSL에서 초고속 디코딩(republish.py)하여 RTAB-Map 공급 (30fps)
if [ "$CAMERA_MODE" = "remote" ]; then
    USE_COMPRESSED=true
fi

# ---------------------------------------------------------
# Windows 원격 카메라 자동 실행/종료 헬퍼 (2026-08-14 갱신)
#   - 실행 방식 자동 감지: WSL interop(cmd.exe) 우선, 불가 시 "신호파일 + Windows watcher" 폴백
#     (이 환경은 interop 이 EINVAL 로 깨짐 → watcher 방식이 기본 동작 경로)
#   - 신호파일 방식: WSL이 공유폴더(/mnt/c/ubuntu_shared/camera_request.txt)에 신호를 쓰면
#     Windows의 camera_guard.ps1 이 감지해 run_camera.bat 을 구동/종료. interop/SSH 불필요.
#   - 카메라 로그: watcher 가 C:\ubuntu_shared\camera_windows.log 에 기록 → WSL이 요약 출력
# ---------------------------------------------------------
WINDOWS_CAM_LOG=""
WINDOWS_CAM_PID=""
WIN_LAUNCH_MODE=""
# 신호파일 방식 경로 (Windows watcher 와 공유)
WIN_SIGNAL_FILE="$SHARED_DIR/camera_request.txt"
WIN_CAM_LOG_SHARED="$SHARED_DIR/camera_windows.log"

detect_windows_launcher() {
    if cmd.exe /c ver >/dev/null 2>&1; then
        WIN_LAUNCH_MODE="interop"
        echo "🔌 Windows 카메라 실행 방식: interop (cmd.exe)"
    else
        WIN_LAUNCH_MODE="signal"
        echo "🔌 Windows 카메라 실행 방식: 신호파일 + watcher (interop 미지원 환경)"
        echo "   - watcher 확인: Windows 시작폴더의 camera_guard.vbs (자동) / C:\\ros2_humble\\camera_guard.ps1"
        echo "   - 신호 파일: $WIN_SIGNAL_FILE / 로그: $WIN_CAM_LOG_SHARED"
    fi
}

start_windows_camera() {
    if [ "$CAMERA_MODE" != "remote" ]; then
        return 0
    fi
    if [ -z "$WIN_LAUNCH_MODE" ]; then
        detect_windows_launcher
    fi
    case "$WIN_LAUNCH_MODE" in
        interop)
            WINDOWS_CAM_LOG="$LOG_DIR/camera_windows_${DB_NAME%.db}.log"
            echo "🪟 Windows 카메라 자동 실행 (run_camera.bat) → 로그: $WINDOWS_CAM_LOG"
            echo "💡 실시간 확인: tail -f \"$WINDOWS_CAM_LOG\""
            cmd.exe /c "call C:\\ros2_humble\\run_camera.bat" > "$WINDOWS_CAM_LOG" 2>&1 &
            WINDOWS_CAM_PID=$!
            ;;
        signal)
            WINDOWS_CAM_LOG="$WIN_CAM_LOG_SHARED"
            echo "🪟 Windows 카메라 시작 신호 전송 → $WIN_SIGNAL_FILE"
            echo "   (Windows watcher 가 감지해 run_camera.bat 구동, 로그: $WINDOWS_CAM_LOG)"
            echo "run_camera" > "$WIN_SIGNAL_FILE" 2>/dev/null || {
                echo "❌ 신호 파일 기록 실패 — /mnt/c/ubuntu_shared 쓰기 권한 확인"
                exit 1
            }
            ;;
        *)
            exit 1
            ;;
    esac
    # 토픽 출현 대기 (직접 구독 프로브, 최대 20회)
    PROBE_TOPIC="$RGB_COMPRESSED_TOPIC"
    for i in $(seq 1 20); do
        if python3 "$PROJECT_DIR/scripts/utils/topic_probe.py" "$PROBE_TOPIC" 4 >/dev/null 2>&1; then
            echo "✅ 카메라 토픽 확인됨 (${i}회 프로브)"
            return 0
        fi
        echo "  ... 카메라 토픽 대기 중 (${i}/20)"
        sleep 1
    done
    echo "❌ [오류] 카메라 토픽이 20초 내 감지되지 않았습니다."
    if [ "$WIN_LAUNCH_MODE" = "signal" ]; then
        echo "👉 Windows watcher(camera_guard.ps1) 가 실행 중인지 확인하세요. (시작 폴더 자동실행 / C:\\ros2_humble\\start_camera_guard.vbs)"
    fi
    echo "📄 최근 카메라 로그:"
    tail -n 25 "$WINDOWS_CAM_LOG" 2>/dev/null || true
    stop_windows_camera
    exit 1
}

stop_windows_camera() {
    case "$WIN_LAUNCH_MODE" in
        interop)
            [ -n "$WINDOWS_CAM_PID" ] && kill "$WINDOWS_CAM_PID" 2>/dev/null || true
            WINDOWS_CAM_PID=""
            cmd.exe /c "call C:\\ros2_humble\\stop_camera.bat" >/dev/null 2>&1 || true
            ;;
        signal)
            rm -f "$WIN_SIGNAL_FILE" 2>/dev/null || true
            ;;
    esac
}

cleanup() {
    kill $GUARD_PID 2>/dev/null || true
    stop_windows_camera
}

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
    echo "=========================================================="

    # 사전 정리 트랩: Ctrl+C/종료 시 신호 파일·Windows 카메라 프로세스 확실히 정리
    # (기존엔 카메라 시작 후에 트랩을 걸어, 대기 중 Ctrl+C 시 신호 파일이 남는 버그가 있었음)
    trap "cleanup; exit 130" INT TERM
    trap "cleanup" EXIT

    # 카메라 토픽 체크: 직접 구독 프로브 사용 (ros2 CLI 데몬이 이 환경에서 hang 하는 문제 회피)
    PROBE_TOPIC="$RGB_TOPIC"
    [ "$CAMERA_MODE" = "remote" ] && PROBE_TOPIC="$RGB_COMPRESSED_TOPIC"
    echo "🔍 카메라 토픽 확인 중: $PROBE_TOPIC ..."
    if ! python3 "$PROJECT_DIR/scripts/utils/topic_probe.py" "$PROBE_TOPIC" 5 >/dev/null 2>&1; then
        if [ "$CAMERA_MODE" = "remote" ]; then
            # Windows 카메라 자동 실행 후 재확인 (실패 시 내부에서 로그 출력 + exit)
            start_windows_camera
        else
            echo "❌ [오류] RealSense 카메라 토픽이 감지되지 않습니다!"
            echo "👉 [터미널 1]에서 먼저 카메라를 구동해주세요:"
            echo "   ros2 launch auto_mobility camera.launch.py"
            exit 1
        fi
    else
        echo "✅ 카메라 토픽 수신 확인됨"
    fi

    # 촬영 품질 모니터링 가드를 백그라운드로 기동 (종료 시 .md 보고서 + .json 요약 저장)
    CAPTURE_GUARD_BASE="$LOG_DIR/capture_guard_${DB_NAME%.db}"
    echo "📊 capture_guard 모니터링 시작 → ${CAPTURE_GUARD_BASE}.md / .json"
    python3 "$PROJECT_DIR/src/auto_mobility/monitor/capture_guard.py" \
        --interval 5 --headless \
        --report "${CAPTURE_GUARD_BASE}.md" \
        --json "${CAPTURE_GUARD_BASE}.json" &
    GUARD_PID=$!

    echo "💡 촬영을 마치려면 터미널에서 Ctrl+C 를 누르세요."

    # RTAB-Map Visual SLAM 실시간 데이터 수집 (종료 시 바로 DB 생성)
    RTAB_LIVE_ARGS=("database_path:=$TARGET_DB_PATH")
    if [ "$CAMERA_MODE" = "remote" ]; then
        echo "🌐 원격 카메라 모드 → use_compressed:=$USE_COMPRESSED"
        RTAB_LIVE_ARGS+=("use_compressed:=$USE_COMPRESSED")
    fi

    # ros2 launch 실행 중에는 SIGINT를 launch 노드들이 받아 클린 커밋하도록 함
    trap "stop_windows_camera" INT TERM
    ros2 launch auto_mobility rtab_live.launch.py "${RTAB_LIVE_ARGS[@]}" || true

    # 촬영 종료 → 모니터 & Windows 카메라 정리
    kill $GUARD_PID 2>/dev/null || true
    wait $GUARD_PID 2>/dev/null || true
    stop_windows_camera
    trap "cleanup; exit 130" INT TERM
    echo ""
    echo "📄 촬영 품질 보고서: ${CAPTURE_GUARD_BASE}.md"
    if [ -f "${CAPTURE_GUARD_BASE}.md" ]; then
        grep -E "시작|종료|Odom 시계열|평균 CPU" "${CAPTURE_GUARD_BASE}.md" | head -5
    fi

    # Windows 카메라 로그의 WARN/ERROR 요약 (에러 파악용, 별도 터미널 대체)
    if [ -n "$WINDOWS_CAM_LOG" ] && [ -f "$WINDOWS_CAM_LOG" ]; then
        echo ""
        echo "📋 Windows 카메라 로그 요약 (WARN/ERROR): $WINDOWS_CAM_LOG"
        grep -iE "WARN|ERROR|fail|drop|unable|cannot" "$WINDOWS_CAM_LOG" | tail -15 || true
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