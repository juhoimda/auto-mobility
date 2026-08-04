#!/bin/bash
# End-to-End Real-to-Sim Pipeline Orchestrator with Integrity Barriers
# Step 1: Real-time Camera Capture + RTAB-Map SLAM -> rtabmap.db & PointCloud
# Step 2: Open3D Mesh Reconstruction (.obj) -> Isaac Sim Digital Twin Ingestion

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="session_${TIMESTAMP}.db"
MESH_NAME="session_${TIMESTAMP}_mesh.obj"
SKIP_CAPTURE=false
SKIP_ISAAC=false

# CLI 옵션 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --db=*)
            DB_NAME="${1#*=}"
            MESH_NAME="${DB_NAME%.db}_mesh.obj"
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
        -h|--help)
            echo "=========================================================="
            echo " 사용법: $0 [--db=DB_NAME.db] [--skip-capture] [--skip-isaac]"
            echo " 예시  : $0 (실시간 캡처부터 Isaac Sim 검증까지 전체 실행)"
            echo " 예시  : $0 --db=my_room.db --skip-capture (기존 DB로 Mesh 및 Isaac Sim 실행)"
            echo "=========================================================="
            exit 0
            ;;
        *)
            echo "⚠️ 알 수 없는 옵션: $1"
            shift
            ;;
    esac
done

TARGET_DB_PATH="$DB_DIR/$DB_NAME"
TARGET_PLY_PATH="$POINTCLOUD_DIR/${DB_NAME%.db}_cloud.ply"
TARGET_MESH_PATH="$MESH_DIR/$MESH_NAME"

echo "=========================================================="
echo " 🌐 Real-to-Sim End-to-End Pipeline Execution (Strict Barriers)"
echo " 📂 Target Database : $TARGET_DB_PATH"
echo " 📂 Output Mesh     : $TARGET_MESH_PATH"
echo "=========================================================="

# ---------------------------------------------------------
# STEP 1: Real-Time Sensor Processing & Visual SLAM
# ---------------------------------------------------------
if [ "$SKIP_CAPTURE" = true ]; then
    echo "⏩ [STEP 1] --skip-capture 옵션 지정됨. 실시간 캡처를 건너끁니다."
else
    echo ""
    echo "=========================================================="
    echo " 🎥 [STEP 1] RTAB-Map SLAM 실시간 데이터 수집 시작"
    echo " 💡 촬영을 마치려면 터미널에서 Ctrl+C 를 누르세요."
    echo "=========================================================="
    
    # 카메라 센서 토픽 발행 여부 사전 체크
    if ! ros2 topic list | grep -q "/camera/camera/color/image_raw"; then
        echo "❌ [오류] RealSense 카메라 토픽이 감지되지 않습니다!"
        echo "👉 [터미널 1]에서 먼저 카메라를 구동해주세요:"
        echo "   ros2 launch auto_mobility camera.launch.py"
        exit 1
    fi

    # RTAB-Map Visual SLAM 실시간 데이터 수집 (종료 시 바로 DB 생성)
    ros2 launch auto_mobility rtab_live.launch.py database_path:="$TARGET_DB_PATH" || true
fi

# 🛡️ BARRIER 1: DB File Integrity Check
echo ""
echo "🛡️ [GATEWAY 1] RTAB-Map DB 파일 무결성 및 구조 검증..."
if ! python3 "$PROJECT_DIR/src/auto_mobility/processing/validate.py" --db "$TARGET_DB_PATH"; then
    echo "❌ [파이프라인 차단] DB 무결성 검증 실패로 인해 이후 단계를 진행할 수 없습니다."
    exit 1
fi

# ---------------------------------------------------------
# STEP 2-1: Point Cloud Extraction & Open3D Mesh Reconstruction
# ---------------------------------------------------------
echo ""
echo "=========================================================="
echo " 🛠️ [STEP 2-1] Open3D 기반 3D Mesh 복원 파이프라인 구동"
echo "=========================================================="

"$PIPELINE_DIR/mesh.sh" "$DB_NAME" "$MESH_NAME" --force

# 🛡️ BARRIER 2: Mesh File Integrity Check
echo ""
echo "🛡️ [GATEWAY 2] 생성된 3D Mesh 무결성 검증..."
if ! python3 "$PROJECT_DIR/src/auto_mobility/processing/validate.py" --mesh "$TARGET_MESH_PATH"; then
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

echo ""
echo "=========================================================="
echo " 🎉 Real-to-Sim 전체 파이프라인 완료 및 무결성 검증 성공!"
echo "=========================================================="
