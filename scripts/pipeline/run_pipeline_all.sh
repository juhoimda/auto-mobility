#!/bin/bash
# End-to-End Real-to-Sim Pipeline Orchestrator
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

echo "=========================================================="
echo " 🌐 Real-to-Sim End-to-End Pipeline Execution"
echo " 📂 Target Database : $DB_NAME"
echo " 📂 Output Mesh     : $MESH_NAME"
echo "=========================================================="

# ---------------------------------------------------------
# STEP 1: Real-Time Sensor Capture & Visual SLAM (RTAB-Map)
# ---------------------------------------------------------
if [ "$SKIP_CAPTURE" = true ]; then
    echo "⏩ [STEP 1] --skip-capture 옵션 지정됨. 실시간 캡처를 건너뜁니다."
else
    echo ""
    echo "=========================================================="
    echo " 🎥 [STEP 1] RealSense & RTAB-Map SLAM 실시간 스캔 시작"
    echo " 💡 촬영을 마치려면 터미널에서 Ctrl+C 를 누르세요."
    echo "=========================================================="
    
    "$PIPELINE_DIR/record.sh" "$DB_NAME" || true
    echo "✅ [STEP 1 완료] RTAB-Map DB 저장 완료 -> $DB_DIR/$DB_NAME"
fi

# ---------------------------------------------------------
# STEP 2-1: Point Cloud Extraction & Open3D Mesh Reconstruction
# ---------------------------------------------------------
echo ""
echo "=========================================================="
echo " 🛠️ [STEP 2-1] Open3D 기반 3D Mesh 복원 파이프라인 구동"
echo "=========================================================="

"$PIPELINE_DIR/mesh.sh" "$DB_NAME" "$MESH_NAME" --force

GENERATED_MESH_PATH="$MESH_DIR/$MESH_NAME"
if [ ! -f "$GENERATED_MESH_PATH" ]; then
    echo "❌ 오류: Mesh 생성이 실패하였거나 파일이 없습니다 -> $GENERATED_MESH_PATH"
    exit 1
fi
echo "✅ [STEP 2-1 완료] 3D Mesh 생성 완료 -> $GENERATED_MESH_PATH"

# ---------------------------------------------------------
# STEP 2-2: NVIDIA Isaac Sim Digital Twin Ingestion & Verification
# ---------------------------------------------------------
if [ "$SKIP_ISAAC" = true ]; then
    echo "⏩ [STEP 2-2] --skip-isaac 옵션 지정됨. Isaac Sim 로더를 건너뜁니다."
else
    echo ""
    echo "=========================================================="
    echo " 🚀 [STEP 2-2] Isaac Sim 디지털 트윈 로드 및 물리 충돌 검증"
    echo "=========================================================="

    "$PIPELINE_DIR/isaac.sh" "$GENERATED_MESH_PATH"
fi

echo ""
echo "=========================================================="
echo " 🎉 Real-to-Sim 전체 파이프라인 완료!"
echo "=========================================================="
