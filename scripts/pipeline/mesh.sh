#!/bin/bash
# mesh.sh — Canonical Frame Dataset / DB + Trajectory로부터 3D Point Cloud(.ply) 및 3D Surface Mesh(.obj)를 생성

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"
export PYTHONWARNINGS="ignore"

SLAM_TYPE="rtab"
VOXEL="0.01"
VIEW_FLAG=""
INPUT_NAME=""
CUSTOM_OUT=""

for arg in "$@"; do
    case $arg in
        --view)
            VIEW_FLAG="--view"
            ;;
        --slam=*)
            SLAM_TYPE="${arg#*=}"
            ;;
        --voxel=*)
            VOXEL="${arg#*=}"
            ;;
        --fine)
            VOXEL="0.005"
            ;;
        rtab|rtabmap)
            SLAM_TYPE="rtab"
            ;;
        orb|orbslam|orbslam3)
            SLAM_TYPE="orb"
            ;;
        -*)
            echo "⚠️ 알 수 없는 옵션: $arg"
            ;;
        *)
            if [ -z "$INPUT_NAME" ]; then
                INPUT_NAME="$arg"
            elif [ -z "$CUSTOM_OUT" ]; then
                CUSTOM_OUT="$arg"
            fi
            ;;
    esac
done

if [ -z "$INPUT_NAME" ]; then
    echo "=========================================================="
    echo " 사용법: $0 DATASET_NAME [--slam=rtab|orb] [--voxel=0.01|0.005|--fine] [--view]"
    echo " 예시 (기본 RTAB-Map 10mm)    : $0 my_dataset --view"
    echo " 예시 (ORB-SLAM3 궤적 적용)   : $0 my_dataset --slam=orb --view"
    echo " 예시 (5mm 초고정밀 메쉬)     : $0 my_dataset --fine --view"
    echo "=========================================================="
    exit 1
fi

BASE_NAME="$(basename "$INPUT_NAME" .db)"

# 1. Canonical Dataset 확인 및 없으면 자동 생성
FRAME_PATH="$FRAME_DIR/$BASE_NAME"
if [ ! -d "$FRAME_PATH" ] || [ ! -f "$FRAME_PATH/frames.csv" ]; then
    if [ -d "$BAG_DIR/$BASE_NAME" ] || [ -f "$BAG_DIR/$BASE_NAME" ]; then
        echo "⚙️ Canonical Frame Dataset이 없어 자동 추출합니다..."
        "$PIPELINE_DIR/prepare_dataset.sh" "$BASE_NAME"
    fi
fi

# 2. SLAM 및 Trajectory 준비
TRAJ_FILE=""
SUFFIX="_rtab_tsdf"

if [ "$SLAM_TYPE" == "orb" ] || [ "$SLAM_TYPE" == "orbslam" ] || [ "$SLAM_TYPE" == "orbslam3" ]; then
    TRAJ_FILE="$TRAJECTORY_DIR/orbslam3_${BASE_NAME}_trajectory.txt"
    if [ ! -f "$TRAJ_FILE" ]; then
        echo "⚙️ ORB-SLAM3 궤적이 없어 SLAM을 실행합니다..."
        "$PIPELINE_DIR/run_slam.sh" "$BASE_NAME" --slam=orb
    fi
    SUFFIX="_orbslam_tsdf"
else
    # RTAB-Map
    TRAJ_FILE="$TRAJECTORY_DIR/rtab_${BASE_NAME}_trajectory.txt"
    DB_FILE="$DB_DIR/${BASE_NAME}.db"
    if [ ! -f "$TRAJ_FILE" ] && [ ! -f "$DB_FILE" ]; then
        echo "⚙️ RTAB-Map SLAM 결과가 없어 SLAM을 자동 실행합니다..."
        "$PIPELINE_DIR/run_slam.sh" "$BASE_NAME" --slam=rtab
    fi
fi

if [ "$VOXEL" == "0.005" ]; then
    SUFFIX="${SUFFIX}_fine"
fi

OUT_MESH="${CUSTOM_OUT:-$MESH_DIR/${BASE_NAME}${SUFFIX}.obj}"
OUT_PCD="$POINTCLOUD_DIR/${BASE_NAME}${SUFFIX}_cloud.ply"

echo "=========================================================="
echo " 🔨 3D Reconstruction 복원 시작 (TSDF Voxel: ${VOXEL}m)"
if [ -d "$FRAME_PATH" ] && [ -f "$FRAME_PATH/frames.csv" ]; then
    echo " 📂 프레임 입력 : $FRAME_PATH"
else
    echo " 📁 DB 파일 입력: $DB_FILE"
fi
echo " 📍 궤적 파일   : $TRAJ_FILE"
echo " 🛠️ SLAM 백엔드 : $SLAM_TYPE"
echo " ☁️ 점군 출력   : $OUT_PCD"
echo " 💾 메쉬 출력   : $OUT_MESH"
echo "=========================================================="

if [ -d "$FRAME_PATH" ] && [ -f "$FRAME_PATH/frames.csv" ] && [ -f "$TRAJ_FILE" ]; then
    python3 "$PROJECT_DIR/src/auto_mobility/mesh/reconstruct_tsdf.py" \
        --dataset="$FRAME_PATH" \
        --trajectory="$TRAJ_FILE" \
        --output="$OUT_MESH" \
        --pcd-output="$OUT_PCD" \
        --voxel="$VOXEL"
else
    # Legacy DB fallback
    python3 "$PROJECT_DIR/src/auto_mobility/mesh/reconstruct_tsdf.py" \
        "$DB_FILE" "$OUT_MESH" \
        --pcd-output="$OUT_PCD" \
        --voxel="$VOXEL" \
        ${TRAJ_FILE:+--trajectory="$TRAJ_FILE"}
fi

if [ $? -eq 0 ]; then
    echo "=========================================================="
    echo " ✅ 3D 복원 완료!"
    echo " ☁️ 점군 확인: ./scripts/utils/view_pointcloud.sh $(basename "$OUT_PCD")"
    echo " 💾 메쉬 확인: ./scripts/utils/view_mesh.sh $(basename "$OUT_MESH")"
    echo "=========================================================="
    
    if [ -n "$VIEW_FLAG" ]; then
        "$PROJECT_DIR/scripts/utils/view_mesh.sh" "$OUT_MESH"
    fi
fi