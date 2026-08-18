#!/bin/bash
# mesh.sh — DB/Rosbag으로부터 3D Point Cloud(.ply) 및 3D Surface Mesh(.obj)를 생성

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"
export PYTHONWARNINGS="ignore"

SLAM_TYPE="rtab"
VOXEL="0.02"
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

# 1. DB 확인 또는 자동 SLAM 생성
DB_FILE="$DB_DIR/${BASE_NAME}.db"
if [ ! -f "$DB_FILE" ]; then
    echo "⚙️ DB 파일이 없어 SLAM을 자동 실행합니다..."
    "$PIPELINE_DIR/run_slam.sh" "$BASE_NAME" --slam=rtab
fi

# 2. ORB-SLAM3 궤적 여부 확인
TRAJ_ARG=""
SUFFIX="_tsdf"
if [ "$SLAM_TYPE" == "orb" ] || [ "$SLAM_TYPE" == "orbslam" ] || [ "$SLAM_TYPE" == "orbslam3" ]; then
    ORB_TRAJ="$TRAJECTORY_DIR/orbslam3_${BASE_NAME}_trajectory.txt"
    if [ ! -f "$ORB_TRAJ" ]; then
        echo "⚙️ ORB-SLAM3 궤적이 없어 궤적 추출을 먼저 실행합니다..."
        "$PIPELINE_DIR/run_slam.sh" "$BASE_NAME" --slam=orb
    fi
    TRAJ_ARG="--trajectory=$ORB_TRAJ"
    SUFFIX="_orbslam"
fi

if [ "$VOXEL" == "0.005" ]; then
    SUFFIX="${SUFFIX}_fine"
fi

OUT_MESH="${CUSTOM_OUT:-$MESH_DIR/${BASE_NAME}${SUFFIX}.obj}"
OUT_PCD="$POINTCLOUD_DIR/${BASE_NAME}${SUFFIX}_cloud.ply"

echo "=========================================================="
echo " 🔨 3D 복원 시작 (TSDF Voxel: ${VOXEL}m)"
echo " 📁 입력 DB   : $DB_FILE"
echo " 🛠️ SLAM 엔진 : $SLAM_TYPE"
echo " ☁️ 점군 출력 : $OUT_PCD"
echo " 💾 메쉬 출력 : $OUT_MESH"
echo "=========================================================="

python3 "$PROJECT_DIR/src/auto_mobility/mesh/reconstruct_tsdf.py" \
    "$DB_FILE" "$OUT_MESH" \
    --pcd-output="$OUT_PCD" \
    --voxel="$VOXEL" \
    $TRAJ_ARG

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