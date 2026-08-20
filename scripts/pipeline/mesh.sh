#!/bin/bash
# mesh.sh — Canonical Frame Dataset / DB + Trajectory로부터 3D Point Cloud(.ply) 및 3D Surface Mesh(.obj)를 생성

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"
export PYTHONWARNINGS="ignore"

# HW 과부하 및 셧다운 방지를 위한 CPU 멀티스레드 상한 제한
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

SLAM_TYPE="rtab"
SURFACE_TYPE="tsdf_direct"
FUSION_TYPE="tsdf"
VOXEL="0.008"
STRIDE="1"
DEPTH_MAX="3.0"
TRUNC_MULT="4"
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
        --surface=*|--method=*)
            SURFACE_TYPE="${arg#*=}"
            ;;
        --fusion=*)
            FUSION_TYPE="${arg#*=}"
            ;;
        --voxel=*)
            VOXEL="${arg#*=}"
            ;;
        --stride=*)
            STRIDE="${arg#*=}"
            ;;
        --depth-max=*)
            DEPTH_MAX="${arg#*=}"
            ;;
        --trunc-mult=*)
            TRUNC_MULT="${arg#*=}"
            ;;
        --fine)
            VOXEL="0.005"
            ;;
        rtab|rtabmap)
            SLAM_TYPE="rtab"
            ;;
        orb|orbslam|orbslam3|orb_rgbd)
            SLAM_TYPE="orb_rgbd"
            ;;
        orb_rgbdi|rgbdi)
            SLAM_TYPE="orb_rgbdi"
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
    echo " 사용법: $0 DATASET_NAME [--slam=rtab|rtab_dense|orb_rgbd|orb_rgbdi|stella_rgbd] [--surface=tsdf_direct|poisson|bpa|alpha|cgal_polygonal] [--voxel=0.008|0.005] [--depth-max=3.0] [--view]"
    echo " 예시 (기본 TSDF 8mm)        : $0 my_dataset --surface=tsdf_direct --view"
    echo " 예시 (Alpha Shape 메쉬)     : $0 my_dataset --surface=alpha --view"
    echo " 예시 (Poisson 8 octree)     : $0 my_dataset --surface=poisson --view"
    echo " 예시 (ORB-SLAM3 RGBD-I 적용) : $0 my_dataset --slam=orb_rgbdi --view"
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
SUFFIX="_${SLAM_TYPE}_${SURFACE_TYPE}"

if [ "$SLAM_TYPE" == "orb_rgbdi" ]; then
    TRAJ_FILE="$TRAJECTORY_DIR/orb_rgbdi_${BASE_NAME}_trajectory.txt"
    if [ ! -f "$TRAJ_FILE" ]; then
        echo "⚙️ ORB-SLAM3 RGB-D-I 궤적이 없어 SLAM을 실행합니다..."
        "$PIPELINE_DIR/run_slam.sh" "$BASE_NAME" --slam=orb_rgbdi
    fi
elif [ "$SLAM_TYPE" == "orb" ] || [ "$SLAM_TYPE" == "orb_rgbd" ] || [ "$SLAM_TYPE" == "orbslam3" ]; then
    TRAJ_FILE="$TRAJECTORY_DIR/orb_rgbd_${BASE_NAME}_trajectory.txt"
    if [ ! -f "$TRAJ_FILE" ]; then
        echo "⚙️ ORB-SLAM3 궤적이 없어 SLAM을 실행합니다..."
        "$PIPELINE_DIR/run_slam.sh" "$BASE_NAME" --slam=orb_rgbd
    fi
elif [ "$SLAM_TYPE" == "stella" ] || [ "$SLAM_TYPE" == "stella_rgbd" ]; then
    TRAJ_FILE="$TRAJECTORY_DIR/stella_${BASE_NAME}_trajectory.txt"
    if [ ! -f "$TRAJ_FILE" ]; then
        echo "⚙️ stella_vslam 궤적이 없어 SLAM을 실행합니다..."
        "$PIPELINE_DIR/run_slam.sh" "$BASE_NAME" --slam=stella_rgbd
    fi
elif [ "$SLAM_TYPE" == "rtab_dense" ]; then
    TRAJ_FILE="$TRAJECTORY_DIR/rtab_dense_${BASE_NAME}_trajectory.txt"
    DB_FILE="$DB_DIR/${BASE_NAME}_dense.db"
    if [ ! -f "$TRAJ_FILE" ] && [ ! -f "$DB_FILE" ]; then
        echo "⚙️ 고밀도 RTAB-Map 궤적이 없어 SLAM을 실행합니다..."
        "$PIPELINE_DIR/run_slam.sh" "$BASE_NAME" --slam=rtab --dense
    fi
else
    # RTAB-Map
    TRAJ_FILE="$TRAJECTORY_DIR/rtab_${BASE_NAME}_trajectory.txt"
    DB_FILE="$DB_DIR/${BASE_NAME}.db"
    if [ ! -f "$TRAJ_FILE" ] && [ ! -f "$DB_FILE" ]; then
        echo "⚙️ RTAB-Map SLAM 결과가 없어 SLAM을 자동 실행합니다..."
        "$PIPELINE_DIR/run_slam.sh" "$BASE_NAME" --slam=rtab
    fi
fi

OUT_MESH="${CUSTOM_OUT:-$MESH_DIR/${BASE_NAME}${SUFFIX}.obj}"
OUT_PCD="$POINTCLOUD_DIR/${BASE_NAME}${SUFFIX}_cloud.ply"

echo "=========================================================="
echo " 🔨 3D Reconstruction 복원 시작"
echo " 📂 프레임 입력 : $FRAME_PATH"
echo " 📍 궤적 파일   : $TRAJ_FILE"
echo " 🛠️ SLAM 백엔드 : $SLAM_TYPE"
echo " 🧱 Surface 방식: $SURFACE_TYPE"
echo " ⚙️ TSDF Voxel  : ${VOXEL}m"
echo " 📏 Depth 범위 : 0.3m ~ ${DEPTH_MAX}m (trunc ${TRUNC_MULT} voxels)"
echo " ☁️ 점군 출력   : $OUT_PCD"
echo " 💾 메쉬 출력   : $OUT_MESH"
echo "=========================================================="

if [ "$SURFACE_TYPE" == "tsdf_direct" ] || [ "$SURFACE_TYPE" == "tsdf" ]; then
    if [ -d "$FRAME_PATH" ] && [ -f "$FRAME_PATH/frames.csv" ] && [ -f "$TRAJ_FILE" ]; then
        python3 "$PROJECT_DIR/src/auto_mobility/mesh/reconstruct_tsdf.py" \
            --dataset="$FRAME_PATH" \
            --trajectory="$TRAJ_FILE" \
            --output="$OUT_MESH" \
            --pcd-output="$OUT_PCD" \
            --voxel="$VOXEL" \
            --depth-max="$DEPTH_MAX" \
            --trunc-mult="$TRUNC_MULT" \
            --stride="$STRIDE"
    else
        python3 "$PROJECT_DIR/src/auto_mobility/mesh/reconstruct_tsdf.py" \
            "$DB_FILE" "$OUT_MESH" \
            --pcd-output="$OUT_PCD" \
            --voxel="$VOXEL" \
            --depth-max="$DEPTH_MAX" \
            --trunc-mult="$TRUNC_MULT" \
            --stride="$STRIDE" \
            ${TRAJ_FILE:+--trajectory="$TRAJ_FILE"}
    fi
else
    # Non-TSDF surface methods (Poisson, BPA, Alpha, CGAL)
    # First generate point cloud via TSDF if not present
    if [ ! -f "$OUT_PCD" ]; then
        python3 "$PROJECT_DIR/src/auto_mobility/mesh/reconstruct_tsdf.py" \
            --dataset="$FRAME_PATH" \
            --trajectory="$TRAJ_FILE" \
            --pcd-output="$OUT_PCD" \
            --voxel="$VOXEL" \
            --depth-max="$DEPTH_MAX" \
            --trunc-mult="$TRUNC_MULT" \
            --stride="$STRIDE"
    fi
    python3 "$PROJECT_DIR/src/auto_mobility/mesh/mesh_open3d.py" \
        "$OUT_PCD" "$OUT_MESH" \
        --method="$SURFACE_TYPE" \
        --voxel="$VOXEL"
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
