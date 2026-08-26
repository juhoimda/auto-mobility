#!/bin/bash
# evaluate.sh — Held-out Sensor Consistency 기반 3D Reconstruction 정량 형상 품질 평가

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

DATASET_NAME="$1"
MESH_PATH="$2"
TRAJ_PATH="$3"

if [ -z "$DATASET_NAME" ] || [ -z "$MESH_PATH" ]; then
    echo "=========================================================="
    echo " 사용법: $0 DATASET_NAME MESH_PATH [TRAJECTORY_PATH] [--name CANDIDATE_NAME]"
    echo " 예시  : $0 room01 ros2_data/meshes/room01_rtab_tsdf.obj ros2_data/trajectories/rtab_room01_trajectory.txt"
    echo " 설명  : Held-out D435i Depth 프레임 및 메쉬 Raycasting을 통해"
    echo "         Depth MAE, P95, Coverage, Point-to-Mesh 거리를 자동 측정하고"
    echo "         PASS/WARN/FAIL 종합 판정 리포트를 생성합니다."
    echo "=========================================================="
    exit 1
fi

# If trajectory not provided as 3rd arg, check default candidates
if [ -z "$TRAJ_PATH" ] || [ "$TRAJ_PATH" == "--name" ]; then
    if [ -f "$TRAJECTORY_DIR/rtab_${DATASET_NAME}_trajectory.txt" ]; then
        TRAJ_PATH="$TRAJECTORY_DIR/rtab_${DATASET_NAME}_trajectory.txt"
    elif [ -f "$TRAJECTORY_DIR/orbslam3_${DATASET_NAME}_trajectory.txt" ]; then
        TRAJ_PATH="$TRAJECTORY_DIR/orbslam3_${DATASET_NAME}_trajectory.txt"
    fi
fi

shift 3 2>/dev/null || shift $# 2>/dev/null || true

python3 "$PROJECT_DIR/src/auto_mobility/evaluation/evaluator.py" \
    "$DATASET_NAME" "$MESH_PATH" "$TRAJ_PATH" "$@"
