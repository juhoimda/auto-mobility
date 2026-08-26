#!/bin/bash
# 3D Point Cloud Viewer Launcher Script (.ply / .pcd)
# 기본적으로 Windows의 MeshLab을 실행하여 GPU 가속으로 부드럽게 시각화합니다.

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

USE_OPEN3D_LOCAL=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --open3d|--wsl|--local)
            USE_OPEN3D_LOCAL=true
            ;;
        -h|--help)
            echo "=========================================================="
            echo " 사용법: $0 [POINTCLOUD_FILE] [--open3d] [--point-size=N] [--voxel=N]"
            echo "  - 기본 동작 : Windows MeshLab을 연동하여 GPU 가속으로 빠르게 실행"
            echo "  - --open3d  : WSL 내부 Open3D 파이썬 뷰어로 강제 실행"
            echo " 예시 (Windows MeshLab) : $0 my_dataset_rtab_tsdf_cloud.ply"
            echo " 예시 (WSL Open3D 뷰어) : $0 my_dataset_rtab_tsdf_cloud.ply --open3d --point-size=4"
            echo "=========================================================="
            exit 0
            ;;
        *)
            if [ -z "$PCD_INPUT" ]; then
                PCD_INPUT="$arg"
            else
                EXTRA_ARGS+=("$arg")
            fi
            ;;
    esac
done

# view_mesh.sh 스크립트로 전달 (동일한 윈도우 MeshLab 연동 및 경로 탐색 지원)
if [ "$USE_OPEN3D_LOCAL" = true ]; then
    if [ -d /mnt/wslg ]; then
        export WAYLAND_DISPLAY=
        export DISPLAY="${DISPLAY:-:0}"
    fi
    # 경로 판별
    if [[ "$PCD_INPUT" == /* ]] && [ -f "$PCD_INPUT" ]; then
        PCD_FILE="$PCD_INPUT"
    elif [ -f "$POINTCLOUD_DIR/$PCD_INPUT" ]; then
        PCD_FILE="$POINTCLOUD_DIR/$PCD_INPUT"
    elif [ -f "$PROJECT_DIR/$PCD_INPUT" ]; then
        PCD_FILE="$PROJECT_DIR/$PCD_INPUT"
    elif [ -f "$PWD/$PCD_INPUT" ]; then
        PCD_FILE="$PWD/$PCD_INPUT"
    else
        PCD_FILE="$PCD_INPUT"
    fi
    exec python3 "$PROJECT_DIR/src/auto_mobility/mesh/view_pointcloud.py" "$PCD_FILE" "${EXTRA_ARGS[@]}"
else
    exec "$PROJECT_DIR/scripts/utils/view_mesh.sh" ${PCD_INPUT:+"$PCD_INPUT"} "${EXTRA_ARGS[@]}"
fi

