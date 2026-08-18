#!/bin/bash
# 3D Point Cloud Viewer Launcher Script (.ply / .pcd)

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

# WSLg(Wayland)에서 Open3D의 GLFW가 wayland-0 소켓을 잡아
# "Failed to initialize GLEW"로 창이 안 뜨는 문제 → X11 백엔드로 강제 폴백
if [ -d /mnt/wslg ]; then
    export WAYLAND_DISPLAY=
    export DISPLAY="${DISPLAY:-:0}"
fi

if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "=========================================================="
    echo " 사용법: $0 POINTCLOUD_FILE [--point-size=N] [--show-normals] [--voxel=N]"
    echo " 예시  : $0 my_dataset_tsdf_cloud.ply"
    echo " 예시  : $0 $POINTCLOUD_DIR/my_dataset_raw_cloud.ply --point-size=4"
    echo " 예시  : $0 my_dataset_tsdf_cloud.ply --voxel=0.02"
    echo "=========================================================="
    exit 0
fi

PCD_INPUT="$1"
shift

# 경로 판별 (절대경로/파일명/pointclouds 디렉터리 탐색)
if [[ "$PCD_INPUT" == /* ]]; then
    PCD_FILE="$PCD_INPUT"
elif [ -f "$POINTCLOUD_DIR/$PCD_INPUT" ]; then
    PCD_FILE="$POINTCLOUD_DIR/$PCD_INPUT"
elif [ -f "$PROJECT_DIR/$PCD_INPUT" ]; then
    PCD_FILE="$PROJECT_DIR/$PCD_INPUT"
else
    PCD_FILE="$PCD_INPUT"
fi

if [ ! -f "$PCD_FILE" ]; then
    echo "❌ 오류: Point Cloud 파일을 찾을 수 없습니다 -> $PCD_FILE"
    echo "💡 팁: $POINTCLOUD_DIR 디렉터리에 .ply/.pcd 파일이 존재하는지 확인하세요."
    exit 1
fi

python3 "$PROJECT_DIR/src/auto_mobility/mesh/view_pointcloud.py" "$PCD_FILE" "$@"
