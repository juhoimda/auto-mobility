#!/bin/bash
# 3D Surface Mesh Viewer Launcher Script (.obj / .stl)

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
    echo " 사용법: $0 MESH_FILE_NAME [--wireframe] [--no-backface]"
    echo " 예시  : $0 my_dataset_tsdf.obj"
    echo " 예시  : $0 $MESH_DIR/my_dataset_mesh.obj --wireframe"
    echo "=========================================================="
    exit 0
fi

MESH_INPUT="$1"
shift

# 경로 판별 (절대경로/파일명/meshes 디렉터리 탐색)
if [[ "$MESH_INPUT" == /* ]]; then
    MESH_FILE="$MESH_INPUT"
elif [ -f "$MESH_DIR/$MESH_INPUT" ]; then
    MESH_FILE="$MESH_DIR/$MESH_INPUT"
elif [ -f "$PROJECT_DIR/$MESH_INPUT" ]; then
    MESH_FILE="$PROJECT_DIR/$MESH_INPUT"
else
    MESH_FILE="$MESH_INPUT"
fi

if [ ! -f "$MESH_FILE" ]; then
    echo "❌ 오류: Mesh 파일을 찾을 수 없습니다 -> $MESH_FILE"
    echo "💡 팁: $MESH_DIR 디렉터리에 Mesh 파일이 존재하는지 확인하세요."
    exit 1
fi

python3 "$PROJECT_DIR/src/auto_mobility/mesh/view_mesh.py" "$MESH_FILE" "$@"
