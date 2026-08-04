#!/bin/bash
# 3D Mesh Viewer Launcher Script

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

if [ -z "$1" ]; then
    echo "=========================================================="
    echo " 사용법: $0 MESH_FILE_NAME [--wireframe] [--no-backface]"
    echo " 예시  : $0 my_room_mesh.obj"
    echo " 예시  : $0 $MESH_DIR/my_room_mesh.obj --wireframe"
    echo "=========================================================="
    exit 1
fi

MESH_INPUT="$1"
shift

# 경로 판별 (절대경로/파일명/상대경로)
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

python3 "$PROJECT_DIR/src/auto_mobility/processing/view_mesh.py" "$MESH_FILE" "$@"
