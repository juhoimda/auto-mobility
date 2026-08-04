#!/bin/bash
# Isaac Sim Digital Twin Mesh Verification Launcher Script

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

if [ -z "$1" ]; then
    echo "=========================================================="
    echo " 사용법: $0 MESH_FILE_NAME [--headless] [--no-physics] [--scale SCALE]"
    echo " 예시  : $0 my_room_mesh.obj"
    echo " 예시  : $0 $MESH_DIR/my_room_mesh.obj --scale 1.0"
    echo "=========================================================="
    exit 1
fi

MESH_INPUT="$1"
shift

# 절대경로/상대경로/파일명 분기 처리
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
    echo "💡 팁: $MESH_DIR 디렉터리에 Mesh 파일이 있는지 확인하세요."
    exit 1
fi

LOADER_SCRIPT="$PROJECT_DIR/src/auto_mobility/processing/load_isaac_mesh.py"

echo "=========================================================="
echo " 🚀 Digital Twin Isaac Sim Launcher"
echo " 📁 Input Mesh : $MESH_FILE"
echo " 📜 Python App : $LOADER_SCRIPT"
echo "=========================================================="

# Isaac Sim 파이썬 실행기 탐색 (Isaac Sim 번들 python.sh 우선)
ISAAC_PYTHON=""
ISAAC_SEARCH_PATHS=(
    "$HOME/.local/share/ov/pkg/isaac-sim-*"
    "$HOME/.local/share/ov/pkg/isaac_sim-*"
    "/opt/nvidia/omniverse/isaac-sim-*"
)

for search_path in "${ISAAC_SEARCH_PATHS[@]}"; do
    expanded_path=$(ls -d $search_path 2>/dev/null | tail -n 1)
    if [ -n "$expanded_path" ] && [ -f "$expanded_path/python.sh" ]; then
        ISAAC_PYTHON="$expanded_path/python.sh"
        break
    fi
done

# VMware 가상머신 환경 감지
IS_VMWARE=false
if grep -qi "vmware" /proc/version 2>/dev/null || systemd-detect-virt 2>/dev/null | grep -qi "vmware"; then
    IS_VMWARE=true
fi

if [ -n "$ISAAC_PYTHON" ]; then
    echo "🔍 Isaac Sim Bundled Python Found: $ISAAC_PYTHON"
    "$ISAAC_PYTHON" "$LOADER_SCRIPT" "$MESH_FILE" "$@"
else
    if [ "$IS_VMWARE" = true ]; then
        USD_FILE="${MESH_FILE%.obj}.usd"
        echo "=========================================================="
        echo " 💡 [VMware 가상머신 환경 안내]"
        echo " 📁 Mesh (.obj) 및 Isaac Sim Ready Scene (.usd) 파일 생성 완료!"
        echo " 🚀 USD 파일: $USD_FILE"
        echo ""
        echo " 👉 Windows의 Isaac Sim GUI 화면으로 '$USD_FILE' 파일(또는 공유폴더 경로)을"
        echo "    드래그 앤 드롭(Drag & Drop)하면 즉시 디지털 트윈 시뮬레이션이 실행됩니다."
        echo "=========================================================="
        exit 0
    fi
    echo "⚠️ Isaac Sim 패키지 경로를 자동으로 찾지 못했습니다. 시스템 기본 python3로 시도합니다."
    python3 "$LOADER_SCRIPT" "$MESH_FILE" "$@"
fi
