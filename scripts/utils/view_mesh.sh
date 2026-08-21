#!/bin/bash
# 3D Mesh / Point Cloud Viewer Launcher Script (.obj / .ply / .stl / .pcd)
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
            echo " 사용법: $0 [3D_FILE_NAME] [--open3d] [--wireframe] [--no-backface]"
            echo "  - 기본 동작 : Windows MeshLab을 연동하여 GPU 가속으로 빠르게 실행"
            echo "  - --open3d  : WSL 내부 Open3D 파이썬 뷰어로 강제 실행"
            echo " 예시 (Windows MeshLab) : $0 my_dataset_rtab_tsdf.obj"
            echo " 예시 (Point Cloud PLY) : $0 my_dataset_rtab_tsdf_cloud.ply"
            echo " 예시 (WSL Open3D 뷰어) : $0 my_dataset_rtab_tsdf.obj --open3d"
            echo "=========================================================="
            exit 0
            ;;
        *)
            if [ -z "$MESH_INPUT" ]; then
                MESH_INPUT="$arg"
            else
                EXTRA_ARGS+=("$arg")
            fi
            ;;
    esac
done

# 파일 지정이 없는 경우 가장 최근 생성된 obj 또는 ply 파일 자동 탐색
if [ -z "$MESH_INPUT" ]; then
    LATEST_FILE=$(ls -t "$MESH_DIR"/*.obj "$POINTCLOUD_DIR"/*.ply 2>/dev/null | head -n 1)
    if [ -n "$LATEST_FILE" ]; then
        MESH_INPUT="$LATEST_FILE"
        echo "💡 파일명이 지정되지 않아 가장 최근 생성된 파일을 엽니다: $(basename "$MESH_INPUT")"
    else
        echo "❌ 오류: 표시할 3D Mesh 또는 Point Cloud 파일이 없습니다."
        echo "사용법: $0 <파일명(.obj, .ply, .stl)>"
        exit 1
    fi
fi

# 경로 판별 (절대경로 / meshes / pointclouds / 프로젝트 루트)
if [[ "$MESH_INPUT" == /* ]] && [ -f "$MESH_INPUT" ]; then
    MESH_FILE="$MESH_INPUT"
elif [ -f "$MESH_DIR/$MESH_INPUT" ]; then
    MESH_FILE="$MESH_DIR/$MESH_INPUT"
elif [ -f "$POINTCLOUD_DIR/$MESH_INPUT" ]; then
    MESH_FILE="$POINTCLOUD_DIR/$MESH_INPUT"
elif [ -f "$PROJECT_DIR/$MESH_INPUT" ]; then
    MESH_FILE="$PROJECT_DIR/$MESH_INPUT"
elif [ -f "$PWD/$MESH_INPUT" ]; then
    MESH_FILE="$PWD/$MESH_INPUT"
else
    MESH_FILE="$MESH_INPUT"
fi

if [ ! -f "$MESH_FILE" ]; then
    echo "❌ 오류: 파일을 찾을 수 없습니다 -> $MESH_FILE"
    echo "💡 팁: $MESH_DIR 또는 $POINTCLOUD_DIR 디렉터리를 확인하세요."
    exit 1
fi

FILE_SIZE=$(du -h "$MESH_FILE" | cut -f1)
FILE_NAME="$(basename "$MESH_FILE")"
echo "=========================================================="
echo " 📂 3D 파일 : $FILE_NAME (용량: $FILE_SIZE)"
echo " 📍 리눅스 경로 : $MESH_FILE"
echo "=========================================================="

# 1. Windows 공유 디렉터리로 자동 복사 (C:\ubuntu_shared\meshes\)
SHARED_DIR="/mnt/c/ubuntu_shared"
WIN_SHARED_PATH="C:\\ubuntu_shared\\meshes\\$FILE_NAME"
if [ -d "$SHARED_DIR" ]; then
    mkdir -p "$SHARED_DIR/meshes" "$SHARED_DIR/pointclouds" 2>/dev/null || true
    DEST_DIR="$SHARED_DIR/meshes"
    [[ "$FILE_NAME" == *.ply || "$FILE_NAME" == *.pcd ]] && DEST_DIR="$SHARED_DIR/pointclouds"
    
    echo "📋 Windows 공유 폴더로 복사 중 ($DEST_DIR/$FILE_NAME)..."
    cp -u "$MESH_FILE" "$DEST_DIR/$FILE_NAME" 2>/dev/null || cp "$MESH_FILE" "$DEST_DIR/$FILE_NAME"
    
    if [[ "$FILE_NAME" == *.ply || "$FILE_NAME" == *.pcd ]]; then
        WIN_SHARED_PATH="C:\\ubuntu_shared\\pointclouds\\$FILE_NAME"
    else
        WIN_SHARED_PATH="C:\\ubuntu_shared\\meshes\\$FILE_NAME"
    fi

    # Windows Auto-Launcher에 실행 요청 전송
    echo "$WIN_SHARED_PATH" > "$SHARED_DIR/view_request.txt"
fi


# 2. WSL 내부 Open3D 강제 실행 옵션이 활성화된 경우
if [ "$USE_OPEN3D_LOCAL" = true ]; then
    if [ -d /mnt/wslg ]; then
        export WAYLAND_DISPLAY=
        export DISPLAY="${DISPLAY:-:0}"
    fi
    echo "🎨 WSL 내부 Open3D 뷰어를 실행합니다..."
    exec python3 "$PROJECT_DIR/src/auto_mobility/mesh/view_mesh.py" "$MESH_FILE" "${EXTRA_ARGS[@]}"
fi

# 3. Windows 환경 (WSL) 감지 및 MeshLab 실행 경로 탐색
MESHLAB_PATH=""
CANDIDATE_PATHS=(
    "/mnt/c/Program Files/VCG/MeshLab/meshlab.exe"
    "/mnt/c/Program Files (x86)/VCG/MeshLab/meshlab.exe"
    "/mnt/c/Program Files/MeshLab/meshlab.exe"
)

for p in "${CANDIDATE_PATHS[@]}"; do
    if [ -f "$p" ]; then
        MESHLAB_PATH="$p"
        break
    fi
done

if [ -z "$MESHLAB_PATH" ] && command -v meshlab.exe &>/dev/null; then
    MESHLAB_PATH="$(command -v meshlab.exe)"
fi

# ── 하드웨어 GPU 가속 렌더링 설정 (D3D12 Intel Arc Pro 140T 48GB GPU 활용) ──
# RViz2용 소프트웨어 렌더링 강제 설정(LIBGL_ALWAYS_SOFTWARE=1)을 해제하고 GPU 하드웨어 가속 사용
export LIBGL_ALWAYS_SOFTWARE=0
unset LIBGL_ALWAYS_SOFTWARE

# WSLg(Wayland)에서 Open3D GLFW가 Wayland 소켓을 잡아 GLEW 초기화 실패하는 문제 방지
# X11 백엔드로 강제 설정
if [ -d /mnt/wslg ]; then
    export WAYLAND_DISPLAY=
    unset WAYLAND_DISPLAY
    export DISPLAY="${DISPLAY:-:0}"
    export XDG_RUNTIME_DIR="/tmp/runtime-$USER"
fi

echo "=========================================================="
echo " 🚀 GPU 하드웨어 가속 3D Viewer를 실행합니다... (D3D12)"
echo "=========================================================="

exec python3 "$PROJECT_DIR/src/auto_mobility/mesh/view_mesh.py" "$MESH_FILE" "${EXTRA_ARGS[@]}"






