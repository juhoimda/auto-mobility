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

if [ ! -f "$PCD_FILE" ]; then
    echo "❌ 오류: 파일을 찾을 수 없습니다 -> $PCD_FILE"
    exit 1
fi

FILE_SIZE=$(du -h "$PCD_FILE" | cut -f1)
FILE_NAME="$(basename "$PCD_FILE")"
echo "=========================================================="
echo " 📂 Point Cloud 파일 : $FILE_NAME (용량: $FILE_SIZE)"
echo " 📍 리눅스 경로 : $PCD_FILE"
echo "=========================================================="

# Windows 공유 디렉터리로 복사 (백그라운드 지원)
SHARED_DIR="/mnt/c/ubuntu_shared"
if [ -d "$SHARED_DIR" ]; then
    mkdir -p "$SHARED_DIR/pointclouds" 2>/dev/null || true
    echo "📋 Windows 공유 폴더로 복사 중 ($SHARED_DIR/pointclouds/$FILE_NAME)..."
    cp -u "$PCD_FILE" "$SHARED_DIR/pointclouds/$FILE_NAME" 2>/dev/null || cp "$PCD_FILE" "$SHARED_DIR/pointclouds/$FILE_NAME"
    echo "C:\\ubuntu_shared\\pointclouds\\$FILE_NAME" > "$SHARED_DIR/view_request.txt" 2>/dev/null || true
fi

# 하드웨어 GPU 가속 설정
export LIBGL_ALWAYS_SOFTWARE=0
unset LIBGL_ALWAYS_SOFTWARE
if [ -d /mnt/wslg ]; then
    export WAYLAND_DISPLAY=
    unset WAYLAND_DISPLAY
    export DISPLAY="${DISPLAY:-:0}"
fi

echo "=========================================================="
echo " 🚀 GPU 하드웨어 가속 Point Cloud Viewer를 실행합니다..."
echo "=========================================================="

exec python3 "$PROJECT_DIR/src/auto_mobility/mesh/view_pointcloud.py" "$PCD_FILE" "${EXTRA_ARGS[@]}"

