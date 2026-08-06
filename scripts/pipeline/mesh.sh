#!/bin/bash

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

POSITIONAL_ARGS=()
VIEW_FLAG=""
FORCE_FLAG=""
METHOD="open3d"

# 파라미터 파싱
for arg in "$@"; do
    case $arg in
        --view)
            VIEW_FLAG="--view"
            ;;
        --force)
            FORCE_FLAG="--force"
            ;;
        --method=*)
            METHOD="${arg#*=}"
            ;;
        rtabmap)
            METHOD="rtabmap"
            ;;
        open3d)
            METHOD="open3d"
            ;;
        -*)
            echo "⚠️ 알 수 없는 옵션: $arg"
            ;;
        *)
            POSITIONAL_ARGS+=("$arg")
            ;;
    esac
done

if [ ${#POSITIONAL_ARGS[@]} -lt 1 ]; then
    echo "=========================================================="
    echo " 사용법: $0 DB_NAME [OUTPUT_MESH_NAME] [--view] [--force] [--method open3d|rtabmap]"
    echo " 예시  : $0 my_room_db.db my_room_mesh.obj --view"
    echo "=========================================================="
    exit 1
fi

DB_INPUT="${POSITIONAL_ARGS[0]}"
OUTPUT_MESH_ARG="${POSITIONAL_ARGS[1]}"

# .db 확장자 처리
if [[ "$DB_INPUT" != *.db ]]; then
    DB_FILE="$DB_DIR/$DB_INPUT.db"
    BASE_NAME="$DB_INPUT"
else
    DB_FILE="$DB_DIR/$DB_INPUT"
    BASE_NAME="${DB_INPUT%.db}"
fi

OUTPUT_MESH_NAME="${OUTPUT_MESH_ARG:-${BASE_NAME}_mesh.obj}"

PLY_PATH="$POINTCLOUD_DIR/${BASE_NAME}_cloud.ply"
MESH_PATH="$MESH_DIR/$OUTPUT_MESH_NAME"

if [ ! -f "$DB_FILE" ]; then
    echo "❌ 오류: DB 파일이 존재하지 않습니다 -> $DB_FILE"
    echo "💡 팁: $DB_DIR 디렉터리에 .db 파일이 있는지 확인하세요."
    exit 1
fi

echo "=========================================================="
echo " 🚀 Digital Twin 3D Mesh 파이프라인 시작"
echo " 📁 입력 DB   : $DB_FILE"
echo " 🛠️ 방식      : $METHOD"
echo " 💾 출력 Mesh : $MESH_PATH"
echo "=========================================================="

if [ "$METHOD" == "rtabmap" ]; then
    echo "1️⃣ DB 검증 실행..."
    python3 "$PROJECT_DIR/src/auto_mobility/utils/validate.py" --db "$DB_FILE" || true
    echo "2️⃣ RTAB-Map 자체 텍스처 Mesh 추출 실행..."
    rtabmap-export --mesh --texture --output "$MESH_PATH" "$DB_FILE"
    echo "✅ RTAB-Map Mesh 추출 완료: $MESH_PATH"
else
    echo "1️⃣ DB에서 Point Cloud (.ply) 추출 중..."
    "$PIPELINE_DIR/../utils/export_ply.sh" "$(basename "$DB_FILE")" "${BASE_NAME}_cloud.ply"

    echo ""
    echo "🔍 [자동 품질 검증] Point Cloud & DB 헬스 체크 실행 중..."
    if ! python3 "$PROJECT_DIR/src/auto_mobility/utils/validate.py" --db "$DB_FILE" --ply "$PLY_PATH"; then
        echo ""
        if [ "$FORCE_FLAG" == "--force" ]; then
            echo "⚠️  [주의] 데이터 품질 경고가 발생했으나 --force 옵션으로 계속 진행합니다."
        else
            echo "❌ [자동 중단] 품질 검증 기준 미달로 Mesh 생성을 중단합니다."
            echo "💡 팁: 경고를 무시하고 강제로 생성하려면 '--force' 옵션을 붙여주세요."
            echo "      예시: $0 $1 --force"
            exit 1
        fi
    fi

    echo ""
    echo "2️⃣ Open3D 기반 3D Mesh 복원 및 정제 중..."
    python3 "$PROJECT_DIR/src/auto_mobility/mesh/mesh_open3d.py" "$PLY_PATH" "$MESH_PATH" $VIEW_FLAG
fi