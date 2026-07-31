#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

if [ -z "$1" ]; then
    echo "사용법: $0 DB_NAME [OUTPUT_NAME]"
    echo "예시: $0 room1_result.db room1_cloud.ply"
    exit 1
fi

DB_INPUT="$1"
if [[ "$DB_INPUT" != *.db ]]; then
    DB_FILE="$DB_DIR/$DB_INPUT.db"
else
    DB_FILE="$DB_DIR/$DB_INPUT"
fi

OUTPUT_ARG="${2:-output_cloud.ply}"
if [[ "$OUTPUT_ARG" == /* ]]; then
    OUTPUT_PATH="$OUTPUT_ARG"
else
    OUTPUT_PATH="$POINTCLOUD_DIR/$OUTPUT_ARG"
fi

if [ ! -f "$DB_FILE" ]; then
    echo "오류: DB 파일이 존재하지 않습니다 -> $DB_FILE"
    exit 1
fi

# rtabmap-export --output 옵션은 파일 이름(확장자 미포함/포함) 지정 시 .ply_cloud.ply 가 붙는 현상을 방지하기 위해 output_dir 방식 사용
OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
OUTPUT_BASE="$(basename "$OUTPUT_PATH" .ply)"

echo "Extracting PointCloud from $DB_FILE -> $OUTPUT_PATH"
rtabmap-export --output_dir "$OUTPUT_DIR" --output "$OUTPUT_BASE" "$DB_FILE"

# rtabmap-export 생성 결과 파일명 맞춤 (_cloud.ply 처리)
GENERATED_FILE="$OUTPUT_DIR/${OUTPUT_BASE}_cloud.ply"
if [ -f "$GENERATED_FILE" ] && [ "$GENERATED_FILE" != "$OUTPUT_PATH" ]; then
    mv "$GENERATED_FILE" "$OUTPUT_PATH"
fi
