#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

if [ -z "$1" ]; then
    echo "사용법: $0 DB_NAME [OUTPUT_NAME]"
    echo "예시: $0 room1_result.db room1_cloud.ply"
    exit 1
fi

DB_FILE="$DB_DIR/$1"
OUTPUT_NAME=${2:-output_cloud.ply}
OUTPUT_PATH="$POINTCLOUD_DIR/$OUTPUT_NAME"

if [ ! -f "$DB_FILE" ]; then
    echo "오류: DB 파일이 존재하지 않습니다 -> $DB_FILE"
    exit 1
fi

echo "Extracting PointCloud from $DB_FILE -> $OUTPUT_PATH"
rtabmap-export --output "$OUTPUT_PATH" "$DB_FILE"
