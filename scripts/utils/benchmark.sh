#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

echo "🚀 [1단계 전용] 센서 취득 및 DDS 하드웨어 자동 벤치마크를 시작합니다..."
python3 "$PROJECT_DIR/src/auto_mobility/processing/benchmark_hw.py" "$@"
