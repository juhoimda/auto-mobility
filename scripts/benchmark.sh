#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/common.sh"

echo "🚀 하드웨어 통신 최적화 (DDS/QoS/해상도/FPS/압축) 벤치마크를 시작합니다..."
python3 "$SCRIPT_DIR/benchmark_hardware.py" "$@"
