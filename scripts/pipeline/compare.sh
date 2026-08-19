#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

if [ -z "$1" ]; then
    echo "=========================================================="
    echo " 사용법: $0 BAG_NAME [--phase=all|a|b|c] [--quick] [--out-dir DIR]"
    echo " 예시 (전체 벤치마크)   : $0 my_dataset --quick"
    echo " 예시 (Phase A: SLAM만) : $0 my_dataset --phase=a"
    echo " 예시 (Phase B: TSDF만) : $0 my_dataset --phase=b"
    echo " 예시 (Phase C: Mesh만) : $0 my_dataset --phase=c"
    echo " 설명  : 동일 데이터셋을 기반으로 SLAM / Fusion / Surface 축을 독립 비교 평가합니다."
    echo "=========================================================="
    exit 1
fi

python3 "$PROJECT_DIR/src/auto_mobility/slam/compare_algorithms.py" "$@"
