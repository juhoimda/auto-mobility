#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

if [ -z "$1" ]; then
    echo "=========================================================="
    echo " 사용법: $0 BAG_NAME [--quick] [--out-dir DIR]"
    echo " 예시  : $0 soak_3min --quick"
    echo " 설명  : 동일 rosbag 데이터셋을 기반으로 다양한 Odometry/SLAM 및"
    echo "         Reconstruction 알고리즘 결과를 자동 생성하고 비교 리포트를 출력합니다."
    echo "=========================================================="
    exit 1
fi

python3 "$PROJECT_DIR/src/auto_mobility/slam/compare_algorithms.py" "$@"
