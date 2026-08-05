#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# 통합 SLAM 파이프라인 벤치마크
#   Stage 1: 카메라 / DDS / QoS 최적 조합 탐색
#   Stage 2: SLAM(RTAB-Map) 파라미터 최적 조합 탐색
#   Stage 3: 전체 파이프라인 종합 진단
#
# 사용법:
#   ./benchmark.sh                      # 전 단계 풀 측정
#   ./benchmark.sh --quick              # 핵심 조합만 빠르게 측정
#   ./benchmark.sh --stage 1            # Stage 1(카메라)만
#   ./benchmark.sh --stage 2            # Stage 2(SLAM)만
#   ./benchmark.sh --stage 3            # Stage 3(진단)만
#
#   # Stage 2/3 단독 실행 시 이전 단계 JSON 경로를 지정해야 합니다:
#   ./benchmark.sh --stage 2 --stage1-json /path/to/slam_bench_s1_XXXXX.json
#   ./benchmark.sh --stage 3 \
#       --stage1-json /path/to/slam_bench_s1_XXXXX.json \
#       --stage2-json /path/to/slam_bench_s2_XXXXX.json
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

echo "🚀 통합 SLAM 파이프라인 벤치마크를 시작합니다..."
python3 "$PROJECT_DIR/src/auto_mobility/processing/benchmark_slam.py" "$@"
