#!/bin/bash
# compare.sh — SLAM / TSDF Fusion / Surface Reconstruction 다축 벤치마크 실행기
#
# 주의: common.sh를 source하지 않는다. common.sh가 ROS setup.bash를 source하여
# LD_LIBRARY_PATH에 /opt/ros/humble/lib 를 주입하면, Open3D가 자체 번들 OpenCV/PCL 대신
# ROS의 불호환 라이브러리를 로드해 SIGSEGV가 발생한다 (실측, 2026-08-19).
# benchmark는 Open3D(TSDF/raycast)만 사용하므로 깨끗한 환경에서 실행한다.
# SLAM이 필요하면 --run-slam 플래그가 별도 프로세스(run_slam.sh, 자체 ROS env 보유)로 실행한다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# HW 과부하 및 WSL2 셧다운/연결 끊김 방지를 위한 CPU 멀티스레드 상한 제한 (기본 4코어)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
_sanitize_path() {
    local var="$1" out=""
    local _oldifs="$IFS"
    IFS=:
    for p in ${!var}; do
        case "$p" in
            *"/opt/ros"*) ;;
            "") ;;
            *) out="${out:+$out:}$p" ;;
        esac
    done
    IFS="$_oldifs"
    if [ -n "$out" ]; then
        export "$var=$out"
    else
        unset "$var"
    fi
}
_sanitize_path LD_LIBRARY_PATH
_sanitize_path PYTHONPATH

if [ -z "$1" ]; then
    echo "=========================================================="
    echo " 사용법: $0 BAG_NAME [옵션]"
    echo ""
    echo " 실행 모드:"
    echo "   --quick     : 빠른 Sanity Check (적은 프레임/후보, 개발/디버그용)"
    echo "   --standard  : 기본 적응형 Coarse-to-Fine 탐색 (권장 Standard)"
    echo "   --full      : 완전 탐색 모드 (Pruning 완화, 연구/벤치마크 검증용)"
    echo ""
    echo " 단계 지정:"
    echo "   --phase=all : 전체 파이프라인 탐색 (기본값)"
    echo "   --phase=a   : Phase A: SLAM 궤적 비교만"
    echo "   --phase=b   : Phase B: TSDF 복셀 해상도 비교만"
    echo "   --phase=c   : Phase C: 표면 복원 알고리즘 비교만"
    echo ""
    echo " 기타 옵션:"
    echo "   --top-k N   : Review 디렉터리에 내보낼 상위 후보 수 (기본: 3)"
    echo "   --run-slam  : 누락된 SLAM 궤적을 run_slam.sh 로 자동 생성 (별도 ROS 프로세스)"
    echo "   --no-cache  : 기존 캐시 무시하고 강제 재생성 (--force 동일)"
    echo "   --no-resume : 이전 실행 상태 복원 비활성화"
    echo "   --output DIR: 커스텀 평가 결과 저장 디렉터리"
    echo "=========================================================="
    exit 1
fi

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR"
exec python3 "$PROJECT_DIR/src/auto_mobility/slam/compare_algorithms.py" "$@"
