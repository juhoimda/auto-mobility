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

# Canonical RGB-D frames are read from rosbag2 before the Open3D benchmark.
# rosbag2_py is installed through the ROS environment, whereas Open3D must not
# inherit ROS's LD_LIBRARY_PATH (see the note below).  Keep these two runtimes
# isolated: extraction runs in a short-lived ROS subprocess and the benchmark
# itself continues in the clean environment prepared by this script.
_extract_frames_with_ros() {
    local bag_input="$1"
    local ros_distro="${ROS_DISTRO:-humble}"
    local ros_setup="/opt/ros/${ros_distro}/setup.bash"

    if [ ! -f "$ros_setup" ]; then
        echo "ERROR: ROS 2 setup file not found: $ros_setup" >&2
        echo "       Canonical frame extraction requires rosbag2_py." >&2
        return 1
    fi

    echo "⚙️ Canonical RGB-D 프레임 준비 중 (격리된 ROS 환경)..."
    bash -c '
        source "$1"
        export PYTHONPATH="$2/src:$2${PYTHONPATH:+:$PYTHONPATH}"
        exec python3 "$2/src/auto_mobility/dataset/extract_frames.py" "$3"
    ' _ "$ros_setup" "$PROJECT_DIR" "$bag_input"
}

# HW 과부하 및 WSL2 셧다운/발열 스로틀링 방지를 위한 CPU 멀티스레드 상한 제한 (ResourcePolicy 단일 소스 준수: 6 OMP, 1 BLAS)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-6}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export OPENCV_NUM_THREADS="${OPENCV_NUM_THREADS:-1}"
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

# Do this before stripping ROS paths below.  The extractor is cache-aware, so
# invoking it on an already prepared dataset only validates and reuses frames.
_extract_frames_with_ros "$1" || exit $?

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR"
exec python3 -m auto_mobility.reconstruction.cli "$@"
