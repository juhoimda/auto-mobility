#!/bin/bash
# extract_db_rgbd 빌드 스크립트 (librtabmap_core + OpenCV + sqlite3 링크)
# 사용처: reconstruct_tsdf.py (DB → RGB-D 추출), 직접 실행도 가능
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$PROJECT_DIR/src/auto_mobility/mesh/extract_db_rgbd.cpp"
OUT_DIR="$PROJECT_DIR/build"
OUT="$OUT_DIR/extract_db_rgbd"

# 설치(share/auto_mobility) 실행 폴백: extract_db_rgbd.cpp 위치 검색
if [ ! -f "$SRC" ]; then
    for cand in \
        "$(dirname "$SCRIPT_DIR")/mesh/extract_db_rgbd.cpp" \
        "$(cd "$SCRIPT_DIR/../.." && pwd)/share/auto_mobility/mesh/extract_db_rgbd.cpp" \
        "$(cd "$SCRIPT_DIR/../.." && pwd)/auto_mobility/mesh/extract_db_rgbd.cpp"; do
        if [ -f "$cand" ]; then SRC="$cand"; break; fi
    done
fi

if [ -f "$OUT" ]; then
    echo "$OUT"
    exit 0
fi

mkdir -p "$OUT_DIR"

# rtabmap include/lib 탐색 (apt 설치 기준, 환경별 폴백)
RTABMAP_INC="$(ls -d /opt/ros/${ROS_DISTRO:-humble}/include/rtabmap-* 2>/dev/null | head -1)"
RTABMAP_LIB="$(dirname "$(find /opt/ros -name 'librtabmap_core.so*' 2>/dev/null | head -1)" 2>/dev/null)"
[ -z "$RTABMAP_INC" ] && RTABMAP_INC="/opt/ros/humble/include/rtabmap-0.23"
[ -z "$RTABMAP_LIB" ] && RTABMAP_LIB="/opt/ros/humble/lib/x86_64-linux-gnu"

OPENCV_CFLAGS="$(pkg-config --cflags opencv4 2>/dev/null || echo '-I/usr/include/opencv4')"
OPENCV_LIBS="$(pkg-config --libs opencv4 2>/dev/null || echo '-lopencv_core -lopencv_imgproc -lopencv_imgcodecs')"
EIGEN_INC="$(ls -d /usr/include/eigen3 2>/dev/null | head -1)"

echo "⚙️  extract_db_rgbd 빌드 중..." >&2
g++ -std=c++17 -O2 "$SRC" -o "$OUT" \
    -I"$RTABMAP_INC" -I"$EIGEN_INC" $OPENCV_CFLAGS \
    -L"$RTABMAP_LIB" -lrtabmap_core \
    $OPENCV_LIBS -lsqlite3 -lpthread \
    -Wl,-rpath,"$RTABMAP_LIB"
echo "$OUT"
