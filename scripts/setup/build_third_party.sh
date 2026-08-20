#!/usr/bin/env bash
set -e

# ==============================================================================
# build_third_party.sh
# ------------------------------------------------------------------------------
# 팀원 및 배포 환경에서 모든 third_party 라이브러리(Pangolin, FBoW, stella_vslam,
# ORB_SLAM3)를 자동으로 컴파일/설치하고 ROS 2 패키지를 빌드하는 통합 스크립트.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
THIRD_PARTY_DIR="$PROJECT_DIR/third_party"
INSTALL_PREFIX="$THIRD_PARTY_DIR/installed"
NUM_PROCS=$(nproc)

echo "=========================================================="
echo " 🚀 Auto-Mobility Third-Party Dependencies Builder"
echo " 📂 Project root: $PROJECT_DIR"
echo " ⚙️ Install prefix: $INSTALL_PREFIX"
echo " ⚡ Parallel jobs: $NUM_PROCS"
echo "=========================================================="

mkdir -p "$INSTALL_PREFIX"

# 1. Build & Install Pangolin
if [ -d "$THIRD_PARTY_DIR/Pangolin" ]; then
    echo "▶️ [1/4] Building Pangolin..."
    mkdir -p "$THIRD_PARTY_DIR/Pangolin/build"
    cd "$THIRD_PARTY_DIR/Pangolin/build"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_PANGOLIN_GUI=ON
    make -j"$NUM_PROCS"
    make install
fi

# 2. Build & Install FBoW
if [ -d "$THIRD_PARTY_DIR/FBoW" ]; then
    echo "▶️ [2/4] Building FBoW..."
    mkdir -p "$THIRD_PARTY_DIR/FBoW/build"
    cd "$THIRD_PARTY_DIR/FBoW/build"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    make -j"$NUM_PROCS"
    make install
fi

# 3. Build & Install stella_vslam
if [ -d "$THIRD_PARTY_DIR/stella_vslam" ]; then
    echo "▶️ [3/4] Building stella_vslam..."
    mkdir -p "$THIRD_PARTY_DIR/stella_vslam/build"
    cd "$THIRD_PARTY_DIR/stella_vslam/build"
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
        -DUSE_PANGOLIN_VIEWER=ON \
        -DBUILD_TESTS=OFF
    make -j"$NUM_PROCS"
    make install
fi

# 4. Build ORB_SLAM3 (DBoW2, g2o, ORB_SLAM3)
if [ -d "$THIRD_PARTY_DIR/ORB_SLAM3" ]; then
    echo "▶️ [4/4] Building ORB_SLAM3..."
    cd "$THIRD_PARTY_DIR/ORB_SLAM3"

    # Vocabulary 압축 해제
    if [ ! -f "Vocabulary/ORBvoc.txt" ] && [ -f "Vocabulary/ORBvoc.txt.tar.gz" ]; then
        echo "   📦 Uncompressing ORBvoc.txt.tar.gz..."
        cd Vocabulary
        tar -xzf ORBvoc.txt.tar.gz
        cd ..
    fi

    # DBoW2
    echo "   ⚙️ Building DBoW2..."
    mkdir -p Thirdparty/DBoW2/build
    cd Thirdparty/DBoW2/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"$NUM_PROCS"

    # g2o
    echo "   ⚙️ Building g2o..."
    cd "$THIRD_PARTY_DIR/ORB_SLAM3"
    mkdir -p Thirdparty/g2o/build
    cd Thirdparty/g2o/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j"$NUM_PROCS"

    # ORB_SLAM3 Core
    echo "   ⚙️ Building libORB_SLAM3..."
    cd "$THIRD_PARTY_DIR/ORB_SLAM3"
    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DPangolin_DIR="$INSTALL_PREFIX/lib/cmake/Pangolin"
    make -j"$NUM_PROCS"
fi

# 5. Build ROS 2 workspace
echo "▶️ [5/5] Building ROS 2 Workspace (auto_mobility)..."
cd "$PROJECT_DIR"
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi
colcon build --packages-select auto_mobility

echo "=========================================================="
echo "✅ All third-party libraries and auto_mobility built successfully!"
echo "💡 To use, run: source install/setup.bash"
echo "=========================================================="
