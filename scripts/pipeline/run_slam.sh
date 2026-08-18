#!/bin/bash
# run_slam.sh — Rosbag 데이터셋에 대해 원하는 SLAM (rtab / orb)을 1-명령어로 실행하고 궤적/DB를 생성

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

SLAM_TYPE="rtab"
RATE="1.0"
BAG_NAME=""

for arg in "$@"; do
    case $arg in
        --slam=*)
            SLAM_TYPE="${arg#*=}"
            ;;
        --rate=*)
            RATE="${arg#*=}"
            ;;
        rtab|rtabmap)
            SLAM_TYPE="rtab"
            ;;
        orb|orbslam|orbslam3)
            SLAM_TYPE="orb"
            ;;
        -*)
            echo "⚠️ 알 수 없는 옵션: $arg"
            ;;
        *)
            if [ -z "$BAG_NAME" ]; then
                BAG_NAME="$arg"
            fi
            ;;
    esac
done

if [ -z "$BAG_NAME" ]; then
    echo "=========================================================="
    echo " 사용법: $0 BAG_NAME [--slam=rtab|orb] [--rate=1.0]"
    echo " 예시  : $0 my_dataset --slam=rtab"
    echo " 예시  : $0 my_dataset --slam=orb"
    echo "=========================================================="
    exit 1
fi

BAG_PATH="$BAG_DIR/$BAG_NAME"
if [ ! -d "$BAG_PATH" ] && [ ! -f "$BAG_PATH" ]; then
    if [ -d "$BAG_NAME" ] || [ -f "$BAG_NAME" ]; then
        BAG_PATH="$BAG_NAME"
        BAG_NAME="$(basename "$BAG_NAME")"
    else
        echo "❌ 오류: Rosbag을 찾을 수 없습니다 -> $BAG_PATH"
        exit 1
    fi
fi

echo "=========================================================="
echo " 🚀 SLAM 오프라인 실행"
echo " 📦 Rosbag : $BAG_PATH"
echo " 🛠️ 엔진   : $SLAM_TYPE"
echo "=========================================================="

if [ "$SLAM_TYPE" == "orb" ] || [ "$SLAM_TYPE" == "orbslam" ] || [ "$SLAM_TYPE" == "orbslam3" ]; then
    OUT_TRAJ="$TRAJECTORY_DIR/orbslam3_${BAG_NAME}_trajectory.txt"
    python3 "$PROJECT_DIR/src/auto_mobility/slam/run_orbslam3_bag.py" "$BAG_PATH" --out "$OUT_TRAJ"
    echo "✅ ORB-SLAM3 완료 -> $OUT_TRAJ"
else
    # RTAB-Map
    DB_PATH="$DB_DIR/${BAG_NAME}.db"
    OUT_TRAJ="$TRAJECTORY_DIR/rtab_${BAG_NAME}_trajectory.txt"
    
    echo "▶️ RTAB-Map 백그라운드 시작..."
    setsid ros2 launch auto_mobility rtab_bag.launch.py \
        database_path:="$DB_PATH" \
        use_compressed:=true >/dev/null 2>&1 &
    SLAM_PID=$!
    
    sleep 3
    echo "▶️ Rosbag 재생 시작 (배속: ${RATE}x)..."
    ros2 bag play "$BAG_PATH" --clock --rate "$RATE"
    
    echo "▶️ RTAB-Map 정리 및 DB 저장..."
    sleep 2
    kill -SIGINT -- -$SLAM_PID 2>/dev/null || pkill -SIGINT -f "rtabmap" 2>/dev/null || true
    sleep 3
    kill -SIGKILL -- -$SLAM_PID 2>/dev/null || pkill -SIGKILL -f "rtabmap" 2>/dev/null || true
    
    # TUM Trajectory 내보내기
    python3 -c "from auto_mobility.trajectory.io import export_from_db; export_from_db('$DB_PATH', '$OUT_TRAJ')" 2>/dev/null || true
    
    echo "✅ RTAB-Map 완료 -> DB: $DB_PATH | Trajectory: $OUT_TRAJ"
fi
