#!/bin/bash
# run_slam.sh — Rosbag 데이터셋에 대해 원하는 SLAM (rtab / orb)을 1-명령어로 실행하고 궤적/DB를 생성

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

# Keep ROS launch logs inside the project output tree. This also makes offline
# runs work in sandboxed/CI environments where ~/.ros is read-only.
export ROS_LOG_DIR="${ROS_LOG_DIR:-$LOG_DIR/ros}"
mkdir -p "$ROS_LOG_DIR"

SLAM_TYPE="rtab"
RATE="1.0"
BAG_NAME=""
DENSE_MAPPING="false"

for arg in "$@"; do
    case $arg in
        --slam=*)
            SLAM_TYPE="${arg#*=}"
            ;;
        --rate=*)
            RATE="${arg#*=}"
            ;;
        --dense)
            DENSE_MAPPING="true"
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
    echo " 사용법: $0 BAG_NAME [--slam=rtab|orb] [--rate=1.0] [--dense]"
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

if [ "$SLAM_TYPE" == "orb_rgbdi" ] || [ "$SLAM_TYPE" == "rgbdi" ]; then
    KEY="orb_rgbdi_rate${RATE}"
    OUT_TRAJ="$TRAJECTORY_DIR/orb_rgbdi_rate${RATE}_${BAG_NAME}_trajectory.txt"
    python3 "$PROJECT_DIR/src/auto_mobility/slam/run_orbslam3_bag.py" "$BAG_PATH" --out "$OUT_TRAJ" --mode rgbdi --rate "$RATE"
    python3 -c "from auto_mobility.benchmark.artifacts import save_trajectory_metadata; from auto_mobility.benchmark.candidate import SlamProfileSpec; save_trajectory_metadata('$OUT_TRAJ', SlamProfileSpec(candidate_key='$KEY', backend='orb_rgbdi', profile='normal', replay_rate=float('$RATE')))" 2>/dev/null || true
    echo "✅ ORB-SLAM3 RGB-D-I 완료 -> $OUT_TRAJ"
elif [ "$SLAM_TYPE" == "orb" ] || [ "$SLAM_TYPE" == "orbslam" ] || [ "$SLAM_TYPE" == "orbslam3" ] || [ "$SLAM_TYPE" == "orb_rgbd" ]; then
    KEY="orb_rgbd_rate${RATE}"
    OUT_TRAJ="$TRAJECTORY_DIR/orb_rgbd_rate${RATE}_${BAG_NAME}_trajectory.txt"
    python3 "$PROJECT_DIR/src/auto_mobility/slam/run_orbslam3_bag.py" "$BAG_PATH" --out "$OUT_TRAJ" --mode rgbd --rate "$RATE"
    python3 -c "from auto_mobility.benchmark.artifacts import save_trajectory_metadata; from auto_mobility.benchmark.candidate import SlamProfileSpec; save_trajectory_metadata('$OUT_TRAJ', SlamProfileSpec(candidate_key='$KEY', backend='orb_rgbd', profile='normal', replay_rate=float('$RATE')))" 2>/dev/null || true
    echo "✅ ORB-SLAM3 RGB-D 완료 -> $OUT_TRAJ"
elif [ "$SLAM_TYPE" == "stella" ] || [ "$SLAM_TYPE" == "stella_rgbd" ]; then
    KEY="stella_rgbd_rate${RATE}"
    OUT_TRAJ="$TRAJECTORY_DIR/stella_rgbd_rate${RATE}_${BAG_NAME}_trajectory.txt"
    python3 "$PROJECT_DIR/src/auto_mobility/slam/run_stella_bag.py" "$BAG_PATH" --out "$OUT_TRAJ" --rate "$RATE"
    python3 -c "from auto_mobility.benchmark.artifacts import save_trajectory_metadata; from auto_mobility.benchmark.candidate import SlamProfileSpec; save_trajectory_metadata('$OUT_TRAJ', SlamProfileSpec(candidate_key='$KEY', backend='stella_rgbd', profile='normal', replay_rate=float('$RATE')))" 2>/dev/null || true
    echo "✅ stella_vslam RGB-D 완료 -> $OUT_TRAJ"
else
    # RTAB-Map
    if [ "$DENSE_MAPPING" = "true" ]; then
        PROFILE="dense"
        KEY="rtab_dense_rate${RATE}"
        DB_PATH="$DB_DIR/${BAG_NAME}_rtab_dense_rate${RATE}.db"
        OUT_TRAJ="$TRAJECTORY_DIR/rtab_dense_rate${RATE}_${BAG_NAME}_trajectory.txt"
        ALT_TRAJ="$TRAJECTORY_DIR/rtab_dense_${BAG_NAME}_trajectory.txt"
    else
        PROFILE="normal"
        KEY="rtab_normal_rate${RATE}"
        DB_PATH="$DB_DIR/${BAG_NAME}_rtab_normal_rate${RATE}.db"
        OUT_TRAJ="$TRAJECTORY_DIR/rtab_normal_rate${RATE}_${BAG_NAME}_trajectory.txt"
        ALT_TRAJ="$TRAJECTORY_DIR/rtab_${BAG_NAME}_trajectory.txt"
    fi
    
    echo "▶️ RTAB-Map 백그라운드 시작..."
    SLAM_LOG="$LOG_DIR/rtab_${BAG_NAME}_${PROFILE}_rate${RATE}.log"
    setsid ros2 launch auto_mobility rtab_bag.launch.py \
        database_path:="$DB_PATH" \
        dense_mapping:="$DENSE_MAPPING" \
        use_compressed:=true >"$SLAM_LOG" 2>&1 &
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
    python3 -c "from auto_mobility.trajectory.export_trajectory import export_from_db; export_from_db('$DB_PATH', '$OUT_TRAJ')"
    if [ ! -s "$OUT_TRAJ" ] || [ "$(grep -vc '^#' "$OUT_TRAJ")" -lt 2 ]; then
        echo "❌ RTAB-Map trajectory export produced fewer than two poses: $OUT_TRAJ" >&2
        exit 1
    fi
    python3 -c "from auto_mobility.benchmark.artifacts import save_trajectory_metadata; from auto_mobility.benchmark.candidate import SlamProfileSpec; save_trajectory_metadata('$OUT_TRAJ', SlamProfileSpec(candidate_key='$KEY', backend='rtab', profile='$PROFILE', replay_rate=float('$RATE')))" 2>/dev/null || true
    cp "$OUT_TRAJ" "$ALT_TRAJ"
    
    echo "✅ RTAB-Map 완료 -> DB: $DB_PATH | Trajectory: $OUT_TRAJ"
fi
