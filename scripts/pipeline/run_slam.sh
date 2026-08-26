#!/bin/bash
# run_slam.sh — Rosbag 데이터셋에 대해 원하는 SLAM (rtab / orb)을 1-명령어로 실행하고 궤적/DB를 생성

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PIPELINE_DIR/../common.sh"

# Keep ROS launch logs inside the project output tree. This also makes offline
# runs work in sandboxed/CI environments where ~/.ros is read-only.
export ROS_LOG_DIR="${ROS_LOG_DIR:-$LOG_DIR/ros}"
mkdir -p "$ROS_LOG_DIR"

SLAM_TYPE="rtab"
BAG_NAME=""
DENSE_MAPPING="false"

for arg in "$@"; do
    case $arg in
        --slam=*)
            SLAM_TYPE="${arg#*=}"
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
    echo " 사용법: $0 BAG_NAME [--slam=rtab|orb] [--dense]"
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
    KEY="orb_rgbdi"
    OUT_TRAJ="$TRAJECTORY_DIR/${KEY}_${BAG_NAME}_trajectory.txt"
    python3 "$PROJECT_DIR/src/auto_mobility/slam/run_orbslam3_bag.py" "$BAG_PATH" --out "$OUT_TRAJ" --mode rgbdi
    python3 -c "from auto_mobility.benchmark.artifacts import save_trajectory_metadata; from auto_mobility.benchmark.candidate import SlamProfileSpec; save_trajectory_metadata('$OUT_TRAJ', SlamProfileSpec(candidate_key='$KEY', backend='orb_rgbdi', profile='normal'))" 2>/dev/null || true
    echo "✅ ORB-SLAM3 RGB-D-I 완료 -> $OUT_TRAJ"
elif [ "$SLAM_TYPE" == "orb" ] || [ "$SLAM_TYPE" == "orbslam" ] || [ "$SLAM_TYPE" == "orbslam3" ] || [ "$SLAM_TYPE" == "orb_rgbd" ]; then
    KEY="orb_rgbd"
    OUT_TRAJ="$TRAJECTORY_DIR/${KEY}_${BAG_NAME}_trajectory.txt"
    python3 "$PROJECT_DIR/src/auto_mobility/slam/run_orbslam3_bag.py" "$BAG_PATH" --out "$OUT_TRAJ" --mode rgbd
    python3 -c "from auto_mobility.benchmark.artifacts import save_trajectory_metadata; from auto_mobility.benchmark.candidate import SlamProfileSpec; save_trajectory_metadata('$OUT_TRAJ', SlamProfileSpec(candidate_key='$KEY', backend='orb_rgbd', profile='normal'))" 2>/dev/null || true
    echo "✅ ORB-SLAM3 RGB-D 완료 -> $OUT_TRAJ"
elif [ "$SLAM_TYPE" == "stella" ] || [ "$SLAM_TYPE" == "stella_rgbd" ]; then
    KEY="stella_rgbd"
    OUT_TRAJ="$TRAJECTORY_DIR/${KEY}_${BAG_NAME}_trajectory.txt"
    python3 "$PROJECT_DIR/src/auto_mobility/slam/run_stella_bag.py" "$BAG_PATH" --out "$OUT_TRAJ"
    python3 -c "from auto_mobility.benchmark.artifacts import save_trajectory_metadata; from auto_mobility.benchmark.candidate import SlamProfileSpec; save_trajectory_metadata('$OUT_TRAJ', SlamProfileSpec(candidate_key='$KEY', backend='stella_rgbd', profile='normal'))" 2>/dev/null || true
    echo "✅ stella_vslam RGB-D 완료 -> $OUT_TRAJ"
else
    # RTAB-Map (Frame-based synchronous zero-drop execution)
    if [ "$DENSE_MAPPING" = "true" ]; then
        PROFILE="dense"
    else
        PROFILE="normal"
    fi
    KEY="rtab_${PROFILE}"
    OUT_TRAJ="$TRAJECTORY_DIR/${KEY}_${BAG_NAME}_trajectory.txt"
    DB_PATH="$DB_DIR/${BAG_NAME}_${KEY}.db"
    
    python3 "$PROJECT_DIR/src/auto_mobility/slam/run_rtabmap_bag.py" "$BAG_PATH" \
        --out "$OUT_TRAJ" \
        --db "$DB_PATH" \
        --profile "$PROFILE"
    
    echo "✅ RTAB-Map 완료 -> DB: $DB_PATH | Trajectory: $OUT_TRAJ"
fi
