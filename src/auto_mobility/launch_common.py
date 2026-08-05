# RTAB-Map 파라미터 단일 소스 (rtab_live / rtab_bag launch 공용)
# 튜닝 시 이 파일만 수정하면 live/bag 양쪽에 동일하게 적용된다.

RTABMAP_ARGS = (
    # [1] 특징점 검출 강화: 흰 벽/빈 벽(텍스처 부재)에서 fromWords=0 방지
    '--Vis/MinInliers 10 '
    '--Vis/MaxFeatures 1500 '
    '--Vis/CornerMinQuality 0.01 '
    '--Vis/CornerGridSize 20 '
    '--Vis/MinDepth 0.3 '
    '--Vis/MaxDepth 8.0 '
    '--Vis/Robust true '
    '--Vis/InlierDistance 1.0 '
    # [2] 잘못된 guess(IMU yaw 드리프트) 의존 제거 (IMU 구독/사용은 유지)
    '--Odom/PoseGuessMode 0 '
    # [3] 추적 끊김 자동 복구
    '--Odom/ResetCountdown 2 '
    '--Rtabmap/ResetCountdown 1 '
    '--RGBD/CreateIntermediateNodes true '
    '--RGBD/ProximityBySpace true '
    '--RGBD/OptimizeFromGraphEnd true '
    # [4] 키프레임 전략: 드리프트 축적 및 정지상태 키프레임 낭비 방지
    '--Mem/STMSize 20 '
    '--RGBD/LinearUpdate 0.2 '
    '--RGBD/AngularUpdate 0.2'
)
