# RTAB-Map 파라미터 단일 소스 (rtab_live / rtab_bag launch 공용)
# 튜닝 시 이 파일만 수정하면 live/bag 양쪽에 동일하게 적용된다.
# VM 기준: nproc=8 (vCPU 8). Vis/CornerNbThreads는 VM vCPU 수에 맞춘다.

RTABMAP_ARGS = (
    # [1] 특징점 검출 강화: 흰 벽/빈 벽(텍스처 부재)에서 fromWords=0 및 Odometry Lost 방지
    '--Vis/MinInliers 8 '
    '--Vis/MaxFeatures 1500 '
    '--Vis/CornerMinQuality 0.01 '
    '--Vis/CornerGridSize 30 '
    '--Vis/MinDepth 0.3 '
    '--Vis/MaxDepth 8.0 '
    '--Vis/Robust true '
    '--Vis/InlierDistance 1.0 '
    # [2] 병렬 검출(VM vCPU 8) + F2M 매칭 부하 축소
    '--Vis/CornerNbThreads 8 '
    '--OdomF2M/MaxFrames 5 '
    # [3] IMU 추정치 활용 (IMU Orientation 초기 추정 + 중력 벡터 정렬)
    '--Odom/PoseGuessMode 1 '
    '--Optimizer/GravityProvided true '
    # [4] 추적 끊김 자동 복구 (Rtabmap/ResetCountdown 0: 지도 전체 리셋 금지, 세션 유지)
    '--Rtabmap/ResetCountdown 0 '
    '--RGBD/CreateIntermediateNodes true '
    '--RGBD/ProximityBySpace true '
    '--RGBD/OptimizeFromGraphEnd true '
    # [5] CPU 분산: 맵핑 루프 5Hz 제한, 루프클로저 후보 축소
    '--Rtabmap/DetectionRate 5 '
    '--Mem/STMSize 10 '
    # [6] 키프레임 전략: 드리프트 축적 및 정지상태 키프레임 낭비 방지
    '--RGBD/LinearUpdate 0.2 '
    '--RGBD/AngularUpdate 0.2 '
    # [7] Point Cloud 품질 & 노이즈 최적화 (VMware 성능 + 고품질 데이터 타협)
    '--Grid/3D true '
    '--Grid/VoxelSize 0.03 '
    '--Grid/RangeMin 0.3 '
    '--Grid/RangeMax 5.0 '
    '--Grid/NoiseFilteringRadius 0.1 '
    '--Grid/NoiseFilteringMinNeighbors 5 '
    '--Grid/RayTracing true'
)
