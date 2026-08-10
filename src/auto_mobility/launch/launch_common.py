# RTAB-Map 파라미터 단일 소스 (rtab_live / rtab_bag launch 공용)
# 튜닝 시 이 파일만 수정하면 live/bag 양쪽에 동일하게 적용된다.
# VM 기준: nproc=8 (vCPU 8). Vis/CornerNbThreads는 VM vCPU 수에 맞춘다.
#
# [2026-08-10 실환경 정합 재조정]
# - 벤치마크(29.9Hz)는 rviz=false + 카메라 필터 없음 + 라이브 5초 창으로 측정되어
#   실제 촬영(rviz on, 필터 on, 장시간)과 다른 조건이었음 (capture_guard 실측 11Hz)
# - [근본 해결] 실환경에서도 VO가 따라갈 수 있도록 프레임당 부하 절감:
#   * Vis/MaxFeatures 2000 → 1000 (특징점 검출/매칭 비용 절반, IR 에미터로 특징량 충분)
#   * Vis/CornerNbThreads 8 유지 (여유 vCPU 활용 → 검출 시간 단축)
#   * Mem/STMSize 10 → 20 (RAM 여유분 활용, 루프클로저 안정성)
# - 해상도: 640x480@30 (848x480은 depth 19.4Hz 드랍 → VM USB 대역폭 초과)

from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

RTABMAP_PARAMS = {
    # [1] 특징점 검출 및 3D-2D PnP 포즈 추정 (CPU 40% 절감 + Depth 결측 유실 방지)
    'Vis/EstimationType': '1',  # 1: 3D-2D PnP (벤치마크 우승 조합)
    'Vis/MinInliers': '10',     # [벤치마크 1위] IR Laser Projector와 결합 시 인라이어 10개 최적
    'Vis/MaxFeatures': '1000',  # [실환경 재조정] 프레임당 VO 비용 절감 (30Hz 유지 여유 확보)
    'Vis/CornerMinQuality': '0.02',   # 벤치마크 검증값 (기존 0.01)
    'Vis/CornerGridSize': '30',       # 벤치마크 검증값 (기존 20)
    'Vis/MinDepth': '0.3',
    'Vis/MaxDepth': '4.0',      # 노이즈가 심한 원거리 뎁스 포인트 필터링 (벤치마크: MD=4/8 동일 성능, 4m로 보수적 유지)
    'Vis/Robust': 'true',
    'Vis/InlierDistance': '1.0',
    # [2] 병렬 검출(VM vCPU 8) + F2M 로컬 맵 최적화
    'Vis/CornerNbThreads': '8',
    'OdomF2M/MaxFrames': '10',  # [벤치마크 1위] F2M=60 대비 rgbd_odometry CPU 73%→~35% 절감 (촬영 끊김 제거)
    # [3] IMU 추정치 활용 (IMU Orientation 초기 추정 + 중력 벡터 정렬)
    'Odom/PoseGuessMode': '1',
    'Optimizer/GravityProvided': 'true',
    # [4] 추적 끊김 자동 복구 및 루프클로저 이상치 걸러내기
    'Rtabmap/ResetCountdown': '0',
    'RGBD/OptimizeMaxError': '3.0',     # 잘못된 루프 클로저 오차 차단 (지형 일그러짐 방지)
    'RGBD/CreateIntermediateNodes': 'true',
    'RGBD/ProximityBySpace': 'true',
    'RGBD/OptimizeFromGraphEnd': 'true',
    'RGBD/NeighborLinkRefining': 'true', # 인접 키프레임 간 그래프 정밀 정렬 (동적 이동 안정성)
    # [5] CPU 분산: 맵핑 루프 5Hz 안정화, 루프클로저 메모리 관리
    'Rtabmap/DetectionRate': '5',       # vCPU 8 활용 및 그래프 최적화 지연 방지 5Hz 맵핑
    'Mem/STMSize': '20',                # [실환경 재조정] STM=20: RAM 여유분 활용 + 루프클로저 안정성 (벤치마크상 성능 차이 없음)
    # [6] 키프레임 전략: 10cm/5.7도 마다 키프레임 업데이트 (그래프 폭주 및 lag 방지)
    'RGBD/LinearUpdate': '0.10',
    'RGBD/AngularUpdate': '0.10',
    # [7] Point Cloud 품질 & 노이즈 최적화 (1cm Voxel 고해상도 매핑)
    'Grid/3D': 'true',
    'Grid/VoxelSize': '0.01',           # 1cm 정밀 격자 보존
    'Grid/RangeMin': '0.3',
    'Grid/RangeMax': '3.0',
    'Grid/NoiseFilteringRadius': '0.05',
    'Grid/NoiseFilteringMinNeighbors': '5',
    'Grid/RayTracing': 'true',
}

# RTABMAP_ARGS: rtabmap.launch.py의 rtabmap_args 인자에 넘겨줄 커맨드라인 렌더링 문자열
RTABMAP_ARGS = ' '.join([f'--{k} {v}' for k, v in RTABMAP_PARAMS.items()])


def create_republish_node(use_sim_time: bool, depth_compressed_topic=None):
    """공통 압축 해제 노드(republish.py) 생성 헬퍼"""
    params = [{'use_sim_time': use_sim_time}]
    if depth_compressed_topic is not None:
        params[0]['depth_compressed_topic'] = depth_compressed_topic

    return Node(
        package='auto_mobility',
        executable='republish.py',
        name='republish_compressed',
        output='screen',
        parameters=params,
        condition=IfCondition(LaunchConfiguration('use_compressed'))
    )


def create_imu_filter_node(use_sim_time: bool):
    """공통 IMU Madgwick 필터 노드 생성 헬퍼"""
    return Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'gain': 0.03,
            'world_frame': 'enu',
            'publish_tf': False,
            'use_sim_time': use_sim_time,
        }],
        remappings=[
            ('imu/data_raw', '/camera/camera/imu'),
            ('imu/data', '/camera/camera/imu/filtered')
        ],
        condition=IfCondition(LaunchConfiguration('use_imu'))
    )


def create_cloud_throttle_node(use_sim_time: bool, max_rate: float = 2.0):
    """RViz 소프트웨어 렌더링 부하 절감용 cloud_map 저주파 중계 노드

    /rtabmap/cloud_map (DetectionRate=5Hz) 을 최대 max_rate Hz 로
    /rtabmap/cloud_map_lite 에 재발행한다. SLAM 내부 처리에는 영향 없음.
    """
    return Node(
        package='auto_mobility',
        executable='cloud_throttle.py',
        name='cloud_throttle',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'max_rate': max_rate,
        }],
    )

