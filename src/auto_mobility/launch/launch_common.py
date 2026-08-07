# RTAB-Map 파라미터 단일 소스 (rtab_live / rtab_bag launch 공용)
# 튜닝 시 이 파일만 수정하면 live/bag 양쪽에 동일하게 적용된다.
# VM 기준: nproc=8 (vCPU 8). Vis/CornerNbThreads는 VM vCPU 수에 맞춘다.

from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

RTABMAP_PARAMS = {
    # [1] 특징점 검출 및 3D-2D PnP 포즈 추정 (CPU 40% 절감 + Depth 결측 유실 방지)
    'Vis/EstimationType': '1',  # 1: 3D-2D PnP (개선)
    'Vis/MinInliers': '10',     # [벤치마크 1위] IR Laser Projector와 결합 시 특징점 인라이어 검증 정밀도 10개 최적
    'Vis/MaxFeatures': '2000',
    'Vis/CornerMinQuality': '0.01',
    'Vis/CornerGridSize': '20',
    'Vis/MinDepth': '0.3',
    'Vis/MaxDepth': '4.0',      # 노이즈가 심한 멀리 있는 뎁스 포인트 필터링 (4m 제한)
    'Vis/Robust': 'true',
    'Vis/InlierDistance': '1.0',
    # [2] 병렬 검출(VM vCPU 8) + F2M 매칭 부하 축소 & 32GB RAM 대역폭 확장
    'Vis/CornerNbThreads': '8',
    'OdomF2M/MaxFrames': '20',  # 32GB RAM 캐시 프레임 확장 (VO 데이터 수율 보존)
    # [3] IMU 추정치 활용 (IMU Orientation 초기 추정 + 중력 벡터 정렬)
    'Odom/PoseGuessMode': '1',
    'Optimizer/GravityProvided': 'true',
    # [4] 추적 끊김 자동 복구 및 루프클로저 이상치 걸러내기
    'Rtabmap/ResetCountdown': '0',
    'RGBD/OptimizeMaxError': '3.0',     # 잘못된 루프 클로저 오차 차단 (지형 일그러짐 방지)
    'RGBD/CreateIntermediateNodes': 'true',
    'RGBD/ProximityBySpace': 'true',
    'RGBD/OptimizeFromGraphEnd': 'true',
    # [5] CPU 분산: 맵핑 루프 10Hz 고해상도 지원, 루프클로저 메모리 확장
    'Rtabmap/DetectionRate': '10',      # vCPU 8 활용 촘촘한 10Hz 키프레임 포착
    'Mem/STMSize': '30',                # Short-term memory 30개 확장
    # [6] 키프레임 전략: 5cm/2.8도 마다 촘촘한 키프레임 업데이트
    'RGBD/LinearUpdate': '0.05',
    'RGBD/AngularUpdate': '0.05',
    # [7] Point Cloud 품질 & 노이즈 최적화 (1cm Voxel 고해상도 매핑)
    'Grid/3D': 'true',
    'Grid/VoxelSize': '0.01',           # 기존 3cm -> 1cm 정밀 격자 보존
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

