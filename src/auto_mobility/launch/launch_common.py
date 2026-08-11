# RTAB-Map 파라미터 단일 소스 (rtab_live / rtab_bag launch 공용)
# 튜닝 시 이 파일만 수정하면 live/bag 양쪽에 동일하게 적용된다.
#
# [하드웨어 / 환경]
# - VMware 가상머신, vCPU 8 (Intel Core Ultra 7 265H 호스트)
# - RAM 31GB, /dev/shm 16GB
# - RealSense D435i @ USB 3.0 (5000 Mbps)
# - 카메라: 640x480@30 (848x480은 depth 19.4Hz 드랍 → VM USB 대역폭 초과)
# - 촬영 중 RViz ON (point cloud 실시간 검증용) → odom 실측 11Hz
# - 벤치마크 29.9Hz는 rviz=false + 필터 없음 + 5초 창 조건 (실환경과 상이)
#
# ⚠️ RTAB-Map 0.23.7 기준 유효 키만 사용 (2026-08-11 검증 완료).

import os
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from auto_mobility.config import (
    CAMERA_RGB_TOPIC,
    CAMERA_IMU_TOPIC,
    CAMERA_IMU_FILTERED_TOPIC,
    CAMERA_INFO_TOPIC,
    CLOUD_MAP_TOPIC,
    CLOUD_MAP_LITE_TOPIC,
)

RTABMAP_PARAMS = {
    # [1] 특징점 검출 및 3D-2D PnP 포즈 추정
    'Vis/EstimationType': '1',  # 1: 3D-2D PnP
    'Vis/MinInliers': '10',     # IR 에미터 환경에서의 최소 인라이어
    'Vis/MaxFeatures': '1000',  # 프레임당 VO 비용 절감 (30Hz 유지)
    'GFTT/QualityLevel': '0.02',# 특징점 품질 임계값 (기본 0.001)
    'Vis/GridRows': '16',       # 특징점 균일 분포용 그리드
    'Vis/GridCols': '21',
    'Vis/MinDepth': '0.3',
    'Vis/MaxDepth': '4.0',      # 원거리 노이즈 뎁스 필터링
    'Vis/InlierDistance': '1.0',
    # [2] IMU 추정치 활용 (중력 벡터 정렬)
    'Odom/GuessMotion': 'true',       # 이전 모션 기반 다음 포즈 추측
    'Optimizer/GravitySigma': '0.3',  # 그래프 최적화 중력 제약
    # [3] 추적 끊김 복구 및 루프클로저 이상치 차단
    'Odom/ResetCountdown': '0',
    'RGBD/OptimizeMaxError': '3.0',   # 잘못된 루프클로저 오차 차단
    'Rtabmap/CreateIntermediateNodes': 'true',  # 키프레임 간 중간 노드 생성 (밀도↑)
    'RGBD/ProximityBySpace': 'true',
    'RGBD/OptimizeFromGraphEnd': 'true',
    'RGBD/NeighborLinkRefining': 'true',
    # [4] CPU 분산: 맵핑 루프 5Hz, 루프클로저 메모리
    'Rtabmap/DetectionRate': '5',
    'OdomF2M/MaxSize': '1000',  # ★ 2026-08-11 벤치마크 1위 — 기본 2000은 odom 17Hz 급락
    'Mem/STMSize': '10',        # ★ 2026-08-11 벤치마크 최적(STM=10), RAM 절감
    # [5] 키프레임 전략: 10cm / 5.7도 마다
    'RGBD/LinearUpdate': '0.10',
    'RGBD/AngularUpdate': '0.10',
    # [6] Point Cloud 품질 & 노이즈
    'Grid/3D': 'true',
    'Grid/DepthDecimation': '2',      # 4→2: 라이브 맵 4배 고밀도 (기본 4)
    'Grid/RangeMin': '0.3',
    'Grid/RangeMax': '4.0',           # Vis/MaxDepth(4.0)와 일치
    'Grid/NoiseFilteringRadius': '0.05',
    'Grid/NoiseFilteringMinNeighbors': '5',
    'Grid/RayTracing': 'true',
}

# RTABMAP_ARGS: rtabmap.launch.py의 rtabmap_args 인자에 넘겨줄 커맨드라인 렌더링 문자열
RTABMAP_ARGS = ' '.join([f'--{k} {v}' for k, v in RTABMAP_PARAMS.items()])

# live / bag 런치에서 서로 다른 값을 쓰는 인자 (단일 소스로 명시적 관리)
RTAB_LIVE_TOPIC_QUEUE_SIZE = '50'
RTAB_BAG_TOPIC_QUEUE_SIZE = '30'


def get_rtabmap_base_args() -> dict:
    """rtabmap.launch.py 에 공통으로 넘기는 인자 (rtab_live / rtab_bag 공용).

    live/bag 간 값이 다른 인자(odom_always_process_most_recent_frame,
    topic_queue_size, use_sim_time 등)는 각 런치 파일에서 덮어쓴다.
    """
    from ament_index_python.packages import get_package_share_directory
    return {
        'rgb_topic': CAMERA_RGB_TOPIC,
        'camera_info_topic': CAMERA_INFO_TOPIC,
        'imu_topic': CAMERA_IMU_FILTERED_TOPIC,
        'frame_id': 'camera_link',
        # QoS profile = [0: system default, 1: Reliable, 2: Best Effort]
        'qos_image': '2',
        'qos_depth': '2',
        'qos_camera_info': '2',
        'qos_imu': '2',
        # Synchronization (0.15s: VO delay 흡수 → 간헐적 "no odometry" 스킵 방지)
        'approx_sync': 'true',
        'approx_sync_max_interval': '0.15',
        'visual_odometry': 'true',
        'rviz': 'true',
        'rviz_cfg': os.path.join(get_package_share_directory('auto_mobility'), 'config', 'rviz', 'rtabmap_vmware.rviz'),
        'rtabmap_viz': 'false',
    }


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
            ('imu/data_raw', CAMERA_IMU_TOPIC),
            ('imu/data', CAMERA_IMU_FILTERED_TOPIC)
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
            'input_topic': CLOUD_MAP_TOPIC,
            'output_topic': CLOUD_MAP_LITE_TOPIC,
        }],
    )

