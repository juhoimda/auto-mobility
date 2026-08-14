# RTAB-Map 파라미터 단일 소스 (rtab_live / rtab_bag launch 공용)
# 튜닝 시 이 파일만 수정하면 live/bag 양쪽에 동일하게 적용된다.
#
# [하드웨어 / 환경] (2026-08-12 WSL2 기준 갱신)
# - WSL2, Intel Core Ultra 7 265H / 32GB RAM / /dev/shm 16GB
# - RealSense D435i @ USB 3.2 패스스루 (usbipd 5.3.0)
# - 카메라: 640x480@30 (WSL2 USB 패스스루에서 848x480 이상은 프레임 손상 → 640x480 고정)
# - 촬영 중 RViz ON (cloud_map_lite 2Hz 중계) → odom 실측 7~11Hz
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
    'Vis/MinInliers': '10',     # 원격 저주파/정지 구간 안정적 매칭 (10)
    'Vis/MaxFeatures': '2000',  # 회전 시 키프레임 중첩 확보
    'GFTT/QualityLevel': '0.01', # 정지 시 영상 노이즈 코너 오인 방지 (안정화 0.01)
    'Vis/GridRows': '16',       # 특징점 균일 분포용 그리드
    'Vis/GridCols': '21',
    'Vis/MinDepth': '0.3',
    'Vis/MaxDepth': '4.0',      # 원거리 노이즈 뎁스 필터링
    'Vis/InlierDistance': '0.1', # ★ 필수 수정: 1.0m -> 0.1m 정상화
    'Vis/RefineIterations': '5', # PnP 포즈 정밀 최적화
    # [2] IMU 추정치 활용 (중력 벡터 정렬)
    'Optimizer/GravitySigma': '0',    # 손떨림/IMU 노이즈로 인한 맵 전체 덜컹거림(진동/왔다갔다) 원천 차단
    # [3] 추적 끊김 복구 및 루프클로저 이상치 차단
    'RGBD/OptimizeMaxError': '3.0',   # 잘못된 루프클로저 오차 차단
    'Rtabmap/CreateIntermediateNodes': 'false', # 정지/미세진동 시 불필요한 중간 노드 및 녹색 경로선 폭발 방지
    # ★ 2026-08-12 euijin 세션(143506/142815) 실측 분석 반영:
    #   재방문 환경에서 ProximityBySpace+NeighborLinkRefining+OptimizeFromGraphEnd 조합이
    #   "Map correction should be identity" 에러 105회 / loop closure 전량 거부 / 맵 스레드 0.94s 스톨 유발.
    #   루프클로저가 그래프를 보정할 수 있도록 전부 비활성화.
    'RGBD/ProximityBySpace': 'false',
    'RGBD/OptimizeFromGraphEnd': 'false',
    'RGBD/NeighborLinkRefining': 'false',
    # [4] CPU 분산: 맵핑 루프 3Hz, 루프클로저 메모리
    'Rtabmap/DetectionRate': '3',
    'Mem/STMSize': '10',        # ★ 2026-08-11 벤치마크 최적(STM=10), RAM 절감
    # [5] 키프레임 전략: 10cm / 5.7도 마다 (안정적인 스캐닝 노드 연속성)
    'RGBD/LinearUpdate': '0.10',
    'RGBD/AngularUpdate': '0.10',
    # [6] Point Cloud 품질 & 노이즈 (선명하고 촘촘한 라이브 포인트클라우드)
    'Grid/3D': 'true',
    'Grid/DepthDecimation': '2',      # 선명하고 촘촘한 3D 포인트 클라우드 표시 (2)
    'Grid/RangeMin': '0.3',
    'Grid/RangeMax': '4.0',           # Vis/MaxDepth(4.0)와 일치
    'Grid/NoiseFilteringRadius': '0.05',
    'Grid/NoiseFilteringMinNeighbors': '5',
    'Grid/RayTracing': 'false',       # 2D 점유격자용 — 구독자 없음 (기존 true)
}

# ⚠️ ODOM_ARGS: rgbd_odometry 노드 전용 파라미터 (2026-08-11 확인)
# rtabmap 메인 노드는 Odom/*, OdomF2M/* 키를 ROS2 파라미터로 선언하지 않아
# rtabmap_args로 전달하면 ParameterNotDeclaredException → SIGABRT 크래시 발생.
# rtabmap.launch.py의 odom_args 인자로 전달해야 rgbd_odometry에서 정상 적용된다.
ODOM_PARAMS = {
    'Odom/Strategy': '0',             # Frame-to-Map Visual Odometry
    'Odom/GuessMotion': 'false',      # 모션 과추정으로 인한 점프/떨림 방지 (안정적인 PnP 추적)
    'Odom/ResetCountdown': '1',       # 연속 실패 시 빠른 자동 복구
    'OdomF2M/MaxSize': '1000',        # ★ 2026-08-11 벤치 1위 — 기본 2000은 odom 17Hz 급락
    'Odom/KeyFrameThr': '0.3',        # 안정적인 키프레임 임계값
    'Odom/FilteringStrategy': '0',    # 불안정한 IMU 필터링 대신 순수 정밀 Visual PnP 추적
    'Odom/GravitySigma': '0',         # 오도메트리 내부 IMU 중력 적분 비활성화 (정지 시 표류/녹색선 일정한 전진 원천 차단)
    'Odom/AlignWithGround': 'false',  # 지면 정렬 모션 강제 비활성화
}

# RTABMAP_ARGS: rtabmap.launch.py의 rtabmap_args 인자에 넘겨줄 커맨드라인 렌더링 문자열
RTABMAP_ARGS = ' '.join([f'--{k} {v}' for k, v in RTABMAP_PARAMS.items()])

# ODOM_ARGS: rtabmap.launch.py의 odom_args 인자 (rgbd_odometry 노드에 전달)
ODOM_ARGS = ' '.join([f'--{k} {v}' for k, v in ODOM_PARAMS.items()])

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
        'wait_imu_to_init': 'false',
        'always_check_imu_tf': 'false',
        # QoS profile = [0: system default, 1: Reliable, 2: Best Effort]
        'qos': '2',
        'qos_image': '2',
        'qos_depth': '2',
        'qos_camera_info': '2',
        'qos_imu': '2',
        'qos_odom': '2',
        # Pre-sync RGBDImage (rtabmap_sync/rgbd_sync 노드로 3개 토픽 묶음 처리)
        # 개별 5개 토픽 직접 동기화 시 발생하는 타임스탬프 불일치 및 큐 드랍 방지
        'rgbd_sync': 'true',
        'approx_rgbd_sync': 'true',
        'subscribe_rgbd': 'true',
        'depth': 'false',
        'subscribe_rgb': 'false',
        # depth_scale 기본값 1.0 유지: republish.py가 depth를 16UC1(mm)로 발행하며,
        # rtabmap core가 16UC1(mm)을 내부적으로 미터 변환해 odometry/mesh 모두 정상 동작.
        # (16UC1 mm에 depth_scale<1을 곱하면 uint16 saturate_cast로 depth 파괴,
        #  32FC1 미터 발행은 rtabmap DB가 8비트 PNG로 저장해 파괴 — 둘 다 실측)
        # Synchronization (0.05s: 50ms 이내의 RGB-Depth만 짝지어 오랜 시차 매칭 방지)
        'approx_sync': 'true',
        'approx_sync_max_interval': '0.05',
        'approx_rgbd_sync': 'true',
        'sync_queue_size': '30',
        'visual_odometry': 'true',
        'rviz': 'true',
        'rviz_cfg': os.path.join(get_package_share_directory('auto_mobility'), 'config', 'rviz', 'rtabmap_live.rviz'),
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
    """공통 IMU Madgwick 필터 노드 생성 헬퍼 (gain 0.01: 센서 노이즈 억제 및 부드러운 중력 벡터 필터링)"""
    return Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'gain': 0.01,
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


def create_point_cloud_node(use_sim_time: bool):
    """실시간 Live PointCloud (/rtabmap/voxel_cloud) 생성 노드"""
    return Node(
        package='rtabmap_util',
        executable='point_cloud_xyzrgb',
        name='point_cloud_xyzrgb_live',
        namespace='rtabmap',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'approx_sync': True,
            'approx_sync_max_interval': 0.0,
            'topic_queue_size': 50,
            'sync_queue_size': 30,
            'qos': 2,
            'qos_camera_info': 2,
            'subscribe_rgbd': True,
            'decimation': 4,
            'voxel_size': 0.0,
        }],
        remappings=[
            ('rgbd_image', 'rgbd_image'),
            ('cloud', 'voxel_cloud'),
        ],
    )


