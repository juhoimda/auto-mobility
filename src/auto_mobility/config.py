"""
Auto-Mobility 중앙 설정 모듈.

토픽명 / 데이터 경로 / 카메라 파라미터 / Mesh 기본값 의 단일 소스(Single Source of Truth).
여러 파일(launch, node, benchmark, shell)에서 하드코딩하던 값을 여기서 관리한다.
수정 시 이 파일(또는 config/topics.yaml)만 바꾸면 전체에 동일하게 적용된다.
"""

import os
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """Return project root directory path"""
    return Path(__file__).resolve().parent.parent.parent


def _find_config_file(filename: str):
    """config/ 하위 파일을 프로젝트 루트 또는 설치 share 디렉터리에서 찾는다."""
    candidates = [
        get_project_root() / "config" / filename,
        Path("/opt/ros") / os.getenv("ROS_DISTRO", "humble")
        / "share" / "auto_mobility" / "config" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_topics_config() -> dict:
    """Load config/topics.yaml"""
    config_path = _find_config_file("topics.yaml")
    if config_path is not None:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


TOPICS_CONFIG = load_topics_config()


def get_topic(name: str, default=None) -> str:
    """config/topics.yaml 에서 토픽명을 조회한다.

    key 형식: '<section>.<field>' (예: 'camera.rgb_topic', 'rtabmap.odom_topic').
    미정의 시 default 를 반환하고, default 도 없으면 KeyError 를 던진다.
    """
    if "." in name:
        section, field = name.split(".", 1)
        section_config = TOPICS_CONFIG.get(section)
        if isinstance(section_config, dict) and field in section_config:
            return section_config[field]
    if default is not None:
        return default
    raise KeyError(f"Unknown topics config key: {name}")


def _topic(key: str, fallback: str) -> str:
    return get_topic(key, fallback)


# ────────────────────────────── 토픽 상수 ──────────────────────────────
CAMERA_RGB_TOPIC               = _topic("camera.rgb_topic", "/camera/camera/color/image_raw")
CAMERA_RGB_COMPRESSED_TOPIC    = _topic("camera.rgb_compressed_topic", "/camera/camera/color/image_raw/compressed")
CAMERA_DEPTH_TOPIC             = _topic("camera.depth_topic", "/camera/camera/depth/image_rect_raw")
CAMERA_DEPTH_COMPRESSED_TOPIC  = _topic("camera.depth_compressed_topic", "/camera/camera/depth/image_rect_raw/compressedDepth")
CAMERA_ALIGNED_DEPTH_TOPIC     = _topic("camera.aligned_depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC = _topic("camera.aligned_depth_compressed_topic",
                                                "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth")
CAMERA_INFO_TOPIC              = _topic("camera.camera_info_topic", "/camera/camera/color/camera_info")
CAMERA_IMU_TOPIC               = _topic("camera.imu_topic", "/camera/camera/imu")
CAMERA_IMU_FILTERED_TOPIC      = _topic("camera.imu_filtered_topic", "/camera/camera/imu/filtered")

ODOM_TOPIC          = _topic("rtabmap.odom_topic", "/rtabmap/odom")
MAP_TOPIC           = _topic("rtabmap.map_topic", "/rtabmap/mapData")
CLOUD_MAP_TOPIC     = _topic("rtabmap.cloud_map_topic", "/rtabmap/cloud_map")
CLOUD_MAP_LITE_TOPIC = _topic("rtabmap.cloud_map_lite_topic", "/rtabmap/cloud_map_lite")


# ────────────────────────────── 경로 상수 ──────────────────────────────
PROJECT_DIR    = get_project_root()
CONFIG_DIR     = PROJECT_DIR / "config"
DATA_DIR       = PROJECT_DIR / "ros2_data"
BAG_DIR        = DATA_DIR / "bags"
DB_DIR         = DATA_DIR / "databases"
POINTCLOUD_DIR = DATA_DIR / "pointclouds"
MESH_DIR       = DATA_DIR / "meshes"
ISAAC_DIR      = DATA_DIR / "isaac_sim"
LOG_DIR        = DATA_DIR / "logs"
FASTDDS_XML    = CONFIG_DIR / "dds" / "fastdds_camera.xml"
SHARED_DIR     = Path("/mnt/hgfs/ubuntu_shared")

# RealSense USB 링크 속도 기준 (Mbps). 5000 이상 = USB 3.x
USB_3_MIN_SPEED_MBPS = 5000


# ──────────────────────── 카메라 노드 파라미터 단일 소스 ────────────────────────
# camera.launch.py 와 benchmark_slam.py / benchmark_hw.py 가 공용으로 사용.
# 해상도/프로파일은 벤치마크가 동적으로 변경하므로 별도 상수로 분리.
CAMERA_PROFILE = "640x480x30"
CAMERA_RESOLUTION = "640x480"

CAMERA_PARAMS = {
    "depth_module.depth_profile": CAMERA_PROFILE,
    "rgb_camera.color_profile": CAMERA_PROFILE,
    "rgb_camera.color_format": "RGB8",
    "align_depth.enable": False,        # vCPU 정렬 병목 제거 (RTAB-Map 자체 정렬 사용)
    "enable_infra1": False,
    "enable_infra2": False,
    "depth_module.emitter_enabled": 1,  # IR Laser Projector
    "enable_accel": True,
    "enable_gyro": True,
    "enable_sync": True,
    "unite_imu_method": 1,              # 1: copy mode
    "enable_metadata": False,
    "global_time_enabled": False,
    "initial_reset": False,
    "rgb_camera.auto_exposure_priority": False,
    "color_qos": "SENSOR_DATA",
    "color_info_qos": "SENSOR_DATA",
    "depth_qos": "SENSOR_DATA",
    "depth_info_qos": "SENSOR_DATA",
    "filters": "spatial,temporal,hole_filling",
    "spatial_filter.spatial_alpha": 0.5,
    "spatial_filter.spatial_delta": 20,
    "temporal_filter.temporal_alpha": 0.4,
    "temporal_filter.temporal_delta": 20,
    "hole_filling_filter.holes_fill": 1,
}


# ──────────────────────── Mesh 기본값 단일 소스 ────────────────────────
# mesh_open3d.py 함수 시그니처 / CLI argparse / run_pipeline_all.sh 가 공용으로 사용.
MESH_DEFAULTS = {
    "depth": 8,             # Poisson octree 깊이
    "voxel_size": 0.005,    # 다운샘플링 voxel 크기 (5mm)
    "method": "poisson",    # poisson | bpa
    "simplify_target": 0.5, # Quadric Decimation 50% 경량화
}
