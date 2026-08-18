"""
Auto-Mobility 중앙 설정 모듈.

토픽명 / 데이터 경로 / 카메라 파라미터 / Mesh 기본값의 단일 소스(Single Source of Truth).
launch, node, benchmark, shell 등 여러 진입점에서 동일한 값을 참조한다.
이 파일(또는 config/topics.yaml)만 수정하면 전체에 동일하게 적용된다.

설정 우선순위 (낮은 → 높은):
  1. 이 파일의 하드코딩 기본값
  2. config/topics.yaml (토픽명 오버라이드)
  3. 환경변수 (common.sh에서 읽음, shell 스크립트 전용)
  4. launch argument / CLI argument (실행 시 명시적 지정)

파이프라인 내 역할:
  SENSOR (D435i)
    ↓  Windows publisher (realsense_pub.py)
    ↓  DDS transport (CycloneDDS, ROS_DOMAIN_ID=42)
  WSL INGRESS
    ↓  rosbag recorder (capture_safe.sh → record.sh)
  IMMUTABLE RAW DATASET (bags/)
    ↓  RTAB-Map SLAM (rtab_live / rtab_bag launch)
  RTAB-Map DATABASE (databases/)
    ↓  Open3D TSDF reconstruction (reconstruct_tsdf.py)
  MESH (meshes/)
    ↓  Isaac Sim (load_isaac_mesh.py)
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
CAMERA_INFO_WINDOWS_TOPIC      = _topic("camera.camera_info_windows_topic", "/camera/camera/color/camera_info_windows")
CAMERA_INFRA1_TOPIC            = _topic("camera.infra1_topic", "/camera/camera/infra1/image_rect_raw")
CAMERA_INFRA1_COMPRESSED_TOPIC = _topic("camera.infra1_compressed_topic", "/camera/camera/infra1/image_rect_raw/compressed")
CAMERA_INFRA1_INFO_WINDOWS_TOPIC = _topic("camera.infra1_camera_info_windows_topic", "/camera/camera/infra1/camera_info_windows")
CAMERA_INFRA2_TOPIC            = _topic("camera.infra2_topic", "/camera/camera/infra2/image_rect_raw")
CAMERA_INFRA2_COMPRESSED_TOPIC = _topic("camera.infra2_compressed_topic", "/camera/camera/infra2/image_rect_raw/compressed")
CAMERA_INFRA2_INFO_WINDOWS_TOPIC = _topic("camera.infra2_camera_info_windows_topic", "/camera/camera/infra2/camera_info_windows")
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
BAG_DIR        = DATA_DIR / "bags"           # raw rosbag2 / MCAP (불변 원본 데이터셋)
DB_DIR         = DATA_DIR / "databases"      # RTAB-Map .db (derived artifact)
POINTCLOUD_DIR = DATA_DIR / "pointclouds"    # 중간 PLY (derived artifact)
MESH_DIR       = DATA_DIR / "meshes"         # 최종 OBJ/PLY mesh (derived artifact)
TRAJECTORY_DIR = DATA_DIR / "trajectories"   # 표준 TUM trajectory (derived artifact)
BENCHMARK_DIR  = DATA_DIR / "benchmarks"     # 알고리즘 비교 벤치마크 결과
ISAAC_DIR      = DATA_DIR / "isaac_sim"      # Isaac Sim 입력용 복사본
LOG_DIR        = DATA_DIR / "logs"           # 파이프라인 / capture_guard 로그
# FASTDDS_XML: FastDDS 프로파일. CycloneDDS(rmw_cyclonedds_cpp)로 전환 후 사용되지 않으나
# 수동으로 FastDDS로 되돌릴 때를 위해 경로 상수만 유지한다 (common.sh 참조).
FASTDDS_XML    = CONFIG_DIR / "dds" / "fastdds_camera.xml"

# SHARED_DIR: WSL2 → Windows 파일 전달 경로 (/mnt/c/ubuntu_shared).
# 과거 VMware(/mnt/hgfs)에서 WSL2(/mnt/c)로 이관되었다 (2026-08-12).
SHARED_DIR     = Path("/mnt/c/ubuntu_shared")

# RealSense USB 링크 속도 기준 (Mbps). 5000 이상 = USB 3.x.
# capture_guard, validate 등이 USB 3.x 여부를 판단할 때 사용한다.
USB_3_MIN_SPEED_MBPS = 5000


# ──────────────────────── 카메라 노드 파라미터 단일 소스 ────────────────────────
# camera.launch.py 와 benchmark_slam.py / benchmark_hw.py 가 공용으로 사용한다.
# 해상도/프로파일은 벤치마크가 동적으로 변경하므로 별도 상수로 분리한다.
#
# 640x480@30: WSL2 USB 3.2 패스스루(usbipd) 환경에서 실측 안정 프로파일.
# 848x480 이상은 USB 대역폭 초과로 프레임 손상 위험 (2026-08-12 실측).
# Windows 원격 카메라 모드에서는 Windows 측 realsense_pub.py 가 이 프로파일을 사용한다.
CAMERA_PROFILE = "640x480x30"
CAMERA_RESOLUTION = "640x480"


CAMERA_PARAMS = {
    "depth_module.depth_profile": CAMERA_PROFILE,
    "rgb_camera.color_profile": CAMERA_PROFILE,
    "rgb_camera.color_format": "RGB8",
    # align_depth: False → CPU 정렬 병목 제거. RTAB-Map 내부에서 rgbd_sync로 정렬.
    # True로 설정하면 D435i가 하드웨어로 정렬하지만, Windows 원격 모드에서는
    # realsense_pub.py가 align_depth_to_color 스트림을 직접 발행하므로 여기서는 무관.
    "align_depth.enable": False,
    # IR 스트림: raw dataset 절약을 위해 기본 비활성. odometry에 IR을 사용하지 않음.
    # stereo 기반 odometry로 전환 시 enable_infra1/2를 True로 바꾸고 rosbag에도 포함해야 한다.
    "enable_infra1": False,
    "enable_infra2": False,
    "depth_module.emitter_enabled": 1,  # IR 레이저 프로젝터 활성 (structured light depth 품질 유지)
    # IMU: raw dataset에 포함하여 향후 visual-inertial odometry 전환에 대비.
    # 현재 RTAB-Map odometry는 Odom/FilteringStrategy=0(Visual PnP 전용)이므로
    # IMU는 rtabmap_sync에서 pose graph 중력 정렬 보조에만 사용된다.
    "enable_accel": True,
    "enable_gyro": True,
    "enable_sync": True,
    "unite_imu_method": 1,              # 1: copy mode (accel과 gyro를 동일 타임스탬프로 합산)
    "global_time_enabled": False,
    "initial_reset": False,
    # NOTE(2026-08-12 WSL2 실측): color_qos/depth_qos 등 QoS 파라미터는
    # realsense2_camera 4.58.3에서 스트림 FPS를 30→~10fps로 급락시킴 (실측 확인).
    # 따라서 제거하고 기본 RELIABLE QoS를 사용한다. (RTAB-Map은 BE로 구독하므로 무해)
    # NOTE: 'filters' 콤마 문자열 키는 이 버전에서 무시됨("Parameter not set").
    #       필터 옵션 키도 'spatial_filter.spatial_alpha'가 아니라
    #       'spatial_filter.filter_smooth_alpha'가 유효함 (실측 param list 확인).
    "spatial_filter.enable": True,
    "temporal_filter.enable": True,
    "hole_filling_filter.enable": True,
    "spatial_filter.filter_smooth_alpha": 0.5,
    "spatial_filter.filter_smooth_delta": 20,
    "temporal_filter.filter_smooth_alpha": 0.4,
    "temporal_filter.filter_smooth_delta": 20,
    "hole_filling_filter.holes_fill": 1,
}



# ──────────────────────── 카메라 파라미터 유효성 검증 ────────────────────────
# realsense2_camera 4.58.3 실측(param list + launch 경고)으로 확인된 문제 키.
# 잘못된 키는 노드가 조용히 무시하므로, 설정 실수 방지를 위해 여기서 사전 차단한다.

# 1) QoS 키: v4.58.3에서 스트림 FPS 급락(30→~10fps)을 유발 (2026-08-12 실측). 사용 금지.
FORBIDDEN_CAMERA_PARAMS = {
    "color_qos": "FPS 급락 유발(실측). 기본 RELIABLE 사용",
    "color_info_qos": "FPS 급락 유발(실측). 기본 RELIABLE 사용",
    "depth_qos": "FPS 급락 유발(실측). 기본 RELIABLE 사용",
    "depth_info_qos": "FPS 급락 유발(실측). 기본 RELIABLE 사용",
}

# 2) 무효 키: 노드가 선언하지 않아 "Parameter not set"로 조용히 무시되는 키들 (실측).
INVALID_CAMERA_PARAMS = {
    "filters": "'spatial_filter.enable' 등 개별 키로 대체",
    "rgb_camera.auto_exposure_priority": "'rgb_camera.enable_auto_exposure'만 존재",
    "spatial_filter.spatial_alpha": "'spatial_filter.filter_smooth_alpha'가 유효",
    "spatial_filter.spatial_delta": "'spatial_filter.filter_smooth_delta'가 유효",
    "temporal_filter.temporal_alpha": "'temporal_filter.filter_smooth_alpha'가 유효",
    "temporal_filter.temporal_delta": "'temporal_filter.filter_smooth_delta'가 유효",
    "enable_metadata": "선언되지 않은 키",
}


def validate_camera_params(params: dict) -> list:
    """CAMERA_PARAMS 를 검증해 문제 목록을 반환한다. (문제 없으면 빈 리스트)"""
    issues = []
    for key, reason in FORBIDDEN_CAMERA_PARAMS.items():
        if key in params:
            issues.append(f"[FORBIDDEN] {key}: {reason}")
    for key, reason in INVALID_CAMERA_PARAMS.items():
        if key in params:
            issues.append(f"[INVALID] {key}: {reason}")
    return issues


if __name__ == "__main__":
    issues = validate_camera_params(CAMERA_PARAMS)
    if issues:
        print("카메라 파라미터 문제 발견:")
        for i in issues:
            print("  -", i)
        raise SystemExit(1)
    print("카메라 파라미터 검증 통과 (문제 없음)")


# ──────────────────────── Mesh 기본값 단일 소스 ────────────────────────
# mesh_open3d.py 함수 시그니처 / CLI argparse 가 공용으로 사용한다.
#
# 주의: 이 값들은 PLY PointCloud → Poisson/BPA mesh 경로(open3d method)에만 해당한다.
#   TSDF 경로(reconstruct_tsdf.py)는 voxel 크기만 참조하고 depth/method는 무시한다.
#
# 현재 production 기본 경로는 TSDF(reconstruct_tsdf.py)이며 run_pipeline_all.sh 에서 호출한다.
# PLY→Poisson 경로는 mesh.sh --method=open3d 로 수동 실행 가능 (비교·레거시용).
MESH_DEFAULTS = {
    "depth": 8,             # Poisson octree 깊이 (PLY→Poisson 경로 전용)
    "voxel_size": 0.02,     # 다운샘플링 voxel 크기 (20mm) — 실측(2026-08-12):
                            # Poisson depth=8 octree 해상도(~2.8cm, 방 7m 기준) 이하라
                            # 10mm 입력이 품질 향상 없이 전처리만 5배 느리게 함.
    "method": "poisson",    # poisson | bpa
    "simplify_target": 0.5, # Quadric Decimation 50% 경량화
}

