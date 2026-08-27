# 🛠️ Auto-Mobility 보조 및 레거시 유틸리티 가이드

> **V2 메인 파이프라인 외에 프로젝트에 포함된 실시간 카메라 녹화, 대체 SLAM(ORB/Stella), 다양한 표면 복원(Poisson/BPA/Alpha/CGAL), Isaac Sim 및 진단 도구 가이드**

Auto-Mobility의 기본 프로덕션 워크플로우는 [docs/guide.md](guide.md)의 V2 파이프라인(RTAB-Map & cuVSLAM 중심)을 권장합니다.  
본 문서는 특정 연구 목적, 센서 디버깅, 과거 호환성 또는 대안적 기법 테스트를 위해 프로젝트에 내장된 **보조 스크립트, 레거시 SLAM 백엔드, 특수 표면 복원 기법 및 유틸리티 도구**의 사용법을 정리한 문서입니다.

---

## 📑 목차
1. [실시간 카메라 스트리밍 & 라이브 SLAM](#1-실시간-카메라-스트리밍--라이브-slam)
2. [대체 SLAM 백엔드 (ORB-SLAM3 & Stella-VSLAM)](#2-대체-slam-백엔드-orb-slam3--stella-vslam)
3. [다양한 3D 표면 복원 기법 (Poisson, BPA, Alpha, CGAL)](#3-다양한-3d-표면-복원-기법-poisson-bpa-alpha-cgal)
4. [단독 정량 품질 평가 (Geometry QA)](#4-단독-정량-품질-평가-geometry-qa)
5. [NVIDIA Isaac Sim / Digital Twin 연동 도구](#5-nvidia-isaac-sim--digital-twin-연동-도구)
6. [센서 진단 및 네트워크 유틸리티](#6-센서-진단-및-네트워크-유틸리티)

---

## 1. 실시간 카메라 스트리밍 & 라이브 SLAM

실시간으로 RealSense D435i 카메라 데이터를 구독하며 RViz2 화면에서 맵을 확인하거나 녹화하는 도구들입니다.

### A. 실시간 RTAB-Map SLAM 실행 (`run_live.sh`)
카메라 드라이버와 RTAB-Map SLAM, RViz2를 동시에 실행하여 실시간 점군 맵핑을 수행합니다.
```bash
# 실시간 SLAM 및 RViz2 실행
./scripts/pipeline/run_live.sh

# 또는 ROS2 Launch 직접 실행
ros2 launch auto_mobility rtab_live.launch.py
```

### B. 실시간 센서 스트림 녹화 (`record.sh`)
실시간으로 들어오는 카메라 토픽을 Rosbag (`.mcap`)으로 직접 기록합니다.
```bash
# 기본 압축 녹화
./scripts/pipeline/record.sh my_room_bag

# 무압축 RAW 녹화
./scripts/pipeline/record.sh my_room_bag --raw
```

### C. Rosbag 재생 기반 라이브 SLAM (`run_bag.sh`, `play.sh`)
기존에 녹화된 Rosbag을 재생하면서 RTAB-Map 실시간 노드로 전달하여 궤적 및 DB를 생성합니다.
```bash
# Rosbag 재생 + RTAB-Map SLAM + RViz2 실행
./scripts/pipeline/run_bag.sh hallway

# Rosbag 단독 재생 (주기 반복 재생)
./scripts/pipeline/play.sh hallway
```

---

## 2. 대체 SLAM 백엔드 (ORB-SLAM3 & Stella-VSLAM)

RTAB-Map과 cuVSLAM 외에 특징점 기반 SLAM 알고리즘을 테스트할 수 있는 오프라인 C++ 바이너리 및 래퍼입니다.

### A. ORB-SLAM3 (RGB-D & Visual-Inertial)
* **관련 파일**: `src/auto_mobility/slam/orbslam3_offline.cpp`, `config/orbslam3_rgbd_custom.yaml`, `config/orbslam3_rgbdi_custom.yaml`

```bash
# 1. ORB-SLAM3 RGB-D 실행 (표준 TUM 궤적 파일 생성)
./scripts/pipeline/run_slam.sh hallway --slam=orb_rgbd

# 2. ORB-SLAM3 RGB-D-Inertial 실행 (IMU 융합 모드)
./scripts/pipeline/run_slam.sh hallway --slam=orb_rgbdi
```
* **출력 파일**: `ros2_data/trajectories/orbslam3_hallway_trajectory.txt`

### B. Stella-VSLAM (경량 오프라인 SLAM)
* **관련 파일**: `src/auto_mobility/slam/stella_offline.cpp`, `config/stella_vslam_d435i.yaml`

```bash
# Stella-VSLAM RGB-D 실행
./scripts/pipeline/run_slam.sh hallway --slam=stella_rgbd
```
* **출력 파일**: `ros2_data/trajectories/stella_hallway_trajectory.txt`

---

## 3. 다양한 3D 표면 복원 기법 (Poisson, BPA, Alpha, CGAL)

`mesh.sh` 스크립트를 통해 TSDF 외의 다양한 3D 점군 기반 표면 재구성 알고리즘을 적용하여 메쉬(.obj)를 생성할 수 있습니다.

```bash
# 1. Screened Poisson Surface Reconstruction (완벽한 폐곡면/Watertight 메쉬)
./scripts/pipeline/mesh.sh hallway --surface=poisson --view

# 2. Ball Pivoting Algorithm (BPA) (비다양체 결함 없는 초경량 표면 삼각망)
./scripts/pipeline/mesh.sh hallway --surface=bpa --view

# 3. Alpha Shape (알파 반경 기반 정밀 경계 메쉬)
./scripts/pipeline/mesh.sh hallway --surface=alpha --voxel=0.02 --view

# 4. CGAL Polygonal Surface Reconstruction (실내 평면 기반 Sharp 메쉬)
./scripts/pipeline/mesh.sh hallway --surface=cgal_polygonal --view

# 5. TSDF Direct 초고해상도 (5mm 복셀) 복원
./scripts/pipeline/mesh.sh hallway --surface=tsdf_direct --voxel=0.005 --view

# 6. ORB-SLAM3 궤적 기반 메쉬 복원
./scripts/pipeline/mesh.sh hallway --slam=orb_rgbdi --surface=tsdf_direct --view
```

### 💡 RTAB-Map DB에서 RGB-D 직접 추출기 (`extract_db_rgbd.cpp`)
RTAB-Map SQLite DB 파일(`.db`)에서 압축된 RGB-D 프레임과 포즈를 고속으로 역추출하는 C++ 도구입니다.
```bash
# C++ 추출기 빌드
./scripts/utils/build_extractor.sh

# 3D 점군(.ply) 단독 내보내기
./scripts/utils/export_ply.sh hallway
```

---

## 4. 단독 정량 품질 평가 (Geometry QA)

`evaluate.sh`는 임의의 메쉬 파일과 궤적 파일에 대해 D435i의 Held-out Depth 관측값과의 오차(MAE, RMSE, P95, Coverage)를 독립적으로 측정합니다.

```bash
# 단독 메쉬 평가 실행
./scripts/pipeline/evaluate.sh hallway \
    ros2_data/meshes/hallway_rtab_tsdf.obj \
    ros2_data/trajectories/rtab_hallway_trajectory.txt

# 특정 후보 이름 태그 부여
./scripts/pipeline/evaluate.sh hallway \
    ros2_data/meshes/hallway_custom.obj \
    --name candidate_custom
```
* **결과 산출물**: `ros2_data/evaluations/hallway/<candidate_name>/evaluation_report.md`

---

## 5. NVIDIA Isaac Sim / Digital Twin 연동 도구

생성된 3D 메쉬를 NVIDIA Isaac Sim 및 Isaac Lab 물리 시뮬레이션 환경에 로드하기 위한 변환 및 검증 도구입니다.
* **관련 파일**: `scripts/pipeline/isaac.sh`, `src/auto_mobility/isaac/load_isaac_mesh.py`

```bash
# 메쉬를 Isaac Sim 호환 포맷으로 패키징 및 로더 실행
./scripts/pipeline/isaac.sh output_standard/hallway/latest/final_candidates/cuvslam/model.obj
```

---

## 6. 센서 진단 및 네트워크 유틸리티

카메라 하드웨어 상태, DDS 통신 무결성, 실시간 스트림 품질을 점검하는 스크립트 모음입니다.

| 스크립트 경로 | 용도 및 설명 | 실행 명령어 |
| :--- | :--- | :--- |
| `scripts/utils/check_camera_live.sh` | RealSense 카메라 라이브 토픽 수신 상태 점검 | `./scripts/utils/check_camera_live.sh` |
| `scripts/utils/check_camera_params.py` | 카메라 내부 캘리브레이션 파라미터(K, D, 해상도) 검증 | `python3 scripts/utils/check_camera_params.py` |
| `scripts/utils/view_camera.py` | OpenCV 기반 실시간 RGB / Depth / IR 윈도우 뷰어 | `python3 scripts/utils/view_camera.py` |
| `scripts/utils/restart_camera.sh` | Windows-WSL2 환경에서 카메라 드라이버 재시작 신호 전송 | `./scripts/utils/restart_camera.sh` |
| `scripts/utils/topic_probe.py` | 토픽별 실제 수신 주기(Hz), 프레임 드롭, 패킷 손실 정밀 프로브 | `python3 scripts/utils/topic_probe.py` |
| `scripts/utils/sync_dds_config.py` | CycloneDDS 및 FastDDS XML 환경 설정 동기화 | `python3 scripts/utils/sync_dds_config.py` |
| `scripts/utils/fix_imu_permissions.sh` | Linux/WSL 환경에서 RealSense IMU USB 권한 자동 수정 | `./scripts/utils/fix_imu_permissions.sh` |
| `scripts/utils/pipeline_log_collector.py` | 전체 서브시스템 로그 수집 및 진단 아카이브 생성 | `python3 scripts/utils/pipeline_log_collector.py` |
| `src/auto_mobility/mesh/view_mesh_web.py` | 로컬 웹 브라우저 기반 3D 메쉬 뷰어 서버 실행 | `python3 src/auto_mobility/mesh/view_mesh_web.py <model.obj>` |
