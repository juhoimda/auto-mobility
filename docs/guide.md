# 🛠️ Auto-Mobility 핵심 워크플로우 가이드

Auto-Mobility (Real-to-Sim) 파이프라인의 **단계별 실행 및 검증 중심 가이드**입니다.  
안전한 Raw 데이터 수집부터 SLAM 맵핑, 3D Mesh 복원, 시뮬레이터 검증까지 로직 흐름 순서대로 구성되어 있습니다.

---

## 🧭 파이프라인 전체 흐름도

```text
[1. 데이터셋 캡처] ──▶ [2. 오프라인 SLAM] ──▶ [3. 3D Mesh 복원] ──▶ [4. Isaac Sim 검증]
  capture_safe.sh        run_bag.sh + play.sh      mesh.sh (TSDF/Open3D)     isaac.sh
  (Raw rosbag2 확보)     (DB 및 Odom/맵 생성)      (3D PointCloud & OBJ)     (디지털 트윈 로드)
```

---

## ⚙️ 0. 필수 환경 초기화

실행 전 워크스페이스와 환경변수를 설정합니다.

```bash
# 1. ROS 2 환경 및 워크스페이스 로드
source /opt/ros/humble/setup.bash
source install/setup.bash

# 2. Python 모듈 및 CycloneDDS 도메인 설정 (기본값: 42)
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export ROS_DOMAIN_ID=42
```

---

## 📹 1단계: 안전한 Raw 데이터셋 수집 (Capture-Safe)

SLAM/RViz 부하와 완전히 분리하여 **무손실 Raw 센서 데이터(RGB-D + IMU + TF)**를 안전하게 기록합니다.

```bash
# [기본 실행] 실시간 카메라 프리뷰 창과 함께 캡처
./scripts/pipeline/capture_safe.sh [DATASET_NAME] --view

# [예시] my_dataset 이름으로 녹화 (기본: RGB JPEG + Depth 16bit PNG lossless)
./scripts/pipeline/capture_safe.sh my_dataset --view

# [초경량 뷰어] CPU 부하를 극소화하고 RGB 화각만 확인할 때
./scripts/pipeline/capture_safe.sh my_dataset --rgb-only
```

* **종료 방법**: 터미널에서 `Ctrl+C`를 누르면 자동 데이터 검증 및 SSD 이관이 진행됩니다.
* **출력 결과물**:
  * **Rosbag2**: `ros2_data/bags/[DATASET_NAME]`
  * **데이터셋 매니페스트**: `ros2_data/bags/[DATASET_NAME]/dataset_manifest.json`
  * **수신 품질 보고서**: `ros2_data/logs/capture_safe_[DATASET_NAME].md`

---

## 🗺️ 2단계: 오프라인 RTAB-Map SLAM & Odom/Map 생성

1단계에서 확보한 Bag 데이터를 재생하여 Visual-Inertial Odometry를 추정하고 3D 지도를 빌드합니다.

### 1) 터미널 1: RTAB-Map SLAM 노드 대기 실행
```bash
./scripts/pipeline/run_bag.sh [DATASET_NAME]
```

### 2) 터미널 2: 데이터셋 재생
```bash
# 기본 1.0배속 재생 (정밀 루프 클로저가 필요할 경우 0.5배속 권장)
./scripts/pipeline/play.sh [DATASET_NAME] 1.0
```

### 3) 실시간 Odom 및 PointCloud 토픽 모니터링 (선택)
```bash
# 실시간 로봇/카메라 위치 궤적 추정 확인
ros2 topic echo /rtabmap/odom

# 생성 중인 3D 포인트클라우드 토픽 수신율 확인
ros2 topic hz /rtabmap/cloud_map

# RViz2 3D 시각화 실행 (필요 시)
rviz2
```

* **완료 방법**: 재생이 끝나면 터미널 1에서 `Ctrl+C`를 눌러 DB를 디스크에 안전하게 저장합니다.
* **출력 결과물**:
  * **SLAM 데이터베이스**: `ros2_data/databases/[DATASET_NAME].db`

---

## 🧊 3단계: 3D Pointcloud & Mesh 재구성 및 시각화

생성된 `.db` 파일로부터 고밀도 3D 형상(PointCloud / Mesh)을 추출하고 Open3D 뷰어로 검토합니다.

```bash
# [방법 A: Open3D TSDF 고정밀 복원 (추천 ⭐)] 원본 RGB-D + 최적화 Pose 적분
./scripts/pipeline/mesh.sh [DATASET_NAME].db --method=tsdf --voxel=0.01 --view

# [방법 B: Open3D Poisson 표면 복원] PointCloud 추출 후 메쉬 생성
./scripts/pipeline/mesh.sh [DATASET_NAME].db --method=open3d --depth=8 --view

# [방법 C: RTAB-Map 자체 텍스처 메쉬 추출]
./scripts/pipeline/mesh.sh [DATASET_NAME].db --method=rtabmap
```

* **출력 결과물**:
  * **3D PointCloud**: `ros2_data/pointclouds/[DATASET_NAME]_cloud.ply`
  * **3D Mesh 모델**: `ros2_data/meshes/[DATASET_NAME]_tsdf.obj` (또는 `_mesh.obj`)

### 💡 RTAB-Map 전용 GUI 분석 뷰어
```bash
rtabmap-databaseViewer ~/auto-mobility/ros2_data/databases/[DATASET_NAME].db
```
* 전체 3D 궤적, 키프레임, 루프 클로저 링크, Depth 포인트클라우드를 3D 화면에서 상세 분석할 수 있습니다.

---

## 🤖 4단계: Isaac Sim 디지털 트윈 검증

재구성된 3D Mesh를 NVIDIA Isaac Sim 환경으로 불러와 물리 충돌체 및 시각적 적합성을 검증합니다.

```bash
# 생성된 TSDF Mesh를 Isaac Sim에 로드
./scripts/pipeline/isaac.sh ros2_data/meshes/[DATASET_NAME]_tsdf.obj

# 물리 시뮬레이션 없이 시각적 검증만 수행할 때
./scripts/pipeline/isaac.sh ros2_data/meshes/[DATASET_NAME]_tsdf.obj --no-physics
```

---

## ⚡ 부록: 기타 유용한 실행 모드

### 1. 실시간 Live SLAM (촬영과 동시에 즉시 매핑)
데이터셋 녹화 과정을 거치지 않고 실시간으로 지도를 생성할 때 사용합니다.
```bash
# 원격 카메라(Windows) 기준 실시간 Live SLAM 실행
CAMERA_MODE=remote ./scripts/pipeline/run_live.sh [DB_NAME] true
```

### 2. 원스톱 전체 파이프라인 (Live 캡처 $\rightarrow$ Isaac Sim 일괄 처리)
```bash
./scripts/pipeline/run_pipeline_all.sh --remote-camera
```

### 3. 시스템 진단 및 데이터셋 무결성 검증
```bash
# 시스템 진단 (DDS 통신, USB, QoS 점검)
./scripts/utils/check.sh

# 개별 Bag 파일 수동 검증 & 매니페스트 생성
python3 src/auto_mobility/utils/validate_bag.py ros2_data/bags/[DATASET_NAME]
```
