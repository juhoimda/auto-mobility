# 🛠️ Auto-Mobility 핵심 워크플로우 가이드

Auto-Mobility (Real-to-Sim) 파이프라인의 **단계별 실행, 점군 분석, 다중 알고리즘 벤치마크 가이드**입니다.  
안전한 Raw 데이터 수집부터 Multi-SLAM 맵핑, 3D Point Cloud 및 Mesh 복원, 시뮬레이터 검증까지 로직 흐름 순서대로 구성되어 있습니다.

---

## 🧭 파이프라인 전체 흐름도

```text
[1. 데이터셋 캡처] ──▶ [2. Multi-SLAM 궤적] ──▶ [3. 3D Point Cloud] ──▶ [4. 3D Mesh 복원] ──▶ [5. Isaac Sim 검증]
  capture_safe.sh        RTAB / ORB-SLAM3          view_pointcloud.sh       reconstruct_tsdf.py       isaac.sh
  (Raw rosbag2 확보)     (DB 및 궤적.txt 생성)     (.ply 노이즈/정합 분석)   (GPU TSDF .obj 생성)     (디지털 트윈 로드)
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

SLAM/RViz 부하와 완전히 분리하여 **무손실 Raw 센서 데이터(RGB-D + Stereo IR + IMU + TF)**를 안전하게 기록합니다.

```bash
# [기본 실행] 실시간 카메라 프리뷰 창과 함께 캡처
./scripts/pipeline/capture_safe.sh [DATASET_NAME] --view

# [예시] my_dataset 이름으로 녹화 (RGB JPEG + Depth 16bit PNG + Stereo IR)
./scripts/pipeline/capture_safe.sh my_dataset --view

# [초경량 모드] CPU 부하를 최소화하고 RGB 화각만 확인할 때
./scripts/pipeline/capture_safe.sh my_dataset --rgb-only
```

* **종료 방법**: 터미널에서 `Ctrl+C`를 누르면 자동 데이터 검증 및 SSD 이관이 진행됩니다.
* **출력 결과물**:
  * **Rosbag2**: `ros2_data/bags/[DATASET_NAME]/` (`.mcap` 포맷)
  * **데이터셋 매니페스트**: `ros2_data/bags/[DATASET_NAME]/dataset_manifest.json`
  * **수신 품질 보고서**: `ros2_data/logs/capture_safe_[DATASET_NAME].md`

---

## 🗺️ 2단계: 오프라인 Multi-SLAM & 궤적(Trajectory) 추출

수집된 Bag 데이터를 재생하여 카메라 궤적 및 SLAM 데이터베이스를 생성합니다.

### 1) RTAB-Map 또는 ORB-SLAM3 단일 명령 실행 (1-명령어)
```bash
# RTAB-Map SLAM 실행 및 DB/궤적 생성
./scripts/pipeline/run_slam.sh [DATASET_NAME] --slam=rtab

# ORB-SLAM3 실행 및 궤적 생성
./scripts/pipeline/run_slam.sh [DATASET_NAME] --slam=orb
```
* **출력 결과물**: 
  * RTAB-Map: `ros2_data/databases/[DATASET_NAME].db`, `ros2_data/trajectories/rtab_[DATASET_NAME]_trajectory.txt`
  * ORB-SLAM3: `ros2_data/trajectories/orbslam3_[DATASET_NAME]_trajectory.txt`

---

## 📊 3단계: 다중 SLAM & 3D 복원 알고리즘 일괄 벤치마크 (추천 ⭐)

동일한 Rosbag에 대해 **RTAB-Map(Global Opt / Raw), ORB-SLAM3, Fine TSDF(5mm)**를 일괄 실행하여 궤적, 3D 점군, 메쉬 품질을 자동 비교 분석합니다.

```bash
# 일괄 비교 벤치마크 실행
python3 src/auto_mobility/slam/compare_algorithms.py [DATASET_NAME]
```

* **출력 결과물**:
  * **비교 리포트**: `ros2_data/benchmarks/bench_[DATASET_NAME]_[DATE]/benchmark_report.md`
  * **정량 수치 JSON**: `ros2_data/benchmarks/bench_[DATASET_NAME]_[DATE]/benchmark_summary.json`
  * **알고리즘별 아티팩트**: 각 알고리즘별 `.txt`(궤적), `.ply`(점군), `.obj`(메쉬) 저장

---

## ☁️ 4단계: 3D Point Cloud & Mesh 재구성 및 시각화

### 1) 3D Point Cloud (.ply) 및 Mesh (.obj) 생성
```bash
# Open3D GPU TSDF (10mm 복셀: 점군 + 메쉬 동시 생성)
python3 src/auto_mobility/mesh/reconstruct_tsdf.py ros2_data/databases/[DATASET_NAME].db ros2_data/meshes/[DATASET_NAME]_tsdf.obj

# 또는 파이프라인 쉘 스크립트 이용
./scripts/pipeline/mesh.sh [DATASET_NAME].db --method=tsdf --voxel=0.01
```

* **출력 결과물**:
  * **3D Point Cloud**: `ros2_data/pointclouds/[DATASET_NAME]_tsdf_cloud.ply`
  * **3D Surface Mesh**: `ros2_data/meshes/[DATASET_NAME]_tsdf.obj`

### 2) 전용 뷰어로 3D 형상 확인

```bash
# 🔹 3D Point Cloud 전용 뷰어 실행 (.ply)
./scripts/utils/view_pointcloud.sh [DATASET_NAME]_tsdf_cloud.ply --point-size=3

# 🔹 3D Surface Mesh 전용 뷰어 실행 (.obj)
./scripts/utils/view_mesh.sh [DATASET_NAME]_tsdf.obj

# 🔹 메쉬 와이어프레임(Wireframe) 구조 확인
./scripts/utils/view_mesh.sh [DATASET_NAME]_tsdf.obj --wireframe
```

> **💡 Windows 탐색기에서 파일 바로 열기**:
> ```bash
> /mnt/c/Windows/explorer.exe ros2_data/pointclouds
> /mnt/c/Windows/explorer.exe ros2_data/meshes
> ```

---

## 🤖 5단계: Isaac Sim 디지털 트윈 검증

생성된 3D Mesh를 NVIDIA Isaac Sim 환경으로 불러와 물리 충돌체 및 시각적 적합성을 검증합니다.

```bash
# 생성된 TSDF Mesh를 Isaac Sim에 로드
./scripts/pipeline/isaac.sh ros2_data/meshes/[DATASET_NAME]_tsdf.obj

# 물리 시뮬레이션 없이 시각적 검증만 수행할 때
./scripts/pipeline/isaac.sh ros2_data/meshes/[DATASET_NAME]_tsdf.obj --no-physics
```

---

## ⚡ 부록: 유용한 진단 및 분석 도구

### 1. RTAB-Map 전용 GUI 분석 뷰어
```bash
rtabmap-databaseViewer ~/auto-mobility/ros2_data/databases/[DATASET_NAME].db
```

### 2. 시스템 진단 및 데이터셋 무결성 검증
```bash
# 시스템 진단 (DDS 통신, USB, QoS 점검)
./scripts/utils/check.sh

# 개별 Bag 파일 수동 검증 & 매니페스트 생성
python3 src/auto_mobility/utils/validate_bag.py ros2_data/bags/[DATASET_NAME]
```
