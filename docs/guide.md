# 🛠️ Auto-Mobility 핵심 워크플로우 가이드

Auto-Mobility (Real-to-Sim) 파이프라인의 **단계별 실행 및 3D 복원 가이드**입니다.  
안전한 Raw 데이터 수집부터 Multi-SLAM 맵핑, 3D Point Cloud 및 Mesh 복원, 시뮬레이터 검증까지 순서대로 구성되어 있습니다.

---

## 🧭 파이프라인 전체 흐름도

```text
[1. 데이터셋 캡처] ──▶ [2. Multi-SLAM 궤적] ──▶ [3. 3D 메쉬/점군 복원] ──▶ [4. Isaac Sim 검증]
  capture_safe.sh        run_slam.sh              mesh.sh (TSDF)           isaac.sh
  (Raw rosbag2 확보)     (DB 및 궤적.txt 생성)     (.obj / .ply 동시 생성)  (디지털 트윈 로드)
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
# [권장] 저지연 RGB 실시간 프리뷰 창과 함께 캡처
./scripts/pipeline/capture_safe.sh [DATASET_NAME] --view

# [헤드리스 모드] 프리뷰 창 없이 백그라운드 무부하 수집
./scripts/pipeline/capture_safe.sh [DATASET_NAME]
```

* **종료 방법**: 터미널에서 `Ctrl+C`를 누르면 자동 데이터 검증 및 저장이 완료됩니다.
* **출력 결과물**:
  * **Rosbag2**: `ros2_data/bags/[DATASET_NAME]/` (`.mcap`)
  * **데이터셋 매니페스트**: `ros2_data/bags/[DATASET_NAME]/dataset_manifest.json`
  * **수신 품질 보고서**: `ros2_data/logs/capture_safe_[DATASET_NAME].md`

---

## 🗺️ 2단계: 오프라인 Multi-SLAM & 궤적(Trajectory) 추출

수집된 Bag 데이터를 재생하여 카메라 궤적 및 SLAM 데이터베이스를 생성합니다.

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

## ☁️ 3단계: 3D Point Cloud & Mesh 재구성 및 시각화

TSDF 알고리즘으로 3D Mesh(.obj) 및 Point Cloud(.ply)를 동시 생성하고 뷰어로 확인합니다.

```bash
# 1. 3D Mesh 및 Point Cloud 생성 (기본: RTAB-Map 10mm TSDF)
./scripts/pipeline/mesh.sh [DATASET_NAME] --voxel=0.01

# (선택) ORB-SLAM3 궤적 기반으로 복원할 때
./scripts/pipeline/mesh.sh [DATASET_NAME] --slam=orb

# 2. 결과물 확인 (메쉬 / 포인트 클라우드 뷰어)
./scripts/utils/view_mesh.sh [DATASET_NAME]_rtab_tsdf.obj
./scripts/utils/view_pointcloud.sh [DATASET_NAME]_rtab_tsdf_cloud.ply
```

* **출력 결과물**:
  * **RTAB-Map 기반**: `ros2_data/meshes/[DATASET_NAME]_rtab_tsdf.obj`, `ros2_data/pointclouds/[DATASET_NAME]_rtab_tsdf_cloud.ply`
  * **ORB-SLAM3 기반**: `ros2_data/meshes/[DATASET_NAME]_orbslam_tsdf.obj`, `ros2_data/pointclouds/[DATASET_NAME]_orbslam_tsdf_cloud.ply`

---

## 🤖 4단계: Isaac Sim 디지털 트윈 검증

생성된 3D Mesh를 NVIDIA Isaac Sim 환경으로 불러와 물리 충돌체 및 시각적 적합성을 검증합니다.

```bash
# 생성된 TSDF Mesh를 Isaac Sim에 로드
./scripts/pipeline/isaac.sh ros2_data/meshes/[DATASET_NAME]_rtab_tsdf.obj

# 물리 시뮬레이션 없이 시각적 검증만 수행할 때
./scripts/pipeline/isaac.sh ros2_data/meshes/[DATASET_NAME]_rtab_tsdf.obj --no-physics
```

---

## 📊 5단계: 다중 알고리즘 일괄 벤치마크 (선택/심화)

동일한 Rosbag에 대해 **RTAB-Map(Global Opt / Raw), ORB-SLAM3, Fine TSDF(5mm)**를 일괄 실행하여 궤적, 점군, 메쉬 품질을 자동 비교 분석합니다.

```bash
# 일괄 비교 벤치마크 실행
python3 src/auto_mobility/slam/compare_algorithms.py [DATASET_NAME]
```

* **출력 결과물**:
  * **비교 리포트**: `ros2_data/benchmarks/bench_[DATASET_NAME]_[DATE]/benchmark_report.md`
  * **정량 수치 JSON**: `ros2_data/benchmarks/bench_[DATASET_NAME]_[DATE]/benchmark_summary.json`

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
