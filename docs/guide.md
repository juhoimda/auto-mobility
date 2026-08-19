# 🛠️ Auto-Mobility 핵심 워크플로우 가이드

Auto-Mobility (Real-to-Sim) 파이프라인의 **단계별 실행, 데이터 구조 및 정량 형상 품질(Geometry Quality) 평가 가이드**입니다.  
안전한 Raw 데이터 수집부터 알고리즘 독립적인 Canonical Dataset 생성, Multi-SLAM 궤적 추적, 3D Mesh 복원, Held-out 센서 관측 데이터 기반 정량 평가 및 후보 랭킹까지 전 과정을 안내합니다.

---

## 🧭 1. 파이프라인 계층 구조 (Layered Architecture)

모든 단계는 이전 특정 알고리즘의 파일 형식에 종속되지 않고, 독립적인 표준 포맷으로 소통합니다.

```text
Windows D435i
      ↓
ROS2 Topics (RGB-D + Stereo IR + IMU + CameraInfo)
      ↓
Rosbag / MCAP (불변 원본 데이터셋, Single Source of Truth)
      ↓
Canonical Frame Dataset (알고리즘 독립 표준 RGB-D + 실제 CameraInfo + IMU)
      ↓
SLAM Backend (RTAB-Map / ORB-SLAM3)
      ↓
Trajectory (TUM 표준 포맷: timestamp tx ty tz qx qy qz qw)
      ↓
Reconstruction Backend (Open3D Tensor TSDF / Poisson / BPA)
      ↓
Point Cloud / Mesh (.ply / .obj)
      ↓
Geometry Quality Evaluator (Held-out 20% Depth Reprojection & Point-to-Mesh)
      ↓
QualityProfile / Evaluation Report / Ranking (PASS / WARN / FAIL)
```

### 🔑 핵심 개념 정의
1. **Rosbag (`bags/`)**: 불변(Immutable) 원본 센서 스트림. 모든 알고리즘의 단일 진실 소스(Single Source of Truth).
2. **Canonical Frame Dataset (`frames/`)**: Rosbag에서 1회 추출되어 알고리즘 독립적으로 재사용되는 표준 RGB-D + 실제 CameraInfo + IMU 데이터셋.
3. **Trajectory (`trajectories/`)**: SLAM 백엔드가 추정한 카메라 6자유도 이동 궤적 (TUM 포맷: timestamp tx ty tz qx qy qz qw).
4. **Reconstruction (`meshes/`, `pointclouds/`)**: FrameDataset과 Trajectory를 결합하여 생성된 3D 기하 형상.
5. **Geometry Evaluation (`evaluations/`)**: Reconstruction에 사용되지 않은 Held-out 센서 데이터를 기준으로 기하학적 정합 오차(Depth MAE/P95, Coverage, Point-to-Mesh)를 객관적으로 측정한 정량 평가.

---

## ⚙️ 2. 필수 환경 초기화

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

## 📹 3. 표준 워크플로우 (Step-by-Step)

### 1단계: 안전한 Raw 데이터셋 수집 (Capture-Safe)
SLAM/RViz 부하와 완전히 분리하여 무손실 Raw 센서 데이터를 안전하게 기록합니다.
```bash
./scripts/pipeline/capture_safe.sh room01 --view
```
* **출력**: `ros2_data/bags/room01/` (`.mcap`)

---

### 2단계: 알고리즘 독립 Canonical Dataset 생성
Rosbag에서 표준 프레임과 실제 CameraInfo를 1회 추출합니다. (한 번 생성 후 모든 SLAM/Reconstruction에서 재사용)
```bash
./scripts/pipeline/prepare_dataset.sh room01
```
* **출력**: `ros2_data/frames/room01/`
  * `rgb/`: `000000.png`, `000001.png`, ...
  * `depth/`: `000000.png`, `000001.png`, ... (16-bit uint16 mm)
  * `frames.csv`: 프레임 타임스탬프, 매칭 delta, 해상도
  * `camera_info.json`: D435i 실제 Intrinsics (K, R, P, 왜곡 계수)
  * `imu.csv`: 각속도 및 선가속도
  * `dataset_info.json`: 프레임 수, Hz, 동기화 품질 통계

---

### 3단계: Multi-SLAM 실행 및 Trajectory 추출
```bash
# RTAB-Map SLAM 실행 및 DB/궤적 생성
./scripts/pipeline/run_slam.sh room01 --slam=rtab

# (선택) ORB-SLAM3 실행 및 궤적 생성
./scripts/pipeline/run_slam.sh room01 --slam=orb
```
* **출력**:
  * `ros2_data/trajectories/rtab_room01_trajectory.txt`
  * `ros2_data/trajectories/orbslam3_room01_trajectory.txt`

---

### 4단계: 3D Mesh & Point Cloud 재구성 (TSDF / Poisson / BPA)
Canonical Dataset과 SLAM Trajectory를 결합하여 3D Mesh를 생성합니다.
```bash
# 기본: RTAB-Map Trajectory + 10mm TSDF 복원
./scripts/pipeline/mesh.sh room01 --voxel=0.01

# ORB-SLAM3 Trajectory + 10mm TSDF 복원
./scripts/pipeline/mesh.sh room01 --slam=orb --voxel=0.01

# 5mm 초고정밀 TSDF 복원
./scripts/pipeline/mesh.sh room01 --fine
```
* **출력**:
  * `ros2_data/meshes/room01_rtab_tsdf.obj`
  * `ros2_data/pointclouds/room01_rtab_tsdf_cloud.ply`

---

### 5단계: Held-out 센서 데이터 기반 자동 Geometry QA & 평가
생성된 Mesh에 대해 사용되지 않은 20% Held-out 센서 뷰포인트에서 가상 Raycasting 렌더링을 수행하여 오차를 정량 측정합니다.
```bash
./scripts/pipeline/evaluate.sh room01 \
    ros2_data/meshes/room01_rtab_tsdf.obj \
    ros2_data/trajectories/rtab_room01_trajectory.txt \
    --name rtab_tsdf_10mm
```
* **출력 (`ros2_data/evaluations/room01/rtab_tsdf_10mm/`)**:
  * `evaluation_summary.json`: Depth MAE, P95, Point-to-Mesh, Coverage, Topology 수치
  * `evaluation_report.md`: PASS/WARN/FAIL 종합 품질 보고서
  * `frame_metrics.csv`: 프레임별 오차 통계
  * `pose_association.csv`: SLERP 보간 및 타임스탬프 매칭 로그
  * `split.json`: 재현 가능한 Train/Holdout 프레임 목록
  * `renders/`: 대표 프레임 실제 Depth, 메쉬 렌더 Depth, 오차 Heatmap 이미지

---

### 6단계: 동일 Dataset 내 다중 후보 자동 비교 & 랭킹
동일 데이터셋에서 생성된 여러 reconstruction 결과들을 공정하게 정규화하여 가중 순위를 산출합니다.
```bash
python3 -m auto_mobility.evaluation.compare_results room01
```
* **출력**:
  * 1위~N위 종합 순위표, Depth MAE/P95, 커버리지, 아티팩트 점수 및 선정 근거 출력.

---

## 📊 4. 핵심 평가 지표 (Evaluation Metrics)

| 지표명 | 단위 | 설명 및 해석 기준 |
| :--- | :---: | :--- |
| **Depth MAE** | mm | Held-out 뷰포인트 실제 Depth와 메쉬 렌더 Depth 간 평균 절대 오차 (낮을수록 우수) |
| **Depth P95** | mm | 95% 신뢰 구간 최대 오차. 국소적 왜곡, 벽 벌어짐 감지 (낮을수록 우수) |
| **Depth Coverage** | % | 실제 센서 유효 관측 영역 중 메쉬 표면이 재구성된 비율 (높을수록 우수) |
| **Within 20mm Ratio** | % | 실제 관측과 2cm 이내로 완벽 정합된 픽셀 비율 (높을수록 우수) |
| **Point-to-Mesh P95** | mm | 센서 3D 포인트에서 메쉬 표면까지의 최단거리 95 백분위수 (낮을수록 우수) |
| **Plane Residual Mean** | mm | RANSAC 실내 주요 벽/바닥 평면의 굴곡 및 잔차 (낮을수록 우수) |
| **Degenerate Ratio** | % | 찌그러진 무효 삼각면 비율 (낮을수록 우수) |
| **Small Shell Area** | % | 전체 메쉬 대비 분리된 미세 부유 아티팩트 면적 비율 (낮을수록 우수) |
