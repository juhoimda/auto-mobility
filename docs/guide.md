# 🛠️ Auto-Mobility 파이프라인 개발자 가이드 (Developer Guide)

RealSense D435i 센서 수집, IMU 융합 SLAM, 3D Mesh 복원 및 시스템 검증을 위한 **개발자 실행 가이드**입니다.

---

## ⚙️ 0. 환경 구축 및 빌드 (Setup & Build)

```bash
# 1. 워크스페이스 빌드 및 환경 로드
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash

# 2. Python 패키지 경로 등록 (src 모듈 참조)
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

---

## 📸 1. 센서 수집 및 IMU 상태 검증 (Hardware Verification)

개발자는 카메라 단독 노드를 실행하고 **RGB-D 프레임 레이트**와 **IMU(가속도/자이로) 200Hz 데이터** 발행 여부를 직접 확인할 수 있습니다.

```bash
# [실행] RealSense D435i 카메라 노드 기동 (IMU enabled, unite_imu_method=1)
ros2 launch auto_mobility camera.launch.py
```

### 🔍 개발자 확인 항목 (Topic Hz & Output Verification)
```bash
# 센서 토픽 출력 상태 확인
ros2 topic echo /camera/camera/imu --once

# 토픽 발행 주파수 검증 (새 터미널)
ros2 topic hz /camera/camera/color/image_raw \
              /camera/camera/depth/image_rect_raw \
              /camera/camera/imu
```
* **정상 확인 기준**:
  * `/camera/camera/color/image_raw`: **~30 Hz** (640x480 RGB8)
  * `/camera/camera/depth/image_rect_raw`: **~30 Hz** (640x480 Z16)
  * `/camera/camera/imu`: **~200 Hz** (Accel 100Hz + Gyro 200Hz Copy 통합)

---

## 🗺️ 2. 실시간 Visual-Inertial SLAM (Live Mapping)

IMU 필터(`imu_filter_madgwick`)와 RTAB-Map Graph SLAM을 융합하여 3D 지도 데이터베이스(`.db`)를 생성합니다.

```bash
# [방법 1: 원스톱 스크립트 실행] 카메라 자동 감지 및 Live SLAM 구동
./scripts/pipeline/run_live.sh [DB_NAME] [USE_COMPRESSED]

# 예시: my_office.db로 저장
./scripts/pipeline/run_live.sh my_office false
```

```bash
# [방법 2: 노드 개별 실행] 로그 및 디버깅용 (터미널 2개 분리)
# Terminal 1: 카메라 실행
ros2 launch auto_mobility camera.launch.py

# Terminal 2: Live SLAM 실행 (Madgwick IMU Filter + RTAB-Map + RViz2)
ros2 launch auto_mobility rtab_live.launch.py database_path:=./ros2_data/databases/my_office.db
```

### 🔍 개발자 확인 항목
* **RViz2 화면**: 3D Point Cloud 지도가 중력 방향(`GravityProvided`)에 맞게 수평 유지되는지 확인
* **Odometry Hz**: `/rtabmap/odom` 토픽이 **~5 Hz**로 오도메트리 손실 없이 유지되는지 확인

---

## 🎬 3. ROS2 Bag 데이터 녹화 및 오프라인 SLAM (Record & Playback)

센서 데이터를 MCAP 포맷으로 녹화한 후 재생하거나 오프라인 SLAM 맵핑을 수행합니다.

```bash
# [녹화] 센서 데이터(RGB, Depth, IMU, TF) MCAP 녹화 (종료: Ctrl+C)
./scripts/pipeline/record.sh capture_test --compressed

# [재생] MCAP 녹화본 재생 (기본 0.5배속)
./scripts/pipeline/play.sh capture_test 0.5

# [오프라인 SLAM] 녹화본(Bag) 기반 오프라인 RTAB-Map 맵핑
./scripts/pipeline/run_bag.sh capture_test
```

---

## 🧊 4. 3D Digital Twin Mesh 복원 및 검증 (Mesh & Inspection)

SLAM 데이터베이스(`.db`)에서 Point Cloud를 추출하고 Open3D 기반 Surface Reconstruction을 통해 `.obj` 3D 모델을 복원합니다.

```bash
# [원스톱 실행] DB -> PLY 추출 -> 품질 검증 -> Open3D Poisson Mesh 생성 및 뷰어 표시
./scripts/pipeline/mesh.sh my_office my_office_mesh --view

# [단독 뷰어] 생성된 3D Mesh (.obj / .ply) 시각화 뷰어 실행
./scripts/utils/view_mesh.sh my_office_mesh.obj
```

### 🛠️ 개발자 개별 모듈 디버깅 명령
```bash
# 1) DB -> PointCloud (.ply) 수동 추출
./scripts/utils/export_ply.sh my_office.db my_cloud.ply

# 2) 데이터 품질 및 점 밀도 무결성 검증
python3 src/auto_mobility/processing/validate.py --db ./ros2_data/databases/my_office.db --ply ./ros2_data/pointclouds/my_cloud.ply

# 3) Open3D Mesh 복원 스크립트 실행
python3 src/auto_mobility/processing/mesh_open3d.py ./ros2_data/pointclouds/my_cloud.ply ./ros2_data/meshes/my_mesh.obj --view
```

---

## 📊 5. 통합 시스템 진단 및 벤치마크 (Diagnostics & Benchmark)

현재 가상머신/하드웨어 환경의 DDS 통신, 패킷 손실, IMU 반응속도 및 SLAM 파이프라인 성능을 전수 검사합니다.

```bash
# [시스템 종합 헬스 체크] DDS, USB, 시스템 소켓 버퍼 검사
./scripts/utils/check.sh

# [통합 SLAM 파이프라인 벤치마크] Stage 1(센서/DDS) -> Stage 2(SLAM) -> Stage 3(종합진단)
python3 src/auto_mobility/processing/benchmark_slam.py --quick
```

### 🔍 벤치마크 결과 보고서 확인
* 실행 완료 후 [`ros2_data/logs/slam_benchmark_YYYYMMDD_HHMMSS.md`](file:///home/kth/auto-mobility/ros2_data/logs) 파일에서 토픽별 실측 Hz 및 CPU 점유율 보고서를 직접 확인할 수 있습니다.
