# 🛠️ Auto-Mobility 개발자 실행 & 검증 가이드 (Developer Operations Guide)

본 문서는 개발자가 터미널 명령어를 통해 **시스템을 단계별로 실행하고, 정상 작동 여부를 즉시 점검/검증**할 수 있도록 작성된 실전 동작 가이드입니다.

---

## ⚙️ 0. 개발 환경 구축 및 최초 빌드 (Setup & Build)

### 1) 시스템 의존성 및 Python 라이브러리 설치
```bash
# ROS 2 Humble & RealSense / RTAB-Map 의존성 설치
sudo apt update && sudo apt install -y \
    python3-pip python3-colcon-common-extensions \
    ros-humble-realsense2-camera ros-humble-rtabmap-ros \
    ros-humble-rtabmap-launch ros-humble-imu-filter-madgwick \
    ros-humble-rosbag2-storage-mcap

# 3D Mesh 가공 및 데이터 분석용 Python 패키지 설치
pip3 install open3d numpy opencv-python
```

### 2) 네트워크 버퍼 증설 및 ROS 2 워크스페이스 빌드
```bash
# 프레임 손실 방지용 UDP 네트워크 버퍼 증설
sudo sysctl -w net.core.rmem_max=10485760
sudo sysctl -w net.core.wmem_max=10485760

# 워크스페이스 빌드 및 환경 로드
cd ~/ros2_ws/src/auto-mobility  # (프로젝트 경로)
cd ../..
colcon build --symlink-install --packages-select auto_mobility
source install/setup.bash
```

---

## 📸 동작 1. 실시간 카메라 구동 및 센서 토픽 점검

실제 RealSense D435i 카메라를 구동하고 센서 토픽이 정상 발행되는지 확인합니다.

```bash
# [터미널 1] RealSense D435i 카메라 노드 실행
ros2 launch auto_mobility camera.launch.py

# [터미널 2] 종합 토픽 및 시스템 헬스 체크
./scripts/utils/check.sh
```

### 🔍 검증 포인트 (Verification Check)
* `check.sh` 실행 결과 RGB(`/camera/camera/color/image_raw`), Depth(`/camera/camera/aligned_depth_to_color/image_raw`), IMU 토픽이 모두 **`[O]` 정상 (15 Hz 이상)**으로 출력되는지 확인.
* 특정 토픽 발행주기 개별 검증:
  ```bash
  ros2 topic hz /camera/camera/color/image_raw
  ```

---

## 🎬 동작 2. ROS 2 Bag 데이터 녹화 및 재생 (Data Collection)

카메라 노드가 켜진 상태에서 장소 데이터를 녹화하고, 추후 재생하여 매핑에 재사용합니다.

### 1) Bag 데이터 녹화
```bash
# ⚠️ [터미널 1]에서 camera.launch.py가 실행 중인 상태에서 진행
# [터미널 2] 데이터 녹화 시작 (원하는 이름 지정)
./scripts/pipeline/record.sh room_sample_01
```
* 녹화를 마치려면 **`Ctrl + C`**를 누릅니다. (RAM 디스크 버퍼에서 `ros2_data/bags/`로 자동 이관)

### 2) Bag 녹화 상태 검증
```bash
ros2 bag info ./ros2_data/bags/room_sample_01
```
* **검증**: `storage_identifier: mcap` 및 RGB/Depth/IMU 토픽 데이터 개수가 수백~수천 개 쌓여있는지 확인.

### 3) Bag 데이터 재생
```bash
# ⚠️ 중요: camera.launch.py를 반드시 종료(Ctrl+C)한 뒤 재생해야 토픽 충돌이 없습니다!
./scripts/pipeline/play.sh room_sample_01 0.5   # (0.5배속 재생)
```

---

## 🗺️ 동작 3. 3D Visual SLAM 지도 생성 (Mapping)

녹화 데이터(Bag) 또는 실시간 카메라 스트림으로부터 RTAB-Map SLAM 지도를 구축합니다.

### [방법 A] 녹화 데이터(Bag) 기반 SLAM 실행
```bash
# [터미널 1] 녹화본 재생
./scripts/pipeline/play.sh room_sample_01 0.5

# [터미널 2] Bag 전용 RTAB-Map SLAM 실행 (RViz2 시각화 포함)
./scripts/pipeline/run_bag.sh room_sample_01_db
```

### [방법 B] 실시간 카메라 기반 SLAM 실행
```bash
# [터미널 1] 카메라 실행
ros2 launch auto_mobility camera.launch.py

# [터미널 2] 실시간 RTAB-Map SLAM 실행
./scripts/pipeline/run_live.sh my_live_db
```

### 🔍 검증 포인트 (Verification Check)
1. **[RViz2 3D 지도]**: 3D Point Cloud 지도가 화면에 잔상 없이 공간 모양으로 누적되는지 확인.
2. **[카메라 궤적]**: `camera_link` TF 좌표축이 빨간 에러 없이 부드럽게 궤적을 그리는지 확인.
3. **[DB 파일 저장]**: SLAM 종료(`Ctrl+C`) 후 DB 용량이 수십 MB 이상 존재하는지 확인:
   ```bash
   ls -lh ros2_data/databases/room_sample_01_db.db
   ```

---

## 🧊 동작 4. Open3D 3D Mesh 모델(.obj) 복원 및 3D 뷰어 검증

SLAM으로 생성된 `.db` 파일에서 3D Point Cloud를 추출하고, Open3D Poisson Reconstruction 및 **RGB Color Transfer**를 적용하여 3D Mesh 모델을 생성합니다.

### 1) 파이프라인 일괄 실행 (추천)
```bash
# DB에서 Point Cloud 추출 -> 자동 데이터 검증 -> Open3D Mesh 생성 -> 3D 뷰어 팝업
./scripts/pipeline/mesh.sh room_sample_01_db --view
```
* 데이터에 경고가 있더라도 강제로 생성할 때: `--force` 옵션 추가
* RTAB-Map 텍스처 맵핑 방식 사용 시: `--method rtabmap` 옵션 추가

### 2) 단계별 수동 디버깅 실행 (개발자 개별 점검용)
```bash
# Step 1: DB에서 Point Cloud (.ply) 추출
./scripts/utils/export_ply.sh room_sample_01_db.db room_sample_cloud.ply

# Step 2: Point Cloud & DB 무결성/품질 자동 검증
python3 src/auto_mobility/processing/validate.py --db ./ros2_data/databases/room_sample_01_db.db --ply ./ros2_data/pointclouds/room_sample_cloud.ply

# Step 3: Open3D Poisson Reconstruction & RGB Color Transfer 실행
python3 src/auto_mobility/processing/mesh_open3d.py ./ros2_data/pointclouds/room_sample_cloud.ply ./ros2_data/meshes/room_sample_mesh.obj --view
```

### 🔍 검증 포인트 (Verification Check)
1. **[품질 검증 통과]**: `validate.py` 실행 시 포인트 수(>10,000개), 공간 스케일, 노이즈 비율 검증 통과 확인.
2. **[3D Mesh 뷰어 창]**: Open3D 대화형 창에서 마우스 드래그로 회전/확대 시 실사 RGB 색상이 입혀진 3D 입체 형상이 깨끗하게 노출되는지 확인.
3. **[최종 결과물 확인]**:
   ```bash
   ls -lh ros2_data/meshes/room_sample_01_db_mesh.obj
   ```

---

## ⚡ 동작 5. 하드웨어 성능 및 DDS/QoS 자동 벤치마크

현재 시스템(CPU/RAM/네트워크) 환경에서 가장 프레임 손실이 적고 지연이 적은 DDS 미들웨어 및 QoS 설정 조합을 자동 측정합니다.

```bash
# 정밀 벤치마크 실행 (DDS x QoS x 해상도 x FPS 조합 테스트)
./scripts/utils/benchmark.sh

# 빠른 벤치마크 실행 (주요 조합만 빠른 테스트)
./scripts/utils/benchmark.sh --quick
```

### 🔍 검증 포인트 (Verification Check)
* `ros2_data/logs/sensor_stage1_benchmark_<타임스탬프>.md` 보고서가 자동 생성되었는지 확인하고, 가장 높은 점수를 받은 DDS 및 QoS 추천 설정값 확인.
