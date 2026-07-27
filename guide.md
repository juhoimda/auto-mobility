# 🚀 Digital Twin 자율주행 프로젝트 사용 가이드 (Guide)

본 프로젝트는 **Intel RealSense D435i** 카메라를 이용하여 실내외 환경 데이터를 취득하고, **RTAB-Map Visual SLAM** 및 **Open3D 3D Reconstruction** 파이프라인을 통해 디지털 트윈(Digital Twin) 및 Isaac Sim 호환 환경을 구축하는 통합 ROS2 패키지입니다.

---

## 📌 1. 데이터 저장 디렉토리 구조 (프로젝트 내부)

스크립트 실행 시 생성되는 촬영 데이터, 데이터베이스, 3D 파일 등은 프로젝트 루트 경로 내부의 `./ros2_data/` 디렉토리에 자동으로 생성되고 저장됩니다.

| 경로 | 저장 내용 |
| :--- | :--- |
| `./ros2_data/bags/` | 취득된 센서 데이터 (rosbag2 폴더) |
| `./ros2_data/databases/` | RTAB-Map SLAM 결과 파일 (`*.db`) |
| `./ros2_data/pointclouds/` | 추출된 Point Cloud 파일 (`*.ply`, `*.pcd`) |
| `./ros2_data/meshes/` | Open3D 필터링 및 3D Mesh 결과 파일 (`*.obj`, `*.ply`) |
| `./ros2_data/isaac_sim/` | Isaac Sim 임포트용 USD 자산 파일 |
| `./ros2_data/logs/` | 실행 로그 파일 |

---

## 🛠️ 2. 빌드 및 최초 설정

프로젝트 루트 디렉토리(`auto-mobility`)에서 패키지를 빌드하고 환경 변수를 로드합니다.

```bash
# 프로젝트 디렉토리 이동
cd /path/to/auto-mobility

# 패키지 빌드 (프로젝트 자체 워크스페이스 빌드)
colcon build --packages-select auto_mobility

# 환경 변수 반영
source install/setup.bash
```

---

## 📜 3. 스크립트 & Launch 파일 상세 기능 안내

### 1) 토픽 상태 점검 (`check.sh`)
* **담당 역할**: 카메라 및 센서 토픽(RGB, Depth, CameraInfo, PointCloud, IMU, TF)이 정상적으로 발행되고 있는지 확인합니다.
* **실행 명령어**:
  ```bash
  ./scripts/check.sh
  # 또는
  ros2 run auto_mobility check.sh
  ```

---

### 2) RealSense D435i 카메라 실행 (`camera.launch.py`)
* **담당 역할**: RGB, Aligned Depth, PointCloud, IMU(가속도계/자이로) 동기화 스트리밍을 시작합니다.
* **실행 명령어**:
  ```bash
  ros2 launch auto_mobility camera.launch.py
  ```

---

### 3) rosbag2 센서 데이터 녹화 (`record.sh`)
* **담당 역할**: 카메라 센서 데이터(RGB, Depth, IMU, PointCloud) 및 TF 트랜스폼 정보를 rosbag2로 녹화합니다.
* **실행 명령어**:
  ```bash
  # 기본 이름(capture_날짜_시간)으로 녹화
  ./scripts/record.sh

  # 특정 이름으로 녹화 (예: room1)
  ./scripts/record.sh room1
  ```
* **저장 위치**: `./ros2_data/bags/room1`

---

### 4) rosbag2 데이터 반복 재생 (`play.sh`)
* **담당 역할**: 녹화한 센서 데이터를 `--loop --clock` 옵션으로 반복 재생합니다.
* **실행 명령어**:
  ```bash
  ./scripts/play.sh BAG_NAME

  # 예시
  ./scripts/play.sh room1
  ```

---

### 5) 실시간 Visual SLAM 실행 (`run_live.sh` / `rtab_live.launch.py`)
* **담당 역할**: 실시간 카메라 센서 입력으로 RTAB-Map SLAM 및 RViz 시각화를 수행하고 지도 DB를 생성합니다.
* **실행 명령어**:
  ```bash
  # DB 이름을 지정하여 실행
  ./scripts/run_live.sh room1_live
  ```
* **저장 위치**: `./ros2_data/databases/room1_live.db`

---

### 6) Bag 재생 기반 Visual SLAM 실행 (`run_bag.sh` / `rtab_bag.launch.py`)
* **담당 역할**: `play.sh`로 재생 중인 rosbag2 데이터 기반으로 `use_sim_time:=true` 설정으로 SLAM을 수행합니다.
* **실행 명령어**:
  ```bash
  ./scripts/run_bag.sh room1_result
  ```
* **저장 위치**: `./ros2_data/databases/room1_result.db`

---

### 7) Point Cloud 추출 (`export_pointcloud.sh`)
* **담당 역할**: 맵핑이 완료된 `rtabmap.db` 파일에서 3D Point Cloud(`.ply`)를 추출합니다.
* **실행 명령어**:
  ```bash
  # 사용법: export_pointcloud.sh <DB_파일명> [출력_파일명.ply]
  ./scripts/export_pointcloud.sh room1_result.db room1_cloud.ply
  ```
* **저장 위치**: `./ros2_data/pointclouds/room1_cloud.ply`

---

### 8) Open3D 기반 Mesh 생성 (`process_mesh_open3d.py`)
* **담당 역할**: 추출된 Point Cloud의 노이즈/아웃라이어를 제거하고 Poisson Surface Reconstruction 알고리즘으로 3D Mesh(`.obj`)를 생성합니다.
* **실행 명령어**:
  ```bash
  python3 scripts/process_mesh_open3d.py ./ros2_data/pointclouds/room1_cloud.ply ./ros2_data/meshes/room1_mesh.obj
  ```
* **저장 위치**: `./ros2_data/meshes/room1_mesh.obj`

---

## 🔄 4. 표준 작업 프로세스 (Step-by-Step Workflow)

### 🔴 Step 1. 실환경 데이터 수집 (Data Collection)
```bash
# [터미널 1] RealSense 카메라 실행
ros2 launch auto_mobility camera.launch.py

# [터미널 2] 토픽 상태 확인
./scripts/check.sh

# [터미널 3] rosbag 녹화 시작 (이동하며 촬영 후 Ctrl+C 종료)
./scripts/record.sh my_room_01
```

### 🔵 Step 2. 데이터 재생 및 3D SLAM (Offline Processing)
```bash
# [터미널 1] 녹화된 bag 반복 재생
./scripts/play.sh my_room_01

# [터미널 2] SLAM 실행 (맵핑 완료 후 Ctrl+C 종료)
./scripts/run_bag.sh my_room_01_db
```

### 🟢 Step 3. Digital Twin용 3D Mesh 생성 (Mesh Reconstruction)
```bash
# 1. DB에서 PointCloud 추출
./scripts/export_pointcloud.sh my_room_01_db.db my_room_01_cloud.ply

# 2. Open3D 기반 Mesh 정제 및 3D 모델(.obj) 생성
python3 scripts/process_mesh_open3d.py ./ros2_data/pointclouds/my_room_01_cloud.ply ./ros2_data/meshes/my_room_01_mesh.obj
```

---

## 💡 유지보수 팁
* **독립적 실행 환경**: 데이터 저장소(`ros2_data/`) 및 실행 경로가 현재 프로젝트 디렉터리를 기준으로 작동하므로 프로젝트 이동 시에도 독립적으로 유지보수 가능합니다.
* **토픽 및 경로 변경**: 센서 토픽 이름이나 세부 설정을 변경하고 싶다면 `scripts/common.sh` 및 `config/topics.yaml` 한 곳만 수정하시면 전체 스크립트에 일괄 적용됩니다.
