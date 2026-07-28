# 📌 Digital Twin 프로젝트 실행 및 검증 가이드 (Quick Guide)

## ⚡ ⚡ [핵심 요약] 카메라 녹화 ➔ 재생 ➔ RTAB-Map 3D 맵핑 3단계 Quick Flow

> ⚠️ **핵심 원칙**: 녹화 파일을 재생할 때는 **실시간 카메라 노드(`camera.launch.py`)를 반드시 종료(`Ctrl+C`)**해야 충돌하지 않습니다.

```bash
# 1단계: 실시간 카메라 실행 & Raw 데이터 녹화 (RAM 초고속 저장)
# [터미널 1]
ros2 launch auto_mobility camera.launch.py
# [터미널 2]
./scripts/record.sh room_301_raw      # (촬영 후 Ctrl+C로 종료)

# 2단계: 카메라 종료 후 녹화 데이터 재생 & RTAB-Map SLAM 실행
# ⚠️ [터미널 1]의 camera.launch.py를 Ctrl+C로 끈 후 진행하세요!
# [터미널 1]
./scripts/play.sh room_301_raw
# [터미널 2]
./scripts/run_bag.sh room_301_db       # (RViz2 자동 통합 실행)

# 3단계: SLAM 정밀 검증 (무엇을 보고 판단하는가?)
# ✅ [RViz2 화면] 3D Point Cloud 지도가 잔상처럼 깔끔하게 공간 모양으로 누적되는가?
# ✅ [RViz2 화면] camera_link (좌표축) 위치가 빨간 에러 없이 궤적을 그리며 이동하는가?
# ✅ [터미널 로그] 'Visual Odometry Lost!' 경고가 도배되지 않는가?
# ✅ [DB 파일] ls -lh ros2_data/databases/room_301_db.db 용량이 수십 MB 이상 증가했는가?
```

---

## ⚙️ 0. 최초 1회 환경 설정 (프로젝트 시작 시 1회만 실행)

```bash
# 1. 네트워크 버퍼 증설 (프레임 유실 방지)
sudo sysctl -w net.core.rmem_max=10485760
sudo sysctl -w net.core.wmem_max=10485760

# 2. 심볼릭 링크 빌드 (앞으로 재빌드 불필요)
cd ~/auto-mobility
colcon build --symlink-install --packages-select auto_mobility
source install/setup.bash
```

---

## 🎬 동작 1. 센서 데이터 녹화하기 (Raw RGB + Depth + IMU 수집)

실제 장소에서 카메라로 환경 데이터를 촬영하고 저장할 때 사용합니다. (RAM 디스크를 초고속 버퍼로 활용하여 프레임 손실 0%)

```bash
# [터미널 1] RealSense 카메라 실행
ros2 launch auto_mobility camera.launch.py

# [터미널 2] 데이터 녹화 시작 (원하는_저장이름 지정)
./scripts/record.sh my_room_01
```
* **촬영 팁**: 카메라를 들고 천천히 60~120초 동안 공간을 둘러본 뒤 **`Ctrl + C`**를 눌러 종료합니다.
* **녹화 검증**:
  ```bash
  ros2 bag info ./ros2_data/bags/my_room_01
  ```
  *(Topic 목록에 `/camera/camera/color/image_raw`가 존재하고 FPS가 20~30 수준이면 정상)*

---

## 🗺️ 동작 2. 녹화본 데이터로 3D Visual SLAM 지도 만들기

수집한 녹화 데이터 기반으로 3D 공간 지도 및 이동 경로를 복원할 때 사용합니다.

```bash
# ⚠️ 중요: 기존 camera.launch.py를 먼저 종료(Ctrl+C)하세요!

# [터미널 1] 녹화 데이터 반복 재생
./scripts/play.sh my_room_01

# [터미널 2] RTAB-Map 3D Visual SLAM 실행 (RViz2 자동 구동)
./scripts/run_bag.sh my_room_01_db
```
* **확인 방법**: RViz2 화면에서 3D 지도가 그려지는 것을 확인한 뒤 **`[터미널 2]`에서 `Ctrl + C`**를 눌러 정지합니다.

---

## 🧊 동작 3. Digital Twin용 3D Mesh 모델(.obj) 생성하기

SLAM 결과 파일(`.db`)을 추출하여 Isaac Sim / 3D 그래픽용 모델(.obj)로 변환할 때 사용합니다.

```bash
# 1. DB에서 3D Point Cloud 추출 (.ply)
./scripts/export_pointcloud.sh my_room_01_db.db my_room_01_cloud.ply

# 2. Open3D 정제 및 3D Mesh 파일(.obj) 생성
python3 scripts/process_mesh_open3d.py ./ros2_data/pointclouds/my_room_01_cloud.ply ./ros2_data/meshes/my_room_01_mesh.obj
```
* **최종 저장 결과물**: `./ros2_data/meshes/my_room_01_mesh.obj`

---

## 🔍 동작 4. 센서 토픽 및 시스템 상태 점검하기

카메라 연결 상태나 토픽 발행 여부, RAM 디스크 환경을 빠르게 확인할 때 사용합니다.

```bash
./scripts/check.sh
```

---

## 🛠️ 자주 쓰는 대처 명령어

* **카메라 Raw 이미지 발행 Hz 측정**:
  ```bash
  ros2 topic hz /camera/camera/color/image_raw
  ```
* **단축어로 빠른 빌드**:
  ```bash
  cb
  ```



