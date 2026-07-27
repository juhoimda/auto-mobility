# 📌 Digital Twin 프로젝트 빠른 실행 가이드 (Quick Guide)

원하는 동작에 맞춰 아래 명령어를 순서대로 실행하시면 됩니다.

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

## 🎬 동작 1. 센서 데이터 녹화하기 (RGB + Depth + IMU 수집)

실제 장소에서 카메라로 환경 데이터를 촬영하고 저장할 때 사용합니다.

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

---

## 🗺️ 동작 2. 녹화본 데이터로 3D Visual SLAM 지도 만들기

수집한 녹화 데이터 기반으로 3D 공간 지도 및 이동 경로를 복원할 때 사용합니다.

```bash
# [터미널 1] 녹화 데이터 반복 재생 (기존 카메라 launch 종료 후 실행)
./scripts/play.sh my_room_01

# [터미널 2] RTAB-Map 3D Visual SLAM 실행 및 지도 생성
./scripts/run_bag.sh my_room_01_db
```
* **확인 방법**: RViz 및 RTAB-Map 화면에서 3D 지도가 그려지는 것을 확인한 뒤 **`[터미널 2]`에서 `Ctrl + C`**를 눌러 정지합니다.

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

카메라 연결 상태나 토픽 발행 여부를 빠르게 확인할 때 사용합니다.

```bash
./scripts/check.sh
```

---

## 🛠️ 자주 쓰는 대처 명령어

* **카메라 생방송 Hz 측정**:
  ```bash
  ros2 topic hz /camera/camera/color/image_raw/compressed
  ```
* **단축어로 빠른 빌드**:
  ```bash
  cb
  ```


