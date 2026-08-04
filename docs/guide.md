# 🛠️ Auto-Mobility 파이프라인 가이드 (Operations Guide)

RealSense D435i 센서 수집부터 3D Digital Twin Mesh 복원까지의 핵심 명령어 모음입니다.

---

## ⚙️ 0. 환경 구축 및 빌드 (Setup & Build)

```bash
# 워크스페이스 빌드 및 환경 로드
colcon build --symlink-install
source install/setup.bash
```

---

## 📊 1. 시스템 헬스 체크 & 토픽 진단 (Health Check)

DDS 설정, USB 모드, 실시간 압축/Raw 토픽 수신 상태 및 용량을 진단합니다.

```bash
./scripts/utils/check.sh
```

---

## 📸 2. 실시간 카메라 구동 및 SLAM 맵핑 (Live Mapping)

카메라 노드를 구동하거나, 실시간 3D Visual SLAM을 실행하여 `.db` 지도를 저장합니다.

```bash
# [단독 실행] RealSense D435i 카메라 노드 실행
ros2 launch auto_mobility camera.launch.py

# [라이브 SLAM] 실시간 카메라 스트림 맵핑 및 DB 저장 (기본값: live_날짜시각.db, USE_COMPRESSED: false)
./scripts/pipeline/run_live.sh [DB_NAME] [USE_COMPRESSED]

```

---

## 🎬 3. 데이터 녹화 및 재생 (Record & Play)

카메라 센서 토픽(RGB/Depth 압축, IMU, TF)을 MCAP 포맷으로 녹화/재생합니다.

```bash
# [녹화] Bag 데이터 녹화 (종료: Ctrl+C)
./scripts/pipeline/record.sh [BAG_NAME]

# [재생] Bag 데이터 재생 (기본 0.5배속)
./scripts/pipeline/play.sh [BAG_NAME] [RATE]

# [오프라인 SLAM] 녹화본(Bag) 데이터 기반 맵핑 및 DB 저장
./scripts/pipeline/run_bag.sh [BAG_NAME]
```

---

## 🧊 4. 3D Mesh 복원 (Mesh Generation)

SLAM 결과인 `.db` 파일에서 Point Cloud를 추출하고 Open3D 3D Mesh(`*.obj`)를 복원합니다.

```bash
# [자동 생성 & 시각화] Open3D Poisson Reconstruction 및 3D 뷰어 실행
./scripts/pipeline/mesh.sh [DB_NAME] [OUTPUT_MESH_NAME] --view

# [옵션] RTAB-Map 텍스처 맵핑 방식 사용 시
./scripts/pipeline/mesh.sh [DB_NAME] [OUTPUT_MESH_NAME] --method rtabmap
```

### 🛠️ 수동 모듈 개별 실행 (디버깅용)
```bash
# 1) DB -> PointCloud (.ply) 추출
./scripts/utils/export_ply.sh my_room.db my_cloud.ply

# 2) PointCloud 품질 및 무결성 자동 검증
python3 src/auto_mobility/processing/validate.py --db ./ros2_data/databases/my_room.db --ply ./ros2_data/pointclouds/my_cloud.ply

# 3) Open3D 3D Mesh 복원
python3 src/auto_mobility/processing/mesh_open3d.py ./ros2_data/pointclouds/my_cloud.ply ./ros2_data/meshes/my_mesh.obj --view
```

---

## ⚡ 5. 하드웨어 자동 벤치마크 (Hardware Benchmark)

현재 하드웨어 환경에서 최적의 DDS/QoS/해상도 조합을 측정하고 보고서를 생성합니다.

```bash
./scripts/utils/benchmark.sh
```
