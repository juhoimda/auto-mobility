# 🛠️ Auto-Mobility 개발자 및 사용 가이드

Auto-Mobility 파이프라인의 핵심 실행 명령어 모음입니다.

---

## ⚙️ 0. 환경 구축 및 빌드

```bash
# 워크스페이스 빌드 및 환경 설정
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

---

## 🔄 1. 전체 파이프라인 원스톱 실행 (Real-to-Sim)

```bash
# [전체 파이프라인] 실시간 수집 -> DB 무결성 검증 -> Mesh 생성 -> Isaac Sim 검증
./scripts/pipeline/run_pipeline_all.sh [--db=DB_NAME.db] [--skip-capture] [--skip-isaac]
```
* **인자 설명**:
  * `--db=DB_NAME`: 저장/사용할 SLAM DB파일명 (기본값: `session_날짜시간.db`)
  * `--skip-capture`: 카메라 실시간 수집을 건너뛰고 기존 DB 사용
  * `--skip-isaac`: Isaac Sim 검증 단계를 건너뜀

---

## 📸 2. 카메라 수집 및 센서 검증

```bash
# 1. RealSense D435i 노드 실행 (RGB-D + IMU 200Hz)
ros2 launch auto_mobility camera.launch.py

# 2. 토픽 수신 상태 및 Hz 확인
ros2 topic echo /camera/camera/imu --once
ros2 topic hz /camera/camera/color/image_raw /camera/camera/depth/image_rect_raw /camera/camera/imu
```

---

## 🗺️ 3. 실시간 Visual-Inertial SLAM

```bash
# [원스톱] Live SLAM 실행 (DB 생성 및 RViz2 시각화)
./scripts/pipeline/run_live.sh [DB_NAME] [USE_COMPRESSED]
# 예시: ./scripts/pipeline/run_live.sh my_office false

# [수동] 카메라 노드 실행 후 Live SLAM 수동 구동
ros2 launch auto_mobility camera.launch.py
ros2 launch auto_mobility rtab_live.launch.py database_path:=./ros2_data/databases/my_office.db
```
* **인자 설명**:
  * `DB_NAME`: 저장할 DB 이름 (확장자 제외, 기본값: `rtabmap`)
  * `USE_COMPRESSED`: 압축 토픽 사용 여부 (`true` / `false`, 기본값: `false`)
  * `database_path`: RTAB-Map DB 파일 저장 경로

---

## 🎬 4. ROS2 Bag 녹화, 재생 및 오프라인 SLAM

```bash
# [녹화] 센서 토픽 MCAP 압축 녹화 (Ctrl+C 종료)
./scripts/pipeline/record.sh [BAG_NAME] [--compressed]

# [재생] 녹화 데이터 재생
./scripts/pipeline/play.sh [BAG_NAME] [RATE]

# [오프라인 SLAM] 녹화본 기반 맵핑
./scripts/pipeline/run_bag.sh [BAG_NAME]
```
* **인자 설명**:
  * `BAG_NAME`: 녹화/재생할 Bag 디렉터리 이름
  * `--compressed`: 이미지 데이터 압축 녹화 옵션
  * `RATE`: 재생 속도 (예: `0.5` = 0.5배속, `1.0` = 정속)

---

## 🧊 5. 3D Mesh 복원 & Isaac Sim 연동

```bash
# [Mesh 복원] DB -> PLY 추출 및 Open3D Mesh 생성 (.obj)
./scripts/pipeline/mesh.sh [DB_NAME] [MESH_NAME] [--view] [--force]

# [Mesh 뷰어] 3D Mesh 모델 단독 시각화
./scripts/utils/view_mesh.sh [MESH_FILE]

# [Isaac Sim] 생성된 Mesh의 물리 충돌 및 USD 로드 검증
./scripts/pipeline/isaac.sh [MESH_PATH]
```
* **인자 설명**:
  * `DB_NAME`: 읽어올 DB 이름 (확장자 선택)
  * `MESH_NAME`: 생성할 Mesh 파일명 (`.obj` / `.ply`)
  * `--view`: Mesh 생성 후 Open3D 3D 뷰어 자동 실행
  * `--force`: 기존 추출본이 있어도 PLY 재추출 강제 진행
  * `MESH_FILE / MESH_PATH`: 시각화/검증할 Mesh 파일 경로

---

## 📊 6. 시스템 진단, 벤치마크 & 테스트

```bash
# [시스템 진단] DDS, USB, 소켓 버퍼 헬스 체크
./scripts/utils/check.sh

# [SLAM 벤치마크] 파이프라인 성능 측정 (결과: ros2_data/logs/)
python3 src/auto_mobility/slam/benchmark_slam.py [--quick]

# [단위/통합 테스트] Pytest 및 커버리지 측정
./scripts/utils/run_tests.sh
```
* **인자 설명**:
  * `--quick`: 빠른 벤치마크 테스트 수행 (측정 시간 단축)


