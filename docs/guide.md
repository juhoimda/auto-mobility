# 🛠️ Auto-Mobility 개발자 및 사용 가이드

Auto-Mobility (Real-to-Sim) 파이프라인의 핵심 실행 명령어 및 사용 가이드입니다.

---

## ⚙️ 0. 환경 구축 및 빌드

```bash
# 1. ROS2 및 프로젝트 워크스페이스 빌드
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash

# 2. Python 경로 및 ROS_DOMAIN_ID 설정 (기본값: 42)
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export ROS_DOMAIN_ID=42
```

---

## 🔄 1. 전체 파이프라인 원스톱 실행 (Real-to-Sim)

실시간 센서 수집(또는 원격 수집), RTAB-Map SLAM, DB 무결성 검증, 3D Mesh 복원(Open3D TSDF), Isaac Sim 검증까지 한 번에 실행합니다.

```bash
# [전체 파이프라인] 기본 실행 (실시간 수집 -> SLAM -> TSDF Mesh -> Isaac Sim)
./scripts/pipeline/run_pipeline_all.sh

# [옵션 활용 예시]
./scripts/pipeline/run_pipeline_all.sh --remote-camera           # Windows 원격 카메라 자동 구동 + 압축 토픽 수신
./scripts/pipeline/run_pipeline_all.sh --skip-isaac               # Mesh 변환까지만 수행 (Isaac Sim 생략)
./scripts/pipeline/run_pipeline_all.sh --db=my_room.db --skip-capture --skip-isaac  # 기존 DB로 TSDF Mesh만 생성
```

* **CLI 옵션 설명**:
  * `--db=DB_NAME.db`: 저장/사용할 SLAM DB 파일명 (기본값: `session_날짜시간.db`)
  * `--skip-capture`: 카메라 실시간 수집을 건너뛰고 기존 DB 파일 재사용
  * `--skip-isaac`: Isaac Sim 물리/시각 검증 단계를 건너뜀
  * `--remote-camera`: Windows 네이티브 RealSense 실행 + 압축 토픽(JPEG/PNG) 수신 및 초고속 디코딩(republish) 모드

---

## 📸 2. 카메라 수집 및 센서 검증

### 2.1 로컬 실행 (WSL 직접 연결 / USB)
```bash
# RealSense D435i 노드 실행 (RGB-D 640x480@30 + IMU 200Hz + Madgwick Filter)
ros2 launch auto_mobility camera.launch.py

# 센서 토픽 수신 상태 및 Hz 확인
ros2 topic echo /camera/camera/imu --once
ros2 topic hz /camera/camera/color/image_raw /camera/camera/depth/image_rect_raw /camera/camera/imu
```

### 2.2 원격 카메라 모드 (Windows 네이티브)
```bash
# Windows 환경에서 C:\ros2_humble\run_camera.bat 실행 중인 경우
# 압축 토픽 수신 확인
ros2 topic hz /camera/camera/color/image_raw/compressed /camera/camera/depth/image_rect_raw/compressedDepth /camera/camera/imu

# 토픽 수신 프로브 도구
python3 scripts/utils/topic_probe.py /camera/camera/color/image_raw/compressed 5
python3 scripts/utils/check_camera_params.py
```

> ⚠️ **타임스탬프 아키텍처 (2026-08-14)**
> - Windows `realsense_pub.py` v2: 캡처/인코딩 스레드 분리 + **프레임 하드웨어 타임스탬프**를 epoch 로 변환해 발행 (RGB/Depth 동일 frameset).
> - WSL `republish.py` v4: 원본 stamp + WSL↔Windows **clock offset 추정값**으로 재발행 → 상대 타이밍이 캡처 시점 그대로 보존.
> - RTAB-Map DB/rosbag 의 timestamp 는 이제 캡처 시각 기반이다.

---

## 🗺️ 3. 실시간 Visual-Inertial SLAM

```bash
# [원스톱] Live SLAM 실행 (카메라 미실행 시 자동 백그라운드 구동 + RViz2 시각화)
./scripts/pipeline/run_live.sh [DB_NAME] [USE_COMPRESSED]
# 예시: ./scripts/pipeline/run_live.sh my_office false

# [원격 카메라 환경 Live SLAM]
CAMERA_MODE=remote ./scripts/pipeline/run_live.sh my_office true

# [수동 실행] 개별 노드 수동 구동
ros2 launch auto_mobility camera.launch.py
ros2 launch auto_mobility rtab_live.launch.py database_path:=./ros2_data/databases/my_office.db use_compressed:=false
```

* **인자 설명**:
  * `DB_NAME`: 저장할 DB 이름 (확장자 제외, 기본값: `live_날짜시간`)
  * `USE_COMPRESSED`: 압축 토픽 사용 여부 (`true` / `false`, 기본값: 로컬 `false`, 원격 `true`)
  * `database_path`: RTAB-Map DB 파일 저장 경로

---

## 🎬 4. ROS2 Bag 녹화, 재생 및 오프라인 SLAM

```bash
# [녹화] MCAP 포맷 저장 (RAM 디스크 /dev/shm 우선 기록 후 SSD 자동 이관)
./scripts/pipeline/record.sh [BAG_NAME] [--compressed | --raw]
# 예시: ./scripts/pipeline/record.sh hall_walk --compressed

# [녹화 후 자동 검증] 메시지 수 / Hz / sync delta / gap / 매니페스트 생성
python3 src/auto_mobility/utils/validate_bag.py ros2_data/bags/hall_walk --out ros2_data/bags/hall_walk/dataset_manifest.json

# [CAPTURE-SAFE] raw dataset 확보 전용 (RViz/SLAM 없이 녹화 + capture_guard 진단)
./scripts/pipeline/capture_safe.sh [BAG_NAME] [--compressed | --raw]
# 예시: ./scripts/pipeline/capture_safe.sh hallway_session --compressed

# [재생] 녹화된 Bag 재생
./scripts/pipeline/play.sh [BAG_NAME] [RATE]
# 예시: ./scripts/pipeline/play.sh hall_walk 1.0

# [오프라인 SLAM] Bag 재생 기반 맵핑 및 DB 생성 (Depth 토픽 자동 감지)
./scripts/pipeline/run_bag.sh [BAG_NAME] [--compressed | --raw]
# 예시: ./scripts/pipeline/run_bag.sh hall_walk --compressed
```

* **인자 설명**:
  * `BAG_NAME`: 녹화/재생할 Bag 디렉터리 이름
  * `--compressed`: 압축 토픽 위주 녹화/처리 (대역폭 및 I/O 절약) — RGB JPEG + Depth PNG(lossless)
  * `--raw`: 무압축 Raw 토픽 녹화/처리 (RGB/Depth bit-exact, 대역폭 큼)
  * `--no-validate`: 녹화 후 자동 검증 건너뜀
  * `RATE`: 재생 속도 (배속, 예: `0.5`, `1.0`, `2.0`)

> 💡 record.sh 는 `camera_info_windows`(Windows 원본 CameraInfo)도 함께 기록하므로,
> Windows 카메라 없이 오프라인 재생(run_bag.sh) 시에도 republish.py 가 올바른
> intrinsics 를 사용할 수 있다. (2026-08-14: 표준 토픽만 기록하던 방식은 replay 시
> 기본 intrinsics(fx=385) 폴백 → 기하 왜곡 잠재 버그였음)

---

## 🧊 5. 3D Mesh 복원 & Isaac Sim 연동

### 5.1 3D Mesh 재구성 (`mesh.sh`)
```bash
# [TSDF 방식 (권장)] Open3D Tensor TSDF 재구성 (GPU 가속, 원본 RGB-D + Pose 적분)
./scripts/pipeline/mesh.sh my_room.db my_room_tsdf.obj --method=tsdf --voxel=0.01 --view

# [Open3D Poisson 방식] 풀해상도 PLY 추출 -> Poisson 표면 복원
./scripts/pipeline/mesh.sh my_room.db my_room_poisson.obj --method=open3d --depth=8 --view

# [RTAB-Map 자체 메시 추출]
./scripts/pipeline/mesh.sh my_room.db my_room_rtab.obj --method=rtabmap

# [단독 PLY 추출] DB -> 풀해상도 Point Cloud(.ply) 추출
./scripts/utils/export_ply.sh my_room.db my_room_cloud.ply
```

* **인자 설명**:
  * `--method=tsdf|open3d|rtabmap`: Mesh 생성 알고리즘 선택 (기본값: `open3d`, `run_pipeline_all.sh`에서는 `tsdf` 기본)
  * `--voxel=VOXEL_SIZE`: TSDF / Voxel 다운샘플링 복셀 크기 (m 단위, 예: `0.005`, `0.01`)
  * `--depth=OCTREE_DEPTH`: Poisson 복원 옥트리 깊이 (기본값: `8`)
  * `--recon-method=poisson|bpa`: Open3D 복원 방식 (`poisson` 또는 `bpa`)
  * `--view`: Mesh 생성 완료 후 3D 뷰어 자동 실행
  * `--force`: DB/Point Cloud 품질 경고 검사를 무시하고 생성 강제 진행

### 5.2 3D Mesh 단독 뷰어 & Isaac Sim 검증
```bash
# [Mesh 뷰어] Open3D 기반 3D 모델 시각화
./scripts/utils/view_mesh.sh [MESH_FILE] [--wireframe] [--no-backface]
# 예시: ./scripts/utils/view_mesh.sh ros2_data/meshes/my_room_tsdf.obj

# [Isaac Sim 연동] 생성된 Mesh 로드 및 물리 충돌/USD 검증
./scripts/pipeline/isaac.sh [MESH_PATH] [--headless] [--no-physics] [--scale SCALE]
# 예시: ./scripts/pipeline/isaac.sh ros2_data/meshes/my_room_tsdf.obj --scale 1.0
```

---

## 📊 6. 시스템 진단, 벤치마크 & 품질 분석

```bash
# [시스템 진단] DDS(FastDDS), USB 대역폭, QoS, RAM 디스크, 소켓 버퍼 점검
./scripts/utils/check.sh

# [통합 SLAM 벤치마크] Stage 1(카메라) / Stage 2(SLAM) / Stage 3(진단) 성능 측정
./scripts/utils/benchmark.sh [--quick]
./scripts/utils/benchmark.sh --stage 1    # 카메라/QoS 조합 측정
./scripts/utils/benchmark.sh --stage 2    # SLAM 파라미터 최적화 측정

# [하드웨어 I/O & Open3D 벤치마크]
python3 src/auto_mobility/utils/benchmark_hw.py

# [세션 사후 분석] 생성된 SLAM DB 궤적, 키프레임, 루프클로저 통계 시각화
python3 src/auto_mobility/monitor/analyze_session.py --db ros2_data/databases/my_room.db

# [데이터셋 검증] 녹화된 bag 메시지 수 / Hz / sync / gap / 단조성 + 매니페스트
python3 src/auto_mobility/utils/validate_bag.py ros2_data/bags/my_bag --out ros2_data/bags/my_bag/dataset_manifest.json

# [단위/통합 테스트] 테스트 스위트 실행 및 코드 커버리지 리포트
./scripts/utils/run_tests.sh
```


