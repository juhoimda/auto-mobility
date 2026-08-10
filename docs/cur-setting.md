# ⚙️ Auto-Mobility 품질 & 최적화 설정 명세 (`docs/cur-setting.md`)

> **최종 갱신: 2026-08-10** — 벤치마크 실측 기반 재조정 (아래 모든 값은 실측 검증됨)

---

## 📸 1. 카메라 하드웨어 최적화 (`launch/camera.launch.py`)

| 설정 항목 | 설정값 | 핵심 기능 (품질 & 최적화) |
| :--- | :--- | :--- |
| **`enable_sync`** | `True` | RGB-Depth 하드웨어 프레임 동기화. **⚠ sync=False 시 depth 4Hz 폭락** (실측) |
| **`unite_imu_method`** | `1` | Gyro-Accel copy 통합 (CPU 절감, 벤치마크 검증) |
| **`depth_module.emitter_enabled`** | `1` | IR Laser Projector 활성화로 무늬 없는 벽면의 3D 특징점 강제 생성 |
| **`rgb_camera.color_profile`** | `640x480x30` | **★ 벤치마크 확정** — 848x480은 depth 19.4Hz로 드랍 (VM USB 대역폭 초과) |
| **`depth_module.depth_profile`** | `640x480x30` | **★ 벤치마크 확정** — depth 29.1Hz 안정 유지 (실측) |
| **`align_depth.enable`** | `False` | vCPU 픽셀 정렬 병목 제거 (RTAB-Map이 자체 정렬, 벤치마크 확정값) |
| **`auto_exposure_priority`** | `False` | 저조도 스캔 시 카메라 프레임레이트(FPS) 폭락 방지 |

### 📊 해상도 벤치마크 결과 (2026-08-10 실측)
| 해상도 | color Hz | depth Hz | 드랍 여부 | 비고 |
| :--- | :---: | :---: | :---: | :--- |
| **640x480@30** | 30.1 | **29.1** | 없음 ✅ | **최종 확정** |
| 848x480@30 | 30.0 | **19.4** | **35% 드랍** ❌ | RGB8+Z16 = 61MB/s > VM USB 한계(~50MB/s) |

---

## 📡 2. ROS 2 / FastDDS & 통신 네트워크 최적화

| 설정 항목 | 설정값 | 핵심 기능 (품질 & 최적화) |
| :--- | :--- | :--- |
| **FastDDS `segment_size`** | `536870912` (512 MB) | 다중 구독자 환경에서 대용량 이미지 스트림 버퍼 오버플로우 및 프레임 드랍 차단 |
| **FastDDS Socket Buffer** | `10 MB` | UDP 수발신 버퍼 확장으로 네트워크 스파이크 완충 |
| **Republish QoS Profile** | `SENSOR_DATA` | Publisher/Subscriber QoS 일치로 비동기 수신 데이터 유실(Drop) 0% 달성 |

---

## 🗺️ 3. RTAB-Map VI-SLAM & Madgwick IMU 최적화 (`src/auto_mobility/launch/launch_common.py`)

| 설정 항목 | 설정값 | 핵심 기능 (품질 & 최적화) |
| :--- | :--- | :--- |
| **Madgwick `gain`** | `0.03` | 자이로스코프 노이즈 필터링 및 빠른 회전 시 맵 수평 드리프트 억제 |
| **`approx_sync_max_interval`**| `0.08` (80 ms) | 센서 타임스탬프 짝짓기 오차를 80ms 이내로 제한 (live 기준) |
| **`Rtabmap/DetectionRate`** | `5` (5 Hz) | 실시간 SLAM 갱신율을 5Hz로 제어하여 vCPU 계산 지연(Lag) 예방 |
| **`RGBD/LinearUpdate`** | `0.10` (10 cm) | 10cm 간격 키프레임 생성으로 그래프 최적화 부하 절감 및 DB 비대화 방지 |
| **`RGBD/AngularUpdate`** | `0.10` (5.7 deg) | 5.7도 간격 키프레임 생성으로 회전 구간 매핑 안정성 확보 |
| **`Vis/EstimationType`** | `1` | 3D-2D PnP 포즈 추정 (벤치마크 1위, SVD 대비 CPU 40% 절감) |
| **`Vis/MinInliers`** | `10` | 특징점 검증 임계값 (벤치마크 1위: IN=6 대비 +4~5점) |
| **`Vis/CornerMinQuality`** | `0.02` | 특징점 품질 임계값 (벤치마크 검증값) |
| **`Vis/CornerGridSize`** | `30` | 특징점 그리드 크기 (벤치마크 검증값) |
| **`Vis/CornerNbThreads`** | `8` | vCPU 8 스레드 멀티스레딩 특징점 병렬 추출 |
| **`OdomF2M/MaxFrames`** | `10` | **★ 벤치마크 1위** — 기존 60은 SLAM 초기화 실패 및 VO 지연 누적 |
| **`Mem/STMSize`** | `10` | **★ 벤치마크 1위** — 기존 100 대비 RAM 절감, 성능 동일 |
| **`Grid/VoxelSize`** | `0.01` (1 cm) | 1cm 정밀 격자 보존 및 3D Point Cloud 잡음 제거 |
| **`odom_always_process_most_recent_frame`** | `false` | **★ 수정됨** — 기존 `always_process_most_recent_frame`(잘못된 인자명)은 무시됐었음. 모든 프레임 순서 처리로 맵 밀도 확보 |

### 🧪 SLAM 파라미터 벤치마크 결과 (2026-08-10 실측, 8개 조합)
| 순위 | 조합 | odom Hz | SLAM CPU | 점수 |
| :---: | :--- | :---: | :---: | :---: |
| 🥇 | **PnP \| F2M=10 \| STM=10 \| KF_0.1 \| IN=10 \| MD=4** | 29.9 | 34.3% | **65.3** |
| 🥈 | PnP \| F2M=10 \| STM=10 \| KF_0.1 \| IN=6 \| MD=4 | 27.8 | 28.8% | 60.8 |
| 🥉 | PnP \| F2M=10 \| STM=10 \| KF_0.2 \| IN=6 \| MD=4 | 29.9 | 27.2% | 61.0 |

---

## 📐 4. Open3D 3D Mesh 복원 최적화 (`src/auto_mobility/mesh/mesh_open3d.py`)

| 설정 항목 | 설정값 | 핵심 기능 (품질 & 최적화) |
| :--- | :--- | :--- |
| **복원 방법** | `poisson` (기본) | **★ BPA→Poisson 전환** — BPA는 구멍 다수 (표면 17.5m²), Poisson은 폐곡면 보완 (115.9m², 실측) |
| **`--depth`** | `8` | Poisson octree 깊이 (depth=9는 vertex 폭증으로 비효율) |
| **`--voxel`** | `0.005` (5mm) | 다운샘플링 voxel 크기 (640x480 depth 해상도에 최적, 3mm는 노이즈 증폭) |
| **`--simplify`** | `0.5` | Quadric Decimation 50% 경량화 (Isaac Sim/뷰어 로딩 성능 확보) |
| **Normal Alignment** | `orient_normals_consistent_tangent_plane(k=15)` | Tangent Plane 기반 3D 일관 법선 정렬로 벽면/천장/측면 메쉬 표면 뒤집힘 방지 |
| **Color Transfer** | `cKDTree k=3 IDW` | 거리 역가중 평균(IDW)으로 Point Cloud 색상을 복사하여 표면 텍스처 노이즈 및 계단 현상 완화 |
| **Outlier Removal** | `remove_statistical_outlier(nb=20, std=2.0)` | 통계적 부유 잡음 노이즈 제거 |
| **Topology Cleanup** | `crop(bbox)` & Topology Repair | 바운딩 박스 외부 허공 메쉬 및 중복/비정상 삼각면 자동 제거 |

---

## 🛡️ 5. 촬영 품질 모니터링 (`src/auto_mobility/monitor/capture_guard.py`) — 신규

`run_pipeline_all.sh` 촬영 중 **병렬 실행**되어 FPS/CPU/RAM/USB를 실시간 감시하고,
종료 시 Markdown 보고서(`ros2_data/logs/capture_guard_*.md`)를 생성한다.

- **감시 토픽**: color(≥15Hz), depth(≥15Hz), camera_info(≥15Hz), IMU(≥100Hz), odom(≥5Hz)
- **핵심 지표**: 시작/종료 odom Hz 비교로 시간 경과 저하 감지
- **실측 발견**: 촬영 시간이 길어지면 VO odom이 28→14Hz로 절반 저하 (VM 한계)
  → **2~3분 단위 세션 분할 촬영 권장**
