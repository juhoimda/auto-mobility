# ⚙️ Auto-Mobility 품질 & 최적화 설정 명세 (`docs/cur-setting.md`)

---

## 📸 1. 카메라 하드웨어 최적화 (`launch/camera.launch.py`)

| 설정 항목 | 설정값 | 핵심 기능 (품질 & 최적화) |
| :--- | :--- | :--- |
| **`enable_sync`** | `True` | RGB-Depth 센서 간 하드웨어 프레임 동기화로 타임스탬프 오차 제거 |
| **`unite_imu_method`** | `2` | Gyro-Accel 데이터 선형 보간(Linear Interpolation) 통합으로 IMU 자세 추정 정밀도 향상 |
| **`depth_module.emitter_enabled`** | `1` | IR Laser Projector 활성화로 무늬 없는 벽면의 3D 특징점 강제 생성 |
| **`rgb_camera.color_profile`** | `848x480x30` | 30 FPS RGB8 하드웨어 압축 스트리밍 및 RTAB-Map 최적 해상도 매칭 |
| **`depth_module.depth_profile`** | `848x480x30` | 30 FPS Z16 비손실 뎁스 스트리밍 |
| **`align_depth.enable`** | `True` | RGB 및 Depth 픽셀 정렬로 정확한 컬러-뎁스 매핑 제공 |
| **`auto_exposure_priority`** | `False` | 저조도 스캔 시 카메라 프레임레이트(FPS) 폭락 방지 |

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
| **`approx_sync_max_interval`**| `0.08` (80 ms) | 센서 타임스탬프 짝짓기 오차를 80ms 이내로 제한하여 Visual Odometry 궤적 오차 감소 |
| **`Rtabmap/DetectionRate`** | `5` (5 Hz) | 실시간 SLAM 갱신율을 5Hz로 제어하여 vCPU 계산 지연(Lag) 예방 |
| **`RGBD/LinearUpdate`** | `0.10` (10 cm) | 10cm 간격 키프레임 생성으로 그래프 최적화 부하 절감 및 DB 비대화 방지 |
| **`RGBD/AngularUpdate`** | `0.10` (5.7 deg) | 5.7도 간격 키프레임 생성으로 회전 구간 매핑 안정성 확보 |
| **`Vis/EstimationType`** | `1` | 3D-2D PnP 포즈 추정으로 Depth 결측 구간 추적 유지력 증대 |
| **`Vis/MinInliers`** | `10` | 특징점 검증 임계값 조율로 오매칭 오도메트리 튀임 현상 방지 |
| **`Vis/CornerNbThreads`** | `8` | vCPU 8 스레드 멀티스레딩 특징점 병렬 추출 |
| **`Grid/VoxelSize`** | `0.01` (1 cm) | 1cm 정밀 격자 보존 및 3D Point Cloud 잡음 제거 |

---

## 📐 4. Open3D 3D Mesh 복원 최적화 (`src/auto_mobility/mesh/mesh_open3d.py`)

| 설정 항목 | 최적화 방식 | 핵심 기능 (품질 & 최적화) |
| :--- | :--- | :--- |
| **Normal Alignment** | `orient_normals_consistent_tangent_plane(k=15)` | Tangent Plane 기반 3D 일관 법선 정렬로 벽면/천장/측면 메쉬 표면 뒤집힘 방지 |
| **Color Transfer** | `cKDTree k=3 IDW` | 거리 역가중 평균(IDW)으로 Point Cloud 색상을 복사하여 표면 텍스처 노이즈 및 계단 현상 완화 |
| **Outlier Removal** | `remove_statistical_outlier(nb=20, std=2.0)` | 통계적 부유 잡음 노이즈 제거 |
| **Topology Cleanup** | `crop(bbox)` & Topology Repair | 바운딩 박스 외부 허공 메쉬 및 중복/비정상 삼각면 자동 제거 |
