# ⚙️ Auto-Mobility 품질 & 최적화 설정 명세 (`docs/cur-setting.md`)

> **최종 갱신: 2026-08-12 (WSL2 이관 + 카메라 QoS 회귀 해결 + IMU 활성화)** — 상세 근거는 [docs/final_report.md](final_report.md)
> 2026-08-11: RTAB-Map 0.23.7 무효 파라미터 9종 유효 키 교정, `OdomF2M/MaxSize=1000` 확정

> **설정 단일 소스(Single Source of Truth) 안내 (리팩토링 적용)**
> 아래 표의 값을 수정할 때는 각 파일을 직접 고치는 대신 다음 중앙 설정에서만 변경하세요:
> - **토픽명**: `config/topics.yaml` → `config.py` / `common.sh` / launch / node 에 자동 반영
> - **카메라 파라미터**: `src/auto_mobility/config.py` 의 `CAMERA_PARAMS` (camera.launch.py / benchmark 공용)
> - **RTAB-Map SLAM 파라미터**: `src/auto_mobility/launch/launch_common.py` 의 `RTABMAP_PARAMS` (live/bag 공용, benchmark 기준으로도 사용)
> - **Mesh 파라미터**: `src/auto_mobility/config.py` 의 `MESH_DEFAULTS` (CLI default 일치)
> - **데이터 경로 / FastDDS XML / USB 기준**: `src/auto_mobility/config.py` 의 경로 상수

---

## 📸 1. 카메라 하드웨어 최적화 (`launch/camera.launch.py`)

| 설정 항목 | 설정값 | 핵심 기능 (품질 & 최적화) |
| :--- | :--- | :--- |
| **`enable_sync`** | `True` | RGB-Depth 하드웨어 프레임 동기화. **⚠ sync=False 시 depth 4Hz 폭락** (실측) |
| **`unite_imu_method`** | `1` | Gyro-Accel copy 통합 (CPU 절감, 벤치마크 검증) |
| **`depth_module.emitter_enabled`** | `1` | IR Laser Projector 활성화로 무늬 없는 벽면의 3D 특징점 강제 생성 |
| **`rgb_camera.color_profile`** | `640x480x30` | **★ 확정** — 848x480 이상은 USB 패스스루에서 프레임 손상 (WSL2 실측) |
| **`depth_module.depth_profile`** | `640x480x30` | **★ 확정** — 29.7fps 안정 (WSL2 실측) |
| **`align_depth.enable`** | `False` | CPU 픽셀 정렬 병목 제거 (RTAB-Map이 자체 정렬, 벤치마크 확정값) |
| **QoS 키 (`color_qos` 등)** | **미사용** | ★ **v4.58.3 FPS 급락 원인(30→~10fps) 실측 → 제거. 기본 RELIABLE 사용** |
| **필터** | `spatial/temporal/hole_filling` (개별 enable 키) | ★ v4.58.3에서 `filters` 문자열 키는 무효. `spatial_filter.enable` 등으로 설정 |
| **IMU** | gyro+accel, ~200Hz | ★ WSL2 커널(HID_SENSOR_HUB) + udev 규칙으로 활성화 (191Hz 실측) |

### 📊 해상도 벤치마크 결과 (WSL2 2026-08-12 실측, rosbag 기준)
| 해상도 | color Hz | depth Hz | 드랍 여부 | 비고 |
| :--- | :---: | :---: | :---: | :--- |
| **640x480@30** | 30.0 | **29.7** | 없음 ✅ | **최종 확정** (필터 ON + IMU ON) |
| 848x480@30 단독 | ~30 | ~30 | **0.35% 손상** ❌ | "Incomplete video frame" 발생 → 미채택 |
| 848x480 + 1280x720 color | 29.2 | 27.3 | 손상 1건/15s | USB 패스스루 한계 |

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
| **`approx_sync_max_interval`**| `0.15` (150 ms) | **★ 실환경 재조정** — VO delay(~70ms, 항상 최신 프레임 처리 특성)를 흡수. 기존 0.08은 간헐적 "no odometry" 스킵(맵 구멍) 유발 |
| **`Rtabmap/DetectionRate`** | `3` (3 Hz) | **★ 2026-08-12 euijin 실측 반영** — 5→3Hz: 맵 스레드 부하 40% 감소 (5Hz 시 RTAB-Map=0.2s/검출로 스레드 100% 점유) |
| **`RGBD/LinearUpdate`** | `0.10` (10 cm) | 10cm 간격 키프레임 생성으로 그래프 최적화 부하 절감 및 DB 비대화 방지 |
| **`RGBD/AngularUpdate`** | `0.10` (5.7 deg) | 5.7도 간격 키프레임 생성으로 회전 구간 매핑 안정성 확보 |
| **`Vis/EstimationType`** | `1` | 3D-2D PnP 포즈 추정 (벤치마크 1위, SVD 대비 CPU 절감) |
| **`Vis/MinInliers`** | `8` | **★ 2026-08-12 조정** — 10→8: 회전 구간에서 불필요한 추적 실패 판정 감소 |
| **`GFTT/QualityLevel`** | `0.005` | **★ 2026-08-12 조정** — 0.02→0.005: blur/회전 프레임에서 특징점 0개(fromWords=0) 방지. `Vis/CornerMinQuality`(0.23.7에 없음)의 유효 대체 키 |
| **`Vis/MaxFeatures`** | `2000` | **★ 2026-08-12 조정** — 1000→2000: 회전 시 키프레임 중첩 확보 (맵 스레드 경량화로 CPU 여유 확보됨) |
| **`Vis/GridRows`** | `16` | **★ 2026-08-11 교정** — `Vis/CornerGridSize`(0.23.7에 없음)의 유효 대체 키. 특징점 균일 분포 그리드 (640x480@30px) |
| **`Vis/GridCols`** | `21` | **★ 2026-08-11 교정** — 상동 |
| **`OdomF2M/MaxSize`** | `1000` | **★ 2026-08-11 벤치 1위** — `OdomF2M/MaxFrames`(0.23.7에 없음)의 유효 대체 키. 기본 2000은 odom 17Hz 급락, 4000은 14Hz |
| **`Odom/GuessMotion`** | `true` | **★ 2026-08-11 교정** — `Odom/PoseGuessMode`(0.23.7에 없음)의 유효 대체 키. 이전 모션 기반 다음 포즈 추측 |
| **`Optimizer/GravitySigma`** | `0.3` | **★ 2026-08-11 교정** — `Optimizer/GravityProvided`(0.23.7에 없음)의 유효 대체 키. 그래프 최적화 중력 제약 |
| **`Mem/STMSize`** | `10` | **★ 2026-08-11 재확정** — STM=100 대비 RAM 절감, 성능 동일 |
| **`RGBD/ProximityBySpace`** | `false` | **★ 2026-08-12 euijin 실측 반영** — 재방문마다 공간 루프클로저 검색 중단. 단일 구역 촬영은 시간 기반으로 충분 |
| **`RGBD/OptimizeFromGraphEnd`** | `false` | **★ 2026-08-12 euijin 실측 반영** — 루프클로저 보정 활성화. true 시 "Map correction should be identity" 에러 105회 + loop closure 전량 거부 (euijin 143506/142815 실측) |
| **`RGBD/NeighborLinkRefining`** | `false` | **★ 2026-08-12 euijin 실측 반영** — 재방문 시 재최적화 폭주 차단 → RTAB-Map 스레드 0.94s 스톨 방지 |
| **`Grid/DepthDecimation`** | `4` | **★ 2026-08-12 조정** — 2→4: 라이브 맵 4배 경량 (기본 4). 최종 메쉬는 offline TSDF에서 생성 |
| **`Grid/RayTracing`** | `false` | **★ 2026-08-12 조정** — 2D 점유격자용, 구독자 없음 |
| **`odom_always_process_most_recent_frame`** | `true` | **★ 근본 수정(2026-08-10 재검증)** — live 촬영은 최신 프레임 처리로 지연 누적 차단. `false`는 rosbag 오프라인 전용 (RTAB-Map 공식 가이드). 기존 false는 DDS 큐 백로그·위상 지연 누적 + 맵핑 동기화 실패 유발 |

> ⚠️ **2026-08-11 파라미터 유효성 검증 결과** — 아래 키는 RTAB-Map 0.23.7에서 **존재하지 않아 조용히 무시**되던 것들:
> `Vis/CornerMinQuality`, `Vis/CornerGridSize`, `Vis/CornerNbThreads`(제거), `OdomF2M/MaxFrames`,
> `Odom/PoseGuessMode`, `Optimizer/GravityProvided`, `Vis/Robust`(→`Optimizer/Robust`),
> `Rtabmap/ResetCountdown`(→`Odom/ResetCountdown`), `RGBD/CreateIntermediateNodes`(→`Rtabmap/CreateIntermediateNodes`), `Grid/VoxelSize`

### 🧪 SLAM 파라미터 벤치마크 결과 (2026-08-11 실측, 8개 조합, 유효 키 기준)
| 순위 | 조합 | odom Hz | SLAM CPU | 점수 |
| :---: | :--- | :---: | :---: | :---: |
| 🥇 | **PnP \| MaxSize=1000 \| STM=10 \| KF_0.1 \| IN=10 \| MD=4** | **30.0** | 40.6% | **64.8** |
| 🥈 | SVD \| MaxSize=1000 \| STM=10 \| KF_0.1 \| IN=6 \| MD=4 | 30.0 | 33.6% | 60.6 |
| 🥉 | PnP \| MaxSize=2000 \| STM=10 \| KF_0.1 \| IN=6 \| MD=4 | 17.4 | 35.4% | 60.4 |

> 🔑 **핵심**: `OdomF2M/MaxSize=1000`이 VO 유지에 결정적 (기본 2000은 odom 17Hz 급락).
> 기존 "F2M=10 CPU 35% 절감" 결론은 무효 키 `OdomF2M/MaxFrames`로 측정된 것 → 2026-08-11 재벤치마크로 교정.

---

## 📐 4. Open3D 3D Mesh 복원 최적화 (`src/auto_mobility/mesh/mesh_open3d.py` + `scripts/utils/export_ply.sh`)

| 설정 항목 | 설정값 | 핵심 기능 (품질 & 최적화) |
| :--- | :--- | :--- |
| **`--decimation`** (export) | `1` | **★ 2026-08-11 — 기본 4는 PLY 밀도 1/16로 mesh 품질 저하의 근본 원인이었음. 오프라인 처리라 캡처 성능 영향 없음** |
| **`--max_range`** (export) | `5` (m) | 4m 이상 벽/천장 포착 (기본 4m) |
| **복원 방법** | `poisson` (기본) | **★ BPA→Poisson 전환** — BPA는 구멍 다수 (표면 17.5m²), Poisson은 폐곡면 보완 (115.9m², 실측) |
| **`--depth`** | `8` | Poisson octree 깊이 (depth=9는 vertex 폭증으로 비효율) |
| **`--voxel`** | `0.01` | 다운샘플링 voxel 크기 (10mm). D435 depth 정밀도(1~2cm) 이내 해상도로 정보 손실 없는 성능 최적화 |
| **voxel_down 연산** | **GPU(CUDA Tensor)** | ★ 2026-08-12 실측: CPU 2183ms → GPU 12.7ms (2.2M 포인트, ~172배). mesh_open3d.py 자동 감지 |
| **`--simplify`** | `0.5` | Quadric Decimation 50% 경량화 (Isaac Sim/뷰어 로딩 성능 확보) |
| **Normal Alignment** | `orient_normals_consistent_tangent_plane(k=15)` | Tangent Plane 기반 3D 일관 법선 정렬로 벽면/천장/측면 메쉬 표면 뒤집힘 방지 |
| **Color Transfer** | `cKDTree k=1` | 최근접 이웃 색상 복사. voxel 다운샘플링이 이미 색상 평균화 → k=3 IDW 보간 중복 제거 |
| **Outlier Removal** | `remove_statistical_outlier(nb=20, std=2.0)` | 통계적 부유 잡음 노이즈 제거 |
| **Topology Cleanup** | `crop(bbox)` & Topology Repair | 바운딩 박스 외부 허공 메쉬 및 중복/비정상 삼각면 자동 제거 |

---

## 🛡️ 5. 촬영 품질 모니터링 (`src/auto_mobility/monitor/capture_guard.py`) — 신규

`run_pipeline_all.sh` 촬영 중 **병렬 실행**되어 FPS/CPU/RAM/USB를 실시간 감시하고,
종료 시 Markdown 보고서(`ros2_data/logs/capture_guard_*.md`)를 생성한다.

- **감시 토픽**: color(≥15Hz), depth(≥15Hz), camera_info(≥15Hz), IMU(≥100Hz), odom(≥5Hz)
- **핵심 지표**: 시작/종료 odom Hz 비교로 시간 경과 저하 감지
- **실측 발견(재검증 후)**: 설정 정합 시 VO delay는 ~70ms로 고정(누적 없음), odom 17~26Hz 유지
  (기존 설정 대비 12Hz→개선). 잔여 저하는 호스트(VMware/thermal) 요인이 주원인

---

## 🖥️ 6. RViz 촬영 뷰어 최적화 (`config/rviz/rtabmap_live.rviz` + `cloud_throttle.py`)

소프트웨어 렌더링(WSLg D3D12 검은 화면 방지)에서 RViz가 CPU를 경쟁하며 VO를 방해하지 않도록 최적화:

| 항목 | 변경 | 효과 |
| :--- | :--- | :--- |
| **`/rtabmap/cloud_map_lite`** | `cloud_throttle.py`가 `/rtabmap/cloud_map`을 **2Hz**로 중계 | PointCloud 소프트웨어 렌더링 부하 ~60% 절감. SLAM 내부 처리 영향 없음 |
| **Image QoS** | **`Reliable`로 변경** | ★ 2026-08-12: 카메라 RELIABLE 발행 ↔ RViz BE 구독 불일치로 이미지 유실 → Reliable로 수정 |
| **죽은 디스플레이 제거** | `/voxel_cloud`(존재하지 않는 토픽) PointCloud2 삭제 | 무의미한 렌더링 제거 |
| **Path (Trajectory)** 추가 | `/rtabmap/odom` 기반 이동 궤적 표시 (경량) | 촬영 진행 확인용 저비용 시각화 |
| **TF 표시 축소** | `camera_link`/`odom`/`map` 외 비활성 | TF 오버레이 부하 감소 |
| **이미지 중복 제거** | RGB 1개만 활성, depth 이미지 미표시 | 텍스처 업로드 부하 절반 |

- 실행: `cloud_throttle.py`는 `rtab_live.launch.py`에서 자동 기동 (CMakeLists에 등록됨)
- 촬영 중 원본 품질 확인 필요 시 rviz에서 `PointCloud2 (RTAB-Map Map)`의 토픽을
  `/rtabmap/cloud_map`으로 일시 전환하면 5Hz 원본을 볼 수 있음
