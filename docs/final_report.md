# D435i → SLAM → Point Cloud → Mesh 최종 보고서 (WSL2)

- 작성일: 2026-08-12
- 환경: WSL2 (VMware에서 이관), ROS2 Humble, auto_mobility 패키지
- 모든 수치는 실측. 측정 도구 제약은 각 절에 명시.

---

## 1. Hardware / WSL Baseline

| 항목 | 값 |
|---|---|
| CPU | Intel Core Ultra 7 265H (16코어/16스레드, WSL vCPU) |
| RAM | 31 GiB 가시 (호스트 32GB), Swap 8GiB, /dev/shm 16GiB |
| GPU (CUDA) | NVIDIA RTX PRO 2000 Blackwell 8GB (드라이버 595.71, CUDA 13.2) |
| Graphics | WSLg D3D12 (Intel Arc Pro 140T) — RViz2는 소프트웨어 렌더링 사용 (아래 §17) |
| WSL | 커널 6.18.40.1-microsoft-standard-WSL2+ (커스텀 빌드, HID_SENSOR_HUB 포함) |
| USB | usbipd-win 5.3.0, D435i USB 3.2 (5Gbps 실측) |
| .wslconfig | kernel=C:\WSL-kernel\custom-bzImage-6.18 (커스텀 커널 지정) |

## 2. Software Stack Baseline

| 컴포넌트 | 버전 | GPU 사용 |
|---|---|---|
| ROS2 | Humble + FastDDS(기본 RMW) | — |
| realsense2_camera / librealsense | 4.58.3 / 2.58.3 | CPU (GLSL 옵션 미사용) |
| RTAB-Map | 0.23.7 (apt) | CPU-only |
| OpenCV | 4.5.4 (시스템) | CPU-only |
| Open3D | 0.19.0 (pip, CUDA 빌드) | **CUDA 사용 가능·일부 채택** |
| pyrealsense2 | 2.58.3 (pip) | — |
| CUDA toolkit (nvcc) | 미설치 (Open3D 내장 CUDA로 충분) | — |

## 3. 발견된 병목

### Critical (해결됨)
1. **카메라 FPS 급락 (30→~10fps)** — 원인: `color_qos/depth_qos` 등 QoS 파라미터가 realsense2_camera 4.58.3에서 스트림 재시작을 유발해 FPS 저하. → QoS 키 제거로 해결 (§4).
2. **IMU 비활성** — 원인: WSL2 기본 커널에 `CONFIG_HID_SENSOR_HUB` 없음 → librealsense가 필요한 `HID-SENSOR-2000e1` 미생성. → 커스텀 커널 빌드로 해결 (§17).

### Major (완화/관리)
3. **USB 패스스루 프레임 손상** — 848x480 이상에서 "Incomplete video frame" (0.3%). 640x480에서 0. → 해상도 고정으로 회피.
4. **카메라 bad-state** — 스트림 시작 실패 후 토픽 0발행 상태 지속. USB 재인식(authorized toggle) 필요.
5. **Python rclpy 소비 유실** — 614KB 이미지 30fps에서 Python 구독자 대량 유실. 측정은 C++(rosbag) 필수.

### Minor
6. RViz2 WSLg D3D12 검은 화면 → 소프트웨어 렌더링으로 회피 (§17).
7. CUDA toolkit/nvcc 미설치 (Open3D 내장 CUDA로 충분, 커스텀 커널 빌드 시 미필요).
8. 타 사용자와 카메라 동시 사용 충돌 위험 (공유 머신 특성).

## 4. D435i 최적 구성 (WSL2 실측 확정)

| 항목 | 값 | 근거 |
|---|---|---|
| RGB | 640x480 @30fps RGB8 | 30.0fps 실측, 848x480+는 USB 손상 |
| Depth | 640x480 @30fps Z16 | 29.7fps 실측 |
| IMU | gyro/accel 통합(~200Hz), unite_imu_method=1 | 191Hz 실측 |
| align_depth | **False** | CPU 정렬 병목 회피 (RTAB-Map 자체 정렬) |
| 필터 | spatial+temporal+hole_filling (개별 enable 키) | FPS 영향 미미(29.7fps) + 노이즈 감소 |
| **QoS 키** | **미사용 (기본 RELIABLE)** | ★ v4.58.3 FPS 급락 원인 |
| enable_sync | True | 필수 (VMware 벤치 근거, 유지) |
| emitter | 1 (IR 패턴) | 무벽면 특징점 확보 |

## 5. ROS2 / QoS 최적 구성

- 발행 QoS: 카메라 **RELIABLE** (QoS 키 제거 시 기본값). RTAB-Map은 BEST_EFFORT로 구독 — DDS에서 정상 동작.
- RViz2 Image display: **Reliable로 변경** (기존 Best Effort 구독이 RELIABLE 발행 대비 프레임 유실 → 부드럽지 않은 원인).
- RMW: rmw_fastrtps_cpp (FastDDS). SHM 프로파일(config/dds/fastdds_camera.xml) 적용.
- topic_queue_size: live 50 / bag 30. approx_sync 0.15s.

## 6. SLAM 후보 비교

| 기준 | RTAB-Map 0.23.7 | ORB-SLAM3 | VINS-Fusion | OpenVINS |
|---|---|---|---|---|
| RGB-D | O | O | X (mono/stereo) | X |
| IMU | O (활성화됨) | O | O | O |
| Loop closure | O (리얼타임) | O (4.0) | X | X |
| ROS2 네이티브 | **O** | X | X | X |
| D435i 호환 | O | O | O | O |
| 설치 복잡도 | 낮음(설치됨) | 높음(빌드) | 높음(ROS1) | 높음(ROS1) |
| 맵 재구성 통합 | O (cloud_map) | X | X | X |
| 유지보수 | 활발 | 중간 | 낮음 | 중간 |

- ROS2 네이티브 + RGB-D + IMU + loop closure + 재구성 통합을 모두 만족하는 후보는 RTAB-Map 유일.
- 나머지는 ROS1 의존/설치 비용 대비 명확한 장점이 실측되지 않음 (plan.md 규정: 장점이 명확할 때만 제안).

## 7. 최종 추천 SLAM
**RTAB-Map 0.23.7 (rgbd_odometry + rtabmap)** — Visual + IMU(Madgwick 융합) 사용.

## 8. 선택 근거
- ROS2 Humble 네이티브 (설치/운영 비용 최소)
- D435i RGB+Depth+IMU 동시 소비 + graph SLAM + loop closure
- cloud_map 실시간 point cloud 생성 (재구성 파이프라인 통합)
- WSL2에서 IMU 활성화로 Visual-Inertial 가능
- 실측: odom 7~11Hz, cloud_map_lite 2.1Hz 안정 (Phase 2)

## 9. Point Cloud 문제 및 최적 구성

- 실시간: RTAB-Map `cloud_map` → `cloud_throttle`(2Hz) → `cloud_map_lite` (RViz 표시용). SLAM 영향 없음, 2.1Hz 실측.
- 오프라인: rtabmap-export로 풀해상도 PLY 추출 (2.2M 포인트/세션).
- 알려진 문제: D435 depth 노이즈(1~2cm), 원거리(>4m) 노이즈 → Grid/RangeMax 4.0 + Statistical Outlier로 처리.
- voxel_down GPU(0.01m) 적용 → 12.7ms (CPU 2.2초 대비 172배).

## 10. Reconstruction 알고리즘 비교 (Open3D 실측)

| 연산 | CPU (legacy) | CUDA (tensor) | 결정 |
|---|---|---|---|
| voxel_down_sample (1M) | 587ms | 23ms | **GPU 채택** |
| estimate_normals (1M) | 1.35s | 9.66s | CPU 유지 |
| remove_statistical_outlier (1M) | 1.12s | 9.53s | CPU 유지 |
| ICP (500K, 100iter) | 288ms | 65.4s | CPU 유지 |
| Poisson / decimation / normal orient | 미지원 | 미지원 | CPU 유지 |

→ **voxel_down만 GPU**, 나머지는 CPU가 압도적으로 빠름/GPU 미지원.

## 11. 최종 Mesh Reconstruction Pipeline

1. RTAB-Map DB → PLY 추출 (풀해상도, rtabmap-export --cloud)
2. **voxel_down 0.01m (CUDA)** + statistical outlier (CPU)
3. normal estimation (CPU, tangent-plane 일관 정렬)
4. **Poisson depth=8** (CPU, watertight)
5. density 3% 제거 + bbox crop
6. quadric decimation 50% + topology 정리
7. RGB nearest-neighbor 전사 → .obj
- 실측 총 시간: 175.6s (66MB ply, 2.2M 포인트)

## 12. GPU를 실제 사용하는 연산
- Open3D voxel_down_sample (mesh 파이프라인, CUDA Tensor API)
- (RViz2는 소프트웨어 렌더링, NVIDIA GPU와 무관)

## 13. CPU에서 유지하는 연산
- realsense2_camera 스트림/필터 (공간·시간 필터)
- RTAB-Map SLAM 전체 (odometry/loop closure/graph — CPU-only 빌드)
- Open3D normals / outlier / Poisson / decimation
- madgwick IMU 필터

## 14. 최종 파라미터 전체 목록
- 카메라: `config.py CAMERA_PARAMS` (QoS 키 제거, 필터 개별 enable, 640x480x30, sync, IMU)
- RTAB-Map: `launch_common.py RTABMAP_PARAMS/ODOM_PARAMS` (PnP, MaxFeatures 1000, DetectionRate 5, STM 10, OdomF2M/MaxSize 1000, LinearUpdate 0.10 등)
- Mesh: `MESH_DEFAULTS` (depth 8, voxel 0.01, poisson, simplify 0.5) + GPU voxel
- RViz: `rtabmap_live.rviz` (Image Reliable, cloud_map_lite 2Hz)

## 15. benchmark 결과 (요약)

| 항목 | 결과 |
|---|---|
| Depth 640x480x30 (필터 ON) | 29.7fps, 손상 0 |
| Color 640x480x30 | 30.0fps |
| IMU | 191Hz (gyro+accel) |
| RTAB-Map odom (실시간) | 7.2Hz (정적 장면), RViz ON |
| cloud_map_lite | 2.1Hz |
| 파이프라인 CPU | ~55% (카메라 19% + VO 25% + 맵 7% + cloud 4%) |
| voxel_down 2.2M 포인트 | CPU 2183ms → GPU 12.7ms |
| 전체 mesh (66MB ply) | 175.6s |

## 16. 기존 VMware 환경 문제의 원인 분석
- (과거 요소는 추정을 명시)
1. **USB 대역폭 제한**: VMware USB 패스스루에서 848x480 depth 19.4Hz 드랍(추정: VM USB 스케줄링 오버헤드). → 640x480 고정.
2. **vCPU 8 제한**: RTAB-Map + RViz 경쟁으로 odom 11Hz 이하.
3. **GPU 부재**: Open3D/렌더링 전부 CPU. RViz 소프트웨어 렌더링으로 VO 부하 간섭.
4. **RViz D-state**: VMware 가상 GPU 경로에서 멈춤 → LIBGL_ALWAYS_SOFTWARE=1로 우회.
5. **IMU**: VMware에서 동작 여부 미검증이었으나, config에 IMU 필터 체인이 이미 구성됨.

## 17. 현재 WSL2 환경에서 해결된 문제
1. **GPU 사용 가능** (RTX PRO 2000): Open3D voxel_down 172배 가속. RTAB-Map은 CPU 유지.
2. **IMU 완전 활성화**: 커널 재빌드(HID_SENSOR_HUB/CUSTOM_SENSOR) + udev 규칙(IIO/scan_elements/buffer/trigger/enable_sensor 권한). pyrealsense2에서 Motion Module + 191Hz 데이터 실측.
3. **WSLg GPU 렌더링**: glxinfo D3D12(Arc). 단 RViz2는 검은 화면 문제로 소프트웨어 렌더링 유지(성능 영향 없음, cloud 2Hz).
4. **USB 속도**: usbipd 5.3.0으로 640x480x30 무손실 (VMware 대비 개선은 없으나 동일 수준 유지).
5. **카메라 FPS 저하(신규 회귀) 해결**: v4.58.3 QoS 파라미터 버그 실측 규명.

## 18. 남아 있는 문제와 한계
1. **USB 패스스루 근본 한계**: 848x480 이상 스트림은 프레임 손상. 고해상도 재구성이 필요하면 별도 수집 전략 필요.
2. **재부팅 후 IMU 재검증 미실시**: udev 규칙이 커널 재부팅 후 적용되는지 다음 부팅 시 확인 필요 (`ls /sys/bus/hid/devices | grep 2000e1`).
3. **IMU 융합 품질 정량 평가 미완료**: madgwick+rtabmap 융합의 trajectory 개선을 rosbag 재현으로 정량화 필요.
4. **RMSE 등 trajectory 정확도 메트릭 미측정**: 기준 거리/GT 부재.
5. **mesh 품질 육안 검증 필요**: 생성 obj(25MB)의 geometry 보존 확인 (뷰어).
6. **CUDA toolkit(nvcc) 미설치**: 커스텀 CUDA 코드 필요 시 설치.

## 19. 향후 Digital Twin / Simulator 연동 대비 데이터 보존
- 보존 대상 (모두 실체 보존됨):
  - raw RGB/Depth/IMU/카메라 정보 → rosbag (record.sh)
  - trajectory/optimized pose → RTAB-Map DB (Node.pose, OptimizedPoses)
  - point cloud → PLY (풀해상도 추출)
  - mesh → OBJ (일반 포맷, 메타데이터 포함)
  - reconstruction 파라미터 → MESH_DEFAULTS/config.py (버전 관리됨)
- 주의점:
  - 좌표계: map(rtabmap) → camera_link(frame_id) 일관 유지. mesh는 RTAB-Map 맵 좌표로 저장되어 simulator에서 anchor로 사용.
  - scale: voxel 0.01m 기준 일관된 미터 단위.
  - **Isaac Sim 설치/실행은 이번 범위에서 제외** — mesh 포맷(OBJ)과 좌표 규약만 보존.
  - 공유 폴더 `/mnt/c/ubuntu_shared`로 mesh 자동 복사 (Windows/Isaac 접근용).

---
*측정 주의: Python rclpy 소비자는 30fps 이미지에서 유실이 크므로 모든 FPS/드랍 수치는 rosbag(C++) 기록 기준. 타 사용자 동시 카메라 사용 시 수치 오염 가능.*
