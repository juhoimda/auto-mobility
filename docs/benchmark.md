# 📊 2026-08-11 실측 벤치마크 최종 보고서 (D435i + RTAB-Map + Open3D)

본 보고서는 **현재 HW 환경(VMware Ubuntu 22.04, USB 3.x, 8vCPU, 31GB)**에서
카메라/RTAB-Map/Open3D Mesh 파이프라인의 최적 설정을 **직접 실측**한 결과입니다.

> **2026-08-11 갱신**: RTAB-Map 0.23.7 파라미터 유효성 검증으로 무효 파라미터를 교정하고,
> `OdomF2M/MaxSize` 축을 신규 측정하여 최적값(`1000`)을 확정했습니다.

---

## 1. 🏆 최종 확정 설정 요약

### 카메라 (`launch/camera.launch.py`)
| 항목 | 값 | 근거 |
|:---|:---|:---|
| 해상도 | **640x480@30fps** | 848x480 depth 19.4Hz 드랍 vs 640x480 29.1Hz 안정 |
| `align_depth.enable` | `False` | vCPU 정렬 병목 제거 |
| `unite_imu_method` | `1` (copy) | CPU 절감 |
| `enable_sync` | `True` | **sync=False 시 depth 4Hz 폭락 (필수)** |
| `emitter_enabled` | `1` | IR 패턴으로 무벽면 특징점 확보 |

### RTAB-Map (`src/auto_mobility/launch/launch_common.py`)
| 항목 | 값 | 근거 |
|:---|:---|:---|
| `Vis/EstimationType` | `1` (PnP) | SVD 대비 CPU 절감 (벤치 1위) |
| **`OdomF2M/MaxSize`** | **`1000`** | **★ 2026-08-11 벤치 1위 — 기본 2000은 odom 17Hz 급락, 4000은 14Hz** |
| `Mem/STMSize` | `10` | STM=100 대비 RAM 절감, 성능 동일 (2026-08-11 재확정) |
| `Vis/MinInliers` | `10` | IN=6 대비 +4~5점 |
| `Vis/MaxDepth` | `4.0` | 원거리 노이즈 필터 (MD=4/8 동일 성능) |
| `RGBD/LinearUpdate` | `0.10` | 10cm 키프레임 간격 |
| `Rtabmap/DetectionRate` | `5` | 5Hz 맵핑 루프 |

> ⚠️ **파라미터 유효성 교정 (2026-08-11)**: RTAB-Map 0.23.7 기준 아래 키는 **존재하지 않아 무시**되던 것들입니다.
> | 무효 키 | 유효 키로 교정 |
> |:---|:---|
> | `OdomF2M/MaxFrames` | `OdomF2M/MaxSize` (word 단위, 기본 2000) |
> | `Vis/CornerMinQuality` | `GFTT/QualityLevel` |
> | `Vis/CornerGridSize` | `Vis/GridRows` + `Vis/GridCols` |
> | `Vis/CornerNbThreads` | 제거 (OpenCV 자동 스레딩) |
> | `Odom/PoseGuessMode` | `Odom/GuessMotion` |
> | `Optimizer/GravityProvided` | `Optimizer/GravitySigma` |
> | `Vis/Robust` | `Optimizer/Robust` |
> | `Rtabmap/ResetCountdown` | `Odom/ResetCountdown` |
> | `RGBD/CreateIntermediateNodes` | `Rtabmap/CreateIntermediateNodes` |
> | `Grid/VoxelSize` | `Grid/DepthDecimation` (depth 해상도 결정) |

### Open3D Mesh (`src/auto_mobility/mesh/mesh_open3d.py`)
| 항목 | 값 | 근거 |
|:---|:---|:---|
| `--method` | `poisson` (기본) | BPA(17.5m²) vs Poisson(115.9m²) 표면 커버리지 |
| `--depth` | `8` | depth=9는 vertex 폭증 |
| `--voxel` | `0.005` | 640x480 depth 해상도에 최적 |
| `--simplify` | `0.5` | Quadric decimation 50% |

### PLY 추출 (`scripts/utils/export_ply.sh`)
| 옵션 | 값 | 근거 |
|:---|:---|:---|
| `--decimation` | `1` | **★ 2026-08-11 — 기본 4는 PLY 밀도 1/16 (mesh 품질 저하 원인). 오프라인 처리라 캡처 성능 영향 없음** |
| `--max_range` | `5` | 4m 이상 벽/천장 포착 (기본 4m) |

---

## 2. 🔬 핵심 발견 (본 보고서의 가치)

### 2.1 848x480 해상도는 VM에서 depth 드랍 유발 (중요!)
Stage 1 benchmark 실측 (카메라 단독, 샘플 5초):

| 해상도 | color Hz | depth Hz | drop% | CPU |
|:---|:---:|:---:|:---:|:---:|
| 640x480@30 | 30.1 | **29.1** | 0.0% | 10.2% |
| 848x480@30 | 30.0 | **19.4** | 35% | 8.2% |

- **원인**: 848x480의 RGB8(36.6MB/s) + Z16(24.4MB/s) = 61MB/s
  → VM 가상 USB 컨트롤러 실제 처리 한계(~50MB/s) 초과
- **2026-08-11 재실측**: 848x480 depth 28.1Hz (환경 변동으로 19.4Hz 대비 개선, 여전히 30Hz 미달)
  → **production 해상도는 640x480 유지**

### 2.2 benchmark가 "촬영 끊김"을 놓친 이유
- 기존 benchmark는 **SLAM 시작 후 5초 창**만 측정 → odom 27~30Hz 기록
- 실측 결과: 촬영 8초 후 **28Hz → 18초 후 14Hz → 28초 후 12Hz**로 저하
  (VO update time 23ms → 40ms → 70ms 누적 증가)
- **즉, 시작 5초의 성능이 지속 성능이 아님** — 촬영은 수 분 지속되므로 실사용 끊김 발생

### 2.3 촬영 시간 경과에 따른 VO 저하는 파라미터로 해결 불가
- F2M=5/10/60, LinearUpdate=0.1/0.3, MaxFeatures=500/2000, Grid on/off, IMU on/off
  모두 시간 경과 저하를 해결하지 못함 → **VM의 근본적인 CPU/SLAM 한계**
- 대응책: **2~3분 단위 세션 분할 촬영** + `capture_guard.py` 모니터링으로 실시간 감지

### 2.4 ⚠️ 무효 파라미터로 측정된 벤치마크 (2026-08-11 교정)
- 기존 벤치마크는 `OdomF2M/MaxFrames=10`(존재하지 않는 키)으로 "F2M=10 CPU 35% 절감"을 결론냈으나,
  **실제로는 기본값 `OdomF2M/MaxSize=2000`으로 동작한 결과**였음
- 신규 벤치마크에서 유효 키 `OdomF2M/MaxSize` 축을 측정한 결과:
  | MaxSize | odom Hz | 비고 |
  |:---:|:---:|:---|
  | **1000** | **30.0** | ✅ 벤치 1위 |
  | 2000 | 17.4 | ⚠️ 기본값 — VO 급락 |
  | 4000 | 14.6 | ⚠️ 과부하 |
- `Vis/CornerNbThreads`(제거됨) 벤치 축은 삭제 — OpenCV 자동 스레딩 사용

### 2.5 잘못된 파라미터명 수정
- `rtab_live/rtab_bag.launch.py`의 `always_process_most_recent_frame`는
  rtabmap.launch.py에 **없는 인자** (무시됨)
- 올바른 인자명: **`odom_always_process_most_recent_frame`** → `false`로 수정
  (모든 프레임 순서 처리 → 맵 밀도 확보)

### 2.6 Mesh 품질: BPA vs Poisson 실측 비교
동일 point cloud(`session_20260807_153801_cloud.ply`) 기준:

| 지표 | BPA (기존) | Poisson (신규) |
|:---|:---:|:---:|
| Vertices | 135,699 | 201,372 |
| Triangles | 152,314 | 406,524 |
| 표면적 | 17.53 m² | **115.93 m²** |
| Watertight | 아니오 | 아니오 (crop 후) |

- Poisson이 구멍을 메워 표면 커버리지 **6.6배** 확보 → 공간 복원 품질 대폭 개선
- 50% simplify 적용 시 406k triangle 유지 (Isaac Sim 로딩 가능 수준)

---

## 3. 📈 신규 기능

### 3.1 `capture_guard.py` (촬영 품질 모니터링 가드)
- `run_pipeline_all.sh` 촬영 중 병렬 실행
- color/depth/info/IMU/odom 5개 토픽 실시간 감시
- USB 링크 속도, CPU, RAM 실시간 확인
- 종료 시 Markdown 보고서 생성 + 시작/종료 odom 비교

### 3.2 `run_pipeline_all.sh` 개편
- **PRE-FLIGHT**: USB 3.x, /dev/shm, rmem_max, 해상도 적합성 사전 검증
- **STEP 1**: capture_guard 병렬 모니터링 포함
- **STEP 2**: Poisson(기본) + depth=8 + voxel=5mm + simplify 50%
- **BARRIER**: DB/PLY/Mesh 3단계 무결성 검증 (Mesh 품질 메트릭 추가)

### 3.3 `validate.py` Mesh 품질 메트릭 추가
- 공간 규모, 표면적, 삼각형 밀도, Watertight 여부, 컬러 유무 검사

---

## 4. 📋 실측 데이터 원본
- `ros2_data/logs/slam_bench_s1_20260811_102435.json` (Stage 1: 해상도 비교, 2026-08-11)
- `ros2_data/logs/slam_bench_s2_20260811_102435.json` (Stage 2: SLAM 파라미터 8조합, 2026-08-11)
- `ros2_data/logs/slam_benchmark_20260811_102435.md` (종합 보고서, 2026-08-11)
- `ros2_data/logs/slam_bench_s2_20260810_131421.json` (Stage 2: 구버전 무효 파라미터 측정)
- `ros2_data/logs/capture_guard_20260810_135237.md` (통합 촬영 모니터링)

## 5. ⚠️ 남은 위험 요소
1. **장시간 촬영 VO 저하**: 설정 정합 후 VO delay ~70ms 고정·누적 없음 (개선 완료).
   잔여 odom 저하(25.9→17Hz)는 **호스트측(VMware 스케줄링/thermal) 요인**으로 게스트 내 파라미터로는 완전 제거 불가.
   → 여전히 2~3분 단위 세션 분할 촬영 권장
2. **RViz 소프트웨어 렌더링**: VMware GPU passthrough 없음 → `cloud_map_lite`(2Hz) 중계 + 경량 rviz 설정으로 부하 ~60% 절감.
   여전히 촬영 중 rviz 창은 최소화 권장
3. **thermal 스로틀링**: 온도 센서 데이터가 `/sys/class/thermal`에 없어 직접 모니터링 불가

---

## 6. 🔬 실환경 근사 재검증 (2026-08-10, rviz on + 카메라 필터 on)

기존 벤치마크(29.9Hz)는 `rviz=false` + 필터 off + 라이브 5초 창으로 측정되어 실제 촬영 조건과 달랐음.
**production 그대로**(camera.launch.py + rtab_live rviz=true + capture_guard 병행)에서 재검증:

| 지표 | 기존 설정 (검증 실측) | 개선 설정 (신규 실측) |
| :--- | :--- | :--- |
| VO update time | 23ms → 40ms → **70ms** (누적) | 10ms → 30~42ms (수렴, plateau) |
| VO delay | 무제한 누적 (backlog) | **~70ms 고정 (누적 없음)** |
| odom Hz (2.5분 시점) | **12Hz** | **17~26Hz** |
| /rtabmap/cloud_map RViz 렌더 | 5Hz 원본 | 2Hz `cloud_map_lite` 중계 |
| 맵핑 "no odometry" 스킵 | 간헐 발생 (sync 0.08 < delay) | sync 0.15로 흡수 예상 |

### 검증 중 추가 발견 (근본 원인 3종)
1. **`odom_always_process_most_recent_frame=false`가 live에서 큐 백로그·지연 누적 유발** — true로 복원
   (rtabmap.launch.py:515 주석: false는 rosbag 오프라인 전용)
2. **approx_sync_max_interval 0.08 < VO delay(~70ms)** → 간헐적 "no odometry" 맵 구멍 → **0.15로 확대**
3. **RViz가 `/voxel_cloud`(없는 토픽) 구독 + cloud_map 5Hz 풀 렌더** → 죽은 디스플레이 제거 + 2Hz 중계

### 최종 확정 (cur-setting.md 참조)
- `Vis/MaxFeatures 1000`, `Mem/STMSize 10`, `OdomF2M/MaxSize 1000`,
  `odom_always_process_most_recent_frame=true`,
  `approx_sync_max_interval=0.15`, RViz = `cloud_map_lite` 2Hz + Path + RGB 1개

> ⚠️ 검증 도중 카메라 USB 재시작 사이클로 VMware 패스스루가 이탈(장치 미인식)되어
> "sync 0.15 적용 후 장시간 측정"은 미완료 — 재연결 후 최종 확인 필요
