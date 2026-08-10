# 📊 2026-08-10 실측 벤치마크 최종 보고서 (D435i + RTAB-Map + Open3D)

본 보고서는 **현재 HW 환경(VMware Ubuntu 22.04, USB 3.2, 8vCPU, 31GB)**에서
카메라/RTAB-Map/Open3D Mesh 파이프라인의 최적 설정을 **직접 실측**한 결과입니다.

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
| `Vis/EstimationType` | `1` (PnP) | SVD 대비 CPU 40% 절감 |
| `OdomF2M/MaxFrames` | `10` | F2M=60은 SLAM 초기화 실패 + VO 지연 누적 |
| `Mem/STMSize` | `10` | STM=100 대비 RAM 절감, 성능 동일 |
| `Vis/MinInliers` | `10` | IN=6 대비 +4~5점 |
| `Vis/MaxDepth` | `4.0` | 원거리 노이즈 필터 (MD=4/8 동일 성능) |
| `RGBD/LinearUpdate` | `0.10` | 10cm 키프레임 간격 |
| `Rtabmap/DetectionRate` | `5` | 5Hz 맵핑 루프 |

### Open3D Mesh (`src/auto_mobility/mesh/mesh_open3d.py`)
| 항목 | 값 | 근거 |
|:---|:---|:---|
| `--method` | `poisson` (기본) | BPA(17.5m²) vs Poisson(115.9m²) 표면 커버리지 |
| `--depth` | `8` | depth=9는 vertex 폭증 |
| `--voxel` | `0.005` | 640x480 depth 해상도에 최적 |
| `--simplify` | `0.5` | Quadric decimation 50% |

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
- **초기 benchmark(30.2Hz)가 맞았던 이유**: 카메라 콜드 상태에선 USB 버퍼 여유가 있었으나,
  발열/지속 스트리밍 시 드랍 시작 → **촬영 중 끊김의 1차 원인**

### 2.2 benchmark가 "촬영 끊김"을 놓친 이유
- 기존 benchmark는 **SLAM 시작 후 5초 창**만 측정 → odom 27~30Hz 기록
- 실측 결과: 촬영 8초 후 **28Hz → 18초 후 14Hz → 28초 후 12Hz**로 저하
  (VO update time 23ms → 40ms → 70ms 누적 증가)
- **즉, 시작 5초의 성능이 지속 성능이 아님** — 촬영은 수 분 지속되므로 실사용 끊김 발생

### 2.3 촬영 시간 경과에 따른 VO 저하는 파라미터로 해결 불가
- F2M=5/10/60, LinearUpdate=0.1/0.3, MaxFeatures=500/2000, Grid on/off, IMU on/off
  모두 시간 경과 저하를 해결하지 못함 → **VM의 근본적인 CPU/SLAM 한계**
- 대응책: **2~3분 단위 세션 분할 촬영** + `capture_guard.py` 모니터링으로 실시간 감지

### 2.4 잘못된 파라미터명 수정
- `rtab_live/rtab_bag.launch.py`의 `always_process_most_recent_frame`는
  rtabmap.launch.py에 **없는 인자** (무시됨)
- 올바른 인자명: **`odom_always_process_most_recent_frame`** → `false`로 수정
  (모든 프레임 순서 처리 → 맵 밀도 확보)

### 2.5 Mesh 품질: BPA vs Poisson 실측 비교
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
- `ros2_data/logs/slam_bench_s1_20260810_133723.json` (Stage 1: 해상도 비교)
- `ros2_data/logs/slam_bench_s2_20260810_131421.json` (Stage 2: SLAM 파라미터 8조합)
- `ros2_data/logs/capture_guard_20260810_135237.md` (통합 촬영 모니터링)

## 5. ⚠️ 남은 위험 요소
1. **장시간 촬영 VO 저하**: 근본 해결 불가 (VM 한계) → 세션 분할 촬영 필수
2. **RViz 소프트웨어 렌더링**: VMware GPU passthrough 없음 → 촬영 중 RViz는 성능 저하 유발
   (촬영 중 RViz 최소화 또는 `rtab_live`에서 rviz=false 권장)
3. **thermal 스로틀링**: 온도 센서 데이터가 `/sys/class/thermal`에 없어 직접 모니터링 불가
