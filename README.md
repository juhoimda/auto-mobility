# 📐 Auto-Mobility

RealSense D435i 기반 **Real-to-Sim 파이프라인** — 실제 공간을 촬영해 3D 메시로 복원하고, Isaac Sim에서 디지털 트윈으로 검증합니다.

```
RealSense D435i ──▶ RTAB-Map SLAM ──▶ RGB-D & Pose / PLY ──▶ Open3D TSDF / Poisson ──▶ Isaac Sim
(RGB+Depth+IMU)      (실시간 맵핑)        (풀해상도 데이터)         (GPU 3D Mesh 복원)       (디지털 트윈)
```

---

## 🚀 Quick Start

```bash
# 1. 전체 파이프라인 실행 (촬영 → TSDF Mesh → Isaac Sim 검증)
./scripts/pipeline/run_pipeline_all.sh

# 2. Windows 원격 카메라 자동 구동 모드
./scripts/pipeline/run_pipeline_all.sh --remote-camera

# 3. Mesh 변환까지만 (Isaac Sim 생략) — 권장
./scripts/pipeline/run_pipeline_all.sh --skip-isaac

# 4. 기존 DB로 Mesh만 재생성
./scripts/pipeline/run_pipeline_all.sh --db=my_room.db --skip-capture --skip-isaac

# 5. RAW 데이터셋 확보 전용 (SLAM/RViz 없이 녹화 + 자동 검증 + 매니페스트)
./scripts/pipeline/capture_safe.sh my_dataset

# 6. 녹화 후 데이터셋 검증 (메시지 수 / Hz / sync / gap / 매니페스트)
python3 src/auto_mobility/utils/validate_bag.py ros2_data/bags/my_dataset
```

자세한 실행 방법: **[docs/guide.md](docs/guide.md)**

---

## 🏗️ 시스템 구성

| 구성요소 | 역할 |
| :--- | :--- |
| **RealSense D435i** | RGB(640x480@30) + Depth(Z16) + IMU(200Hz) (로컬 USB 또는 Windows 원격 구동) |
| **RTAB-Map** | 실시간 Visual-Inertial SLAM → 공간 DB(`.db`) 생성 |
| **Open3D** | DB의 원본 RGB-D + Pose 기반 Tensor TSDF(GPU) 복원 또는 풀해상도 PLY Poisson 복원 |
| **Isaac Sim** | 생성된 Mesh(`.obj`)의 물리 충돌 및 USD 씬 로드 검증 |

### 디렉터리 구조

```
auto-mobility/
├── src/auto_mobility/     # Python 모듈 (config / launch / mesh / monitor / nodes / slam / utils)
├── scripts/               # 셸 실행 도구 (pipeline / utils)
├── config/                # FastDDS / RViz2 / 토픽 설정
├── launch/                # ROS2 launch 파일 (camera / rtab_live / rtab_bag)
├── ros2_data/             # 생성 데이터 (databases / pointclouds / meshes / bags / logs)
└── docs/                  # 사용자 가이드 및 개발 문서
```

---

## ⚙️ 핵심 설정 요약

> 모든 설정은 **단일 소스**(`config.py`, `topics.yaml`)로 관리됩니다.

### 카메라 — `src/auto_mobility/config.py` (`CAMERA_PARAMS`)
| 설정 | 값 | 비고 |
| :--- | :--- | :--- |
| 해상도 / FPS | `640x480@30` | 848x480은 VM에서 depth 드랍 방지 최적치 |
| IR 에미터 | `emitter_enabled: 1` | 텍스처 부족 벽면 특징점 확보 |
| Depth 필터 | `spatial + temporal + hole_filling` | 노이즈 제거 |

### SLAM — `src/auto_mobility/launch/launch_common.py` (`RTABMAP_PARAMS`)
| 설정 | 값 | 비고 |
| :--- | :--- | :--- |
| 포즈 추정 | `Vis/EstimationType 1` (PnP) | 벤치마크 1위 |
| 특징점 | `Vis/MaxFeatures 2000` | 회전 시 키프레임 중첩 확보 |
| 특징점 품질 | `GFTT/QualityLevel 0.005` | blur/회전 프레임 특징점 누락 방지 |
| 로컬 맵 크기 | `OdomF2M/MaxSize 1000` | **★ 벤치 1위** — 기본 2000 대비 안정적 실시간 연산 |
| 키프레임 간격 | `RGBD/LinearUpdate 0.10` | 10cm 마다 키프레임 갱신 |
| 루프클로저 | `OptimizeFromGraphEnd/NeighborLinkRefining/ProximityBySpace false` | 루프클로저 보정 활성화, 맵 스레드 스톨 방지 |
| 맵핑 주기 | `Rtabmap/DetectionRate 3` | 맵 스레드 부하 최적화 (3Hz) |
| Point Cloud 밀도 | `Grid/DepthDecimation 4` | 라이브 맵 경량화 (최종 메쉬는 offline TSDF) |

### 3D Mesh 복원
| 방식 | 대상 스크립트 | 주요 설정 | 비고 |
| :--- | :--- | :--- | :--- |
| **TSDF (기본)** | `src/auto_mobility/mesh/reconstruct_tsdf.py` | `--voxel 0.01` | Open3D Tensor GPU 적분, 원본 RGB-D + Pose 통합 |
| **Poisson (PLY)** | `scripts/utils/export_ply.sh` + `mesh_open3d.py` | `--decimation 1 --max_range 4 --depth 8` | 풀해상도 PLY 추출 기반 폐곡면 표면 복원 |

---

## ✨ 핵심 특징

1. **IMU 융합 Visual SLAM** — D435i IMU(400Hz)를 Madgwick 필터로 처리해 수평 보정 및 오도메트리 추적 안정화
2. **WSL2 / Windows 하이브리드 연동** — 신호 기반 자동 카메라 watcher, CycloneDDS 정적 피어, 압축 토픽 디코딩(`republish.py`) + **원본 타임스탬프 보존**
3. **GPU 가속 TSDF 3D 재구성** — Open3D Tensor TSDF Voxel Grid로 고밀도 실내 공간 3D Mesh(`.obj`) 생성
4. **자동 품질 검증 및 안전 장치** — DB 무결성 검증(`validate.py`), 촬영 품질 모니터링(`capture_guard`, sync/gap 포함), **녹화 데이터셋 자동 검증 + 매니페스트**(`validate_bag.py`), RAM 디스크(`/dev/shm`) 우선 녹화 및 SSD 자동 이관

---

## 📚 문서

| 문서 | 내용 |
| :--- | :--- |
| [docs/guide.md](docs/guide.md) | 전체 실행 명령어 및 단계별 사용법 |
| [docs/proposal_2026.pdf](docs/proposal_2026.pdf) | 프로젝트 제안서 및 기술 문서 |

