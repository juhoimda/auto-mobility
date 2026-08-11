# 📐 Auto-Mobility

RealSense D435i 기반 **Real-to-Sim 파이프라인** — 실제 공간을 촬영해 3D 메시로 복원하고, Isaac Sim에서 디지털 트윈으로 검증합니다.

```
RealSense D435i ──▶ RTAB-Map SLAM ──▶ Point Cloud(.ply) ──▶ Mesh(.obj) ──▶ Isaac Sim
(RGB+Depth+IMU)      (실시간 맵핑)      (풀해상도 추출)      (Poisson 복원)   (디지털 트윈)
```

---

## 🚀 Quick Start

```bash
# 1. 전체 파이프라인 실행 (촬영 → Mesh → Isaac Sim 검증)
./scripts/pipeline/run_pipeline_all.sh

# 2. Mesh 변환까지만 (Isaac Sim 생략) — 권장
./scripts/pipeline/run_pipeline_all.sh --skip-isaac

# 3. 기존 DB로 Mesh만 재생성
./scripts/pipeline/run_pipeline_all.sh --db=my_room.db --skip-capture --skip-isaac
```

자세한 실행 방법: **[docs/guide.md](docs/guide.md)**

---

## 🏗️ 시스템 구성

| 구성요소 | 역할 |
| :--- | :--- |
| **RealSense D435i** | RGB(640x480@30) + Depth(Z16) + IMU(200Hz) |
| **RTAB-Map** | 실시간 Visual-Inertial SLAM → 공간 DB(.db) 생성 |
| **Open3D** | DB에서 풀해상도 Point Cloud(.ply) 추출 후 Poisson Mesh 복원 |
| **Isaac Sim** | 생성된 Mesh(.obj)의 물리 충돌 검증 |

### 디렉터리 구조

```
auto-mobility/
├── src/auto_mobility/     # Python 모듈 (config / launch / mesh / slam / utils)
├── scripts/               # 셸 실행 도구 (pipeline / utils)
├── config/                # FastDDS / RViz2 / 토픽 설정
├── launch/                # ROS2 launch 파일 (camera / rtab_live / rtab_bag)
├── ros2_data/             # 생성 데이터 (databases / pointclouds / meshes / logs)
└── docs/                  # 가이드 및 벤치마크 보고서
```

---

## ⚙️ 핵심 설정 요약

> 모든 설정은 **단일 소스**로 관리됩니다. 값을 수정하려면 아래 파일만 수정하면 전체에 적용됩니다.

### 카메라 — `src/auto_mobility/config.py` (`CAMERA_PARAMS`)
| 설정 | 값 | 비고 |
| :--- | :--- | :--- |
| 해상도 / FPS | `640x480@30` | 848x480은 VM에서 depth 드랍 |
| IR 에미터 | `emitter_enabled: 1` | 무벽면 특징점 확보 |
| Depth 필터 | `spatial + temporal + hole_filling` | 노이즈 제거 |

### SLAM — `src/auto_mobility/launch/launch_common.py` (`RTABMAP_PARAMS`)
| 설정 | 값 | 비고 |
| :--- | :--- | :--- |
| 포즈 추정 | `Vis/EstimationType 1` (PnP) | 벤치마크 1위 |
| 특징점 | `Vis/MaxFeatures 1000` | CPU/성능 균형 |
| 로컬 맵 크기 | `OdomF2M/MaxSize 1000` | **★ 벤치 1위** — 기본 2000은 odom 17Hz 급락 |
| 키프레임 간격 | `RGBD/LinearUpdate 0.10` | 10cm 마다 |
| Point Cloud 밀도 | `Grid/DepthDecimation 2` | 기본 4 대비 4배 고밀도 |

### Mesh — `scripts/utils/export_ply.sh` + `src/auto_mobility/mesh/mesh_open3d.py`
| 단계 | 값 | 비고 |
| :--- | :--- | :--- |
| PLY 추출 | `--decimation 1 --max_range 5` | **풀해상도** — 기본 4는 1/16 밀도 |
| 복원 | Poisson `--depth 8` | 폐곡면 보완 |
| 해상도 | `--voxel 0.005` | 5mm |
| 경량화 | `simplify 0.5` | Isaac Sim 로딩 성능 확보 |

---

## ✨ 특징

1. **IMU 융합 Visual SLAM** — D435i IMU(200Hz)를 Madgwick 필터로 처리해 맵 수평 유지 및 추적 안정화
2. **VMware 최적화** — FastDDS SHM + vCPU 8 활용, 640x480에서 안정적인 30FPS 확보
3. **자동 품질 검증** — DB → Point Cloud → Mesh 3단계 무결성 검사 + 촬영 품질 모니터링(`capture_guard`)

---

## 📚 문서

| 문서 | 내용 |
| :--- | :--- |
| [docs/guide.md](docs/guide.md) | 실행 명령어 및 단계별 사용법 |
| [docs/cur-setting.md](docs/cur-setting.md) | 현재 적용된 설정 상세 명세 |
| [docs/benchmark.md](docs/benchmark.md) | 하드웨어 실측 벤치마크 보고서 |
