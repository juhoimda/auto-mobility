# 📐 Auto-Mobility

RealSense D435i 기반 **Real-to-Sim 파이프라인** — 실제 공간을 촬영해 다중 SLAM(RTAB-Map / ORB-SLAM3)으로 궤적을 추적하고, Open3D GPU TSDF로 고정밀 3D 점군(Point Cloud) 및 표면 메시(Mesh)를 복원하여 Isaac Sim 디지털 트윈으로 검증합니다.

```text
RealSense D435i ──▶ Multi-SLAM (RTAB / ORB-SLAM3) ──▶ Point Cloud (.ply) ──▶ Open3D TSDF (GPU) ──▶ 3D Mesh (.obj) ──▶ Isaac Sim
(RGB+Depth+IR+IMU)        (Visual / Feature SLAM)           (3D 점군 데이터)       (VoxelBlockGrid)      (디지털 트윈)        (물리 검증)
```

---

## 🚀 Quick Start (초간단 핵심 명령어)

```bash
# 1. RAW 데이터셋 녹화 (RGB-D + Stereo IR + IMU)
./scripts/pipeline/capture_safe.sh my_dataset --view

# 2. [선택 A] RTAB-Map SLAM 실행 및 메쉬+점군 생성 (1-명령어)
./scripts/pipeline/mesh.sh my_dataset --view

# 3. [선택 B] ORB-SLAM3 실행 및 메쉬+점군 생성 (1-명령어)
./scripts/pipeline/mesh.sh my_dataset --slam=orb --view

# 4. [선택 C] 5mm 초고정밀 TSDF 메쉬 생성
./scripts/pipeline/mesh.sh my_dataset --fine --view

# 5. [선택 D] 모든 알고리즘 한 번에 비교 분석 (리포트 자동 생성)
python3 src/auto_mobility/slam/compare_algorithms.py my_dataset

# 6. 생성된 3D 결과물 확인
./scripts/utils/view_pointcloud.sh my_dataset_tsdf_cloud.ply  # 3D 점군 보기
./scripts/utils/view_mesh.sh my_dataset_tsdf.obj              # 3D 메쉬 보기
```

자세한 단계별 실행 방법: **[docs/guide.md](docs/guide.md)**

---

## 🏗️ 시스템 구성

| 구성요소 | 역할 |
| :--- | :--- |
| **RealSense D435i** | RGB(640x480@30) + Depth(Z16) + Stereo IR(Infra1/2) + IMU(200Hz) (로컬 USB 또는 Windows 원격 구동) |
| **RTAB-Map** | 실시간 Visual-Inertial SLAM → 공간 DB(`.db`) 및 전역 포즈 그래프 최적화 |
| **ORB-SLAM3** | 특징점 기반 고감도 Visual SLAM (RGB-D / Stereo) → 정밀 궤적(`.txt`) 추적 |
| **Open3D (CUDA TSDF)** | VoxelBlockGrid GPU 가속 기반 고밀도 3D 점군(`.ply`) 및 표면 메쉬(`.obj`) 추출 |
| **Isaac Sim** | 생성된 Mesh(`.obj`)의 물리 충돌체 검증 및 USD 디지털 트윈 씬 로드 |

### 📁 표준 데이터 아키텍처 (`ros2_data/`)

```bash
ros2_data/
├── bags/                        # 1단계: 원본 센서 스트림 (MCAP 불변 데이터셋)
│   └── <dataset_name>/
│       ├── <dataset_name>_0.mcap
│       └── dataset_manifest.json
├── databases/                   # 2단계: RTAB-Map SLAM 세션 DB (.db)
├── trajectories/                # 2단계: 6자유도 카메라 이동 궤적 (TUM .txt)
├── pointclouds/                 # 3단계: [분석용] 3D 점군 데이터 (PLY)
│   ├── <dataset_name>_raw_cloud.ply     # Depth 투영 원본 점군 (노이즈/정합 분석용)
│   └── <dataset_name>_tsdf_cloud.ply    # TSDF 복셀 가중치 필터링 점군
├── meshes/                      # 4단계: [디지털트윈] 최종 3D 표면 메쉬 (OBJ)
│   └── <dataset_name>_tsdf.obj          # Open3D GPU TSDF 표면 메쉬
└── benchmarks/                  # 5단계: 다중 알고리즘 비교 리포트
    └── bench_<dataset_name>_<date>/
        ├── benchmark_report.md          # 점군+메쉬+궤적 비교 요약 마크다운
        └── benchmark_summary.json       # 정량 수치 JSON
```

---

## ⚙️ 핵심 설정 요약

> 모든 설정은 **단일 소스**(`config.py`, `topics.yaml`)로 관리됩니다.

### 1. 센서 스트림 — `config/topics.yaml` & `src/auto_mobility/config.py`
| 설정 | 값 | 비고 |
| :--- | :--- | :--- |
| 해상도 / FPS | `640x480@30` | WSL2 및 원격 네트워크 대역폭 최적치 |
| Stereo IR | `infra1/infra2` Y8 포맷 | 스테레오 매칭 및 저조도/질감 부족 환경 지원 |
| IMU 필터 | Madgwick (400Hz) | 자세(Roll/Pitch) 수평 보정 |

### 2. 3D 점군 및 메쉬 복원
| 방식 | 대상 스크립트 | 주요 파라미터 | 산출물 |
| :--- | :--- | :--- | :--- |
| **TSDF (기본)** | `src/auto_mobility/mesh/reconstruct_tsdf.py` | `--voxel 0.01` (10mm)<br>`--voxel 0.005` (5mm Fine) | `pointclouds/*_tsdf_cloud.ply`<br>`meshes/*_tsdf.obj` |
| **Poisson (PLY)** | `scripts/utils/export_ply.sh` + `mesh_open3d.py` | `--decimation 1 --max_range 4 --depth 8` | `pointclouds/*_raw_cloud.ply`<br>`meshes/*_mesh.obj` |

---

## ✨ 핵심 특징

1. **Multi-SLAM 백엔드 지원** — RTAB-Map(OdomF2M)과 ORB-SLAM3 C++14 엔진을 모두 지원하여 상황에 맞는 최적 궤적 선택 가능
2. **효율+효과 통합 데이터 관리** — 원본 Bag부터 SLAM 궤적, 3D 점군(`.ply`), 최종 메쉬(`.obj`)까지 단계별 독립 저장 및 분석
3. **독립된 3D 뷰어 환경** — 3D 점군 전용 뷰어(`view_pointcloud.sh`)와 메쉬 전용 뷰어(`view_mesh.sh`)로 분리 제공
4. **GPU 가속 TSDF 재구성** — Open3D Tensor VoxelBlockGrid(CUDA)를 통해 수백만 폴리곤의 고정밀 메쉬를 10초 내외로 고속 복원
5. **Windows ↔ WSL2 하이브리드 연동** — Windows 측 초경량 퍼블리셔(`realsense_pub.py`)와 WSL2 CycloneDDS/압축 스트림 완벽 연동
