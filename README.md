# 📐 Auto-Mobility

RealSense D435i 기반 **Real-to-Sim 파이프라인** — 실제 공간을 촬영해 다중 SLAM(RTAB-Map / ORB-SLAM3 / ORB RGB-D-I / stella_vslam)으로 궤적을 추적하고, Open3D GPU TSDF 및 다양한 표면 복원(TSDF Direct / Poisson / BPA / Alpha Shape / CGAL Polygonal)으로 고정밀 3D 점군 및 표면 메시를 생성하며, Held-out 센서 관측 기반 정량 형상 품질 평가(Geometry QA) 및 다축 벤치마크 랭킹을 제공합니다.

```text
RealSense D435i ──▶ Rosbag (MCAP) ──▶ Canonical Dataset ──▶ Multi-SLAM Trajectory ──▶ Surface Reconstruction ──▶ Multi-Axis Evaluator
 (RGB-D+IR+IMU)     (불변 원본)         (알고리즘 독립)      (RTAB/ORB-RGBDI/stella)   (TSDF/Poisson/Alpha/CGAL)    (Depth MAE/P95 QA)
```

---

## 🚀 Quick Start (권장 워크플로우)

```bash
# 1. RAW 데이터셋 녹화 (RGB-D + Stereo IR + IMU)
./scripts/pipeline/capture_safe.sh room01 --view

# 2. 알고리즘 독립적인 Canonical Frame Dataset 생성 (1회 추출 후 재사용)
./scripts/pipeline/prepare_dataset.sh room01

# 3. SLAM 궤적 생성 (원하는 백엔드 선택)
./scripts/pipeline/run_slam.sh room01 --slam=rtab        # RTAB-Map
./scripts/pipeline/run_slam.sh room01 --slam=orb_rgbdi   # ORB-SLAM3 RGB-D-Inertial

# 4. 3D Mesh 및 Point Cloud 복원 (Surface 백엔드 및 해상도 선택)
./scripts/pipeline/mesh.sh room01 --surface=tsdf_direct --voxel=0.01
./scripts/pipeline/mesh.sh room01 --surface=alpha --voxel=0.02

# 5. Held-out 센서 데이터 기반 정량 형상 품질 평가 (Depth Reprojection & Point-to-Mesh)
./scripts/pipeline/evaluate.sh room01 \
    ros2_data/meshes/room01_rtab_tsdf.obj \
    ros2_data/trajectories/rtab_room01_trajectory.txt

# 6. Multi-Axis 독립 벤치마크 & 랭킹 리포트 (Phase A: SLAM / Phase B: TSDF / Phase C: Surface)
./scripts/pipeline/compare.sh room01 --quick
```

자세한 단계별 가이드: **[docs/guide.md](docs/guide.md)**

---

## 🏗️ 시스템 아키텍처 (Layered Architecture)

모든 모듈은 특정 알고리즘의 내부 파일 포맷에 종속되지 않고 독립적인 표준 포맷으로 연결됩니다.

| 계층 | 대상 모듈 | 입력 | 산출물 | 지원 알고리즘 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Sensor Ingress** | `republish.py` | D435i Topics | Rosbag (`.mcap`) | 무손실 패스스루 / 시간 동기화 |
| **2. Canonical Dataset** | `dataset/` | Rosbag | `frames/` | 표준 RGB-D + IMU + CameraInfo |
| **3. SLAM Backend** | `slam/` | Rosbag / Frames | `trajectories/` (TUM `.txt`) | `rtab`, `orb_rgbd`, `orb_rgbdi`, `stella_rgbd` |
| **4. Reconstruction** | `mesh/` | Frames + Trajectory | `meshes/`, `pointclouds/` | `tsdf_direct`, `poisson`, `bpa`, `alpha_shape`, `cgal_polygonal` |
| **5. Geometry Evaluator** | `evaluation/` | Mesh + Trajectory + Held-out Depth | `evaluations/`, `benchmarks/` | Raycast Depth MAE/P95, Coverage, 축별 랭킹 |

### 📁 표준 디렉터리 구조 (`ros2_data/`)

```bash
ros2_data/
├── bags/                        # 1단계: 원본 센서 스트림 (MCAP 불변 데이터셋)
├── frames/                      # 2단계: Canonical Frame Dataset (표준 RGB-D, camera_info.json, imu.csv)
├── databases/                   # 3단계: RTAB-Map SLAM DB (.db)
├── trajectories/                # 3단계: 표준 TUM 포맷 카메라 이동 궤적 (.txt)
├── pointclouds/                 # 4단계: 3D 점군 데이터 (.ply)
├── meshes/                      # 4단계: 최종 3D 표면 메쉬 (.obj)
├── evaluations/                 # 5단계: 단일 후보 정량 품질 평가 결과 (JSON, MD, frame_metrics.csv)
└── benchmarks/                  # 6단계: Multi-Axis 벤치마크 결과 및 Manifest 리포트
```

---

## 📊 정량 품질 평가 지표 (Geometry QA)

* **Depth MAE / RMSE / Median (mm)**: Held-out 센서 프레임의 실제 Depth와 가상 Mesh 렌더링 Depth 간의 절대 오차.
* **Depth P90 / P95 (mm)**: 90%/95% 신뢰 구간 최대 오차 (벽면 굴곡 및 국소적 왜곡 감지).
* **Depth Coverage (%)**: 실제 센서 유효 관측 영역 중 메쉬가 존재하는 비율.
* **Point-to-Mesh Distance (mm)**: 역투영된 3D 포인트들과 메쉬 표면 간의 최단거리.
* **Mesh Topology & Plane Residuals**: 비다양체 엣지, 찌그러진 삼각면, 부유 파편 비율 및 RANSAC 평면 잔차.
