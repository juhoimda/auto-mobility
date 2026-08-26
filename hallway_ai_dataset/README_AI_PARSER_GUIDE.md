# Hallway Dataset & Output Preview AI Data Parsing Guide

본 문서는 **`hallway`** 원시/전처리 데이터셋과 **`output_preview(hallway)`** 3D 재구성 및 SLAM 평가 파이프라인의 모든 세부 데이터를 AI 및 LLM 에이전트가 완벽하게 구조화하여 파싱할 수 있도록 작성된 종합 메타데이터 가이드입니다.

---

## 📁 생성된 파일 목록 및 구조

모든 데이터는 `hallway_ai_dataset/` 디렉토리에 용도별로 분리 저장되어 있습니다.

```
hallway_ai_dataset/
├── 00_manifest_index.json                    # 전체 데이터셋 및 파이프라인 마스터 색인
├── 01_sensor_and_bag_data.json               # ROS2 MCAP Bag 메타데이터, 토픽 통계, 카메라 내부 파라미터(K, R, P), 센서 무결성 진단
├── 02_frames_and_imu_summary.json            # 5,625개 프레임 동기화 통계, 40,395개 IMU 6-DOF 가속도/각속도 통계
├── 03_slam_trajectories_analysis.json        # cuVSLAM vs RTAB-Map 궤적 비교 (TUM 포맷 포즈 5625개, 이동 거리, Bounding Box) 및 RTAB DB 통계
├── 04_preview_pipeline_decisions.json        # 3D Preview 파이프라인 의사결정 추적 (VRAM 예산 관리, 복셀 10mm->15.6mm 강등, 800 프레임 선별)
├── 05_reconstruction_and_quality_comparison.json  # cuVSLAM vs RTAB 3D 메쉬 및 깊이/외관 품질 정밀 벤치마크 (MAE, RMSE, P95, 텍스처 커버리지, 실행시간)
├── 06_artifacts_and_machine_profile.json     # 아티팩트 해시(SHA256), 작업자 스펙, 호스트 머신 사양 (RTX PRO 2000 Blackwell, VRAM, WSL)
└── README_AI_PARSER_GUIDE.md                 # 본 가이드 문서
```

---

## 📊 핵심 데이터 요약 (Key Statistics)

### 1. 센서 & Bag (`01_sensor_and_bag_data.json`)
- **Bag 경로**: `ros2_data/bags/hallway/hallway_0.mcap` (크기: 1,378,407,440 bytes ≈ 1.38 GB)
- **총 메시지 수**: 52,213개
- **기록 시간**: 200.29초 (약 3분 20초)
- **카메라 해상도 & 내부 파라미터**:
  - Resolution: 640x480
  - Focal Length: $f_x = 606.5387$, $f_y = 606.4935$
  - Principal Point: $c_x = 324.4991$, $c_y = 241.7047$
  - Distortion: Plumb Bob ($k_1=0, k_2=0, p_1=0, p_2=0, k_3=0$)
- **센서 무결성 판정**: `PASS` (RGB 실측 28.19 Hz, Depth 실측 29.80 Hz, IMU 실측 201.68 Hz)

### 2. 프레임 & IMU (`02_frames_and_imu_summary.json`)
- **추출된 RGB-D 프레임 수**: 총 5,625쌍 (000000.png ~ 005624.png)
- **RGB ↔ Depth 동기화 오차**: 평균 0.065 ms, P95 0.0 ms, Max 33.566 ms (비동기율 0%)
- **IMU 샘플 수**: 40,395개 (201.66 Hz)
- **최대 회전속도**: 66°/s (평균 9.7°/s)

### 3. SLAM 궤적 비교 (`03_slam_trajectories_analysis.json`)
| 항목 | cuVSLAM (`cuvslam`) | RTAB-Map (`rtab_normal`) |
|---|---|---|
| **총 포즈 수** | 5,625개 | 5,625개 |
| **추적 커버리지** | 99.98% (트래킹 끊김 0회) | 99.98% (트래킹 끊김 0회) |
| **총 궤적 이동 거리** | **43.62 m** | **64.36 m** |
| **3D Bounding Box ($X \times Y \times Z$)** | $1.96\text{m} \times 0.64\text{m} \times 9.28\text{m}$ | $9.36\text{m} \times 4.38\text{m} \times 3.43\text{m}$ |
| **평균 이동 스텝 크기** | 0.0078 m (Max: 0.126 m) | 0.0114 m (Max: 0.281 m) |
| **RTAB Database** | N/A | SQLite 199.3MB (Node 529개, Link 1,240개, Feature 261,885개) |

### 4. 의사결정 파이프라인 (`04_preview_pipeline_decisions.json`)
1. **Frame Quality Classification**: 5,625개 프레임 단일 디코딩 품질 검사 (64.1초 소요, 불량 프레임 REJECT).
2. **Preview Frame Selection**: 4,782개의 공통 후보군 중 포즈 공간 커버리지 기반으로 **800개 대표 FUSE 프레임** 선별.
3. **VRAM Preflight & Voxel Coarsening**:
   - VRAM Budget: 3,554.0 MB (GPU 가용: 7,899 MB)
   - 요청된 10.0 mm 복셀에 필요한 VRAM(6,398 MB)이 예산을 초과함에 따라 GPU 메모리 내 융합 유지를 위해 **15.6 mm 복셀로 자동 강등 (1,685 MB)**.
4. **Backend Selection**: **`cuvslam` 최종 우승 (Winner)**.

### 5. 3D 재구성 & 품질 벤치마크 (`05_reconstruction_and_quality_comparison.json`)
| 평가 지표 (Metric) | cuVSLAM (권장 백엔드) | RTAB-Map | 우수 백엔드 |
|---|---|---|---|
| **Held-out Depth MAE** | **479.54 mm** | 736.83 mm | 🏆 **cuVSLAM** (-35% 오차) |
| **Held-out Depth RMSE** | **701.37 mm** | 1005.78 mm | 🏆 **cuVSLAM** (-30% 오차) |
| **Held-out Depth P95 Error** | **1667.0 mm** | 2000.7 mm | 🏆 **cuVSLAM** (-16.7%) |
| **Depth Coverage Ratio** | **53.20%** | 36.28% | 🏆 **cuVSLAM** (+46.6%) |
| **Free-space Correctness** | **74.19%** | 72.22% | 🏆 **cuVSLAM** |
| **Texture Atlas Coverage** | **51.81%** | 35.94% | 🏆 **cuVSLAM** |
| **Untextured Face Ratio** | **5.74%** | 12.60% | 🏆 **cuVSLAM** |
| **Delivery Mesh 정점 수 (Vertices)** | 1,518,842개 | 896,111개 | cuVSLAM (더 정밀) |
| **Delivery Mesh 면 수 (Triangles)** | 2,387,279개 | 1,233,980개 | cuVSLAM (더 정밀) |
| **총 파이프라인 소요 시간** | **233.1 초** (약 3.8분) | 297.1 초 (약 4.9분) | 🏆 **cuVSLAM** (21.5% 더 빠름) |

---

## 🤖 AI / LLM 파싱 코드 예시 (Python)

```python
import json

# 1. 00_manifest_index 로드
with open("hallway_ai_dataset/00_manifest_index.json") as f:
    manifest = json.load(f)

# 2. SLAM 비교 데이터 로드
with open("hallway_ai_dataset/03_slam_trajectories_analysis.json") as f:
    slam_data = json.load(f)
    print("cuVSLAM Path Length:", slam_data["trajectories"]["cuvslam"]["analysis"]["total_path_length_meters"])

# 3. 3D 품질 비교 데이터 로드
with open("hallway_ai_dataset/05_reconstruction_and_quality_comparison.json") as f:
    recon_data = json.load(f)
    print("Winner Backend:", recon_data["summary"]["recommended_backend"])
    print("Recommendation Reason:", recon_data["summary"]["recommendation_reason"])
```
