# 🚀 Auto-Mobility 실전 파이프라인 가이드

RealSense D435i 데이터 수집부터 3D SLAM, 정밀 3D 표면 메쉬(Mesh) 복원, 그리고 센서 기반 정량 품질 평가(QA)까지 **실제 작업에 필요한 핵심 명령어 중심의 실전 워크플로우**입니다.

---

## ⚡ 1. 실전 3단계 핵심 워크플로우 (Quick Start)

가장 빠르고 표준적인 3D 공간 복원 절차입니다.

```text
[1. 원본 촬영] ──────▶ [2. SLAM 궤적] ──────▶ [3. 3D 메쉬 생성 & 뷰어]
 capture_safe.sh        run_slam.sh             mesh.sh --view
 (무손실 Rosbag 확보)   (카메라 이동 궤적 계산)  (3D 메쉬 자동 생성 및 확인)
```

```bash
# [Step 1] 데이터셋 촬영 및 녹화 (RGB-D + Stereo IR + IMU)
./scripts/pipeline/capture_safe.sh room01 --view

# [Step 2] RTAB-Map SLAM 실행 및 궤적(.txt) 생성
./scripts/pipeline/run_slam.sh room01 --slam=rtab

# [Step 3] 3D TSDF 메쉬 생성 및 뷰어 자동 실행
./scripts/pipeline/mesh.sh room01 --view
```

> 💡 **Tip (자동 프레임 추출)**:  
> `prepare_dataset.sh`를 별도로 실행하지 않아도 `mesh.sh`나 `evaluate.sh`가 원본 Rosbag에서 필요한 프레임을 자동으로 추출합니다.

---

## 📊 2. 정량 품질 평가 및 비교 워크플로우

생성된 3D 메쉬가 실제 센서 관측과 얼마나 일치하는지 정량 오차를 측정하고 알고리즘 간 순위를 비교합니다.

### 4단계: 센서 데이터 기반 정량 형상 품질 평가 (Geometry QA)
실제 D435i Depth 관측값과 메쉬의 오차(Depth MAE, P95, Coverage, Point-to-Mesh)를 측정하여 PASS/WARN/FAIL을 판정합니다.
```bash
./scripts/pipeline/evaluate.sh room01 ros2_data/meshes/room01_rtab_tsdf.obj
```
* **결과 보고서**: `ros2_data/evaluations/room01/rtab_tsdf_10mm/evaluation_report.md`

### 5단계: 다중 알고리즘 일괄 벤치마크 및 비교 랭킹
동일 데이터셋에 대해 RTAB-Map, ORB-SLAM3, TSDF, BPA, Poisson 등을 일괄 실행하고 순위표를 출력합니다.
```bash
# 동일 데이터셋 내 다중 후보 랭킹 비교
python3 -m auto_mobility.evaluation.compare_results room01
```

---

## 🛠️ 3. 상황별 실전 명령어 치트시트 (Cheat Sheet)

### A. 3D 복원 옵션 다양화 (`mesh.sh`)
```bash
# 기본 10mm TSDF 메쉬 생성
./scripts/pipeline/mesh.sh room01

# 5mm 초고정밀 TSDF 메쉬 생성
./scripts/pipeline/mesh.sh room01 --fine --view

# ORB-SLAM3 궤적 기반 TSDF 메쉬 생성
./scripts/pipeline/mesh.sh room01 --slam=orb --view
```

### B. TSDF 없이 메쉬 생성 (Non-TSDF)
```bash
# 1. BPA (Ball Pivoting) — TSDF 없는 초경량/비다양체 결함 0% 메쉬 (Isaac Sim 충돌체용)
python3 src/auto_mobility/mesh/mesh_open3d.py \
    ros2_data/pointclouds/room01_rtab_tsdf_cloud.ply \
    ros2_data/meshes/room01_bpa.obj \
    --method bpa --voxel 0.02

# 2. Poisson — 구멍 없이 매끄러운 완벽 폐곡면(Watertight) 메쉬
python3 src/auto_mobility/mesh/mesh_open3d.py \
    ros2_data/pointclouds/room01_rtab_tsdf_cloud.ply \
    ros2_data/meshes/room01_poisson.obj \
    --method poisson --depth 8
```

### C. 결과물 시각화 뷰어
```bash
# 3D 메쉬(.obj) 확인
./scripts/utils/view_mesh.sh room01_rtab_tsdf.obj

# 3D 점군(.ply) 확인
./scripts/utils/view_pointcloud.sh room01_rtab_tsdf_cloud.ply

# RTAB-Map SLAM 세션 DB 3D GUI 뷰어
rtabmap-databaseViewer ~/auto-mobility/ros2_data/databases/room01.db
```

---

## 📁 4. 표준 데이터 디렉터리 구조 (`ros2_data/`)

모든 결과물은 파일 형식별로 독립 관리됩니다.

```text
ros2_data/
├── bags/          # [원본] 무손실 Rosbag (.mcap) — 단 하나의 불변 원본
├── frames/        # [추출] 표준 RGB-D 이미지 + 실제 카메라 파라미터 (자동 추출됨)
├── databases/     # [SLAM] RTAB-Map 세션 DB (.db)
├── trajectories/  # [SLAM] 카메라 6자유도 이동 궤적 (.txt, TUM 포맷)
├── pointclouds/   # [3D]   추출된 3D 점군 데이터 (.ply)
├── meshes/        # [3D]   최종 3D 표면 메쉬 (.obj)
└── evaluations/   # [평가] 정량 품질 리포트 (JSON, MD, 오차 Heatmap 이미지)
```

---

## 📐 5. 핵심 품질 평가 지표 (Geometry QA)

| 지표명 | 단위 | 설명 | 목표 기준 |
| :--- | :---: | :--- | :---: |
| **Depth MAE** | mm | 실제 Depth와 메쉬 렌더 Depth 간 평균 오차 | **$\le 25\text{ mm}$ (PASS)** |
| **Depth P95** | mm | 95% 신뢰구간 최대 오차 (벽면 휨/모서리 벌어짐) | **$\le 60\text{ mm}$ (PASS)** |
| **센서 커버리지** | % | 실제 센서 유효 관측 영역 중 메쉬가 복원된 비율 | **$\ge 75\%$ (PASS)** |
| **Within 20mm** | % | 실제 관측과 2cm 이내로 완벽 정합된 픽셀 비율 | **$\ge 65\%$ (PASS)** |
| **Point-to-Mesh**| mm | 역투영된 센서 3D 점군과 메쉬 표면 간의 최단거리 | **$\le 50\text{ mm}$ (PASS)** |
