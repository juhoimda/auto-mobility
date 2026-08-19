# 🚀 Auto-Mobility 실전 파이프라인 가이드

RealSense D435i 데이터 수집부터 3D SLAM, 정밀 3D 표면 메쉬(Mesh) 복원, 그리고 센서 기반 정량 품질 평가(QA) 및 다축 벤치마크까지 **실제 작업에 필요한 핵심 명령어 중심의 실전 워크플로우**입니다.

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

# [Step 2] 원하는 SLAM 백엔드 실행 및 표준 TUM 궤적(.txt) 생성
./scripts/pipeline/run_slam.sh room01 --slam=rtab        # RTAB-Map Visual SLAM
./scripts/pipeline/run_slam.sh room01 --slam=orb_rgbdi   # ORB-SLAM3 RGB-D-Inertial (IMU 융합)
./scripts/pipeline/run_slam.sh room01 --slam=orb_rgbd    # ORB-SLAM3 RGB-D
./scripts/pipeline/run_slam.sh room01 --slam=stella_rgbd # stella_vslam RGB-D

# [Step 3] 3D 표면 메쉬 생성 및 뷰어 자동 실행
./scripts/pipeline/mesh.sh room01 --surface=tsdf_direct --view # 20mm TSDF 메쉬 (기본)
```

> 💡 **Tip (자동 프레임 추출)**:  
> `prepare_dataset.sh`를 별도로 실행하지 않아도 `mesh.sh`나 `evaluate.sh`, `compare.sh`가 원본 Rosbag에서 필요한 Canonical Frames를 자동으로 추출합니다.

---

## 📊 2. 정량 품질 평가 및 다축 벤치마크 워크플로우

생성된 3D 메쉬가 실제 센서 관측과 얼마나 일치하는지 정량 오차를 측정하고 알고리즘 축별로 독립 비교합니다.

### 4단계: 센서 데이터 기반 정량 형상 품질 평가 (Geometry QA)
실제 D435i Depth 관측값과 메쉬의 오차(Depth MAE, P95, Coverage, Point-to-Mesh)를 측정하여 PASS/WARN/FAIL을 판정합니다.
```bash
./scripts/pipeline/evaluate.sh room01 ros2_data/meshes/room01_rtab_tsdf.obj
```
* **결과 보고서**: `ros2_data/evaluations/room01/rtab_tsdf_20mm/evaluation_report.md`

### 5단계: Multi-Axis 모듈식 벤치마크 및 축별 랭킹 (`compare.sh`)
SLAM, TSDF 해상도, Surface 표현 방식을 직교 분리하여 독립 비교합니다.
```bash
# 전체 단계 일괄 실행 (Phase A + B + C + D)
./scripts/pipeline/compare.sh room01 --quick

# 단계별 독립 벤치마크
./scripts/pipeline/compare.sh room01 --phase=a   # PHASE A: SLAM 궤적 및 센서 일치도 비교
./scripts/pipeline/compare.sh room01 --phase=b   # PHASE B: TSDF 복원 해상도(5/10/20mm) 비교
./scripts/pipeline/compare.sh room01 --phase=c   # PHASE C: Surface 방식(TSDF/Poisson/BPA/Alpha/CGAL) 비교
```
* **결과 산출물**: `ros2_data/benchmarks/bench_room01_<timestamp>/`
  * `experiment_manifest.json`: 하드웨어(GPU/RAM), 소프트웨어(Open3D/ROS), Git Commit SHA 기록
  * `benchmark_report.md`: `[SLAM Ranking]`, `[TSDF Ranking]`, `[Surface Ranking]` 분리 리포트

---

## 🛠️ 3. 상황별 실전 명령어 치트시트 (Cheat Sheet)

### A. 3D 복원 알고리즘 및 해상도 선택 (`mesh.sh`)
```bash
# 1. TSDF Direct Mesh (기본 20mm - 고속 & 연산/메모리 최적화 표준)
./scripts/pipeline/mesh.sh room01 --surface=tsdf_direct --voxel=0.02 --view

# 2. 10mm 고정밀 TSDF 복원
./scripts/pipeline/mesh.sh room01 --surface=tsdf_direct --voxel=0.01 --view

# 3. Open3D Alpha Shape (점간 간격 기반 동적 스케일 표면 복원)
./scripts/pipeline/mesh.sh room01 --surface=alpha --voxel=0.02 --view

# 4. Poisson Surface Reconstruction (완벽 폐곡면/Watertight)
./scripts/pipeline/mesh.sh room01 --surface=poisson --view

# 5. Ball Pivoting Algorithm (BPA) (비다양체 결함 없는 초경량 메쉬)
./scripts/pipeline/mesh.sh room01 --surface=bpa --view

# 6. CGAL Polygonal Surface Reconstruction (실내 벽/바닥/천장 Sharp 메쉬)
./scripts/pipeline/mesh.sh room01 --surface=cgal_polygonal --view

# 7. ORB-SLAM3 RGB-D-Inertial 궤적 기반 메쉬 생성
./scripts/pipeline/mesh.sh room01 --slam=orb_rgbdi --surface=tsdf_direct --view
```

### B. 결과물 시각화 뷰어
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
├── evaluations/   # [평가] 정량 품질 리포트 (JSON, MD, 오차 Heatmap 이미지)
└── benchmarks/    # [비교] Multi-Axis 벤치마크 결과 및 Manifest 리포트
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
| **Non-Manifold** | 개 | 2개 이상의 면이 공유하는 비정상 엣지 수 | **$0$ (PASS)** |
