# 🚀 Auto-Mobility 실전 파이프라인 가이드

> **RTAB-Map & NVIDIA cuVSLAM Dual-Backend 기반 3D 공간 복원 및 품질 검증(QA) 실전 명령어 가이드**

---

## ⚡ 1. 핵심 3단계 워크플로우 (Quick Start)

```text
[1. 원본 센서 녹화] ────────▶ [2. Preview 신속 검증] ────────▶ [3. Standard 정밀 복원 & 듀얼 비교]
 capture_safe.sh               compare.sh --preview           compare.sh --standard
 (무손실 MCAP Bag)             (RTAB & cuVSLAM 듀얼 OBJ)      (전체 프레임 고정밀 복원 & 공정 QA)
```

```bash
# [Step 1] RealSense D435i 데이터셋 녹화 (RGB-D + Stereo IR + IMU)
./scripts/pipeline/capture_safe.sh hallway --view

# [Step 2] Preview 모드: 신속 시각화 및 SLAM 궤적 생성 (15분 이내, 800프레임)
./scripts/pipeline/compare.sh hallway --preview --compare-backends --run-slam

# [Step 3] Standard 모드: 프로덕션 고정밀 3D 복원 및 RTAB vs cuVSLAM 공정 벤치마크
./scripts/pipeline/compare.sh hallway --standard --compare-backends
```

---

## 🛠️ 2. 핵심 실행 명령어 (Command Cheat Sheet)

### A. V2 통합 복원 파이프라인 (`compare.sh`)

`compare.sh`는 Canonical 데이터셋 추출, SLAM 궤적 연동, 3-Way Split, GPU TSDF 복원 및 정량 QA를 일괄 수행합니다.

```bash
# 1. Preview 모드 (신속한 육안 검사용 듀얼 OBJ 생성)
# - RTAB-Map 및 cuVSLAM 양쪽 모두 동일한 800 FUSE 프레임 조건에서 메쉬 생성
./scripts/pipeline/compare.sh hallway --preview --compare-backends --run-slam

# 2. Standard 모드 (프로덕션 고정밀 복원 & 듀얼 공정 비교)
# - 검증된 궤적 캐시 재사용, Zero Data Leakage 3-Way Split 기반 최종 QA 평가
./scripts/pipeline/compare.sh hallway --standard --compare-backends

# 3. Fast-Compare 모드 (Standard 기반 고속 듀얼 비교)
# - 적응형 1,600~2,000 프레임 샘플링 및 신속 벤치마크
./scripts/pipeline/compare.sh hallway --standard --fast-compare --compare-backends

# 4. 특정 백엔드 단일 복원 전달
./scripts/pipeline/compare.sh hallway --standard --deliver-backends rtab
./scripts/pipeline/compare.sh hallway --standard --deliver-backends cuvslam

# 5. 캐시 강제 무시 및 전체 재생성 (--no-cache)
./scripts/pipeline/compare.sh hallway --standard --compare-backends --run-slam --no-cache

# 6. 커스텀 출력 디렉터리 지정
./scripts/pipeline/compare.sh hallway --standard --compare-backends --output output_standard/hallway/run_01
```

---

### B. 센서 데이터 수집 (`capture_safe.sh`)

SLAM/TSDF 연산 부하와 격리하여 무손실 불변 데이터셋(`ros2_data/bags/<bag_name>`)을 안전하게 확보합니다.

```bash
# 기본 압축 토픽 녹화 + 실시간 프리뷰 뷰어
./scripts/pipeline/capture_safe.sh room01 --view

# 무압축 RAW 토픽 녹화 (고대역폭, bit-exact)
./scripts/pipeline/capture_safe.sh room01 --raw --view

# 특정 시간(초) 동안만 자동 녹화
./scripts/pipeline/capture_safe.sh room01 --duration=60
```

---

### C. 개별 SLAM 실행 및 궤적 생성

`--run-slam` 플래그를 통해 자동 실행되지만, 독립적으로 SLAM 궤적만 먼저 생성할 수도 있습니다.

```bash
# RTAB-Map C++ 오프라인 SLAM 실행 (ros2_data/trajectories/rtab_<bag>_trajectory.txt 생성)
./scripts/pipeline/run_slam.sh hallway --slam=rtab

# NVIDIA cuVSLAM Worker 독립 실행 (CUDA 격리 서브프로세스)
python3 -m auto_mobility.reconstruction.pose.backends.cuvslam_worker \
    --dataset ros2_data/frames/hallway \
    --out ros2_data/trajectories/cuvslam_hallway_trajectory.txt
```

---

### D. 3D 메쉬 및 점군 시각화 뷰어

생성된 3D 메쉬와 점군을 Open3D 뷰어로 즉시 확인합니다.

```bash
# Preview 결과 메쉬 확인
./scripts/utils/view_mesh.sh output_preview/hallway/latest/final_candidates/rtab/model.obj
./scripts/utils/view_mesh.sh output_preview/hallway/latest/final_candidates/cuvslam/model.obj

# Standard 결과 메쉬 확인
./scripts/utils/view_mesh.sh output_standard/hallway/latest/final_candidates/rtab/model.obj
./scripts/utils/view_mesh.sh output_standard/hallway/latest/final_candidates/cuvslam/model.obj

# 3D 점군(.ply) 확인
./scripts/utils/view_pointcloud.sh ros2_data/pointclouds/hallway_cloud.ply

# RTAB-Map 데이터베이스 3D GUI 뷰어
rtabmap-databaseViewer ros2_data/databases/hallway.db
```

---

## 📂 3. 결과 산출물 및 리포트 확인

실행 완료 후 아래 디렉터리에서 결과물과 정량 검증 리포트를 확인합니다.

```bash
# Preview 산출물 확인
cat output_preview/hallway/latest/report.md
cat output_preview/hallway/latest/decision_trace.json

# Standard 듀얼 비교 산출물 확인
cat output_standard/hallway/latest/report.md
cat output_standard/hallway/latest/standard_comparison.json
cat output_standard/hallway/latest/run_manifest.json
```

* **`final_candidates/<backend>/`**: 각 백엔드별 최종 복원 결과 (`model.obj`, `model.mtl`, `textures/`, `texture_contract.json`)
* **`standard_comparison.json`**: 동일 Held-out 프레임셋 기준 RTAB vs cuVSLAM의 Depth MAE/RMSE/P95, Coverage, Point-to-Mesh 거리 비교 및 최종 승자(Winner) 명시
* **`decision_trace.json`**: Coarse-to-Fine 복셀 탐색 및 품질 판정 결정 히스토리
* **`telemetry.json`**: 실행 중 GPU/CPU 온도 및 메모리 자원 소비 추적 데이터
