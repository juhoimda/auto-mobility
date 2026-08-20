# 📊 Multi-Axis Robotics SLAM & 3D Reconstruction Benchmark Guide

Production-ready, repeatable, explainable multi-axis optimization and benchmarking pipeline for autonomous robotics.

---

## 🎯 1. 최적화 목표 및 원칙

Auto-Mobility 벤치마크는 SLAM, 3D TSDF Voxel 해상도, 표면 메쉬(Surface Mesh) 표현 알고리즘을 직교 분리하여, **최소한의 연산 자원으로 센서 관측 일치도가 가장 높은 최적의 3D 복원 파이프라인(Winner Pipeline)을 자동 선정**합니다.

1. **후보별 프로세스 격리 (Crash Isolation)**:
   - Native C++/Open3D/CGAL의 예외, SIGSEGV, OOM(Out of Memory), 타임아웃이 발생해도 부모 벤치마크 프로세스는 중단 없이 다음 후보 평가를 지속합니다.
2. **결정론적 캐시 및 재사용 (Artifact Cache & Resume)**:
   - 파라미터/데이터셋 SHA-256 캐시 키를 통해 기완료된 메쉬 및 평가 결과를 100% 재사용하며, 중단 후 재실행 시 기존 진행 상태에서 자동 복원(Resume)됩니다.
3. **Cheap Screening → Top-1 Full Fidelity 2단계 평가**:
   - 탐색 단계에서는 Holdout 서브샘플링(최대 12장)과 렌더링 I/O 생략으로 5~10배 빠르게 후보를 스크리닝하고, 최종 우승 후보에 대해서만 Full Fidelity(고해상도 렌더링 및 50k 점 거리 연산)를 수행합니다.
4. **설명 가능한 의사결정 (Explainable Winner Rationale & Decision Trace)**:
   - 우승 후보가 선정된 기하학적/자원적 근거(Quality 차이 vs Cost Tie-break) 및 후보별 가지치기(Pruning) 사유가 리포트에 명시됩니다.
5. **Isaac Sim과의 관계**:
   - 현재 벤치마크 파이프라인은 정밀 3D 표면 복원 및 센서 기반 정량 평가를 담당하며, Isaac Sim은 벤치마크가 생성한 최종 `final/best.obj` 및 `best_config.json`을 소비하는 Downstream 시뮬레이션 단계입니다 (벤치마크 실행 범위 외).

---

## 🚀 2. CLI 실행 방법

표준 진입점 스크립트:
```bash
./scripts/pipeline/compare.sh BAG_NAME [옵션]
```

### 실행 모드 (Execution Modes)
| 모드 | CLI 플래그 | 설명 |
|---|---|---|
| **STANDARD** *(기본값)* | *(옵션 없음)* 또는 `--mode=standard` | **권장 모드**: 20mm/10mm 탐색 후 유의미한 이득(+1.5점 이상) & 가용 RAM(3GB 이상) 시에만 5mm 미세 TSDF 실행. Tier 1 우수(>=70점) 시 Tier 2 생략. |
| **QUICK** | `--quick` 또는 `--mode=quick` | **개발/디버그 모드**: 20mm/10mm TSDF 및 Tier 1 표면만 빠르게 검증. |
| **FULL** | `--full` 또는 `--mode=full` | **연구/완전 탐색 모드**: 가지치기 완화. 모든 TSDF 복셀(20/10/5mm) 및 모든 표면 알고리즘(Tier 1 + Tier 2)을 완전 탐색. |

### 주요 옵션
```bash
# 기본 Standard 모드 실행
./scripts/pipeline/compare.sh room01

# 빠른 개발용 Quick 실행
./scripts/pipeline/compare.sh room01 --quick

# 완전 탐색 Full 모드 실행
./scripts/pipeline/compare.sh room01 --full

# 특정 단계만 독립 실행
./scripts/pipeline/compare.sh room01 --phase=a   # SLAM 백엔드 비교
./scripts/pipeline/compare.sh room01 --phase=b   # TSDF 복셀 해상도 비교
./scripts/pipeline/compare.sh room01 --phase=c   # 표면 복원 방식 비교

# 캐시 무시 강제 재생성
./scripts/pipeline/compare.sh room01 --no-cache

# 상위 5개 후보 Review 디렉터리 내보내기
./scripts/pipeline/compare.sh room01 --top-k 5
```

---

## 📂 3. 디렉터리 및 산출물 구조

벤치마크 실행 결과는 `ros2_data/evaluations/<BAG_NAME>/` 에 저장됩니다:

```text
ros2_data/evaluations/<BAG_NAME>/
├── experiment_manifest.json     # 전체 벤치마크 메타데이터, 하드웨어, 소프트웨어, Decision Trace
├── benchmark_report.md          # 마크다운 벤치마크 리포트 (단계별 랭킹, 선정 이유, 뷰어 명령어)
├── rankings.json                # 전체 후보군 통합 순위 및 세부 점수
├── holdout_split.json           # Train / Hold-out 프레임 분할 정보
├── final/                       # 최종 우승 후보 전달 패키지
│   ├── best.obj                 # 최적 3D 복원 표면 메쉬
│   ├── best_config.json         # 최적 복원에 사용된 파라미터 및 SHA-256 해시
│   └── quality_report.json      # 우승 후보의 센서 기반 상세 기하 정량 평가 결과
└── review/                      # 수동 시각 검수를 위한 Top-K 후보 메쉬
    ├── rank_01.obj              # 1위 후보 메쉬
    ├── rank_02.obj              # 2위 후보 메쉬
    └── rank_03.obj              # 3위 후보 메쉬
```

---

## 🔍 4. 3D 시각 검수 (Interactive Viewer)

벤치마크가 생성한 상위 후보 메쉬를 대화형 3D 뷰어로 즉시 열어볼 수 있습니다:

```bash
# 1위 후보 메쉬 시각화
./scripts/utils/view_mesh.sh ros2_data/evaluations/room01/review/rank_01.obj

# 2위 후보 메쉬 시각화
./scripts/utils/view_mesh.sh ros2_data/evaluations/room01/review/rank_02.obj

# 최종 확정 우승 메쉬 시각화
./scripts/utils/view_mesh.sh ros2_data/evaluations/room01/final/best.obj
```

---

## 🧬 5. 재현성 (Reproducibility) 보장 체계

모든 벤치마크 실행은 `experiment_manifest.json` 및 `best_config.json`에 다음의 재현성 메타데이터를 원자적(Atomic Write)으로 기록합니다:

- **Git Commit SHA & Dirty Status**: 소스 코드 형상 및 미커밋 수정 여부
- **Dataset Fingerprint**: `frames.csv` 및 `camera_info.json` 기반 16자리 SHA-256 지문
- **Deterministic Random Seed**: 프레임 분할, Point Cloud 서브샘플링, RANSAC 평면 추정에 고정 Seed(`42`) 강제
- **Artifact SHA-256 Hashes**: `best.obj` 및 `trajectory.txt`의 해시 무결성 검증
- **Software & Hardware Environment**: OS, ROS2 Distro, Open3D, Python, CPU, RAM, GPU VRAM
