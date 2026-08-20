#!/usr/bin/env python3
"""
compare_algorithms.py — Multi-Axis SLAM / Fusion / Surface Reconstruction Benchmarking Tool

설계 원칙:
  - 후보별 격리 실행: Subprocess crash isolation (SIGSEGV/OOM 방지)
  - 결정적 캐시 및 재사용: 동일 파라미터/데이터셋 artifact 재계산 방지 및 resume 지원
  - 승자 전파 (Winner propagation): Phase A 승자 SLAM -> Phase B TSDF 전달, Phase B 승자 TSDF PCD -> Phase C 전달
  - Phase B 우수 artifact를 Phase C에서 재사용: TSDF direct 메쉬 및 PCD 중복 생성 완전 제거
  - 표준 산출물 생성: final/best.obj, best_config.json, quality_report.json, rankings.json, manifest, report

하위 호환성 유지:
  - 기존 CLI 및 `./scripts/pipeline/compare.sh` 호출과 100% 호환
  - 모듈화된 `auto_mobility.benchmark` 패키지로 핵심 로직 위임
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from auto_mobility.benchmark.manifest import (
    get_git_commit,
    get_system_hardware_info,
    get_software_info,
    BenchmarkManifestExporter
)
from auto_mobility.benchmark.orchestrator import (
    BenchmarkOrchestrator,
    run_benchmark,
    SLAM_TRAJ_FILES,
    SLAM_RUN_ARGS
)
from auto_mobility.benchmark.workers import (
    run_tsdf_worker as _run_reconstruct_worker,
    WorkerStatus
)
from auto_mobility.config import MESH_DIR, POINTCLOUD_DIR, EVALUATION_DIR


def _mesh_path(dataset: str, slam: str, voxel_mm: int, method: Optional[str] = None) -> Path:
    base = f"{dataset}_{slam}_voxel{voxel_mm}mm"
    if method:
        return MESH_DIR / f"{base}_{method}.obj"
    return MESH_DIR / f"{base}.obj"


def _pcd_path(dataset: str, slam: str, voxel_mm: int) -> Path:
    return POINTCLOUD_DIR / f"{dataset}_{slam}_voxel{voxel_mm}mm_cloud.ply"


def _eval_dir(dataset: str, cand: str) -> Path:
    return EVALUATION_DIR / dataset / cand


def _eval_summary(eval_dir: Path) -> Optional[dict]:
    f = eval_dir / "evaluation_summary.json"
    if f.exists():
        import json
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _generate_modular_markdown_report(manifest: dict, report_path: Path):
    from auto_mobility.benchmark.scoring import rank_candidate_summaries
    all_results = manifest.get("phase_a_slam_results", []) + manifest.get("phase_b_tsdf_results", []) + manifest.get("phase_c_surface_results", [])
    ranked = rank_candidate_summaries(all_results)
    BenchmarkManifestExporter.generate_markdown_report(manifest, ranked, report_path)


def main():
    parser = argparse.ArgumentParser(description="Multi-Axis Modular SLAM & Reconstruction Benchmarking Tool")
    parser.add_argument("bag", help="Rosbag name or path")
    parser.add_argument("--phase", choices=["all", "a", "b", "c", "slam", "tsdf", "surface"], default="all", help="Target benchmark phase (default: all)")
    parser.add_argument("--mode", choices=["quick", "standard", "full"], default="standard", help="Execution mode (quick: fast check, standard: adaptive search, full: complete exploration)")
    parser.add_argument("--quick", action="store_true", help="Run quick comparison (alias for --mode=quick)")
    parser.add_argument("--standard", action="store_true", help="Run standard adaptive comparison (alias for --mode=standard)")
    parser.add_argument("--full", action="store_true", help="Run full exploration (alias for --mode=full)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top candidates to export to review/ directory (default: 3)")
    parser.add_argument("--run-slam", action="store_true", help="누락된 SLAM 궤적을 run_slam.sh 로 자동 생성 (별도 ROS 프로세스)")
    parser.add_argument("--force", "--no-cache", dest="force", action="store_true", help="기존 mesh/pcd/평가 결과를 무시하고 강제 재생성")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="이전 실행 상태 복원 (기본값: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="이전 실행 상태 복원 비활성화")
    parser.add_argument("--output-dir", "--output", dest="output_dir", default=None, help="커스텀 평가 결과 저장 디렉터리")
    args = parser.parse_args()

    # Determine mode
    mode = "standard"
    if args.quick or args.mode == "quick":
        mode = "quick"
    elif args.full or args.mode == "full":
        mode = "full"
    elif args.standard or args.mode == "standard":
        mode = "standard"

    run_benchmark(
        bag_input=args.bag,
        phase=args.phase,
        quick=(mode == "quick"),
        full=(mode == "full"),
        mode=mode,
        run_slam=args.run_slam,
        force=args.force,
        resume=args.resume,
        top_k=args.top_k,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()