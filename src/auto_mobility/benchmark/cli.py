"""
cli.py — CLI Entrypoint for Multi-Axis Benchmark.
"""

import sys
import argparse
from auto_mobility.benchmark.orchestrator import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Multi-Axis Modular SLAM & 3D Reconstruction Benchmarking Tool")
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
