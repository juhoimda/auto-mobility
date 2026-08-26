#!/usr/bin/env python3
"""
worker.py — 단일 후보의 3D 재구성(reconstruct)을 서브프로세스로 실행하는 워커.

benchmark(run_benchmark)에서 각 후보별로 별도 프로세스로 호출되며,
SIGSEGV(OOM 등) 크래시가 발생해도 benchmark 자체는 생존한다.

사용법:
  python3 worker.py --dataset=PATH --trajectory=PATH --output-mesh=PATH       --pcd-output=PATH --voxel=0.010 --split=PATH --no-color --no-gpu

종료 코드:
  0 = 성공
  1 = Python 예외
  139 = SIGSEGV (Open3D 크래시 등)
"""

import os
import sys
import json
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.resources import DEFAULT_RESOURCE_POLICY
from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.mesh.reconstruct_tsdf import reconstruct
from auto_mobility.evaluation.split import load_split_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output-mesh", default=None)
    parser.add_argument("--pcd-output", default=None)
    parser.add_argument("--voxel", type=float, default=0.010)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--trunc-mult", type=float, default=4.0)
    parser.add_argument("--split", default=None, help="shared_holdout_split.json 경로")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    # load train indices if split provided
    train_indices = None
    if args.split and os.path.exists(args.split):
        split = load_split_json(args.split)
        train_indices = split.get("train_indices")

    dataset = FrameDataset(args.dataset)

    print(f"Worker: reconstruct {args.dataset} voxel={args.voxel} depth_max={args.depth_max} trunc={args.trunc_mult}", flush=True)
    reconstruct(
        dataset=dataset,
        trajectory=args.trajectory,
        voxel_size=args.voxel,
        depth_max=args.depth_max,
        trunc_mult=args.trunc_mult,
        output_mesh=args.output_mesh,
        output_pcd=args.pcd_output,
        train_indices=train_indices,
        stride=args.stride,
        no_color=args.no_color,
        no_gpu=args.no_gpu,
    )
    print(f"Worker: done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
