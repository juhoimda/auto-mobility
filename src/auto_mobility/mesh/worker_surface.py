#!/usr/bin/env python3
"""
worker_surface.py — 단일 Surface Reconstruction(Poisson/BPA/Alpha/CGAL)을 서브프로세스로 실행하는 워커.

benchmark(run_benchmark)에서 각 surface 후보별로 별도 프로세스로 호출되며,
SIGSEGV(Open3D/CGAL crash 등) 크래시가 발생해도 benchmark orchestrator 자체는 생존한다.

종료 코드:
  0 = 성공
  1 = Python 예외
  2 = CGAL 사용 불가 (미설치)
  139 / -11 = SIGSEGV
"""

import os
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.resources import DEFAULT_RESOURCE_POLICY
from auto_mobility.mesh.mesh_open3d import generate_mesh
from auto_mobility.mesh.cgal_surface import is_cgal_available


def main():
    parser = argparse.ArgumentParser(description="Surface Reconstruction Subprocess Worker")
    parser.add_argument("--input-ply", required=True, help="Input Point Cloud PLY file")
    parser.add_argument("--output-mesh", required=True, help="Output Mesh OBJ file")
    parser.add_argument("--method", default="poisson", choices=["poisson", "bpa", "alpha", "alpha_shape", "cgal", "cgal_polygonal"])
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--voxel", type=float, default=0.010)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--alpha-factor", type=float, default=3.0)
    parser.add_argument("--simplify", type=float, default=0.5)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--no-color-transfer", action="store_true", help="Skip vertex RGB color transfer")
    parser.add_argument("--benchmark-mode", action="store_true", help="Do not use fallback proxy for CGAL if unavailable")
    args = parser.parse_args()

    t0 = time.time()
    m_lower = args.method.lower()

    if m_lower in ("cgal", "cgal_polygonal"):
        available, msg = is_cgal_available()
        if not available:
            print(f"SKIPPED_UNAVAILABLE: {msg}", flush=True)
            if args.benchmark_mode:
                sys.exit(2)

    print(f"SurfaceWorker: method={args.method} input={args.input_ply} output={args.output_mesh}", flush=True)
    generate_mesh(
        input_ply=args.input_ply,
        output_mesh=args.output_mesh,
        depth=args.depth,
        voxel_size=args.voxel,
        method=args.method,
        alpha=args.alpha,
        alpha_factor=args.alpha_factor,
        view_result=False,
        clean_density=not args.no_clean,
        simplify_target=0.0 if args.no_simplify else args.simplify,
        use_cuda=False,
        color_transfer=not args.no_color_transfer
    )
    print(f"SurfaceWorker: done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
