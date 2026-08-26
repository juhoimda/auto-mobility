#!/usr/bin/env python3
"""
worker_poisson_fallback.py — extract_triangle_mesh 크래시 시 pcd → Poisson 메쉬 대체 워커

Usage:
  python3 worker_poisson_fallback.py --pcd=PATH --output-mesh=PATH [--depth=9]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("OMP_NUM_THREADS", "8")

import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True, help="입력 point cloud (.ply)")
    parser.add_argument("--output-mesh", required=True, help="출력 mesh (.obj)")
    parser.add_argument("--depth", type=int, default=9, help="Poisson 옥트리 깊이 (기본 9)")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.pcd)
    if len(pcd.points) == 0:
        print(f"ERROR: pcd 비어있음: {args.pcd}", flush=True)
        sys.exit(1)

    print(f"_worker_poisson: {len(pcd.points):,} 포인트 → Poisson depth={args.depth}", flush=True)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)

    # 저밀도 꼭짓점 제거 (노이즈 필터링)
    import numpy as np
    densities_arr = np.asarray(densities)
    thr = float(np.quantile(densities_arr, 0.01))
    vertices_to_remove = densities_arr < thr
    mesh.remove_vertices_by_mask(vertices_to_remove)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_mesh)), exist_ok=True)
    o3d.io.write_triangle_mesh(args.output_mesh, mesh)
    print(f"_worker_poisson: 저장 완료 → {args.output_mesh} ({len(mesh.vertices):,} verts)", flush=True)


if __name__ == "__main__":
    main()
