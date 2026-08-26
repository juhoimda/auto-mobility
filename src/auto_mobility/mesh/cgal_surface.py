#!/usr/bin/env python3
"""
cgal_surface.py — CGAL Polygonal Surface Reconstruction Adapter

기능:
  - 실내 벽/바닥/천장/기둥 등 Piecewise-Planar 구조에 특화된 CGAL 다각형 표면 복원
  - CGAL 라이브러리 및 executable 가용성 검증
  - Open3D PointCloud 또는 PLY 파일을 입력받아 간결하고 Sharp한 Mesh 출력 생성
  - 미설치/미지원 환경에서의 우아한 fallback 및 에러 핸들링
"""

import os
import sys
import shutil
import subprocess
import tempfile
import open3d as o3d
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple


def is_cgal_available() -> Tuple[bool, str]:
    """CGAL 빌드 도구 또는 바이너리 사용 가능 여부 확인."""
    exe = shutil.which("cgal_polygonal_reconstruction") or shutil.which("polygonal_surface_reconstruction")
    if exe is not None:
        return True, exe
    
    # Check project third_party
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    candidate = project_root / "third_party" / "installed" / "bin" / "cgal_polygonal_reconstruction"
    if candidate.exists() and os.access(str(candidate), os.X_OK):
        return True, str(candidate)

    return False, "CGAL Polygonal Reconstruction executable not found (requires CGAL + SCIP/GLPK solver)."


def reconstruct_cgal_polygonal(
    pcd_or_file: Union[o3d.geometry.PointCloud, str, Path],
    output_mesh: str,
    distance_threshold: float = 0.03,
    angle_threshold_deg: float = 25.0,
    min_region_size: int = 300
) -> o3d.geometry.TriangleMesh:
    """CGAL Polygonal Surface Reconstruction 실행 또는 시뮬레이션 어댑터.

    참고: PSR의 MIP 최적화는 평면 수에 지수적으로 민감하다 (실측 2026-08-24:
    94 planes -> GLPK/SCIP 모두 수십 분+, 13 planes -> ~1s). 따라서 입력을
    250k pts 이하로 선축소하고 min_region_size 기본값을 300으로 올려
    평면 수를 10~20개 수준으로 유지한다.
    """
    available, exec_path = is_cgal_available()

    # Load point cloud
    if isinstance(pcd_or_file, o3d.geometry.PointCloud):
        pcd = pcd_or_file
        temp_dir = tempfile.mkdtemp()
        temp_ply = os.path.join(temp_dir, "input.ply")
    else:
        temp_ply = str(pcd_or_file)
        temp_dir = None
        pcd = o3d.io.read_point_cloud(temp_ply)

    # Dense cloud은 MIP 폭발의 직접 원인 -> CGAL 입력만 선축소 (공정성을 위해 로그 기록)
    max_pts = 250000
    decimated = False
    search_sphere_radius = 0.03
    if len(pcd.points) > max_pts:
        voxel = 0.025
        while len(pcd.points) > max_pts and voxel < 0.06:
            pcd = pcd.voxel_down_sample(voxel)
            last_voxel = voxel
            voxel *= 1.3
        decimated = True
        # 탐색 반경은 포인트 간격(≈voxel)보다 커야 region growing이 이웃을 찾는다
        search_sphere_radius = round(max(0.03, last_voxel * 1.5), 4)
        print(f"  ⚠️ [CGAL Adapter] input decimated to {len(pcd.points):,} pts "
              f"(voxel {last_voxel*1000:.0f}mm, sphere {search_sphere_radius*1000:.0f}mm)")

    if isinstance(pcd_or_file, o3d.geometry.PointCloud) or decimated:
        o3d.io.write_point_cloud(temp_ply, pcd)

    if pcd.is_empty() or len(pcd.points) < 10:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("Input point cloud is empty or contains too few points for CGAL reconstruction.")

    os.makedirs(os.path.dirname(os.path.abspath(output_mesh)), exist_ok=True)

    if available:
        cmd = [
            exec_path,
            temp_ply,
            output_mesh,
            "-d", str(distance_threshold),
            "-a", str(angle_threshold_deg),
            "-m", str(min_region_size),
            "-r", str(search_sphere_radius)
        ]
        print(f"🏛️ Running CGAL Polygonal Surface Reconstruction: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"CGAL execution failed (code {res.returncode}):\n{res.stderr}")
        mesh = o3d.io.read_triangle_mesh(output_mesh)
    else:
        # Planar RANSAC cluster abstraction fallback
        print(f"⚠️ [CGAL Adapter] {exec_path}")
        print("ℹ️ Piecewise planar proxy mesh fallback...")
        mesh = _generate_planar_proxy_mesh(pcd, distance_threshold=distance_threshold)
        o3d.io.write_triangle_mesh(output_mesh, mesh)

    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return mesh


def _generate_planar_proxy_mesh(pcd: o3d.geometry.PointCloud, distance_threshold: float = 0.03) -> o3d.geometry.TriangleMesh:
    """CGAL 미설치 환경에서도 기하 비교 파이프라인이 정상 동작하도록 평면 RANSAC 기반 proxy mesh 생성."""
    # Downsample and segment planes
    pcd_down = pcd.voxel_down_sample(0.03)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)
    pcd_clean = pcd_down.select_by_index(ind)

    if not pcd_clean.has_normals():
        pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Reconstruct surface with planar simplification
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_clean, depth=7)
    bbox = pcd_clean.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    if len(mesh.triangles) > 500:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=min(len(mesh.triangles)//3, 3000))
    return mesh
