"""
auto_mobility.evaluation.mesh_metrics

3D Surface Mesh 기하학적/위상학적 품질 지표 및 실내 평면(Plane) 분석 모듈.
"""

import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import open3d as o3d


def compute_mesh_quality_metrics(mesh_or_path: Union[str, o3d.geometry.TriangleMesh]) -> dict:
    """3D TriangleMesh의 포괄적 위상(Topology) 및 기하(Geometry) 품질 지표 계산."""
    if isinstance(mesh_or_path, str):
        if not os.path.exists(mesh_or_path):
            return {"exists": False, "valid": False, "error": "File not found"}
        mesh = o3d.io.read_triangle_mesh(mesh_or_path)
    else:
        mesh = mesh_or_path

    num_v = len(mesh.vertices)
    num_t = len(mesh.triangles)

    if num_v == 0 or num_t == 0:
        return {
            "exists": True,
            "valid": False,
            "num_vertices": 0,
            "num_triangles": 0,
            "surface_area_m2": 0.0,
            "is_watertight": False,
            "error": "Empty mesh"
        }

    # Bounding Box & Surface Area
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = [round(float(x), 3) for x in bbox.get_extent()]
    try:
        area = float(mesh.get_surface_area())
    except Exception:
        area = 0.0

    try:
        watertight = bool(mesh.is_watertight())
    except Exception:
        watertight = False

    # Triangle & Edge metrics calculation
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    all_edges = np.concatenate([e0, e1, e2])

    cross_prod = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    # Degenerate triangles (area < 1e-10 or min edge < 1e-5)
    degenerate_mask = (tri_areas < 1e-10) | (np.minimum(np.minimum(e0, e1), e2) < 1e-5)
    num_degenerate = int(np.sum(degenerate_mask))
    degenerate_ratio = round(num_degenerate / max(num_t, 1), 6)

    # Triangle aspect ratios: max_edge / min_altitude
    max_e = np.maximum(np.maximum(e0, e1), e2)
    min_altitude = (2.0 * tri_areas) / np.maximum(max_e, 1e-8)
    aspect_ratios = max_e / np.maximum(min_altitude, 1e-8)
    aspect_ratios_clean = aspect_ratios[~degenerate_mask] if np.any(~degenerate_mask) else np.array([1.0])

    # Topology checks using Open3D built-ins
    non_manifold_edges = mesh.get_non_manifold_edges()
    num_non_manifold_edges = len(non_manifold_edges)
    non_manifold_edge_ratio = round(num_non_manifold_edges / max(len(all_edges) // 2, 1), 6)

    # Connected components / Floating clusters
    try:
        triangle_clusters, num_clusters, cluster_areas = mesh.cluster_connected_triangles()
        cluster_counts = np.bincount(np.asarray(triangle_clusters), minlength=num_clusters)
        largest_comp_triangles = int(np.max(cluster_counts)) if len(cluster_counts) else num_t
        largest_comp_ratio = round(largest_comp_triangles / max(num_t, 1), 4)

        # Small components: clusters with < 1% of total triangles
        small_clusters_mask = cluster_counts < (num_t * 0.01)
        small_comp_count = int(np.sum(small_clusters_mask))
        small_comp_triangles = int(np.sum(cluster_counts[small_clusters_mask]))
        small_comp_ratio = round(small_comp_triangles / max(num_t, 1), 4)
    except Exception:
        num_clusters = 1
        largest_comp_ratio = 1.0
        small_comp_count = 0
        small_comp_ratio = 0.0

    return {
        "exists": True,
        "valid": True,
        "num_vertices": int(num_v),
        "num_triangles": int(num_t),
        "surface_area_m2": round(area, 4),
        "density_tri_per_m2": round(num_t / max(area, 1e-5), 1),
        "bbox_extent_m": extent,
        "is_watertight": watertight,
        "connected_component_count": int(num_clusters),
        "largest_component_ratio": largest_comp_ratio,
        "small_component_count": small_comp_count,
        "small_component_area_ratio": small_comp_ratio,
        "degenerate_triangle_count": num_degenerate,
        "degenerate_triangle_ratio": degenerate_ratio,
        "non_manifold_edge_count": num_non_manifold_edges,
        "non_manifold_edge_ratio": non_manifold_edge_ratio,
        "triangle_area_median_m2": round(float(np.median(tri_areas)), 8),
        "triangle_area_p95_m2": round(float(np.percentile(tri_areas, 95)), 8),
        "edge_length_median_m": round(float(np.median(all_edges)), 4),
        "edge_length_p95_m": round(float(np.percentile(all_edges, 95)), 4),
        "triangle_aspect_ratio_median": round(float(np.median(aspect_ratios_clean)), 2),
        "triangle_aspect_ratio_p95": round(float(np.percentile(aspect_ratios_clean, 95)), 2)
    }


def compute_plane_quality_metrics(
    mesh_or_pcd: Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, np.ndarray],
    distance_threshold_m: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    max_planes: int = 5,
    min_plane_points: int = 500
) -> dict:
    """실내 환경 주요 평면(벽, 바닥) 검출 및 평면 오차(Planar Residual) 통계 계산."""
    if isinstance(mesh_or_pcd, o3d.geometry.TriangleMesh):
        # 대형 메쓰(수백만 정점)의 uniform sampling은 수십 초 소요 -> 30k로 상한 (2026-08-24)
        pcd = mesh_or_pcd.sample_points_uniformly(number_of_points=min(len(mesh_or_pcd.vertices) * 2, 30000))
    elif isinstance(mesh_or_pcd, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh_or_pcd)
    else:
        pcd = mesh_or_pcd

    total_points = len(pcd.points)
    if total_points < min_plane_points:
        return {
            "num_major_planes": 0,
            "plane_inlier_ratio": 0.0,
            "plane_residual_mean_mm": None,
            "plane_residual_p95_mm": None,
            "planes": []
        }

    remaining_pcd = pcd
    detected_planes = []
    total_inliers = 0
    all_residuals_mm = []

    for _ in range(max_planes):
        if len(remaining_pcd.points) < min_plane_points:
            break
        try:
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=distance_threshold_m,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
        except Exception:
            break

        if len(inliers) < min_plane_points:
            break

        total_inliers += len(inliers)
        inlier_cloud = remaining_pcd.select_by_index(inliers)
        pts = np.asarray(inlier_cloud.points)

        a, b, c, d = plane_model
        normal_norm = np.sqrt(a**2 + b**2 + c**2)
        dist_m = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / max(normal_norm, 1e-8)
        dist_mm = dist_m * 1000.0
        all_residuals_mm.extend(dist_mm.tolist())

        detected_planes.append({
            "model": [round(float(x), 4) for x in plane_model],
            "inlier_count": len(inliers),
            "residual_mean_mm": round(float(np.mean(dist_mm)), 2),
            "residual_p95_mm": round(float(np.percentile(dist_mm, 95)), 2)
        })

        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    inlier_ratio = round(total_inliers / max(total_points, 1), 4)
    res_arr = np.array(all_residuals_mm) if all_residuals_mm else np.array([0.0])

    return {
        "num_major_planes": len(detected_planes),
        "plane_inlier_ratio": inlier_ratio,
        "plane_residual_mean_mm": round(float(np.mean(res_arr)), 2) if len(all_residuals_mm) else None,
        "plane_residual_p95_mm": round(float(np.percentile(res_arr, 95)), 2) if len(all_residuals_mm) else None,
        "planes": detected_planes
    }
