"""
tests/unit/test_mesh_metrics.py

Unit tests for Mesh Quality metrics, Point-to-Mesh distance, and Planar Residuals.
"""

import numpy as np
import pytest
import open3d as o3d
from auto_mobility.evaluation.mesh_metrics import compute_mesh_quality_metrics, compute_plane_quality_metrics
from auto_mobility.evaluation.render_depth import create_raycasting_scene
from auto_mobility.evaluation.geometry_metrics import compute_point_to_mesh_metrics


def test_mesh_quality_metrics_cube():
    # Create unit cube (1m x 1m x 1m)
    cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    cube.compute_vertex_normals()

    metrics = compute_mesh_quality_metrics(cube)

    assert metrics["valid"] is True
    assert metrics["num_vertices"] == 8
    assert metrics["num_triangles"] == 12
    assert metrics["surface_area_m2"] == 6.0
    assert metrics["is_watertight"] is True
    assert metrics["degenerate_triangle_count"] == 0
    assert metrics["connected_component_count"] == 1
    assert metrics["largest_component_ratio"] == 1.0


def test_point_to_mesh_distance():
    # XY Plane at Z=0 (1m x 1m)
    vertices = np.array([
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [-0.5,  0.5, 0.0]
    ], dtype=np.float64)
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    scene = create_raycasting_scene(mesh)

    # Known query points at Z = 0.05m (50mm away from plane)
    query_pts = np.array([
        [0.0, 0.0, 0.05],
        [0.1, 0.1, 0.05],
        [-0.2, 0.3, 0.05]
    ], dtype=np.float32)

    p2m = compute_point_to_mesh_metrics(scene, query_pts)

    assert p2m["num_sampled_points"] == 3
    # Distance should be 50.0 mm
    assert abs(p2m["point_to_mesh_mean_mm"] - 50.0) < 1e-1
    assert abs(p2m["point_to_mesh_median_mm"] - 50.0) < 1e-1
    assert abs(p2m["point_to_mesh_p95_mm"] - 50.0) < 1e-1


def test_planar_residual_analysis():
    # Point cloud on flat plane z=0 with 1mm Gaussian noise
    np.random.seed(42)
    x = np.random.uniform(-1, 1, 2000)
    y = np.random.uniform(-1, 1, 2000)
    z = np.random.normal(0.0, 0.001, 2000) # 1mm std dev

    pts = np.stack([x, y, z], axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    planes = compute_plane_quality_metrics(pcd, distance_threshold_m=0.01)

    assert planes["num_major_planes"] >= 1
    assert planes["plane_inlier_ratio"] > 0.95
    # Residual mean should be around ~0.8mm
    assert planes["plane_residual_mean_mm"] is not None
    assert planes["plane_residual_mean_mm"] < 2.0
