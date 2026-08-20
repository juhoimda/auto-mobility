"""
auto_mobility.evaluation.geometry_metrics

Held-out Depth Reprojection 오차 및 Point-to-Mesh Distance 계산 모듈.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
import open3d as o3d
import open3d.core as o3c
from auto_mobility.dataset.frame_dataset import CameraIntrinsics


def compute_depth_metrics(
    real_depth_mm: np.ndarray,
    rendered_depth_mm: np.ndarray,
    depth_min_mm: float = 300.0,
    depth_max_mm: float = 5000.0,
    free_space_margin_mm: float = 50.0
) -> dict:
    """단일 프레임 또는 누적 프레임의 실제 관측 Depth와 Rendered Mesh Depth 간 정밀 오차 지표 계산."""
    # 유효 마스크 정의
    ref_valid = (real_depth_mm >= depth_min_mm) & (real_depth_mm <= depth_max_mm)
    pred_valid = (rendered_depth_mm >= depth_min_mm) & (rendered_depth_mm <= depth_max_mm)
    overlap = ref_valid & pred_valid

    n_ref = int(np.sum(ref_valid))
    n_pred = int(np.sum(pred_valid))
    n_overlap = int(np.sum(overlap))

    if n_ref == 0:
        return {
            "valid_reference_pixels": 0,
            "valid_prediction_pixels": n_pred,
            "overlapping_pixels": 0,
            "depth_coverage_ratio": 0.0,
            "observed_surface_completeness": 0.0,
            "free_space_violation_ratio": 0.0,
            "free_space_correctness_ratio": 1.0,
            "depth_mae_mm": None,
            "depth_rmse_mm": None,
            "depth_median_error_mm": None,
            "depth_p90_mm": None,
            "depth_p95_mm": None,
            "within_10mm_ratio": 0.0,
            "within_20mm_ratio": 0.0,
            "within_50mm_ratio": 0.0
        }

    coverage = round(n_overlap / max(n_ref, 1), 4)

    if n_overlap == 0:
        return {
            "valid_reference_pixels": n_ref,
            "valid_prediction_pixels": n_pred,
            "overlapping_pixels": 0,
            "depth_coverage_ratio": coverage,
            "observed_surface_completeness": 0.0,
            "free_space_violation_ratio": 0.0,
            "free_space_correctness_ratio": 1.0,
            "depth_mae_mm": None,
            "depth_rmse_mm": None,
            "depth_median_error_mm": None,
            "depth_p90_mm": None,
            "depth_p95_mm": None,
            "within_10mm_ratio": 0.0,
            "within_20mm_ratio": 0.0,
            "within_50mm_ratio": 0.0
        }

    real_overlap = real_depth_mm[overlap].astype(np.float64)
    rend_overlap = rendered_depth_mm[overlap].astype(np.float64)
    errors = np.abs(real_overlap - rend_overlap)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median_err = float(np.median(errors))
    p90 = float(np.percentile(errors, 90))
    p95 = float(np.percentile(errors, 95))

    w10 = float(np.mean(errors <= 10.0))
    w20 = float(np.mean(errors <= 20.0))
    w50 = float(np.mean(errors <= 50.0))

    # Free-space violation: rendered depth is significantly closer than measured real depth
    # indicating phantom geometry / false surface blocking empty ray space.
    free_space_violations = np.sum(rend_overlap < (real_overlap - free_space_margin_mm))
    fs_violation_ratio = float(free_space_violations / max(n_overlap, 1))
    fs_correctness = float(np.clip(1.0 - fs_violation_ratio, 0.0, 1.0))

    # Observed surface completeness = coverage * ratio of points within 50mm bound
    surface_completeness = float(round(coverage * w50, 4))

    return {
        "valid_reference_pixels": n_ref,
        "valid_prediction_pixels": n_pred,
        "overlapping_pixels": n_overlap,
        "depth_coverage_ratio": coverage,
        "observed_surface_completeness": surface_completeness,
        "free_space_violation_ratio": round(fs_violation_ratio, 4),
        "free_space_correctness_ratio": round(fs_correctness, 4),
        "depth_mae_mm": round(mae, 2),
        "depth_rmse_mm": round(rmse, 2),
        "depth_median_error_mm": round(median_err, 2),
        "depth_p90_mm": round(p90, 2),
        "depth_p95_mm": round(p95, 2),
        "within_10mm_ratio": round(w10, 4),
        "within_20mm_ratio": round(w20, 4),
        "within_50mm_ratio": round(w50, 4)
    }


def backproject_depth_to_world_points(
    depth_mm: np.ndarray,
    T_world_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_min_mm: float = 300.0,
    depth_max_mm: float = 5000.0,
    stride: int = 2
) -> np.ndarray:
    """Depth 이미지의 유효 픽셀을 3D World 좌표계 Point Cloud (N, 3)로 역투영 (단위: m)."""
    h, w = depth_mm.shape
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    v_grid, u_grid = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing='ij')
    depth_sub = depth_mm[v_grid, u_grid]

    valid = (depth_sub >= depth_min_mm) & (depth_sub <= depth_max_mm)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    u_valid = u_grid[valid].astype(np.float32)
    v_valid = v_grid[valid].astype(np.float32)
    z_m = depth_sub[valid].astype(np.float32) / 1000.0  # meters

    x_m = (u_valid - cx) * z_m / fx
    y_m = (v_valid - cy) * z_m / fy

    pts_cam = np.stack([x_m, y_m, z_m], axis=-1)  # (N, 3)

    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    pts_world = np.dot(pts_cam, R.T) + t
    return pts_world.astype(np.float32)


def compute_point_to_mesh_metrics(
    scene: o3d.t.geometry.RaycastingScene,
    world_points: np.ndarray,
    max_sample_points: int = 50000,
    max_distance_mm: float = 500.0
) -> dict:
    """World 좌표계의 Point Cloud와 Mesh 간의 최단 거리(Point-to-Mesh Distance) 통계 계산."""
    n_pts = len(world_points)
    if n_pts == 0:
        return {
            "num_sampled_points": 0,
            "point_to_mesh_mean_mm": None,
            "point_to_mesh_median_mm": None,
            "point_to_mesh_p90_mm": None,
            "point_to_mesh_p95_mm": None,
            "point_to_mesh_max_mm": None
        }

    if n_pts > max_sample_points:
        indices = np.random.choice(n_pts, max_sample_points, replace=False)
        pts_sampled = world_points[indices]
    else:
        pts_sampled = world_points

    query_t = o3c.Tensor(pts_sampled.astype(np.float32))
    distances_m = scene.compute_distance(query_t).numpy()  # meters
    distances_mm = distances_m * 1000.0

    # Filter out extreme outliers
    valid_dist = distances_mm[distances_mm <= max_distance_mm]
    if len(valid_dist) == 0:
        valid_dist = distances_mm

    return {
        "num_sampled_points": int(len(pts_sampled)),
        "point_to_mesh_mean_mm": round(float(np.mean(valid_dist)), 2),
        "point_to_mesh_median_mm": round(float(np.median(valid_dist)), 2),
        "point_to_mesh_p90_mm": round(float(np.percentile(valid_dist, 90)), 2),
        "point_to_mesh_p95_mm": round(float(np.percentile(valid_dist, 95)), 2),
        "point_to_mesh_max_mm": round(float(np.max(valid_dist)), 2)
    }


def generate_error_visualization(
    real_depth_mm: np.ndarray,
    rendered_depth_mm: np.ndarray,
    out_real_path: Union[str, Path],
    out_rendered_path: Union[str, Path],
    out_heatmap_path: Union[str, Path],
    depth_min_mm: float = 300.0,
    depth_max_mm: float = 5000.0
) -> None:
    """실제 Depth, 예측 Rendered Depth, 그리고 범례가 포함된 오차 Heatmap 이미지 저장."""
    os.makedirs(os.path.dirname(os.path.abspath(out_real_path)), exist_ok=True)

    # 1. Real Depth Colorization
    real_norm = np.clip((real_depth_mm - depth_min_mm) / (depth_max_mm - depth_min_mm), 0.0, 1.0)
    real_vis = (real_norm * 255.0).astype(np.uint8)
    real_color = cv2.applyColorMap(real_vis, cv2.COLORMAP_TURBO)
    real_color[real_depth_mm < depth_min_mm] = [0, 0, 0]
    cv2.imwrite(str(out_real_path), real_color)

    # 2. Rendered Depth Colorization
    rend_norm = np.clip((rendered_depth_mm - depth_min_mm) / (depth_max_mm - depth_min_mm), 0.0, 1.0)
    rend_vis = (rend_norm * 255.0).astype(np.uint8)
    rend_color = cv2.applyColorMap(rend_vis, cv2.COLORMAP_TURBO)
    rend_color[rendered_depth_mm < depth_min_mm] = [0, 0, 0]
    cv2.imwrite(str(out_rendered_path), rend_color)

    # 3. Discrete Error Heatmap with Legend
    # 0-10 mm : Green [40, 200, 40]
    # 10-20 mm: Yellow-Green [160, 220, 30]
    # 20-50 mm: Orange [30, 140, 255]
    # 50-100 mm: Red [30, 30, 230]
    # >100 mm : Dark Red [10, 10, 130]
    # Invalid : Black [0, 0, 0]

    h, w = real_depth_mm.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)

    ref_valid = (real_depth_mm >= depth_min_mm) & (real_depth_mm <= depth_max_mm)
    pred_valid = (rendered_depth_mm >= depth_min_mm) & (rendered_depth_mm <= depth_max_mm)
    overlap = ref_valid & pred_valid

    err = np.zeros((h, w), dtype=np.float32)
    err[overlap] = np.abs(real_depth_mm[overlap] - rendered_depth_mm[overlap])

    heatmap[overlap & (err <= 10.0)] = [40, 200, 40]             # 0-10mm Green
    heatmap[overlap & (err > 10.0) & (err <= 20.0)] = [30, 220, 200]  # 10-20mm Yellow
    heatmap[overlap & (err > 20.0) & (err <= 50.0)] = [30, 140, 255]  # 20-50mm Orange
    heatmap[overlap & (err > 50.0) & (err <= 100.0)] = [30, 30, 230]  # 50-100mm Red
    heatmap[overlap & (err > 100.0)] = [10, 10, 130]             # >100mm Dark Red
    heatmap[ref_valid & (~pred_valid)] = [80, 80, 80]            # Uncovered reference (Gray)

    # Add legend bar at the bottom
    legend_h = 30
    combined = np.zeros((h + legend_h, w, 3), dtype=np.uint8)
    combined[:h, :, :] = heatmap
    combined[h:, :, :] = [30, 30, 30]

    labels = [("0-10mm", (40, 200, 40)), ("10-20mm", (30, 220, 200)),
              ("20-50mm", (30, 140, 255)), ("50-100mm", (30, 30, 230)), (">100mm", (10, 10, 130))]
    step_w = w // len(labels)
    for i, (text, col) in enumerate(labels):
        x0 = i * step_w
        cv2.rectangle(combined, (x0 + 5, h + 5), (x0 + 25, h + 25), col, -1)
        cv2.putText(combined, text, (x0 + 30, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)

    cv2.imwrite(str(out_heatmap_path), combined)
