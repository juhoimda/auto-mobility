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
    free_space_margin_mm: float = 50.0,
    return_errors: bool = False
) -> dict:
    """단일 프레임의 실제 관측 Depth와 Rendered Mesh Depth 간 정밀 오차 지표 계산."""
    ref_valid = (real_depth_mm >= depth_min_mm) & (real_depth_mm <= depth_max_mm)
    pred_valid = (rendered_depth_mm >= depth_min_mm) & (rendered_depth_mm <= depth_max_mm)
    overlap = ref_valid & pred_valid

    n_ref = int(np.sum(ref_valid))
    n_pred = int(np.sum(pred_valid))
    n_overlap = int(np.sum(overlap))

    if n_ref == 0 or n_overlap == 0:
        res = {
            "valid_reference_pixels": n_ref,
            "valid_prediction_pixels": n_pred,
            "overlapping_pixels": n_overlap,
            "depth_coverage_ratio": round(n_overlap / max(n_ref, 1), 4) if n_ref > 0 else 0.0,
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
            "within_50mm_ratio": 0.0,
            "sum_abs_error": 0.0,
            "sum_sq_error": 0.0,
            "count_w10": 0,
            "count_w20": 0,
            "count_w50": 0,
            "count_fs_viol": 0
        }
        if return_errors:
            res["errors"] = np.empty((0,), dtype=np.float32)
        return res

    real_overlap = real_depth_mm[overlap].astype(np.float64)
    rend_overlap = rendered_depth_mm[overlap].astype(np.float64)
    errors = np.abs(real_overlap - rend_overlap)

    sum_abs = float(np.sum(errors))
    sum_sq = float(np.sum(errors ** 2))
    mae = sum_abs / n_overlap
    rmse = float(np.sqrt(sum_sq / n_overlap))
    median_err = float(np.median(errors))
    p90 = float(np.percentile(errors, 90))
    p95 = float(np.percentile(errors, 95))

    count_w10 = int(np.sum(errors <= 10.0))
    count_w20 = int(np.sum(errors <= 20.0))
    count_w50 = int(np.sum(errors <= 50.0))

    free_space_violations = int(np.sum(rend_overlap < (real_overlap - free_space_margin_mm)))
    fs_violation_ratio = float(free_space_violations / max(n_overlap, 1))
    fs_correctness = float(np.clip(1.0 - fs_violation_ratio, 0.0, 1.0))
    coverage = round(n_overlap / max(n_ref, 1), 4)
    surface_completeness = float(round(coverage * (count_w50 / max(n_overlap, 1)), 4))

    res = {
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
        "within_10mm_ratio": round(count_w10 / max(n_overlap, 1), 4),
        "within_20mm_ratio": round(count_w20 / max(n_overlap, 1), 4),
        "within_50mm_ratio": round(count_w50 / max(n_overlap, 1), 4),
        "sum_abs_error": sum_abs,
        "sum_sq_error": sum_sq,
        "count_w10": count_w10,
        "count_w20": count_w20,
        "count_w50": count_w50,
        "count_fs_viol": free_space_violations
    }
    if return_errors:
        res["errors"] = errors.astype(np.float32)
    return res


def aggregate_depth_metrics(frame_metrics_list: List[dict], pooled_errors: Optional[np.ndarray] = None) -> dict:
    """여러 프레임의 depth 메트릭을 통계적으로 올바르게 결합하여 전체 집계 메트릭을 계산."""
    if not frame_metrics_list:
        return {
            "depth_mae_mm": None,
            "depth_rmse_mm": None,
            "depth_median_error_mm": None,
            "depth_p90_mm": None,
            "depth_p95_mm": None,
            "depth_coverage_ratio": 0.0,
            "observed_surface_completeness": 0.0,
            "free_space_violation_ratio": 0.0,
            "free_space_correctness_ratio": 1.0,
            "within_10mm_ratio": 0.0,
            "within_20mm_ratio": 0.0,
            "within_50mm_ratio": 0.0
        }

    total_ref = sum(m.get("valid_reference_pixels", 0) for m in frame_metrics_list)
    total_overlap = sum(m.get("overlapping_pixels", 0) for m in frame_metrics_list)
    total_sum_abs = sum(m.get("sum_abs_error", (m.get("depth_mae_mm") or 0.0) * m.get("overlapping_pixels", 0)) for m in frame_metrics_list)
    total_sum_sq = sum(m.get("sum_sq_error", ((m.get("depth_rmse_mm") or 0.0) ** 2) * m.get("overlapping_pixels", 0)) for m in frame_metrics_list)
    total_w10 = sum(m.get("count_w10", int(round(m.get("within_10mm_ratio", 0.0) * m.get("overlapping_pixels", 0)))) for m in frame_metrics_list)
    total_w20 = sum(m.get("count_w20", int(round(m.get("within_20mm_ratio", 0.0) * m.get("overlapping_pixels", 0)))) for m in frame_metrics_list)
    total_w50 = sum(m.get("count_w50", int(round(m.get("within_50mm_ratio", 0.0) * m.get("overlapping_pixels", 0)))) for m in frame_metrics_list)
    total_fs_viol = sum(m.get("count_fs_viol", int(round(m.get("free_space_violation_ratio", 0.0) * m.get("overlapping_pixels", 0)))) for m in frame_metrics_list)

    if total_overlap == 0:
        cov = round(total_overlap / max(total_ref, 1), 4) if total_ref > 0 else 0.0
        return {
            "depth_mae_mm": None,
            "depth_rmse_mm": None,
            "depth_median_error_mm": None,
            "depth_p90_mm": None,
            "depth_p95_mm": None,
            "depth_coverage_ratio": cov,
            "observed_surface_completeness": 0.0,
            "free_space_violation_ratio": 0.0,
            "free_space_correctness_ratio": 1.0,
            "within_10mm_ratio": 0.0,
            "within_20mm_ratio": 0.0,
            "within_50mm_ratio": 0.0
        }

    cov = round(total_overlap / max(total_ref, 1), 4)
    mae = total_sum_abs / total_overlap
    rmse = float(np.sqrt(total_sum_sq / total_overlap))
    w10 = total_w10 / total_overlap
    w20 = total_w20 / total_overlap
    w50 = total_w50 / total_overlap
    fs_viol = total_fs_viol / total_overlap
    fs_corr = float(np.clip(1.0 - fs_viol, 0.0, 1.0))
    surface_compl = round(cov * w50, 4)

    if pooled_errors is not None and len(pooled_errors) > 0:
        med = float(np.median(pooled_errors))
        p90 = float(np.percentile(pooled_errors, 90))
        p95 = float(np.percentile(pooled_errors, 95))
    else:
        med = float(np.mean([m["depth_median_error_mm"] for m in frame_metrics_list if m.get("depth_median_error_mm") is not None])) if any(m.get("depth_median_error_mm") is not None for m in frame_metrics_list) else None
        p90 = float(np.mean([m["depth_p90_mm"] for m in frame_metrics_list if m.get("depth_p90_mm") is not None])) if any(m.get("depth_p90_mm") is not None for m in frame_metrics_list) else None
        p95 = float(np.mean([m["depth_p95_mm"] for m in frame_metrics_list if m.get("depth_p95_mm") is not None])) if any(m.get("depth_p95_mm") is not None for m in frame_metrics_list) else None

    return {
        "depth_mae_mm": round(float(mae), 2),
        "depth_rmse_mm": round(float(rmse), 2),
        "depth_median_error_mm": round(float(med), 2) if med is not None else None,
        "depth_p90_mm": round(float(p90), 2) if p90 is not None else None,
        "depth_p95_mm": round(float(p95), 2) if p95 is not None else None,
        "depth_coverage_ratio": cov,
        "observed_surface_completeness": round(float(surface_compl), 4),
        "free_space_violation_ratio": round(float(fs_viol), 4),
        "free_space_correctness_ratio": round(float(fs_corr), 4),
        "within_10mm_ratio": round(float(w10), 4),
        "within_20mm_ratio": round(float(w20), 4),
        "within_50mm_ratio": round(float(w50), 4)
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
