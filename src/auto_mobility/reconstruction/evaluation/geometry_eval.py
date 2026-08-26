"""Held-out depth evaluation (Tier 1, #60).

Renders the reconstructed mesh at holdout views and compares against measured
depth using the pooled metric functions from the audited KEEP module
(auto_mobility.evaluation.geometry_metrics).

Time  O(V * raycast_cost), V = sampled holdout frames.
Memory O(1) per frame: metrics pooled incrementally, residuals not retained (#82).
"""

from __future__ import annotations

import numpy as np

from auto_mobility.reconstruction.model import CameraIntrinsics


def assess_geometry_quality(metrics: dict, *, max_mae_mm: float = 50.0,
                            max_p95_mm: float = 120.0,
                            min_coverage: float = 0.50,
                            min_within_20mm: float = 0.40) -> dict:
    """Return an explicit acceptance decision, separate from evaluator health."""
    if metrics.get("status") != "ok":
        return {"status": "FAIL", "reasons": ["evaluation_unavailable"]}
    checks = {
        "depth_mae_mm": float(metrics.get("depth_mae_mm", float("inf"))) <= max_mae_mm,
        "depth_p95_mm": float(metrics.get("depth_p95_mm", float("inf"))) <= max_p95_mm,
        "depth_coverage_ratio": float(metrics.get("depth_coverage_ratio", 0.0)) >= min_coverage,
        "within_20mm_ratio": float(metrics.get("within_20mm_ratio", 0.0)) >= min_within_20mm,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "reasons": [name for name, passed in checks.items() if not passed],
            "thresholds": {"max_mae_mm": max_mae_mm, "max_p95_mm": max_p95_mm,
                           "min_coverage": min_coverage,
                           "min_within_20mm": min_within_20mm}}


def evaluate_geometry(mesh, frames, pose_by_frame, cam: CameraIntrinsics,
                      depth_loader, max_views: int = 12,
                      depth_min_mm: float = 300.0,
                      depth_max_mm: float = 5000.0) -> dict:
    import open3d as o3d

    from auto_mobility.evaluation.geometry_metrics import (
        aggregate_depth_metrics, compute_depth_metrics,
    )
    from auto_mobility.evaluation.render_depth import (
        create_raycasting_scene, render_depth_map,
    )

    if mesh is None or len(mesh.triangles) == 0 or not frames:
        return {"status": "EVALUATION_FAILED", "reason": "empty mesh or no frames"}

    scene = create_raycasting_scene(mesh)
    per_frame = []
    stride = max(1, len(frames) // max_views)
    for f in frames[::stride][:max_views]:
        T_wc = pose_by_frame.get(f.frame_id)
        real = depth_loader(f.frame_id)
        if T_wc is None or real is None:
            continue
        rendered = render_depth_map(scene, T_wc, cam,
                                    width=cam.width, height=cam.height)
        if rendered.shape != real.shape[:2]:
            import cv2
            rendered = cv2.resize(rendered, (real.shape[1], real.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        m = compute_depth_metrics(real.astype(np.float64), rendered.astype(np.float64),
                                  depth_min_mm=depth_min_mm, depth_max_mm=depth_max_mm)
        m["frame_id"] = f.frame_id
        per_frame.append(m)

    if not per_frame:
        return {"status": "EVALUATION_FAILED", "reason": "no evaluable views"}

    agg = aggregate_depth_metrics(per_frame)
    agg["n_eval_views"] = len(per_frame)
    agg["status"] = "ok"
    agg["quality"] = assess_geometry_quality(agg)
    return agg
