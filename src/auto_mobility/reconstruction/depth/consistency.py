"""Depth Consistency 2-pass masking (#46-48).

Pass 1 coarse mesh -> render each candidate frame -> mask pixels whose
measured depth disagrees with the rendered surface beyond an adaptive
allowed_error = sensor_uncertainty(depth) + pose_uncertainty + base_margin.
Depth-edge / silhouette neighborhoods get conservative protection (#48).

Time  O(V * raycast_cost + V * pixels)
Memory O(1) per frame (mask emitted streaming).
"""

from __future__ import annotations

import numpy as np

from auto_mobility.dataset.frame_dataset import CameraIntrinsics


def allowed_error_mm(depth_mm: np.ndarray, base_margin_mm: float = 25.0,
                     sensor_ratio: float = 0.02,
                     pose_uncertainty_mm: float = 8.0) -> np.ndarray:
    """Adaptive per-pixel tolerance; grows quadratically-ish with depth."""
    z = np.maximum(depth_mm.astype(np.float32), 1e-3)
    return (sensor_ratio * z * z / 1000.0 * 8.0 + pose_uncertainty_mm + base_margin_mm).astype(np.float32)


def _edge_protection(real_mm: np.ndarray, radius: int = 2, jump_mm: float = 60.0) -> np.ndarray:
    """Dilate depth-discontinuity regions so thin structures survive (#48)."""
    valid = real_mm > 0
    edge = np.zeros_like(valid)
    for shift in range(1, radius + 1):
        for ax in (0, 1):
            a = np.roll(real_mm, shift, axis=ax)
            b = np.roll(real_mm, -shift, axis=ax)
            va = np.roll(valid, shift, axis=ax)
            vb = np.roll(valid, -shift, axis=ax)
            with np.errstate(invalid="ignore"):
                edge |= valid & va & (np.abs(real_mm - a) > jump_mm)
                edge |= valid & vb & (np.abs(real_mm - b) > jump_mm)
    return edge


def compute_consistency_mask(
    real_mm: np.ndarray,
    rendered_mm: np.ndarray,
    pose_uncertainty_mm: float = 8.0,
    edge_protect: bool = True,
) -> np.ndarray:
    """True = keep pixel for final fusion."""
    both = (real_mm > 0) & (rendered_mm > 0)
    keep = real_mm <= 0
    err = np.abs(real_mm - rendered_mm)
    tol = allowed_error_mm(real_mm, pose_uncertainty_mm=pose_uncertainty_mm)
    agree = both & (err <= tol)
    disagree = both & (err > 3.0 * tol)

    out = keep | agree
    if edge_protect:
        protected = _edge_protection(real_mm)
        out = keep | agree | (disagree & protected)
    else:
        out = keep | (both & ~disagree)
    return out


def render_frame_depth(scene, T_wc, cam: CameraIntrinsics):
    from auto_mobility.evaluation.render_depth import render_depth_map

    return render_depth_map(scene, T_wc, cam, width=cam.width, height=cam.height)
