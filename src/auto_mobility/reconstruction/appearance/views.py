"""Appearance helpers: exposure normalization + Tier-4 metrics (#63, #73)."""

from __future__ import annotations

import numpy as np


def normalize_exposure(bgr: np.ndarray, target_brightness: float = 0.45) -> np.ndarray:
    """Match mean luminance to a fixed reference so multi-view blending
    does not bake exposure differences into the atlas (#73)."""
    gray = bgr.mean(axis=2) / 255.0
    cur = float(gray.mean())
    if cur < 1e-3:
        return bgr
    gain = float(np.clip(target_brightness / cur, 0.7, 1.4))
    out = np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out


def atlas_metrics(atlas_bgr: np.ndarray, untextured_faces: int,
                  total_faces: int) -> dict:
    """Tier-4 appearance metrics (#63).

    Atlas cells hold one solid color per face, so per-cell Laplacian blur is
    meaningless; coverage = occupied cells, uniformity = cross-face color
    variance (low variance reads as washed-out texture).
    """
    h, w = atlas_bgr.shape[:2]
    cell = max(1, min(h, w) // 64)
    cell_means = []
    for y in range(0, h - cell + 1, cell):
        for x in range(0, w - cell + 1, cell):
            patch = atlas_bgr[y : y + cell, x : x + cell]
            if patch.any():
                cell_means.append(patch.reshape(-1, 3).mean(axis=0))
    if not cell_means:
        return {"texture_coverage": 0.0,
                "untextured_face_ratio": round(untextured_faces / max(1, total_faces), 4),
                "color_variance": 0.0}
    cm = np.asarray(cell_means)
    color_var = float(cm.std(axis=0).mean())
    return {
        "texture_coverage": round(len(cell_means) / max(1, ((h // cell) * (w // cell))), 4),
        "untextured_face_ratio": round(untextured_faces / max(1, total_faces), 4),
        "color_variance": round(color_var, 2),
        "mean_brightness": round(float(cm.mean()) / 255.0, 4),
        "n_atlas_cells": len(cell_means),
    }
