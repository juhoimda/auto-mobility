"""Cheap canonical RGB-D input suitability checks for SLAM."""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from auto_mobility.dataset.frame_dataset import FrameDataset


def analyze_frame_quality(
    dataset: FrameDataset,
    sample_count: int = 100,
    depth_min_mm: float = 300.0,
    depth_max_mm: float = 5000.0,
) -> Dict[str, Any]:
    """Sample canonical frames and report image/depth usability indicators."""
    if len(dataset) == 0:
        return {"overall_status": "FAIL", "sample_count": 0, "issues": ["Dataset is empty"]}
    indices = np.linspace(0, len(dataset) - 1, min(sample_count, len(dataset)), dtype=int)
    blur_values, brightness_values, invalid_ratios, shape_mismatches = [], [], [], 0
    missing_rgb = missing_depth = 0
    for idx in indices:
        rgb = dataset.get_rgb(int(idx))
        depth = dataset.get_depth(int(idx))
        if rgb is None:
            missing_rgb += 1
        else:
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            brightness_values.append(float(np.mean(gray)))
        if depth is None:
            missing_depth += 1
        else:
            valid = (depth >= depth_min_mm) & (depth <= depth_max_mm)
            invalid_ratios.append(float(1.0 - np.mean(valid)))
            if rgb is not None and depth.shape[:2] != rgb.shape[:2]:
                shape_mismatches += 1

    n = max(len(indices), 1)
    issues = []
    if missing_rgb:
        issues.append(f"Missing RGB samples: {missing_rgb}/{n}")
    if missing_depth:
        issues.append(f"Missing Depth samples: {missing_depth}/{n}")
    if invalid_ratios and float(np.mean(invalid_ratios)) > 0.35:
        issues.append(f"High invalid depth ratio: {np.mean(invalid_ratios) * 100:.1f}%")
    if shape_mismatches:
        issues.append(f"RGB/Depth shape mismatch: {shape_mismatches}/{n}")
    if blur_values and float(np.mean(blur_values)) < 20.0:
        issues.append(f"Low RGB sharpness (Laplacian variance {np.mean(blur_values):.1f})")
    if brightness_values:
        dark = float(np.mean(np.asarray(brightness_values) < 15.0))
        bright = float(np.mean(np.asarray(brightness_values) > 245.0))
        if dark > 0.30 or bright > 0.30:
            issues.append(f"Exposure outliers: dark={dark * 100:.1f}%, bright={bright * 100:.1f}%")

    status = "FAIL" if missing_rgb / n > 0.05 or missing_depth / n > 0.05 or (invalid_ratios and np.mean(invalid_ratios) > 0.5) else ("WARN" if issues else "PASS")
    return {
        "overall_status": status,
        "sample_count": int(n),
        "missing_rgb": int(missing_rgb),
        "missing_depth": int(missing_depth),
        "mean_invalid_depth_ratio": round(float(np.mean(invalid_ratios)), 4) if invalid_ratios else None,
        "mean_rgb_sharpness_laplacian": round(float(np.mean(blur_values)), 2) if blur_values else None,
        "mean_rgb_brightness": round(float(np.mean(brightness_values)), 2) if brightness_values else None,
        "shape_mismatch_count": int(shape_mismatches),
        "issues": issues,
    }

