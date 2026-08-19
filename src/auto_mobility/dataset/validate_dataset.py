#!/usr/bin/env python3
"""
validate_dataset.py — Canonical Frame Dataset 무결성 및 품질 검증 도구
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import FRAME_DIR, get_evaluation_config
from auto_mobility.dataset.frame_dataset import FrameDataset


def validate_dataset(dataset_path: str) -> dict:
    p = Path(dataset_path)
    if not p.is_absolute():
        if (FRAME_DIR / dataset_path).exists():
            p = FRAME_DIR / dataset_path
        elif not p.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    dataset = FrameDataset(p)
    cfg = get_evaluation_config()
    thresholds = cfg.get("quality_thresholds", {}).get("dataset", {})

    checks = {}
    warnings = []
    errors = []

    # 1. Basic frame count check
    n_frames = len(dataset)
    checks["frame_count"] = {"pass": n_frames > 0, "count": n_frames}
    if n_frames == 0:
        errors.append("Dataset contains 0 frames!")

    # 2. Check file integrity & sample images
    missing_rgb = 0
    missing_depth = 0
    invalid_depth_type = 0
    sample_indices = np.linspace(0, max(0, n_frames - 1), min(n_frames, 20), dtype=int)

    for idx in sample_indices:
        rgb_path = dataset.get_rgb_path(idx)
        depth_path = dataset.get_depth_path(idx)

        if not rgb_path.exists() or rgb_path.stat().st_size == 0:
            missing_rgb += 1
        if not depth_path.exists() or depth_path.stat().st_size == 0:
            missing_depth += 1

        dimg = dataset.get_depth(idx)
        if dimg is not None:
            if dimg.dtype != np.uint16:
                invalid_depth_type += 1

    checks["rgb_files"] = {"pass": missing_rgb == 0, "missing": missing_rgb}
    checks["depth_files"] = {"pass": missing_depth == 0, "missing": missing_depth}
    checks["depth_format_16uc1"] = {"pass": invalid_depth_type == 0, "invalid": invalid_depth_type}

    if missing_rgb > 0 or missing_depth > 0:
        errors.append(f"Missing frame image files: RGB {missing_rgb}, Depth {missing_depth}")
    if invalid_depth_type > 0:
        warnings.append(f"Depth format is not 16-bit uint16 mm: {invalid_depth_type} samples")

    # 3. Timestamps & Monotonicity
    rgb_stamps = dataset.get_timestamps(use_rgb=True)
    depth_stamps = dataset.get_timestamps(use_rgb=False)

    rgb_diffs = np.diff(rgb_stamps) if len(rgb_stamps) > 1 else np.array([0.0])
    monotonic_violations = int(np.sum(rgb_diffs < 0))
    duplicate_stamps = int(np.sum(rgb_diffs == 0))

    checks["monotonicity"] = {"pass": monotonic_violations == 0, "violations": monotonic_violations}
    checks["duplicate_stamps"] = {"pass": duplicate_stamps == 0, "duplicates": duplicate_stamps}

    if monotonic_violations > 0:
        errors.append(f"Timestamp monotonicity violations found: {monotonic_violations}")
    if duplicate_stamps > 0:
        warnings.append(f"Duplicate timestamps found: {duplicate_stamps}")

    # 4. Sync Delta stats
    sync_deltas_ms = np.abs(depth_stamps - rgb_stamps) * 1000.0
    sync_mean_ms = float(np.mean(sync_deltas_ms)) if len(sync_deltas_ms) else 0.0
    sync_p95_ms = float(np.percentile(sync_deltas_ms, 95)) if len(sync_deltas_ms) else 0.0
    sync_max_ms = float(np.max(sync_deltas_ms)) if len(sync_deltas_ms) else 0.0

    sync_pass_thr = thresholds.get("sync_delta_p95_ms", {}).get("pass", 25.0)
    sync_warn_thr = thresholds.get("sync_delta_p95_ms", {}).get("warn", 50.0)

    if sync_p95_ms > sync_warn_thr:
        warnings.append(f"RGB-Depth sync delta P95 is high: {sync_p95_ms:.1f}ms > {sync_warn_thr}ms")

    # 5. Estimated Hz
    duration = float(rgb_stamps[-1] - rgb_stamps[0]) if len(rgb_stamps) > 1 else 0.0
    est_hz = round(n_frames / duration, 2) if duration > 0 else 0.0
    hz_pass_thr = thresholds.get("rgb_hz", {}).get("pass", 20.0)
    hz_warn_thr = thresholds.get("rgb_hz", {}).get("warn", 10.0)

    if est_hz < hz_warn_thr:
        warnings.append(f"Estimated FPS is low: {est_hz} Hz < {hz_warn_thr} Hz")

    # 6. CameraInfo check
    intr = dataset.intrinsics
    has_cam_info = intr.fx > 0 and intr.fy > 0 and intr.cx > 0 and intr.cy > 0
    is_fallback = dataset.dataset_info.get("is_camera_info_fallback", False)

    checks["camera_info"] = {"pass": has_cam_info, "is_fallback": is_fallback}
    if is_fallback:
        warnings.append("CameraInfo used fallback intrinsics! Actual calibration parameters missing.")

    overall_pass = len(errors) == 0

    result = {
        "dataset_name": p.name,
        "dataset_path": str(p),
        "pass": overall_pass,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "quality_metrics": {
            "num_frames": n_frames,
            "duration_sec": round(duration, 3),
            "estimated_hz": est_hz,
            "sync_delta_mean_ms": round(sync_mean_ms, 2),
            "sync_delta_p95_ms": round(sync_p95_ms, 2),
            "sync_delta_max_ms": round(sync_max_ms, 2),
            "monotonic_violations": monotonic_violations,
            "duplicate_stamps": duplicate_stamps,
            "is_camera_info_fallback": is_fallback
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate Canonical Frame Dataset")
    parser.add_argument("dataset", help="Dataset name or path (e.g. ros2_data/frames/room01)")
    args = parser.parse_args()

    res = validate_dataset(args.dataset)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if not res["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
