"""
tests/unit/test_depth_metrics.py

Unit tests for depth reprojection error metrics (MAE, RMSE, Median, P95, Coverage, Within-N-mm).
"""

import numpy as np
import pytest
from auto_mobility.evaluation.geometry_metrics import compute_depth_metrics, aggregate_depth_metrics


def test_depth_metrics_exact_match():
    # 100x100 constant depth of 1500mm (1.5m)
    real = np.full((100, 100), 1500.0, dtype=np.float32)
    rendered = np.full((100, 100), 1500.0, dtype=np.float32)

    metrics = compute_depth_metrics(real, rendered)

    assert metrics["valid_reference_pixels"] == 10000
    assert metrics["overlapping_pixels"] == 10000
    assert metrics["depth_coverage_ratio"] == 1.0
    assert metrics["depth_mae_mm"] == 0.0
    assert metrics["depth_rmse_mm"] == 0.0
    assert metrics["depth_median_error_mm"] == 0.0
    assert metrics["depth_p95_mm"] == 0.0
    assert metrics["within_10mm_ratio"] == 1.0
    assert metrics["within_20mm_ratio"] == 1.0
    assert metrics["within_50mm_ratio"] == 1.0


def test_depth_metrics_known_error_distribution():
    # 100x100 depth
    # 50% with 10mm error, 30% with 30mm error, 20% with 100mm error
    real = np.full((100, 100), 2000.0, dtype=np.float32)
    rendered = np.full((100, 100), 2000.0, dtype=np.float32)

    # 51% with 10mm error, 29% with 30mm error, 20% with 100mm error
    rendered[:51, :] += 10.0   # 5100 pixels with 10mm error
    rendered[51:80, :] += 30.0 # 2900 pixels with 30mm error
    rendered[80:, :] += 100.0  # 2000 pixels with 100mm error

    metrics = compute_depth_metrics(real, rendered)

    # MAE = 0.51*10 + 0.29*30 + 0.20*100 = 5.1 + 8.7 + 20 = 33.8 mm
    assert metrics["depth_mae_mm"] == 33.8
    assert metrics["depth_median_error_mm"] == 10.0
    assert metrics["within_10mm_ratio"] == 0.51
    assert metrics["within_20mm_ratio"] == 0.51
    assert metrics["within_50mm_ratio"] == 0.80
    assert metrics["depth_p95_mm"] == 100.0


def test_depth_metrics_coverage_and_unobserved_regions():
    real = np.full((100, 100), 2000.0, dtype=np.float32)
    rendered = np.zeros((100, 100), dtype=np.float32) # No hit

    # Only 40% rendered
    rendered[:40, :] = 2020.0 # 20mm error

    metrics = compute_depth_metrics(real, rendered)

    assert metrics["valid_reference_pixels"] == 10000
    assert metrics["overlapping_pixels"] == 4000
    assert metrics["depth_coverage_ratio"] == 0.40
    assert metrics["depth_mae_mm"] == 20.0


def test_free_space_violation_detection():
    # Real depth is at 2000mm (2.0m)
    real = np.full((100, 100), 2000.0, dtype=np.float32)
    rendered = np.full((100, 100), 2000.0, dtype=np.float32)

    # 30% of pixels have phantom surfaces in free space at 1500mm (500mm closer than real wall)
    rendered[:30, :] = 1500.0

    metrics = compute_depth_metrics(real, rendered, free_space_margin_mm=50.0)

    # 3000 out of 10000 pixels violate free space
    assert metrics["free_space_violation_ratio"] == 0.30
    assert metrics["free_space_correctness_ratio"] == 0.70


def test_surface_completeness():
    # Real depth is at 2000mm
    real = np.full((100, 100), 2000.0, dtype=np.float32)
    rendered = np.zeros((100, 100), dtype=np.float32)

    # 80% coverage, all with 10mm error (within 50mm)
    rendered[:80, :] = 2010.0

    metrics = compute_depth_metrics(real, rendered)

    assert metrics["depth_coverage_ratio"] == 0.80
    assert metrics["within_50mm_ratio"] == 1.0
    # Observed surface completeness = coverage * within_50mm_ratio = 0.80 * 1.0 = 0.80
    assert metrics["observed_surface_completeness"] == 0.80


def test_aggregate_depth_metrics_exact_pooling():
    # Frame 1: 100 pixels, all error = 10mm
    r1 = np.full((10, 10), 1000.0, dtype=np.float32)
    p1 = np.full((10, 10), 1010.0, dtype=np.float32)
    m1 = compute_depth_metrics(r1, p1, return_errors=True)
    err1 = m1.pop("errors")

    # Frame 2: 300 pixels, all error = 30mm
    r2 = np.full((10, 30), 1000.0, dtype=np.float32)
    p2 = np.full((10, 30), 1030.0, dtype=np.float32)
    m2 = compute_depth_metrics(r2, p2, return_errors=True)
    err2 = m2.pop("errors")

    # Total pixels = 400 (100 @ 10mm, 300 @ 30mm)
    # Expected pooled MAE = (100*10 + 300*30) / 400 = (1000 + 9000)/400 = 25.0 mm
    # Expected pooled RMSE = sqrt((100*100 + 300*900)/400) = sqrt((10000 + 270000)/400) = sqrt(700) = 26.4575... mm
    # Expected pooled median = 30.0 mm
    # Expected pooled P95 = 30.0 mm
    pooled = np.concatenate([err1, err2])
    agg = aggregate_depth_metrics([m1, m2], pooled_errors=pooled)

    assert agg["depth_mae_mm"] == 25.0
    assert abs(agg["depth_rmse_mm"] - 26.46) <= 0.02
    assert agg["depth_median_error_mm"] == 30.0
    assert agg["depth_p95_mm"] == 30.0
    assert agg["depth_coverage_ratio"] == 1.0
    assert agg["within_10mm_ratio"] == 0.25
    assert agg["within_50mm_ratio"] == 1.0


