"""Streaming dataset audit over frame metadata.

Never loads the full sequence into RAM: one metadata pass plus optional
sampled image probes. Reports timestamp monotonicity, RGB-depth sync quality,
depth validity, and corruption at sample points.

Complexity: Time O(N + S) with S sampled frames.
Memory: O(1) / O(window) — only running stats retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import cv2

from auto_mobility.reconstruction.data.frame_selector import FrameSelector


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: str
    detail: str


@dataclass
class DatasetAuditResult:
    n_frames: int = 0
    monotonic_rgb: bool = True
    monotonic_depth: bool = True
    sync_dt_ms_p50: float = 0.0
    sync_dt_ms_p95: float = 0.0
    sync_dt_ms_max: float = 0.0
    sync_failures: int = 0
    depth_valid_ratio_median: float = 1.0
    corrupt_frames: list = field(default_factory=list)
    issues: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "FAIL" for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "monotonic_rgb": self.monotonic_rgb,
            "monotonic_depth": self.monotonic_depth,
            "sync_dt_ms_p50": round(self.sync_dt_ms_p50, 2),
            "sync_dt_ms_p95": round(self.sync_dt_ms_p95, 2),
            "sync_dt_ms_max": round(self.sync_dt_ms_max, 2),
            "sync_failures": self.sync_failures,
            "depth_valid_ratio_median": round(self.depth_valid_ratio_median, 4),
            "corrupt_frames": self.corrupt_frames,
            "issues": [
                {"code": i.code, "severity": i.severity, "detail": i.detail}
                for i in self.issues
            ],
            "ok": self.ok,
        }


def _percentile(values: list, q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), q))


def audit_dataset(
    frames,
    load_rgb: Callable[[object], np.ndarray],
    load_depth: Callable[[object], np.ndarray],
    probe_count: int = 24,
    selector: Optional[FrameSelector] = None,
    max_sync_dt_ms: float = 50.0,
) -> DatasetAuditResult:
    """Metadata streaming pass; image probes are evenly strided samples."""
    sel = selector or FrameSelector()
    result = DatasetAuditResult()
    result.n_frames = len(frames)
    if result.n_frames == 0:
        result.issues.append(AuditIssue("EMPTY_DATASET", "FAIL", "no frames found"))
        return result

    prev_rgb_ts = -np.inf
    prev_depth_ts = -np.inf
    dt_samples = []
    valid_ratios = []
    rng = np.random.RandomState(42)
    probe_idx = set(
        int(i) for i in rng.choice(result.n_frames, size=min(probe_count, result.n_frames), replace=False)
    )

    for idx, frame in enumerate(frames):
        if frame.rgb_timestamp <= prev_rgb_ts:
            result.monotonic_rgb = False
        if frame.depth_timestamp <= prev_depth_ts:
            result.monotonic_depth = False
        prev_rgb_ts = frame.rgb_timestamp
        prev_depth_ts = frame.depth_timestamp

        dt_ms = abs(frame.rgb_timestamp - frame.depth_timestamp) * 1000.0
        dt_samples.append(dt_ms)
        if dt_ms > max_sync_dt_ms or not (frame.rgb_path and frame.depth_path):
            result.sync_failures += 1

        if idx in probe_idx:
            try:
                rgb = load_rgb(frame)
                depth = load_depth(frame)
                if rgb is None or depth is None:
                    result.corrupt_frames.append(frame.frame_id)
                    continue
                q = sel.quality_from_arrays(frame, rgb, depth)
                valid_ratios.append(q.depth_valid_ratio)
            except (OSError, ValueError, cv2.error) as exc:
                result.corrupt_frames.append(frame.frame_id)
                result.issues.append(
                    AuditIssue("DECODE_FAILED", "WARN", f"frame {frame.frame_id}: {exc}")
                )

    result.sync_dt_ms_p50 = _percentile(dt_samples, 50)
    result.sync_dt_ms_p95 = _percentile(dt_samples, 95)
    result.sync_dt_ms_max = max(dt_samples) if dt_samples else 0.0
    result.depth_valid_ratio_median = (
        float(np.median(valid_ratios)) if valid_ratios else 1.0
    )

    if not result.monotonic_rgb:
        result.issues.append(AuditIssue("RGB_TS_NON_MONOTONIC", "WARN", "rgb timestamps regress"))
    if not result.monotonic_depth:
        result.issues.append(AuditIssue("DEPTH_TS_NON_MONOTONIC", "WARN", "depth timestamps regress"))
    fail_ratio = result.sync_failures / max(1, result.n_frames)
    if fail_ratio > 0.05:
        result.issues.append(
            AuditIssue(
                "SYNC_QUALITY_FAIL",
                "FAIL",
                f"{result.sync_failures}/{result.n_frames} frames exceed {max_sync_dt_ms} ms",
            )
        )
    elif fail_ratio > 0.01:
        result.issues.append(
            AuditIssue("SYNC_QUALITY_WARN", "WARN", f"sync failures: {result.sync_failures}")
        )
    if result.corrupt_frames:
        result.issues.append(
            AuditIssue("CORRUPT_FRAMES", "WARN", f"undecodable frames: {result.corrupt_frames}")
        )
    if valid_ratios and result.depth_valid_ratio_median < sel.min_depth_valid_ratio:
        result.issues.append(
            AuditIssue(
                "DEPTH_SPARSE",
                "FAIL",
                f"median depth validity {result.depth_valid_ratio_median:.3f}",
            )
        )
    return result
