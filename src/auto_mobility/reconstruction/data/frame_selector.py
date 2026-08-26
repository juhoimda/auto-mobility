"""Frame quality metrics and TRACK / FUSE / REJECT classification.

Metrics are computed on a downscaled copy (single decode per frame, shared
across all metrics). A bounded LRU cache holds decoded frames only.

Complexity: Time O(F) over F frames (O(pixels/decimation^2) each).
Memory: O(cache_frames * frame_bytes) bounded.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Callable, Optional

import cv2
import numpy as np


class FrameRole(str, Enum):
    TRACK = "TRACK"
    FUSE = "FUSE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class FrameQuality:
    frame_id: int
    sharpness: float
    brightness: float
    clipped_dark_ratio: float
    clipped_bright_ratio: float
    depth_valid_ratio: float
    depth_edge_ratio: float

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "sharpness": round(self.sharpness, 3),
            "brightness": round(self.brightness, 3),
            "clipped_dark_ratio": round(self.clipped_dark_ratio, 4),
            "clipped_bright_ratio": round(self.clipped_bright_ratio, 4),
            "depth_valid_ratio": round(self.depth_valid_ratio, 4),
            "depth_edge_ratio": round(self.depth_edge_ratio, 4),
        }


class _BoundedFrameCache:
    def __init__(self, capacity: int):
        self._cap = max(1, capacity)
        self._store: OrderedDict = OrderedDict()

    def get_or_load(self, key, loader: Callable):
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        value = loader()
        self._store[key] = value
        if len(self._store) > self._cap:
            self._store.popitem(last=False)
        return value


class FrameSelector:
    def __init__(
        self,
        min_depth_valid_ratio: float = 0.30,
        blur_percentile: float = 15.0,
        bright_range: tuple = (0.25, 0.85),
        downscale: int = 2,
        cache_frames: int = 12,
    ):
        if downscale < 1:
            raise ValueError("downscale must be >= 1")
        self.min_depth_valid_ratio = min_depth_valid_ratio
        self.blur_percentile = blur_percentile
        self.bright_lo, self.bright_hi = bright_range
        self.downscale = downscale
        self.cache = _BoundedFrameCache(cache_frames)

    def compute_quality(
        self,
        frame,
        load_rgb: Callable[[object], np.ndarray],
        load_depth: Callable[[object], np.ndarray],
    ) -> FrameQuality:
        rgb = self.cache.get_or_load(("rgb", id(frame)), lambda: load_rgb(frame))
        depth = self.cache.get_or_load(("d", id(frame)), lambda: load_depth(frame))
        return self.quality_from_arrays(frame, rgb, depth)

    def quality_from_arrays(self, frame, rgb: np.ndarray, depth: np.ndarray) -> FrameQuality:
        gray = self._to_gray_downscaled(rgb)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean()) / 255.0
        dark = float((gray <= 5).mean())
        bright = float((gray >= 250).mean())

        d = np.asarray(depth)
        valid = d > 0
        valid_ratio = float(valid.mean()) if valid.size else 0.0
        edge_ratio = self._depth_edge_ratio(d, valid)

        return FrameQuality(
            frame_id=int(getattr(frame, "frame_id")),
            sharpness=laplacian_var,
            brightness=brightness,
            clipped_dark_ratio=dark,
            clipped_bright_ratio=bright,
            depth_valid_ratio=valid_ratio,
            depth_edge_ratio=edge_ratio,
        )

    def classify(
        self, qualities: list, sync_dt_ms_by_frame: Optional[dict] = None
    ) -> dict:
        """Assign FrameRole to every frame.

        REJECT: blur below global percentile floor, extreme exposure,
                insufficient depth validity, or failed RGB-depth sync.
        FUSE:   passes all quality gates.
        TRACK:  usable for SLAM tracking but excluded from fusion.
        """
        sharp_values = sorted(q.sharpness for q in qualities)
        k = int(len(sharp_values) * self.blur_percentile / 100.0)
        sharp_floor = sharp_values[max(0, min(k, len(sharp_values) - 1))]

        roles = {}
        for q in qualities:
            sync_ok = True
            if sync_dt_ms_by_frame is not None:
                dt = sync_dt_ms_by_frame.get(q.frame_id)
                sync_ok = dt is None or dt <= 50.0
            if (
                not sync_ok
                or q.depth_valid_ratio < self.min_depth_valid_ratio
                or q.brightness < self.bright_lo
                or q.brightness > self.bright_hi
                or q.clipped_bright_ratio > 0.30
                or q.clipped_dark_ratio > 0.30
            ):
                roles[q.frame_id] = FrameRole.REJECT
            elif q.sharpness < sharp_floor and q.depth_edge_ratio > 0.20:
                roles[q.frame_id] = FrameRole.REJECT
            elif q.sharpness >= sharp_floor and q.depth_edge_ratio <= 0.35:
                roles[q.frame_id] = FrameRole.FUSE
            else:
                roles[q.frame_id] = FrameRole.TRACK
        return roles

    def _to_gray_downscaled(self, rgb: np.ndarray) -> np.ndarray:
        img = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // self.downscale, h // self.downscale),
                             interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _depth_edge_ratio(depth: np.ndarray, valid: np.ndarray) -> float:
        if depth.ndim != 2 or depth.shape[0] < 3 or depth.shape[1] < 3:
            return 0.0
        d = depth.astype(np.float32)
        gx = np.abs(np.diff(d, axis=1))
        gy = np.abs(np.diff(d, axis=0))
        valid_gx = valid[:, 1:] & valid[:, :-1]
        valid_gy = valid[1:, :] & valid[:-1, :]
        jump_x = (gx > 80.0) & valid_gx
        jump_y = (gy > 80.0) & valid_gy
        denom = max(1, valid_gx.sum() + valid_gy.sum())
        return float(jump_x.sum() + jump_y.sum()) / float(denom)
