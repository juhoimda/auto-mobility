"""Typed core models and the canonical coordinate/unit contract.

Complexity: pure dataclasses and O(1) math helpers. No IO, no heavy deps.
"""

from dataclasses import dataclass, field
import numpy as np

SCHEMA_VERSION = "recon-v2/2026-08-26"

PoseMatrix = np.ndarray


def invert_pose(t_world_camera: PoseMatrix) -> PoseMatrix:
    """Closed-form SE(3) inverse: T_camera_world."""
    R = t_world_camera[:3, :3]
    t = t_world_camera[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def validate_pose(pose: PoseMatrix, atol: float = 1e-6) -> None:
    """Raise ValueError unless pose is a valid rigid transform (T_world_camera contract)."""
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"pose must be 4x4, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pose contains NaN/Inf")
    R = arr[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError("rotation block is not orthonormal")
    bottom = arr[3, :]
    if not np.allclose(bottom, np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        raise ValueError("bottom row must be [0,0,0,1]")


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1000.0

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "depth_scale": self.depth_scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CameraIntrinsics":
        known = {"width", "height", "fx", "fy", "cx", "cy", "depth_scale"}
        missing = known - set(data)
        if missing:
            raise ValueError(f"missing intrinsics keys: {sorted(missing)}")
        extra = set(data) - known
        if extra:
            raise ValueError(f"unknown intrinsics keys: {sorted(extra)}")
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            depth_scale=float(data.get("depth_scale", 1000.0)),
        )


@dataclass(frozen=True)
class FrameMeta:
    frame_id: int
    rgb_timestamp: float
    depth_timestamp: float
    rgb_path: str
    depth_path: str
    rgb_depth_dt_ms: float

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "rgb_timestamp": self.rgb_timestamp,
            "depth_timestamp": self.depth_timestamp,
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "rgb_depth_dt_ms": self.rgb_depth_dt_ms,
        }


@dataclass(frozen=True)
class StageDecision:
    """One adaptive decision recorded into decision_trace.json."""

    stage: str
    decision: str
    reason: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "decision": self.decision,
            "reason": self.reason,
            "evidence": self.evidence,
        }
