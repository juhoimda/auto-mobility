"""
candidate.py — Candidate Specification, Identity, and Parameter Tracking.

Provides:
  - CandidateSpec: Explicit definition of the entire pipeline configuration
    (SLAM backend + profile + rate, frame selection, fusion, surface reconstruction, postprocessing).
  - Deterministic hashing and requested_params vs effective_params separation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import hashlib
import json


@dataclass
class CandidateSpec:
    dataset_name: str
    slam_backend: str               # "rtab", "orb_rgbd", "orb_rgbdi", "stella_rgbd"
    slam_profile: str = "normal"    # "normal", "dense"
    replay_rate: float = 1.0        # 1.0, 0.5, 0.75, 0.25
    frame_selector: str = "all"     # "all", "stride", "motion"
    frame_stride: int = 1
    fusion_method: str = "tsdf"     # "tsdf", "direct_pointcloud"
    fusion_params: Dict[str, Any] = field(default_factory=lambda: {
        "voxel_size_m": 0.010,
        "depth_min_m": 0.3,
        "depth_max_m": 3.0,
        "trunc_mult": 4.0,
        "weight_threshold": 1.5
    })
    surface_method: str = "tsdf_direct"  # "tsdf_direct", "poisson", "bpa", "alpha_shape", "cgal_polygonal"
    surface_params: Dict[str, Any] = field(default_factory=lambda: {
        "depth": 8,
        "alpha_factor": 3.0,
        "orient": "centroid"
    })
    postprocess_params: Dict[str, Any] = field(default_factory=lambda: {
        "clean_density": True,
        "simplify_target": 0.5
    })
    is_full_rebuild: bool = False
    git_commit: str = "unknown"

    def compute_candidate_id(self) -> str:
        """Compute a human-readable unique identifier."""
        parts = [self.slam_backend]
        if self.slam_profile and self.slam_profile != "normal":
            parts.append(self.slam_profile)
        if self.replay_rate != 1.0:
            parts.append(f"rate{self.replay_rate:.2f}".rstrip("0").rstrip("."))

        v_mm = int(round(self.fusion_params.get("voxel_size_m", 0.010) * 1000))
        if self.fusion_method == "direct_pointcloud":
            parts.append(f"direct{v_mm}mm")
        else:
            parts.append(f"tsdf{v_mm}mm")

        if self.surface_method != "tsdf_direct":
            parts.append(self.surface_method)

        if self.is_full_rebuild:
            parts.append("fullrebuild")

        return "_".join(parts)

    def compute_spec_hash(self) -> str:
        """Compute a deterministic 16-character SHA-256 hash for cache validation."""
        d = asdict(self)
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def to_metadata_dict(self) -> dict:
        return {
            "candidate_id": self.compute_candidate_id(),
            "spec_hash": self.compute_spec_hash(),
            "requested_params": asdict(self),
            "effective_params": asdict(self)
        }
