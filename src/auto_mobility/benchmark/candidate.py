"""
candidate.py — Candidate Specification, Identity, and Parameter Tracking.

Provides:
  - CandidateSpec: Explicit definition of the entire pipeline configuration
    (SLAM backend + profile + rate, frame selection, fusion, surface reconstruction, postprocessing).
  - SlamProfileSpec: SLAM execution profile definition registry.
  - Deterministic hashing and requested_params vs effective_params separation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import hashlib
import json


@dataclass
class SlamProfileSpec:
    candidate_key: str              # e.g. "rtab_dense_rate0.5", "orb_rgbd_rate0.5"
    backend: str                    # "rtab", "orb_rgbd", "orb_rgbdi", "stella_rgbd"
    profile: str = "normal"         # "normal", "dense"
    replay_rate: float = 1.0        # 1.0, 0.5, 0.75, 0.25
    run_args: List[str] = field(default_factory=list)
    description: str = ""

    def compute_spec_hash(self) -> str:
        d = asdict(self)
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# Standard SLAM Profile Registry
STANDARD_SLAM_PROFILES: Dict[str, SlamProfileSpec] = {
    "rtab_dense_rate0.5": SlamProfileSpec(
        candidate_key="rtab_dense_rate0.5",
        backend="rtab",
        profile="dense",
        replay_rate=0.5,
        run_args=["--slam=rtab", "--dense", "--rate=0.5"],
        description="RTAB-Map dense offline mapping at 0.5x playback"
    ),
    "rtab_dense_rate1.0": SlamProfileSpec(
        candidate_key="rtab_dense_rate1.0",
        backend="rtab",
        profile="dense",
        replay_rate=1.0,
        run_args=["--slam=rtab", "--dense", "--rate=1.0"],
        description="RTAB-Map dense offline mapping at 1.0x playback"
    ),
    "rtab_normal_rate0.5": SlamProfileSpec(
        candidate_key="rtab_normal_rate0.5",
        backend="rtab",
        profile="normal",
        replay_rate=0.5,
        run_args=["--slam=rtab", "--rate=0.5"],
        description="RTAB-Map default keyframe mapping at 0.5x playback"
    ),
    "rtab_normal_rate1.0": SlamProfileSpec(
        candidate_key="rtab_normal_rate1.0",
        backend="rtab",
        profile="normal",
        replay_rate=1.0,
        run_args=["--slam=rtab", "--rate=1.0"],
        description="RTAB-Map default keyframe mapping at 1.0x playback"
    ),
    "orb_rgbd_rate0.5": SlamProfileSpec(
        candidate_key="orb_rgbd_rate0.5",
        backend="orb_rgbd",
        profile="normal",
        replay_rate=0.5,
        run_args=["--slam=orb_rgbd", "--rate=0.5"],
        description="ORB-SLAM3 RGB-D at 0.5x playback"
    ),
    "orb_rgbd_rate1.0": SlamProfileSpec(
        candidate_key="orb_rgbd_rate1.0",
        backend="orb_rgbd",
        profile="normal",
        replay_rate=1.0,
        run_args=["--slam=orb_rgbd", "--rate=1.0"],
        description="ORB-SLAM3 RGB-D at 1.0x playback"
    ),
    "orb_rgbdi_rate0.5": SlamProfileSpec(
        candidate_key="orb_rgbdi_rate0.5",
        backend="orb_rgbdi",
        profile="normal",
        replay_rate=0.5,
        run_args=["--slam=orb_rgbdi", "--rate=0.5"],
        description="ORB-SLAM3 RGB-D-Inertial at 0.5x playback"
    ),
    "orb_rgbdi_rate1.0": SlamProfileSpec(
        candidate_key="orb_rgbdi_rate1.0",
        backend="orb_rgbdi",
        profile="normal",
        replay_rate=1.0,
        run_args=["--slam=orb_rgbdi", "--rate=1.0"],
        description="ORB-SLAM3 RGB-D-Inertial at 1.0x playback"
    ),
    "stella_rgbd_rate0.5": SlamProfileSpec(
        candidate_key="stella_rgbd_rate0.5",
        backend="stella_rgbd",
        profile="normal",
        replay_rate=0.5,
        run_args=["--slam=stella_rgbd", "--rate=0.5"],
        description="stella_vslam RGB-D at 0.5x playback"
    ),
    "stella_rgbd_rate1.0": SlamProfileSpec(
        candidate_key="stella_rgbd_rate1.0",
        backend="stella_rgbd",
        profile="normal",
        replay_rate=1.0,
        run_args=["--slam=stella_rgbd", "--rate=1.0"],
        description="stella_vslam RGB-D at 1.0x playback"
    ),
}


def get_slam_profile_spec(key: str) -> SlamProfileSpec:
    """Retrieve or construct a SlamProfileSpec from key."""
    if key in STANDARD_SLAM_PROFILES:
        return STANDARD_SLAM_PROFILES[key]
    
    # Parse candidate key components if custom
    backend = "rtab"
    profile = "normal"
    rate = 1.0
    if "rtab" in key:
        backend = "rtab"
        profile = "dense" if "dense" in key else "normal"
    elif "orb_rgbdi" in key or "rgbdi" in key:
        backend = "orb_rgbdi"
    elif "orb" in key:
        backend = "orb_rgbd"
    elif "stella" in key:
        backend = "stella_rgbd"

    if "0.5" in key:
        rate = 0.5
    elif "0.25" in key:
        rate = 0.25
    elif "0.75" in key:
        rate = 0.75

    return SlamProfileSpec(
        candidate_key=key,
        backend=backend,
        profile=profile,
        replay_rate=rate,
        run_args=[f"--slam={backend}", f"--rate={rate}"] + (["--dense"] if profile == "dense" else [])
    )


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
        "simplify_target": 0.0      # Screening default: no simplification
    })
    evaluation_profile: str = "screening" # "screening", "expanded", "full"
    is_full_rebuild: bool = False
    code_version: str = "unknown"
    cache_schema_version: str = "v2"

    def compute_candidate_id(self, include_hash: bool = False) -> str:
        """Compute a human-readable unique identifier."""
        parts = [self.slam_backend]
        if self.slam_profile and self.slam_profile != "normal":
            parts.append(self.slam_profile)
        if self.replay_rate != 1.0:
            parts.append(f"rate{self.replay_rate:g}")

        v_mm = int(round(self.fusion_params.get("voxel_size_m", 0.010) * 1000))
        if self.fusion_method == "direct_pointcloud":
            parts.append(f"direct{v_mm}mm")
        else:
            parts.append(f"tsdf{v_mm}mm")

        if self.surface_method != "tsdf_direct":
            parts.append(self.surface_method)

        if self.is_full_rebuild:
            parts.append("fullrebuild")

        base_id = "_".join(parts)
        if include_hash:
            h = self.compute_spec_hash()[:8]
            return f"{base_id}_{h}"
        return base_id

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
