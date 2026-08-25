"""
candidate.py — Candidate Specification, Identity, and Parameter Tracking.

Provides:
  - CandidateSpec: Explicit definition of the entire pipeline configuration
    (SLAM backend + profile + rate, frame selection, fusion, surface reconstruction, postprocessing).
  - SlamProfileSpec: SLAM execution profile definition registry.
  - Deterministic hashing and requested_params vs effective_params separation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
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


@dataclass
class SlamChampion:
    profile_spec: SlamProfileSpec
    trajectory_path: str
    trajectory_sha256: str
    phase_a_summary: dict

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.profile_spec.candidate_key
        elif idx == 1:
            return self.trajectory_path
        elif idx == 2:
            return self.trajectory_sha256
        elif idx == 3:
            return self.phase_a_summary
        raise IndexError(f"SlamChampion index out of range: {idx}")


def get_trajectory_filename(bag_name: str, key_or_spec: Union[str, SlamProfileSpec]) -> str:
    """Returns deterministic standard trajectory filename for any SLAM profile."""
    spec = get_slam_profile_spec(key_or_spec) if isinstance(key_or_spec, str) else key_or_spec
    return f"{spec.candidate_key}_{bag_name}_trajectory.txt"


def get_rtab_db_filename(bag_name: str, profile: str = "normal", rate: float = 1.0) -> str:
    """Returns deterministic standard RTAB database filename."""
    return f"{bag_name}_rtab_{profile}_rate{rate:g}.db"


# Standard SLAM Profile Registry (Frame-based direct execution)
STANDARD_SLAM_PROFILES: Dict[str, SlamProfileSpec] = {
    "rtab_normal": SlamProfileSpec(
        candidate_key="rtab_normal",
        backend="rtab",
        profile="normal",
        replay_rate=1.0,
        run_args=["--slam=rtab"],
        description="RTAB-Map default keyframe mapping"
    ),
    "rtab_dense": SlamProfileSpec(
        candidate_key="rtab_dense",
        backend="rtab",
        profile="dense",
        replay_rate=1.0,
        run_args=["--slam=rtab", "--dense"],
        description="RTAB-Map dense keyframe mapping"
    ),
    "orb_rgbd": SlamProfileSpec(
        candidate_key="orb_rgbd",
        backend="orb_rgbd",
        profile="normal",
        replay_rate=1.0,
        run_args=["--slam=orb_rgbd"],
        description="ORB-SLAM3 RGB-D"
    ),
    "orb_rgbdi": SlamProfileSpec(
        candidate_key="orb_rgbdi",
        backend="orb_rgbdi",
        profile="normal",
        replay_rate=1.0,
        run_args=["--slam=orb_rgbdi"],
        description="ORB-SLAM3 RGB-D-Inertial"
    ),
    "stella_rgbd": SlamProfileSpec(
        candidate_key="stella_rgbd",
        backend="stella_rgbd",
        profile="normal",
        replay_rate=1.0,
        run_args=["--slam=stella_rgbd"],
        description="stella_vslam RGB-D"
    ),
}


def get_slam_profile_spec(key: str) -> SlamProfileSpec:
    """Retrieve or construct a SlamProfileSpec from key."""
    if key in STANDARD_SLAM_PROFILES:
        return STANDARD_SLAM_PROFILES[key]

    # Normalize key by stripping legacy _rateX.X tag
    norm_key = key
    if "_rate" in key:
        norm_key = key.split("_rate")[0]
    if norm_key in STANDARD_SLAM_PROFILES:
        return STANDARD_SLAM_PROFILES[norm_key]
    
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

    return SlamProfileSpec(
        candidate_key=norm_key,
        backend=backend,
        profile=profile,
        replay_rate=rate,
        run_args=[f"--slam={backend}"] + (["--dense"] if profile == "dense" else [])
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
        parts = [str(self.slam_backend or "slam")]
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

    def compute_fusion_hash(self) -> str:
        """Deterministic hash of ONLY the parameters that affect raw reconstruction
        output (mesh/PCD from the fusion worker).

        The TSDF/direct-fusion workers receive voxel/depth_max/trunc_mult/stride and
        the split — surface method, postprocess, and unused passthrough params
        (e.g. weight_threshold) do not influence reconstruction bytes. Candidates
        sharing this hash can safely reuse the same mesh/PCD artifacts.
        """
        d = {
            "schema": "fh1",
            "dataset_name": self.dataset_name,
            "slam_backend": self.slam_backend,
            "slam_profile": self.slam_profile,
            "replay_rate": self.replay_rate,
            "frame_stride": self.frame_stride,
            "fusion_method": self.fusion_method,
            "fusion_params": {
                k: v for k, v in self.fusion_params.items() if k != "weight_threshold"
            },
        }
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def to_metadata_dict(self) -> dict:
        return {
            "candidate_id": self.compute_candidate_id(),
            "spec_hash": self.compute_spec_hash(),
            "requested_params": asdict(self),
            "effective_params": asdict(self)
        }
