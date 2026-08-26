"""Fusion backend abstraction (next.md #38, #39).

Backend choice is a machine capability problem decided ONCE per calibration,
never re-raced per dataset. GPU-heavy integrate() must run under the
scheduler's single GPU slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

from auto_mobility.reconstruction.runtime.machine_profile import MachineProfile


class FusionBackendName(str, Enum):
    NVBLOX = "nvblox"
    OPEN3D_VBG_CUDA = "open3d_vbg_cuda"
    OPEN3D_VBG_CPU = "open3d_vbg_cpu"


@dataclass(frozen=True)
class CapabilityResult:
    backend: FusionBackendName
    available: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {"backend": self.backend.value, "available": self.available, "reason": self.reason}


@dataclass(frozen=True)
class ResourceEstimate:
    voxel_m: float
    estimated_voxel_count: int
    vram_mb: int
    ram_mb: int
    gpu_slots: int

    @property
    def fits_gpu(self) -> bool:
        return self.gpu_slots > 0 and self.vram_mb > 0


@dataclass(frozen=True)
class EffectiveFusionParams:
    """requested='auto' resolved against measured scene/HW (#42/#88)."""

    requested_voxel_mm: float
    effective_voxel_mm: float
    truncation_multiplier: float = 4.0

    @property
    def truncation_m(self) -> float:
        return self.effective_voxel_mm * self.truncation_multiplier / 1000.0

    @property
    def voxel_m(self) -> float:
        return self.effective_voxel_mm / 1000.0


def choose_fusion_backend(profile: MachineProfile) -> CapabilityResult:
    """nvblox PASS -> primary; else Open3D CUDA VBG; else CPU fallback (#39)."""
    if profile.nvblox_available:
        return CapabilityResult(FusionBackendName.NVBLOX, True)
    if profile.open3d_cuda and profile.gpu.present and profile.gpu.vram_free_mb >= 1500:
        return CapabilityResult(FusionBackendName.OPEN3D_VBG_CUDA, True)
    return CapabilityResult(
        FusionBackendName.OPEN3D_VBG_CPU,
        True,
        reason="no CUDA-capable stack detected",
    )


def estimate_resources(
    params: EffectiveFusionParams,
    trajectory_bbox_diagonal_m: float,
    n_frames: int,
    profile: MachineProfile,
    occupancy_factor: float = 0.22,
) -> ResourceEstimate:
    """Block-count style estimate (reconstruct_tsdf idea, V2-clean form).

    Time O(1), Memory O(1).
    """
    voxels_along_diag = trajectory_bbox_diagonal_m / max(params.voxel_m, 1e-6)
    est_voxels = int(occupancy_factor * 2.5 * voxels_along_diag**3 / 1000.0) * 1000
    bytes_per_voxel = 20.0
    total_mb = int(est_voxels * bytes_per_voxel / (1024 * 1024))

    use_gpu = (
        choose_fusion_backend(profile).backend == FusionBackendName.OPEN3D_VBG_CUDA
        or profile.nvblox_available
    )
    if use_gpu and profile.gpu.present:
        free = profile.gpu.vram_free_mb
        budget = min(int(free * 0.65), free - 1250)
        if total_mb <= budget:
            return ResourceEstimate(params.voxel_m, est_voxels, total_mb, 0, 1)
    return ResourceEstimate(params.voxel_m, est_voxels, 0, total_mb, 0)


def resolve_effective_voxel(
    requested_voxel_mm: float,
    median_depth_m: float,
    pose_residual_mm: float,
    depth_noise_mm: float,
    profile: MachineProfile,
) -> EffectiveFusionParams:
    """Auto voxel selection heuristic (#42): never finer than the noise floor."""
    if requested_voxel_mm > 0:
        return EffectiveFusionParams(requested_voxel_mm, requested_voxel_mm)

    noise_floor = max(depth_noise_mm, pose_residual_mm) * 1.5
    scale_term = max(6.0, min(12.0, median_depth_m * 4.0))
    effective = max(noise_floor, scale_term)

    small_gpu = profile.gpu.present and profile.gpu.vram_total_mb < 6000
    if small_gpu:
        effective *= 1.25
    return EffectiveFusionParams(requested_voxel_mm, round(effective, 1))
