"""Fusion layer: backend selection, resource estimates, Open3D VBG."""

from auto_mobility.reconstruction.fusion.backend import (
    CapabilityResult,
    EffectiveFusionParams,
    FusionBackendName,
    choose_fusion_backend,
    estimate_resources,
    resolve_effective_voxel,
)
from auto_mobility.reconstruction.fusion.open3d_vbg import (
    FusionInput,
    FusionOutput,
    integrate_frames,
)

__all__ = [
    "CapabilityResult",
    "EffectiveFusionParams",
    "FusionBackendName",
    "choose_fusion_backend",
    "estimate_resources",
    "resolve_effective_voxel",
    "FusionInput",
    "FusionOutput",
    "integrate_frames",
]
