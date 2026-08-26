"""Typed reconstruction configuration.

Defaults are code-owned; an optional YAML file may override known keys only.
No deep raw-dict access in business logic: everything lands in frozen dataclasses.
"""

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Optional
import yaml

from auto_mobility.reconstruction.model import SCHEMA_VERSION


@dataclass(frozen=True)
class ResourcePolicyConfig:
    ram_budget_fraction: float = 0.55
    reserve_min_gb: float = 6.0
    reserve_total_fraction: float = 0.15
    # Sweet spot for an 8GB-class laptop GPU: enough room for a real TSDF
    # buffer plus its Marching-Cubes assistance structure, while keeping clear
    # of the VRAM ceiling where CUDA OOM / driver instability lives.
    vram_free_fraction: float = 0.55
    vram_reserve_gb: float = 1.5
    cpu_headroom_cores: int = 1
    gpu_heavy_slots: int = 1
    process_poll_interval_s: float = 0.5
    rss_violation_kill_after: int = 3


@dataclass(frozen=True)
class BudgetConfig:
    total_minutes: float = 30.0
    rank01_reserve_fraction: float = 0.30
    pose_exploration_fraction: float = 0.25
    geometry_exploration_fraction: float = 0.25
    optional_improvement_fraction: float = 0.20


@dataclass(frozen=True)
class FrameSelectionConfig:
    blur_downscale: int = 2
    min_depth_valid_ratio: float = 0.30
    max_sync_dt_ms: float = 50.0
    cache_frames: int = 12


@dataclass(frozen=True)
class ReconstructionConfig:
    schema_version: str = SCHEMA_VERSION
    resources: ResourcePolicyConfig = field(default_factory=ResourcePolicyConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    frame_selection: FrameSelectionConfig = field(default_factory=FrameSelectionConfig)


_CONFIG_SECTIONS = {
    "resources": ResourcePolicyConfig,
    "budget": BudgetConfig,
    "frame_selection": FrameSelectionConfig,
}


def _section_from_dict(cls, data: dict):
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items()})


def default_config() -> ReconstructionConfig:
    return ReconstructionConfig()


def load_config(path: Optional[Path]) -> ReconstructionConfig:
    """Load config from YAML; unknown keys are rejected (fail fast, no silent drift)."""
    cfg = default_config()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    updates = {}
    for key, cls in _CONFIG_SECTIONS.items():
        if key in raw:
            updates[key] = _section_from_dict(cls, raw[key])
    unknown_top = set(raw) - set(_CONFIG_SECTIONS) - {"schema_version"}
    if unknown_top:
        raise ValueError(f"unknown config sections: {sorted(unknown_top)}")
    return replace(cfg, **updates)
