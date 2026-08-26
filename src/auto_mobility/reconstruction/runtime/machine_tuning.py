"""Machine-specific tuning cache (§27).

machine_tuning.json stores per-hardware sweet-spot calibration results
so global magic constants can be replaced by Lenovo-measured values.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MachineTuning:
    hardware_fingerprint: str = ""
    software_fingerprint: str = ""
    fusion_cpu_threads: int = 4
    rtab_cpu_threads: int = 6
    cuvslam_cpu_threads: int = 3
    safe_vram_delta_mb: int = 0
    hard_vram_ceiling_mb: int = 0
    measured_context_mb: int = 0
    safe_active_blocks: int = 0
    depth_max_m: float = 4.0
    max_continuous_gpu_s: float = 0.0
    cooldown_s: float = 8.0
    last_sustained_test_status: str = "unknown"
    updated_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MachineTuning":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "MachineTuning | None":
        if not path.is_file():
            return None
        try:
            return MachineTuning.from_dict(json.loads(path.read_text()))
        except Exception:
            return None
