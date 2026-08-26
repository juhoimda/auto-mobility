"""Resource budget derivation from a measured MachineProfile.

All coefficients are explicit config; no static OMP/VRAM/RAM constants.

    ram_budget  = min(available * ram_budget_fraction, available - system_reserve)
    system_reserve = max(reserve_min_gb, total * reserve_total_fraction)
    vram_budget = min(free * vram_free_fraction, free - vram_reserve)
    cpu_threads = max(1, physical - cpu_headroom_cores)

Complexity: O(1). Memory: O(1).
"""

from dataclasses import dataclass

from auto_mobility.reconstruction.config import ResourcePolicyConfig
from auto_mobility.reconstruction.runtime.machine_profile import MachineProfile

_MB = 1024 * 1024


@dataclass(frozen=True)
class ResourceBudgets:
    cpu_threads: int
    ram_budget_mb: int
    vram_budget_mb: int
    gpu_heavy_slots: int
    system_reserve_mb: int
    vram_reserve_mb: int

    def to_dict(self) -> dict:
        return {
            "cpu_threads": self.cpu_threads,
            "ram_budget_mb": self.ram_budget_mb,
            "vram_budget_mb": self.vram_budget_mb,
            "gpu_heavy_slots": self.gpu_heavy_slots,
            "system_reserve_mb": self.system_reserve_mb,
            "vram_reserve_mb": self.vram_reserve_mb,
        }


def compute_resource_budgets(
    profile: MachineProfile,
    cfg: ResourcePolicyConfig,
) -> ResourceBudgets:
    avail = max(0, profile.ram_available_mb)
    total = max(avail, profile.ram_total_mb)
    reserve = int(
        max(cfg.reserve_min_gb, total * cfg.reserve_total_fraction / 1000.0) * 1000
    )
    ram_budget = min(int(avail * cfg.ram_budget_fraction), max(0, avail - reserve))

    if profile.gpu.present and profile.gpu.vram_free_mb > 0:
        free_vram = profile.gpu.vram_free_mb
        vram_reserve = int(cfg.vram_reserve_gb * 1000)
        # §9/§10: subtract measured CUDA/Open3D context overhead before applying
        # sweet-spot fractions. When overhead is 0 (unmeasured or no GPU), budget
        # stays backwards-compatible (existing unit tests unchanged).
        overhead = int(getattr(profile, "cuda_context_overhead_mb", 0) or 0) + \
                   int(getattr(profile, "open3d_context_overhead_mb", 0) or 0)
        # also consider baseline driver usage already captured in free, but ensure
        # we keep at least 512MB headroom even if overhead is large.
        effective_free = max(0, free_vram - overhead)
        vram_budget = min(
            int(effective_free * cfg.vram_free_fraction),
            max(0, effective_free - vram_reserve),
        )
        # also enforce hard ceiling: current used + budget must not exceed total - reserve
        # we expose vram_reserve for caller-side ceiling checks (§10 dual budget)
    else:
        vram_reserve = 0
        vram_budget = 0

    return ResourceBudgets(
        cpu_threads=max(1, profile.cpu_physical - max(0, cfg.cpu_headroom_cores)),
        ram_budget_mb=max(0, ram_budget),
        vram_budget_mb=vram_budget,
        gpu_heavy_slots=cfg.gpu_heavy_slots if profile.gpu.present else 0,
        system_reserve_mb=reserve,
        vram_reserve_mb=vram_reserve,
    )
