"""Machine hardware/software profiling.

Every resource decision is derived from a measured MachineProfile, never from
hardcoded VRAM/RAM constants. Probing is side-effect free and cheap; optional
heavy backends are detected via importlib spec lookup only (never imported here).

Complexity: O(1) subprocess probes. Memory: O(1).
"""

from dataclasses import dataclass, field, fields, replace
import importlib.util
import json
import hashlib
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from auto_mobility.reconstruction.model import SCHEMA_VERSION


@dataclass(frozen=True)
class GpuInfo:
    model: str = "none"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    driver: str = ""
    cuda_version: str = ""

    @property
    def present(self) -> bool:
        return self.model != "none" and self.vram_total_mb > 0


@dataclass(frozen=True)
class MachineProfile:
    schema_version: str = SCHEMA_VERSION
    cpu_physical: int = 1
    cpu_logical: int = 1
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    gpu: GpuInfo = field(default_factory=GpuInfo)
    is_wsl: bool = False
    filesystem: str = "unknown"
    open3d_version: str = "absent"
    open3d_cuda: bool = False
    cuvslam_available: bool = False
    nvblox_available: bool = False
    python_version: str = ""
    numpy_version: str = ""

    @property
    def software_fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "python": self.python_version,
            "numpy": self.numpy_version,
            "open3d": self.open3d_version,
            "open3d_cuda": self.open3d_cuda,
            "gpu_model": self.gpu.model,
            "driver": self.gpu.driver,
            "cuda": self.gpu.cuda_version,
            "wsl": self.is_wsl,
            "cuvslam": self.cuvslam_available,
            "nvblox": self.nvblox_available,
            "cpu_physical": self.cpu_physical,
            "cpu_logical": self.cpu_logical,
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "cpu_physical": self.cpu_physical,
            "cpu_logical": self.cpu_logical,
            "ram_total_mb": self.ram_total_mb,
            "ram_available_mb": self.ram_available_mb,
            "gpu": {
                "model": self.gpu.model,
                "vram_total_mb": self.gpu.vram_total_mb,
                "vram_free_mb": self.gpu.vram_free_mb,
                "driver": self.gpu.driver,
                "cuda_version": self.gpu.cuda_version,
            },
            "is_wsl": self.is_wsl,
            "filesystem": self.filesystem,
            "open3d_version": self.open3d_version,
            "open3d_cuda": self.open3d_cuda,
            "cuvslam_available": self.cuvslam_available,
            "nvblox_available": self.nvblox_available,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "software_fingerprint": self.software_fingerprint,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        tmp.replace(path)

    @classmethod
    def from_dict(cls, data: dict) -> "MachineProfile":
        gpu = GpuInfo(
            model=data.get("gpu", {}).get("model", "none"),
            vram_total_mb=int(data.get("gpu", {}).get("vram_total_mb", 0)),
            vram_free_mb=int(data.get("gpu", {}).get("vram_free_mb", 0)),
            driver=data.get("gpu", {}).get("driver", ""),
            cuda_version=data.get("gpu", {}).get("cuda_version", ""),
        )
        known = {f.name for f in fields(cls) if f.name != "gpu"}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(gpu=gpu, **kwargs)


def _probe_gpu() -> GpuInfo:
    if shutil.which("nvidia-smi") is None:
        return GpuInfo()
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return GpuInfo()
    if out.returncode != 0 or not out.stdout.strip():
        return GpuInfo()
    first = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 4:
        return GpuInfo()
    try:
        total, free = int(parts[1]), int(parts[2])
    except ValueError:
        return GpuInfo()
    cuda_version = _query_field("cuda_version")
    return GpuInfo(model=parts[0], vram_total_mb=total, vram_free_mb=free,
                   driver=parts[3], cuda_version=cuda_version)


def _query_field(field_name: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field_name}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip().splitlines()[0].strip() if out.returncode == 0 and out.stdout.strip() else ""


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _filesystem_type(path: Path) -> str:
    stat_cmd = shutil.which("stat")
    if stat_cmd is None:
        return "unknown"
    try:
        out = subprocess.run(
            [stat_cmd, "-f", "-c", "%T", str(path)],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return out.stdout.strip() or "unknown"


def probe_machine(workdir: Optional[Path] = None, probe_open3d: bool = True) -> MachineProfile:
    """Measure the actual machine. Never hardcode HW values."""
    import psutil

    vm = psutil.virtual_memory()
    pcpu = psutil.cpu_count(logical=False) or 1
    lcpu = psutil.cpu_count(logical=True) or pcpu

    open3d_version = "absent"
    open3d_cuda = False
    if probe_open3d and _module_available("open3d"):
        try:
            import open3d as o3d
            import open3d.core as o3c

            open3d_version = o3d.__version__
            try:
                open3d_cuda = bool(
                    o3c.cuda.is_available() and o3c.cuda.device_count() > 0
                )
            except Exception:
                open3d_cuda = False
        except Exception:
            open3d_version = "broken"

    proc_version_path = Path("/proc/version")
    is_wsl = False
    if proc_version_path.is_file():
        try:
            is_wsl = "microsoft" in proc_version_path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            pass

    numpy_version = "absent"
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        pass

    return MachineProfile(
        cpu_physical=max(1, pcpu),
        cpu_logical=max(1, lcpu),
        ram_total_mb=int(vm.total // (1024 * 1024)),
        ram_available_mb=int(vm.available // (1024 * 1024)),
        gpu=_probe_gpu(),
        is_wsl=is_wsl,
        filesystem=_filesystem_type(Path(workdir) if workdir else Path.cwd()),
        open3d_version=open3d_version,
        open3d_cuda=open3d_cuda,
        cuvslam_available=_module_available("cuVSLAM") or _module_available("cuvslam"),
        nvblox_available=_module_available("nvblox"),
        python_version=platform.python_version(),
        numpy_version=numpy_version,
    )


def load_or_probe_profile(cache_dir: Path, workdir: Optional[Path] = None,
                          max_age_s: float = 7 * 24 * 3600.0) -> MachineProfile:
    """Reuse cached profile when the software fingerprint matches; else re-probe."""
    cache_file = Path(cache_dir) / "machine_profile.json"
    fresh = probe_machine(workdir=workdir, probe_open3d=False)
    if cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if abs(time.time() - cache_file.stat().st_mtime) <= max_age_s:
                cached = MachineProfile.from_dict(data)
                if cached.software_fingerprint == fresh.software_fingerprint:
                    return replace(
                        cached,
                        ram_available_mb=fresh.ram_available_mb,
                        gpu=replace(cached.gpu, vram_free_mb=fresh.gpu.vram_free_mb),
                    )
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass
    full = probe_machine(workdir=workdir, probe_open3d=True)
    full.save(cache_file)
    return full
