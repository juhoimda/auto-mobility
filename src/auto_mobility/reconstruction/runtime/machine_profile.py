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
    # §9 CUDA context overhead measured via nvidia-smi delta (0 when unmeasured)
    gpu_baseline_used_mb: int = 0
    cuda_context_overhead_mb: int = 0
    open3d_context_overhead_mb: int = 0

    @property
    def software_fingerprint(self) -> str:
        # Cache-validity fingerprint: stable across probe_open3d=True/False.
        # open3d version/cuda probe must NOT create parent CUDA context (#8),
        # so we either use subprocess-isolated probe or metadata.  The
        # fingerprint therefore relies on importlib.metadata for open3d version
        # when probe_open3d=False, keeping fresh vs cached equality.
        payload = {
            "schema_version": self.schema_version,
            "python": self.python_version,
            "numpy": self.numpy_version,
            "open3d": self.open3d_version,
            # NOTE: open3d_cuda intentionally excluded — the lightweight
            # refresh path must not spawn a CUDA probe subprocess just to
            # compare fingerprints (#13).  The measured flag stays on the
            # cached profile and is refreshed on full probes.
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

    @property
    def hardware_fingerprint(self) -> str:
        """Stable hardware fingerprint excluding volatile available/free fields."""
        payload = {
            "schema_version": self.schema_version,
            "cpu_physical": self.cpu_physical,
            "cpu_logical": self.cpu_logical,
            "gpu_model": self.gpu.model,
            "driver": self.gpu.driver,
            "cuda": self.gpu.cuda_version,
            "wsl": self.is_wsl,
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
            "gpu_baseline_used_mb": int(self.gpu_baseline_used_mb),
            "cuda_context_overhead_mb": int(self.cuda_context_overhead_mb),
            "open3d_context_overhead_mb": int(self.open3d_context_overhead_mb),
            "software_fingerprint": self.software_fingerprint,
            "hardware_fingerprint": self.hardware_fingerprint,
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


def _open3d_version_via_metadata() -> str:
    """Read open3d version without importing open3d (no CUDA context)."""
    try:
        import importlib.metadata as _im
        return _im.version("open3d")
    except Exception:
        return "absent"


def _probe_open3d_cuda_subprocess(timeout_s: float = 15.0) -> tuple[str, bool]:
    """Subprocess-isolated Open3D CUDA probe — never imports in parent (§8)."""
    ver = _open3d_version_via_metadata()
    if ver == "absent":
        return ver, False
    # short-lived child that may create CUDA context, parent stays clean
    code = (
        "import json, sys; "
        "ver='absent'; cuda=False; "
        "try:\n"
        " import open3d as o3d, open3d.core as o3c; ver=o3d.__version__; "
        " cuda=bool(o3c.cuda.is_available() and o3c.cuda.device_count()>0)\n"
        "except Exception: pass\n"
        "print(json.dumps({'ver':ver,'cuda':cuda}))"
    )
    # use python -c with proper newlines via subprocess
    py = (
        "import json\n"
        "ver='absent'\n"
        "cuda=False\n"
        "try:\n"
        " import open3d as o3d\n"
        " import open3d.core as o3c\n"
        " ver=o3d.__version__\n"
        " cuda=bool(o3c.cuda.is_available() and o3c.cuda.device_count()>0)\n"
        "except Exception:\n"
        " pass\n"
        "print(json.dumps({'ver':ver,'cuda':cuda}))\n"
    )
    try:
        out = subprocess.run(
            [shutil.which("python3") or "python3", "-c", py],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout.strip().splitlines()[-1])
            return str(data.get("ver", ver)), bool(data.get("cuda", False))
    except Exception:
        pass
    return ver, False


def _measure_cuda_overhead_mb(timeout_s: float = 15.0) -> tuple[int, int, int]:
    """Measure CUDA context overhead via nvidia-smi delta (§9).

    Returns (baseline_used_mb, cuda_overhead_mb, open3d_overhead_mb).
    Uses nvidia-smi memory.used before/after a short-lived Open3D CUDA probe.
    """
    if shutil.which("nvidia-smi") is None:
        return 0, 0, 0
    def _used():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5.0,
            )
            if r.returncode == 0 and r.stdout.strip():
                return int(float(r.stdout.strip().splitlines()[0].strip()))
        except Exception:
            pass
        return 0
    baseline = _used()
    # baseline 0 when idle is valid; treat None as failure (handled inside _used ->0)
    # we still measure delta; only fail if _used returns 0 and GPU not present
    if baseline is None:
        return 0, 0, 0
    # spawn child that initializes Open3D CUDA context then sleeps briefly
    py = (
        "import time\n"
        "try:\n"
        " import open3d.core as o3c\n"
        " import open3d as o3d\n"
        " dev=o3c.Device('CUDA:0')\n"
        " o3c.cuda.synchronize_device(0)\n"
        " time.sleep(0.8)\n"
        "except Exception:\n"
        " time.sleep(0.5)\n"
    )
    proc = None
    try:
        proc = subprocess.Popen([shutil.which("python3") or "python3", "-c", py])
        time.sleep(0.6)
        peak = _used()
        overhead = max(0, peak - baseline)
        proc.wait(timeout=5.0)
        time.sleep(0.3)
        recovered = _used()
        # if not recovered to baseline, part of overhead may be persistent driver alloc
        open3d_overhead = overhead
        # second child for raw CUDA without Open3D (cupy/torch not available, approximate)
        return baseline, overhead, open3d_overhead
    except Exception:
        return baseline, 0, 0
    finally:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass


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


def probe_machine(workdir: Optional[Path] = None, probe_open3d: bool = True,
                  measure_overhead: bool = False) -> MachineProfile:
    """Measure the actual machine. Never hardcode HW values."""
    import psutil

    vm = psutil.virtual_memory()
    pcpu = psutil.cpu_count(logical=False) or 1
    lcpu = psutil.cpu_count(logical=True) or pcpu

    # §8: parent must never import open3d.cuda directly — use subprocess.
    # _open3d_version_via_metadata is context-free; cuda probe is subprocess
    # and only paid when the caller actually requests a full probe.
    if probe_open3d and _module_available("open3d"):
        open3d_version, open3d_cuda = _probe_open3d_cuda_subprocess()
    elif _module_available("open3d"):
        # lightweight path: metadata version only, no CUDA subprocess (#13)
        open3d_version = _open3d_version_via_metadata()
        open3d_cuda = False
    else:
        open3d_version = "absent"
        open3d_cuda = False

    gpu_baseline = 0
    cuda_overhead = 0
    open3d_overhead = 0
    if measure_overhead:
        gpu_baseline, cuda_overhead, open3d_overhead = _measure_cuda_overhead_mb()

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
        gpu_baseline_used_mb=int(gpu_baseline),
        cuda_context_overhead_mb=int(cuda_overhead),
        open3d_context_overhead_mb=int(open3d_overhead),
    )


def load_or_probe_profile(cache_dir: Path, workdir: Optional[Path] = None,
                          max_age_s: float = 7 * 24 * 3600.0,
                          measure_overhead: bool = False) -> MachineProfile:
    """Reuse cached profile when the software fingerprint matches; else re-probe."""
    cache_file = Path(cache_dir) / "machine_profile.json"
    fresh = probe_machine(workdir=workdir, probe_open3d=False)
    if cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if abs(time.time() - cache_file.stat().st_mtime) <= max_age_s:
                cached = MachineProfile.from_dict(data)
                # §8 bug fixed: software_fingerprint now stable across probe_open3d modes
                # because both paths read open3d version via metadata/subprocess.
                if cached.software_fingerprint == fresh.software_fingerprint:
                    return replace(
                        cached,
                        ram_available_mb=fresh.ram_available_mb,
                        gpu=replace(cached.gpu, vram_free_mb=fresh.gpu.vram_free_mb),
                    )
                # Fallback: also accept hardware_fingerprint match when only open3d
                # cuda flag drifted; treat as same hardware generation.
                if cached.hardware_fingerprint == fresh.hardware_fingerprint:
                    # refresh open3d fields but keep budget-relevant GPU free
                    return replace(
                        cached,
                        ram_available_mb=fresh.ram_available_mb,
                        gpu=replace(cached.gpu, vram_free_mb=fresh.gpu.vram_free_mb),
                        open3d_version=fresh.open3d_version,
                        open3d_cuda=fresh.open3d_cuda,
                    )
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            pass
    full = probe_machine(workdir=workdir, probe_open3d=True,
                         measure_overhead=measure_overhead)
    full.save(cache_file)
    return full
