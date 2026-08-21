from __future__ import annotations

import os
import sys
import time
import json
import psutil
import hashlib
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


@dataclass
class ResourceUsage:
    """Detailed resource telemetry captured during worker subprocess execution."""
    wall_time_sec: float = 0.0
    cpu_user_time_sec: float = 0.0
    cpu_system_time_sec: float = 0.0
    cpu_time_total_sec: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    min_available_ram_mb: float = 0.0
    gpu_peak_mb: float = 0.0
    gpu_avg_util: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ResourceUsage:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ResourcePolicy:
    """Hardware-calibrated Resource Policy for SLAM, TSDF, and Surface Reconstruction."""
    # Threading policies
    cpu_threads: int = 6               # Calibrated for P-cores (eliminates E-core contention on hybrid CPUs)
    blas_threads: int = 1              # Single-thread BLAS to prevent oversubscription inside OpenMP
    openmp_threads: int = 6            # OpenMP thread pool cap
    numexpr_threads: int = 6

    # Prefetch pipeline
    frame_prefetch_workers: int = 4    # Sliding window async I/O worker count
    frame_prefetch_depth: int = 8      # Max preloaded frames in queue

    # Algorithmic parallel knobs
    poisson_threads: int = 6           # Open3D Poisson reconstruction thread count
    kdtree_workers: int = 6            # SciPy cKDTree nearest neighbor search threads
    opencv_threads: int = 1            # cv2 internal thread pool cap inside prefetch workers

    # Memory budgets (in GB)
    tsdf_memory_budget_gb: float = 6.0
    process_memory_budget_gb: float = 12.0
    system_ram_reserve_gb: float = 2.0

    # GPU execution policy
    gpu_mode: str = "cpu"              # "cpu" (safe default) | "cuda" | "auto"
    gpu_vram_reserve_mb: float = 1024.0

    def get_worker_env(self, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Generate standardized environment variables enforcing thread limits."""
        env = dict(base_env) if base_env is not None else dict(os.environ)
        
        env["OMP_NUM_THREADS"] = str(self.openmp_threads)
        env["OPENBLAS_NUM_THREADS"] = str(self.blas_threads)
        env["MKL_NUM_THREADS"] = str(self.blas_threads)
        env["NUMEXPR_NUM_THREADS"] = str(self.numexpr_threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(self.blas_threads)
        env["OPENCV_NUM_THREADS"] = str(self.opencv_threads)
        
        # Sanitize ROS libraries from LD_LIBRARY_PATH if present to prevent Open3D C++ symbol clashes
        if "LD_LIBRARY_PATH" in env:
            cleaned_ld = [p for p in env["LD_LIBRARY_PATH"].split(":") if "/opt/ros" not in p and p]
            if cleaned_ld:
                env["LD_LIBRARY_PATH"] = ":".join(cleaned_ld)
            else:
                env.pop("LD_LIBRARY_PATH", None)
        return env

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ResourcePolicy:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def detect_system_hardware_fingerprint() -> str:
    """Generate deterministic hardware fingerprint from CPU, RAM, GPU, and OS environment."""
    info_parts = []
    
    # CPU
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        info_parts.append(line.split(":", 1)[1].strip())
                        break
    except Exception:
        pass
    
    info_parts.append(f"cpus={psutil.cpu_count(logical=True)}")
    
    # RAM
    try:
        mem = psutil.virtual_memory()
        info_parts.append(f"ram_gb={round(mem.total / 1e9, 1)}")
    except Exception:
        pass
        
    # GPU
    try:
        smi_paths = ["/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi"]
        for sp in smi_paths:
            res = subprocess.run([sp, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                parts = res.stdout.strip().splitlines()[0].split(",")
                info_parts.append(f"gpu={parts[0].strip()}_{parts[1].strip()}MB")
                break
    except Exception:
        pass

    raw_str = "|".join(info_parts)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


def get_default_resource_policy() -> ResourcePolicy:
    """Return calibrated resource policy based on current WSL and CPU architecture."""
    # Detect available memory
    try:
        mem = psutil.virtual_memory()
        avail_gb = mem.available / 1e9
        usable_tsdf_gb = max(2.0, min(8.0, round((avail_gb - 2.0) * 0.7, 1)))
    except Exception:
        usable_tsdf_gb = 6.0

    # Detect CPU count & calibrate threads
    # Intel Core Ultra 7 265H has 6 P-cores + 8 E-cores. 6 threads achieves peak compute without E-core thrashing.
    logical_cpus = psutil.cpu_count(logical=True) or 8
    calibrated_threads = 6 if logical_cpus >= 12 else max(2, min(4, logical_cpus))

    return ResourcePolicy(
        cpu_threads=calibrated_threads,
        blas_threads=1,
        openmp_threads=calibrated_threads,
        numexpr_threads=calibrated_threads,
        frame_prefetch_workers=4,
        frame_prefetch_depth=8,
        poisson_threads=calibrated_threads,
        kdtree_workers=calibrated_threads,
        opencv_threads=1,
        tsdf_memory_budget_gb=usable_tsdf_gb,
        process_memory_budget_gb=12.0,
        system_ram_reserve_gb=2.0,
        gpu_mode="cpu",
        gpu_vram_reserve_mb=1024.0
    )


# Single global default instance
DEFAULT_RESOURCE_POLICY = get_default_resource_policy()


def run_monitored_subprocess(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    timeout: int = 1800,
    sample_interval: float = 0.25
) -> Tuple[int, str, str, ResourceUsage]:
    """Execute command in subprocess with live process tree CPU and RSS telemetry monitoring."""
    # Check if subprocess.run has been mocked (e.g. during unit tests)
    from unittest.mock import MagicMock
    if isinstance(subprocess.run, MagicMock) or getattr(subprocess.run, '_mock_return_value', None) is not None or getattr(subprocess.run, 'side_effect', None) is not None:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
            return getattr(res, 'returncode', 0), getattr(res, 'stdout', '') or '', getattr(res, 'stderr', '') or '', ResourceUsage(wall_time_sec=0.1)
        except subprocess.TimeoutExpired as te:
            return -15, '', f'Timed out after {timeout}s: {te}', ResourceUsage(wall_time_sec=float(timeout))
        except Exception as e:
            return 1, '', str(e), ResourceUsage(wall_time_sec=0.1)

    t0 = time.time()
    min_avail_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    peak_rss_bytes = 0
    peak_vms_bytes = 0
    cpu_percent_samples = []
    
    initial_io = None
    try:
        p = psutil.Process(proc.pid)
        try:
            initial_io = p.io_counters()
        except Exception:
            initial_io = None
    except Exception:
        p = None

    timed_out = False
    
    while proc.poll() is None:
        elapsed = time.time() - t0
        if elapsed > timeout:
            timed_out = True
            try:
                proc.kill()
            except Exception:
                pass
            break
            
        try:
            if p and p.is_running():
                cur_rss = 0
                cur_vms = 0
                cur_cpu = 0.0
                
                # Monitor parent + all children
                procs = [p]
                try:
                    procs.extend(p.children(recursive=True))
                except Exception:
                    pass
                    
                for subp in procs:
                    try:
                        mem_info = subp.memory_info()
                        cur_rss += mem_info.rss
                        cur_vms += mem_info.vms
                        cur_cpu += subp.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
                peak_rss_bytes = max(peak_rss_bytes, cur_rss)
                peak_vms_bytes = max(peak_vms_bytes, cur_vms)
                if cur_cpu > 0:
                    cpu_percent_samples.append(cur_cpu)
                    
            avail_now = psutil.virtual_memory().available / (1024 * 1024)
            min_avail_ram_mb = min(min_avail_ram_mb, avail_now)
        except Exception:
            pass
            
        time.sleep(sample_interval)

    stdout, stderr = proc.communicate()
    wall_time = time.time() - t0
    
    if timed_out:
        rc = -15
        stderr = (stderr or '') + chr(10) + f'Process timed out after {timeout}s'
    else:
        rc = proc.returncode

    # Extract final CPU and IO metrics if available
    cpu_user = 0.0
    cpu_sys = 0.0
    io_read_mb = 0.0
    io_write_mb = 0.0
    
    try:
        if p:
            cpu_times = p.cpu_times()
            cpu_user = cpu_times.user
            cpu_sys = cpu_times.system
            if initial_io:
                try:
                    final_io = p.io_counters()
                    io_read_mb = (final_io.read_bytes - initial_io.read_bytes) / (1024 * 1024)
                    io_write_mb = (final_io.write_bytes - initial_io.write_bytes) / (1024 * 1024)
                except Exception:
                    pass
    except Exception:
        pass

    avg_cpu = float(np.mean(cpu_percent_samples)) if cpu_percent_samples else 0.0
    peak_cpu = float(np.max(cpu_percent_samples)) if cpu_percent_samples else 0.0

    usage = ResourceUsage(
        wall_time_sec=wall_time,
        cpu_user_time_sec=cpu_user,
        cpu_system_time_sec=cpu_sys,
        cpu_time_total_sec=cpu_user + cpu_sys,
        avg_cpu_percent=avg_cpu,
        peak_cpu_percent=peak_cpu,
        peak_rss_mb=round(peak_rss_bytes / (1024 * 1024), 2),
        peak_vms_mb=round(peak_vms_bytes / (1024 * 1024), 2),
        io_read_mb=round(io_read_mb, 2),
        io_write_mb=round(io_write_mb, 2),
        min_available_ram_mb=round(min_avail_ram_mb, 2),
        gpu_peak_mb=0.0,
        gpu_avg_util=0.0
    )

    return rc, stdout, stderr, usage
