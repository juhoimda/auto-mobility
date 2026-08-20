"""
workers.py — Subprocess Candidate Execution & Crash Isolation.

Provides process-level isolation for heavy native Open3D / CGAL / C++ tasks.
Traps SIGSEGV, OOM, and timeouts without crashing the parent benchmark orchestrator.
"""

import os
import sys
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

from auto_mobility.config import PROJECT_DIR
from auto_mobility.mesh.cgal_surface import is_cgal_available


class WorkerStatus:
    SUCCESS = "SUCCESS"
    FAIL_EXCEPTION = "FAIL_EXCEPTION"
    FAIL_SEGFAULT = "FAIL_SEGFAULT"
    FAIL_OOM = "FAIL_OOM"
    FAIL_TIMEOUT = "FAIL_TIMEOUT"
    SKIPPED_UNAVAILABLE = "SKIPPED_UNAVAILABLE"
    CACHED = "CACHED"


class CandidateLifecycleStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    INVALID = "INVALID"
    FAIL_EXCEPTION = "FAIL_EXCEPTION"
    FAIL_SEGFAULT = "FAIL_SEGFAULT"
    FAIL_OOM = "FAIL_OOM"
    FAIL_TIMEOUT = "FAIL_TIMEOUT"
    SKIPPED_UNAVAILABLE = "SKIPPED_UNAVAILABLE"
    PRUNED = "PRUNED"
    REUSED = "REUSED"


@dataclass
class WorkerResult:
    status: str
    returncode: int
    runtime_sec: float
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status in (WorkerStatus.SUCCESS, WorkerStatus.CACHED, CandidateLifecycleStatus.REUSED)


def _classify_returncode(rc: int, stderr: str = "", stdout: str = "") -> str:
    """Map subprocess returncode and output to a standardized WorkerStatus."""
    if rc == 0:
        return WorkerStatus.SUCCESS
    if rc in (-11, 139):
        return WorkerStatus.FAIL_SEGFAULT
    if rc in (-9, 137):
        return WorkerStatus.FAIL_OOM
    if rc == 2:
        return WorkerStatus.SKIPPED_UNAVAILABLE

    # Inspect stderr/stdout for clues
    combined = (stderr + " " + stdout).lower()
    if "segmentation fault" in combined or "sigsegv" in combined:
        return WorkerStatus.FAIL_SEGFAULT
    if "out of memory" in combined or "killed" in combined or "std::bad_alloc" in combined:
        return WorkerStatus.FAIL_OOM
    if "skipped_unavailable" in combined:
        return WorkerStatus.SKIPPED_UNAVAILABLE

    return WorkerStatus.FAIL_EXCEPTION


def run_tsdf_worker(
    dataset_dir: str,
    traj_file: str,
    mesh_path: Optional[str] = None,
    pcd_path: Optional[str] = None,
    voxel: float = 0.010,
    split_file: Optional[str] = None,
    stride: int = 1,
    quick: bool = False,
    timeout: int = 1800,
    env: Optional[Dict[str, str]] = None
) -> WorkerResult:
    """Run TSDF integration in an isolated subprocess."""
    worker_script = PROJECT_DIR / "src" / "auto_mobility" / "mesh" / "worker.py"
    cmd = [
        sys.executable, "-u", str(worker_script),
        f"--dataset={dataset_dir}",
        f"--trajectory={traj_file}",
        f"--voxel={voxel}",
        f"--stride={stride}",
    ]
    if mesh_path:
        cmd.append(f"--output-mesh={mesh_path}")
    if pcd_path:
        cmd.append(f"--pcd-output={pcd_path}")
    if split_file:
        cmd.append(f"--split={split_file}")
    if quick:
        cmd.append("--no-gpu")

    worker_env = os.environ.copy()
    worker_env.setdefault("OMP_NUM_THREADS", "4")
    worker_env.setdefault("OPENBLAS_NUM_THREADS", "4")
    worker_env.setdefault("MKL_NUM_THREADS", "4")
    worker_env.setdefault("NUMEXPR_NUM_THREADS", "4")
    worker_env.setdefault("VECLIB_MAXIMUM_THREADS", "4")
    if env:
        worker_env.update(env)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=worker_env
        )
        runtime = time.time() - t0
        status = _classify_returncode(proc.returncode, proc.stderr, proc.stdout)
        err_msg = proc.stderr.strip() if proc.returncode != 0 else None
        return WorkerResult(
            status=status,
            returncode=proc.returncode,
            runtime_sec=runtime,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error_message=err_msg
        )
    except subprocess.TimeoutExpired as te:
        runtime = time.time() - t0
        return WorkerResult(
            status=WorkerStatus.FAIL_TIMEOUT,
            returncode=-15,
            runtime_sec=runtime,
            stdout=te.stdout.decode() if te.stdout else "",
            stderr=te.stderr.decode() if te.stderr else "",
            error_message=f"Timed out after {timeout}s"
        )
    except Exception as e:
        runtime = time.time() - t0
        return WorkerResult(
            status=WorkerStatus.FAIL_EXCEPTION,
            returncode=1,
            runtime_sec=runtime,
            error_message=str(e)
        )


def run_surface_worker(
    input_ply: str,
    output_mesh: str,
    method: str = "poisson",
    voxel: float = 0.010,
    depth: int = 8,
    alpha: Optional[float] = None,
    alpha_factor: float = 3.0,
    simplify: float = 0.5,
    no_clean: bool = False,
    no_simplify: bool = False,
    timeout: int = 600,
    env: Optional[Dict[str, str]] = None
) -> WorkerResult:
    """Run Surface Reconstruction (Poisson/BPA/Alpha/CGAL) in an isolated subprocess."""
    m_lower = method.lower()
    if m_lower in ("cgal", "cgal_polygonal"):
        available, msg = is_cgal_available()
        if not available:
            return WorkerResult(
                status=WorkerStatus.SKIPPED_UNAVAILABLE,
                returncode=2,
                runtime_sec=0.0,
                error_message=f"CGAL not available: {msg}"
            )

    worker_script = PROJECT_DIR / "src" / "auto_mobility" / "mesh" / "worker_surface.py"
    cmd = [
        sys.executable, "-u", str(worker_script),
        f"--input-ply={input_ply}",
        f"--output-mesh={output_mesh}",
        f"--method={method}",
        f"--voxel={voxel}",
        f"--depth={depth}",
        f"--alpha-factor={alpha_factor}",
        f"--simplify={simplify}",
        "--benchmark-mode"
    ]
    if alpha is not None:
        cmd.append(f"--alpha={alpha}")
    if no_clean:
        cmd.append("--no-clean")
    if no_simplify:
        cmd.append("--no-simplify")

    worker_env = os.environ.copy()
    worker_env.setdefault("OMP_NUM_THREADS", "4")
    worker_env.setdefault("OPENBLAS_NUM_THREADS", "4")
    worker_env.setdefault("MKL_NUM_THREADS", "4")
    worker_env.setdefault("NUMEXPR_NUM_THREADS", "4")
    worker_env.setdefault("VECLIB_MAXIMUM_THREADS", "4")
    if env:
        worker_env.update(env)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=worker_env
        )
        runtime = time.time() - t0
        status = _classify_returncode(proc.returncode, proc.stderr, proc.stdout)
        err_msg = proc.stderr.strip() if proc.returncode != 0 else None
        return WorkerResult(
            status=status,
            returncode=proc.returncode,
            runtime_sec=runtime,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error_message=err_msg
        )
    except subprocess.TimeoutExpired as te:
        runtime = time.time() - t0
        return WorkerResult(
            status=WorkerStatus.FAIL_TIMEOUT,
            returncode=-15,
            runtime_sec=runtime,
            stdout=te.stdout.decode() if te.stdout else "",
            stderr=te.stderr.decode() if te.stderr else "",
            error_message=f"Timed out after {timeout}s"
        )
    except Exception as e:
        runtime = time.time() - t0
        return WorkerResult(
            status=WorkerStatus.FAIL_EXCEPTION,
            returncode=1,
            runtime_sec=runtime,
            error_message=str(e)
        )
