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
    FAIL_CRASH = "FAIL_CRASH"
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
    FAIL_CRASH = "FAIL_CRASH"
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
    depth_max: float = 3.0,
    trunc_mult: float = 4.0,
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
        f"--depth-max={depth_max}",
        f"--trunc-mult={trunc_mult}",
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
    worker_env.setdefault("OMP_NUM_THREADS", "8")
    worker_env.setdefault("OPENBLAS_NUM_THREADS", "8")
    worker_env.setdefault("MKL_NUM_THREADS", "8")
    worker_env.setdefault("NUMEXPR_NUM_THREADS", "8")
    worker_env.setdefault("VECLIB_MAXIMUM_THREADS", "8")
    if "LD_LIBRARY_PATH" in worker_env:
        cleaned_ld = [p for p in worker_env["LD_LIBRARY_PATH"].split(":") if "/opt/ros" not in p and p]
        if cleaned_ld:
            worker_env["LD_LIBRARY_PATH"] = ":".join(cleaned_ld)
        else:
            worker_env.pop("LD_LIBRARY_PATH", None)
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
    simplify: float = 0.0,
    no_clean: bool = False,
    no_simplify: bool = True,
    timeout: int = 600,
    env: Optional[Dict[str, str]] = None
) -> WorkerResult:
    """Run Surface Reconstruction (Poisson/BPA/Alpha/CGAL) in an isolated subprocess."""
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
    if no_simplify or simplify <= 0.0:
        cmd.append("--no-simplify")

    worker_env = os.environ.copy()
    worker_env.setdefault("OMP_NUM_THREADS", "8")
    worker_env.setdefault("OPENBLAS_NUM_THREADS", "8")
    worker_env.setdefault("MKL_NUM_THREADS", "8")
    worker_env.setdefault("NUMEXPR_NUM_THREADS", "8")
    worker_env.setdefault("VECLIB_MAXIMUM_THREADS", "8")
    if "LD_LIBRARY_PATH" in worker_env:
        cleaned_ld = [p for p in worker_env["LD_LIBRARY_PATH"].split(":") if "/opt/ros" not in p and p]
        if cleaned_ld:
            worker_env["LD_LIBRARY_PATH"] = ":".join(cleaned_ld)
        else:
            worker_env.pop("LD_LIBRARY_PATH", None)
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


def run_direct_fusion_worker(
    dataset_dir: str,
    traj_file: str,
    pcd_path: str,
    voxel: float = 0.010,
    depth_min: float = 0.3,
    depth_max: float = 3.0,
    stride: int = 1,
    split_file: Optional[str] = None,
    no_color: bool = False,
    orient: str = "centroid",
    timeout: int = 1800,
    env: Optional[Dict[str, str]] = None
) -> WorkerResult:
    """Run Direct Point Cloud Fusion in an isolated subprocess."""
    worker_script = PROJECT_DIR / "src" / "auto_mobility" / "mesh" / "direct_fusion.py"
    cmd = [
        sys.executable, "-u", str(worker_script),
        dataset_dir,
        traj_file,
        f"--output={pcd_path}",
        f"--voxel={voxel}",
        f"--depth-min={depth_min}",
        f"--depth-max={depth_max}",
        f"--stride={stride}",
        f"--orient={orient}"
    ]
    if split_file:
        cmd.append(f"--split={split_file}")
    if no_color:
        cmd.append("--no-color")

    worker_env = os.environ.copy()
    worker_env.setdefault("OMP_NUM_THREADS", "8")
    worker_env.setdefault("OPENBLAS_NUM_THREADS", "8")
    worker_env.setdefault("MKL_NUM_THREADS", "8")
    worker_env.setdefault("NUMEXPR_NUM_THREADS", "8")
    worker_env.setdefault("VECLIB_MAXIMUM_THREADS", "8")
    if "LD_LIBRARY_PATH" in worker_env:
        cleaned_ld = [p for p in worker_env["LD_LIBRARY_PATH"].split(":") if "/opt/ros" not in p and p]
        if cleaned_ld:
            worker_env["LD_LIBRARY_PATH"] = ":".join(cleaned_ld)
        else:
            worker_env.pop("LD_LIBRARY_PATH", None)
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

