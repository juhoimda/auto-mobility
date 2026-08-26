"""Resource-token scheduler with the GPU-heavy serialization invariant.

Jobs declare their resource tokens (cpu_threads, ram_mb, gpu_slots, vram_mb,
timeout); the scheduler admits a job only when all tokens fit remaining
capacity. In particular gpu_slots >= 1 jobs are serialized because capacity is
1 (MachineProfile-derived), preventing cuVSLAM + GPU-TSFDF-style contention.

A job that can never fit declared capacity is rejected at submit time
(preflight reject instead of mid-run OOM).

Complexity: O(J log J) scheduling for J jobs. Memory: O(J).
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
import traceback
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class JobSpec:
    name: str
    cpu_threads: float = 1.0
    ram_mb: int = 0
    gpu_slots: int = 0
    vram_mb: int = 0
    io_weight: float = 0.0
    timeout_s: Optional[float] = None
    priority: int = 0


@dataclass
class _QueuedJob:
    spec: JobSpec
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict
    future: Future = field(default_factory=Future)


class CapacityError(ValueError):
    pass


class Scheduler:
    """Deterministic admission-control scheduler running jobs in worker threads."""

    def __init__(self, cpu_threads: int, ram_mb: int, gpu_slots: int, vram_mb: int):
        if cpu_threads < 1 or ram_mb < 0 or gpu_slots < 0 or vram_mb < 0:
            raise ValueError("invalid capacities")
        self._cap_cpu = float(cpu_threads)
        self._cap_ram = ram_mb
        self._cap_gpu = gpu_slots
        self._cap_vram = vram_mb
        self._used_cpu = 0.0
        self._used_ram = 0
        self._used_gpu = 0
        self._used_vram = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: deque = deque()
        self._running = 0
        self._shutdown = False
        self._dispatch_thread: Optional[threading.Thread] = None
        self.max_concurrent_gpu_jobs = 0

    def start(self) -> "Scheduler":
        if self._dispatch_thread is None:
            self._dispatch_thread = threading.Thread(
                target=self._dispatch_loop, name="recon-scheduler", daemon=True
            )
            self._dispatch_thread.start()
        return self

    def submit(self, fn: Callable[..., Any], spec: JobSpec, /, *args, **kwargs) -> Future:
        self._validate_fits_capacity(spec)
        job = _QueuedJob(spec=spec, fn=fn, args=args, kwargs=kwargs)
        with self._cond:
            if self._shutdown:
                raise RuntimeError("scheduler is shut down")
            self._pending.append(job)
            self._cond.notify_all()
        return job.future

    def _validate_fits_capacity(self, spec: JobSpec) -> None:
        if spec.cpu_threads > self._cap_cpu:
            raise CapacityError(
                f"{spec.name}: needs {spec.cpu_threads} cpu threads > capacity {self._cap_cpu}"
            )
        if spec.ram_mb > self._cap_ram:
            raise CapacityError(
                f"{spec.name}: needs {spec.ram_mb} MB RAM > budget {self._cap_ram} MB"
            )
        if spec.gpu_slots > self._cap_gpu:
            raise CapacityError(
                f"{spec.name}: needs {spec.gpu_slots} gpu slots > capacity {self._cap_gpu}"
            )
        if spec.vram_mb > self._cap_vram:
            raise CapacityError(
                f"{spec.name}: needs {spec.vram_mb} MB VRAM > budget {self._cap_vram} MB"
            )

    def _fits(self, spec: JobSpec) -> bool:
        return (
            self._used_cpu + spec.cpu_threads <= self._cap_cpu + 1e-9
            and self._used_ram + spec.ram_mb <= self._cap_ram
            and self._used_gpu + spec.gpu_slots <= self._cap_gpu
            and self._used_vram + spec.vram_mb <= self._cap_vram
        )

    def _dispatch_loop(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._shutdown:
                    self._cond.wait(timeout=0.2)
                if self._shutdown and not self._pending:
                    return
                selected_idx = None
                for idx, job in enumerate(self._pending):
                    if self._fits(job.spec):
                        selected_idx = idx
                        break
                if selected_idx is None:
                    continue
                job = self._pending[selected_idx]
                del self._pending[selected_idx]
                s = job.spec
                self._used_cpu += s.cpu_threads
                self._used_ram += s.ram_mb
                self._used_gpu += s.gpu_slots
                self._used_vram += s.vram_mb
                self._running += 1
                self.max_concurrent_gpu_jobs = max(
                    self.max_concurrent_gpu_jobs,
                    1 if self._used_gpu > 0 else 0,
                )
            worker = threading.Thread(target=self._run_job, args=(job,), daemon=True)
            worker.start()

    def _run_job(self, job: _QueuedJob) -> None:
        s = job.spec
        try:
            result = job.fn(*job.args, **job.kwargs)
            if not job.future.set_running_or_notify_cancel():
                pass
            job.future.set_result(result)
        except BaseException as exc:
            job.future.set_exception(exc)
            tb = traceback.format_exc()
            print(f"[scheduler] job {s.name} failed: {exc}\n{tb}")
        finally:
            with self._cond:
                self._used_cpu -= s.cpu_threads
                self._used_ram -= s.ram_mb
                self._used_gpu -= s.gpu_slots
                self._used_vram -= s.vram_mb
                self._running -= 1
                self._cond.notify_all()

    def wait_all(self, poll_interval_s: float = 0.05) -> None:
        import time

        while True:
            with self._cond:
                busy = bool(self._pending) or self._running > 0
            if not busy:
                return
            time.sleep(poll_interval_s)

    def shutdown(self) -> None:
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=5.0)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def running_count(self) -> int:
        with self._lock:
            return self._running
