"""Subprocess runner with log-file redirect and resource watchdog.

stdout/stderr stream directly into a candidate.log file (never PIPE buffers).
The parent samples only process-tree state (RSS, CPU, elapsed) and kills the
whole session on timeout or sustained budget violation.

Complexity: O(samples) monitoring; sampling is O(processes). Memory: O(1).
"""

from dataclasses import dataclass
import os
import signal
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Optional


class ProcessStatus(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    KILLED_RSS = "killed_rss_budget"
    SEGFAULT = "segfault"
    OOM_KILLED = "oom_killed"
    FAILED = "failed"


@dataclass(frozen=True)
class ProcessOutcome:
    status: ProcessStatus
    returncode: Optional[int]
    elapsed_s: float
    peak_rss_mb: float
    avg_cpu_percent: float

    @property
    def ok(self) -> bool:
        return self.status == ProcessStatus.OK

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "returncode": self.returncode,
            "elapsed_s": round(self.elapsed_s, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "avg_cpu_percent": round(self.avg_cpu_percent, 1),
        }


def _tree_stats(proc: "psutil.Process") -> tuple[float, float]:
    import psutil

    procs = [proc]
    try:
        procs.extend(proc.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    rss_mb = 0.0
    cpu_total = 0.0
    for p in procs:
        try:
            rss_mb += p.memory_info().rss / (1024 * 1024)
            cpu_total += p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return rss_mb, cpu_total


def _kill_session(popen_obj: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(popen_obj.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            popen_obj.kill()
        except OSError:
            pass


def _gpu_sample() -> tuple[float, float, float] | None:
    """(vram_used_mb, temp_c, power_w) or None when unreadable."""
    import shutil
    import subprocess as sp

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = sp.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0)
    except (OSError, sp.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        vals = [float(p.strip()) for p in out.stdout.splitlines()[0].split(",")]
        return vals[0], vals[1], vals[2]
    except (ValueError, IndexError):
        return None


def vram_barrier_breach(used_mb: float, baseline_mb: Optional[float],
                        limit_mb: Optional[float],
                        hard_ceiling_mb: Optional[float]):
    """§10/§11 VRAM barrier semantics (pure; unit-testable).

    Returns (kill_reason_or_None, effective_baseline).
    - hard_ceiling_mb compares ABSOLUTE memory.used against the global
      ceiling (baseline + delta <= total - reserve invariant);
    - limit_mb is an INCREMENTAL job budget and must compare
      (used - baseline), never absolute usage.
    When baseline is None it is lazily established from the first sample so
    the incremental budget is never misused as an absolute barrier.
    """
    if hard_ceiling_mb is not None and used_mb > hard_ceiling_mb:
        return (f"hard_ceiling {used_mb:.0f}MB > {hard_ceiling_mb}MB",
                baseline_mb)
    if limit_mb is not None:
        if baseline_mb is None:
            baseline_mb = used_mb
        delta = used_mb - baseline_mb
        if delta > limit_mb:
            return (f"job_delta {delta:.0f}MB > budget {limit_mb}MB "
                    f"(used {used_mb:.0f} base {baseline_mb:.0f})", baseline_mb)
    return (None, baseline_mb)


def run_monitored_process(
    cmd: list,
    log_path: Path,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
    timeout_s: float = 3600.0,
    ram_limit_mb: Optional[int] = None,
    poll_interval_s: float = 0.5,
    rss_violation_kill_after: int = 3,
    gpu_limits: Optional[dict] = None,
) -> ProcessOutcome:
    """Run cmd as a detached session; logs go straight to log_path.

    gpu_limits (HW barrier, L2): {"vram_mb", "temp_c", "power_w",
       "hard_ceiling_mb", "baseline_used_mb"}  §10 dual budget:
       - incremental job budget: vram_mb (job delta must stay < limit)
       - global hard ceiling: hard_ceiling_mb = total_vram - hard_reserve
       Both are checked; either breach kills the session.
       Baseline is sampled at job start so delta = current_used - baseline.
    """
    import psutil

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    peak_rss = 0.0
    cpu_samples = []
    violations = 0
    timed_out = False
    budget_killed = False
    gpu_killed_reason = ""
    gpu_poll_tick = 0
    gpu_temp_strikes = 0
    gpu_power_strikes = 0

    # §10 sample baseline at job start for incremental budget
    baseline_vram = None
    if gpu_limits and (gpu_limits.get("vram_mb") or gpu_limits.get("hard_ceiling_mb")):
        samp = _gpu_sample()
        if samp is not None:
            baseline_vram = samp[0]
        else:
            baseline_vram = gpu_limits.get("baseline_used_mb")
    with open(log_path, "w", encoding="utf-8") as log_fh:
        popen_obj = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=str(cwd) if cwd else None,
            start_new_session=True,
        )
        psutil_proc = None
        try:
            psutil_proc = psutil.Process(popen_obj.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        try:
            while True:
                rc = popen_obj.poll()
                if rc is not None:
                    break
                if psutil_proc is not None:
                    rss_mb, cpu_pct = _tree_stats(psutil_proc)
                    peak_rss = max(peak_rss, rss_mb)
                    cpu_samples.append(cpu_pct)
                    if ram_limit_mb is not None and rss_mb > ram_limit_mb:
                        violations += 1
                        if violations >= max(1, rss_violation_kill_after):
                            _kill_session(popen_obj)
                            budget_killed = True
                            break
                    else:
                        violations = 0

                # ---- HW barrier (L2): bounded-frequency GPU sampling (§10/§12) ----
                # This is LAST defense; primary is predict/preflight + scheduler admission.
                # GPU memory can spike GBs between polls, so never rely solely on watchdog.
                if gpu_limits:
                    gpu_poll_tick += 1
                    if gpu_poll_tick % 4 == 0:  # ~2s at default interval
                        s = _gpu_sample()
                        if s is not None:
                            vram_mb, temp_c, power_w = s
                            limit_vram = gpu_limits.get("vram_mb")
                            hard_ceiling = gpu_limits.get("hard_ceiling_mb")
                            limit_temp = gpu_limits.get("temp_c")
                            limit_power = gpu_limits.get("power_w")
                            # §10/§11 dual check via shared pure helper:
                            # hard ceiling = absolute, vram budget = job delta.
                            # No baseline -> lazily established from first
                            # sample (never absolute-vs-incremental mismatch).
                            breached = False
                            reason, baseline_vram = vram_barrier_breach(
                                vram_mb, baseline_vram, limit_vram,
                                hard_ceiling)
                            if reason:
                                gpu_killed_reason = reason
                                breached = True
                            if not breached and limit_temp and temp_c >= limit_temp:
                                gpu_temp_strikes += 1
                                if gpu_temp_strikes >= 3:
                                    gpu_killed_reason = (
                                        f"temp {temp_c:.0f}C >= barrier {limit_temp}C x3")
                            elif not breached:
                                gpu_temp_strikes = 0
                            if not gpu_killed_reason and not breached and limit_power and power_w > 0 \
                                    and power_w >= limit_power:
                                gpu_power_strikes += 1
                                if gpu_power_strikes >= 5:
                                    gpu_killed_reason = (
                                        f"power {power_w:.0f}W >= barrier "
                                        f"{limit_power}W x5")
                            elif not gpu_killed_reason and not breached:
                                gpu_power_strikes = 0
                            if gpu_killed_reason:
                                print(f"[hw_barrier] killing session: "
                                      f"{gpu_killed_reason}", flush=True)
                                _kill_session(popen_obj)
                                break

                if time.monotonic() - start > timeout_s:
                    _kill_session(popen_obj)
                    timed_out = True
                    break
                time.sleep(poll_interval_s)
        finally:
            if psutil_proc is not None:
                try:
                    psutil_proc.wait(timeout=5.0)
                except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                    pass
            try:
                popen_obj.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                _kill_session(popen_obj)
                popen_obj.wait(timeout=10.0)

    elapsed = time.monotonic() - start
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    rc = popen_obj.returncode

    if timed_out:
        status = ProcessStatus.TIMEOUT
    elif budget_killed:
        status = ProcessStatus.KILLED_RSS
    elif gpu_killed_reason:
        status = ProcessStatus.KILLED_RSS  # barrier kill: resource breach
        log_path.with_suffix(".barrier").write_text(gpu_killed_reason + "\n")
    elif rc == 0:
        status = ProcessStatus.OK
    elif rc == -signal.SIGKILL:
        status = ProcessStatus.OOM_KILLED
    elif rc == -signal.SIGSEGV:
        status = ProcessStatus.SEGFAULT
    else:
        status = ProcessStatus.FAILED

    return ProcessOutcome(
        status=status,
        returncode=rc,
        elapsed_s=elapsed,
        peak_rss_mb=peak_rss,
        avg_cpu_percent=avg_cpu,
    )
