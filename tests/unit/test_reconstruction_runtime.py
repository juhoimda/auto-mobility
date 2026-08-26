"""V2 runtime invariants: budgets, GPU-heavy serialization, OOM preflight,
process isolation (next.md #16..#20, #99, §93)."""

import time
from pathlib import Path

import pytest

from auto_mobility.reconstruction.config import ResourcePolicyConfig
from auto_mobility.reconstruction.runtime import (
    CapacityError,
    GpuInfo,
    JobSpec,
    MachineProfile,
    Scheduler,
    compute_resource_budgets,
    run_monitored_process,
)


def _profile(gpu=False):
    return MachineProfile(
        cpu_physical=4,
        cpu_logical=8,
        ram_total_mb=32000,
        ram_available_mb=16000,
        gpu=GpuInfo(model="X", vram_total_mb=8192, vram_free_mb=6000) if gpu else GpuInfo(),
    )


def test_resource_budget_prevents_oom_job():
    cfg = ResourcePolicyConfig()
    b = compute_resource_budgets(_profile(), cfg)

    avail = 16000
    reserve = max(cfg.reserve_min_gb, 32.0 * cfg.reserve_total_fraction / 1000.0) * 1000
    assert b.ram_budget_mb <= avail * cfg.ram_budget_fraction
    assert b.ram_budget_mb <= avail - reserve
    assert b.system_reserve_mb >= cfg.reserve_min_gb * 1000


def test_vram_budget_min_rule():
    cfg = ResourcePolicyConfig()
    b = compute_resource_budgets(_profile(gpu=True), cfg)
    free_vram = 6000
    expected = min(
        int(free_vram * cfg.vram_free_fraction),
        free_vram - int(cfg.vram_reserve_gb * 1000),
    )
    assert b.vram_budget_mb == expected
    assert b.gpu_heavy_slots == 1


def test_no_gpu_means_zero_vram_slots():
    b = compute_resource_budgets(_profile(gpu=False), ResourcePolicyConfig())
    assert b.vram_budget_mb == 0
    assert b.gpu_heavy_slots == 0


def test_resource_budget_preflight_rejects_impossible_job():
    s = Scheduler(cpu_threads=2, ram_mb=4000, gpu_slots=1, vram_mb=2000)
    with pytest.raises(CapacityError):
        s.submit(lambda: None, JobSpec(name="oom-job", ram_mb=8000))
    with pytest.raises(CapacityError):
        s.submit(lambda: None, JobSpec(name="gpu-hog", gpu_slots=2))


def test_gpu_heavy_jobs_are_serialized():
    state = {"cur": 0, "max": 0}
    lock = __import__("threading").Lock()

    def gpu_job():
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.15)
        with lock:
            state["cur"] -= 1

    s = Scheduler(cpu_threads=8, ram_mb=64000, gpu_slots=1, vram_mb=8000).start()
    try:
        futs = [
            s.submit(gpu_job, JobSpec(name=f"g{i}", gpu_slots=1, vram_mb=100))
            for i in range(4)
        ]
        for f in futs:
            f.result(timeout=10)
    finally:
        s.shutdown()

    assert state["max"] == 1
    assert s.max_concurrent_gpu_jobs == 1


def test_cpu_jobs_parallelize_within_capacity():
    state = {"cur": 0, "max": 0}
    lock = __import__("threading").Lock()

    def cpu_job():
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.1)
        with lock:
            state["cur"] -= 1

    s = Scheduler(cpu_threads=3, ram_mb=64000, gpu_slots=0, vram_mb=0).start()
    try:
        futs = [
            s.submit(cpu_job, JobSpec(name=f"c{i}", cpu_threads=1)) for i in range(6)
        ]
        for f in futs:
            f.result(timeout=10)
    finally:
        s.shutdown()

    assert state["max"] <= 3
    assert state["max"] >= 2


def test_run_monitored_process_success_and_log_redirect(tmp_path):
    log = tmp_path / "candidate.log"
    out = run_monitored_process(
        ["python3", "-c", "print('hello-from-worker')"],
        log_path=log,
        timeout_s=30,
        poll_interval_s=0.05,
    )
    assert out.ok, out.to_dict()
    assert out.returncode == 0
    text = log.read_text(encoding="utf-8")
    assert "hello-from-worker" in text


def test_run_monitored_process_timeout_cleanup(tmp_path):
    log = tmp_path / "slow.log"
    start = time.monotonic()
    out = run_monitored_process(
        ["python3", "-c", "import time; time.sleep(60)"],
        log_path=log,
        timeout_s=2.0,
        poll_interval_s=0.05,
    )
    elapsed = time.monotonic() - start

    from auto_mobility.reconstruction.runtime.process import ProcessStatus

    assert out.status == ProcessStatus.TIMEOUT
    assert elapsed < 15.0


def test_run_monitored_process_rss_budget_kill(tmp_path):
    log = tmp_path / "hog.log"
    out = run_monitored_process(
        ["python3", "-c", "x = bytearray(600 * 1024 * 1024); import time; time.sleep(30)"],
        log_path=log,
        timeout_s=60,
        ram_limit_mb=200,
        poll_interval_s=0.05,
        rss_violation_kill_after=2,
    )

    from auto_mobility.reconstruction.runtime.process import ProcessStatus

    assert out.status in {ProcessStatus.KILLED_RSS, ProcessStatus.OOM_KILLED}
    assert out.elapsed_s < 30
