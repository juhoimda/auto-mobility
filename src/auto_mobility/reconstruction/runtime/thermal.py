"""HW safety guards: thermal/power gate before GPU-heavy submissions.

Priority one is that a run must NEVER destabilize the host (laptop forced
shutdown / WSL kill). Before every fusion submission we check GPU temperature
and wait for cooldown when it is hot. All limits are deliberately conservative;
a slow run is always preferable to a host crash.

Complexity: O(1) subprocess probes per call.
"""

from __future__ import annotations

import shutil
import subprocess
import time

# Laptop Blackwell throttles well above these values; they gate sustained
# load only, leaving margin to throttle/shutdown territory.
MAX_GPU_TEMP_C = 82.0
RESUME_GPU_TEMP_C = 74.0
COOLDOWN_POLL_S = 15.0
MAX_WAIT_S = 900.0


def _query(fields: list) -> list | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def gpu_thermals() -> dict:
    vals = _query(["temperature.gpu", "power.draw"])
    if vals is None:
        return {"available": False}
    return {"available": True, "temp_c": vals[0], "power_w": vals[1]}


def power_source() -> str:
    """'ac' | 'battery' | 'unknown' — L0 barrier input.

    Sustained combined CPU+GPU draw on battery is the prime suspect for host
    forced-poweroffs; heavy GPU stages refuse to start while discharging.
    """
    from pathlib import Path

    supplies = Path("/sys/class/power_supply")
    if not supplies.is_dir():
        return "unknown"
    for entry in sorted(supplies.iterdir()):
        online = entry / "online"
        if online.is_file():
            try:
                return "ac" if online.read_text().strip() == "1" else "battery"
            except OSError:
                continue
        status = entry / "status"
        if status.is_file():
            try:
                s = status.read_text().strip().lower()
                if s in ("discharging",):
                    return "battery"
                if s in ("charging", "full", "not charging", "ac"):
                    return "ac"
            except OSError:
                continue
    return "unknown"


def wait_for_thermal_headroom(
    max_temp_c: float = MAX_GPU_TEMP_C,
    resume_temp_c: float = RESUME_GPU_TEMP_C,
    max_wait_s: float = MAX_WAIT_S,
) -> dict:
    """Block until GPU temperature allows another heavy submission.

    Returns evidence dict for the decision trace; never raises. If sensors are
    unavailable or the wait times out, proceed (caller keeps its own resource
    caps) but record it.
    """
    t0 = time.monotonic()
    waited_s = 0.0
    peak_seen = 0.0
    announced = False
    while True:
        th = gpu_thermals()
        if not th.get("available"):
            return {"waited_s": 0.0, "reason": "no_sensor"}
        temp = float(th["temp_c"])
        peak_seen = max(peak_seen, temp)
        if temp <= max_temp_c:
            return {"waited_s": round(waited_s, 1), "peak_temp_c": peak_seen,
                    "temp_c": temp, "power_w": th.get("power_w")}
        if time.monotonic() - t0 > max_wait_s:
            print(f"[thermal] cooldown timeout (peak {peak_seen:.0f}C); proceeding")
            return {"waited_s": round(waited_s, 1), "peak_temp_c": peak_seen,
                    "reason": "cooldown_timeout"}
        if not announced:
            print(f"[thermal] GPU hot ({temp:.0f}C >= {max_temp_c:.0f}C); "
                  f"cooling down until <={resume_temp_c:.0f}C")
            announced = True
        time.sleep(COOLDOWN_POLL_S)
        waited_s = time.monotonic() - t0
