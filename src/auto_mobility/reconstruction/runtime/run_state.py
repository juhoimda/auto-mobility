"""Unclean-run detection & SAFE MODE (§25/§26).

compare 시작 시 run_state.json 을 atomic write. 포함 boot_id, started_at,
hardware_fingerprint, status=RUNNING. 정상 종료 시 COMPLETED.

다음 실행에서 previous RUNNING + different boot_id 이면 previous_host_reset=true
로 SAFE MODE 진입 (더 작은 block target, 더 많은 reserve, sequential).
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def _boot_id() -> str:
    for p in [Path("/proc/sys/kernel/random/boot_id")]:
        try:
            if p.is_file():
                return p.read_text().strip()[:32]
        except Exception:
            pass
    # fallback: uptime-based pseudo
    try:
        import subprocess
        r = subprocess.run(["uptime", "-s"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def write_run_state(out_dir: Path, profile, status: str = "RUNNING") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rs = {
        "boot_id": _boot_id(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_fingerprint": getattr(profile, "hardware_fingerprint", "unknown"),
        "software_fingerprint": getattr(profile, "software_fingerprint", "unknown"),
        "status": status,
    }
    # also include gpu model for diagnostics
    try:
        rs["gpu_model"] = profile.gpu.model
    except Exception:
        pass
    p = out_dir / "run_state.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rs, indent=2))
    tmp.replace(p)
    return p


def detect_previous_host_reset(out_dir: Path, current_boot: str | None = None) -> dict:
    p = Path(out_dir) / "run_state.json"
    if not p.is_file():
        return {"previous_host_reset": False, "reason": "no_previous_state"}
    try:
        data = json.loads(p.read_text())
        prev_status = data.get("status")
        prev_boot = data.get("boot_id", "")
        cur = current_boot or _boot_id()
        if prev_status == "RUNNING" and prev_boot and cur and prev_boot != cur:
            return {"previous_host_reset": True, "prev_boot": prev_boot, "cur_boot": cur,
                    "reason": "RUNNING + boot_id changed"}
        return {"previous_host_reset": False, "prev_boot": prev_boot, "cur_boot": cur,
                "prev_status": prev_status}
    except Exception as e:
        return {"previous_host_reset": False, "reason": f"read_error:{e}"}


def mark_completed(out_dir: Path):
    p = Path(out_dir) / "run_state.json"
    if not p.is_file():
        return
    try:
        data = json.loads(p.read_text())
        data["status"] = "COMPLETED"
        data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(p)
    except Exception:
        pass
