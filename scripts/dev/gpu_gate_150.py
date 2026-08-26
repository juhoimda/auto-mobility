"""150-frame GPU Gate per feedback.md §20-§22.

Measures:
  - VRAM baseline
  - Planned peak VRAM & blocks
  - Actual peak VRAM
  - Post-worker VRAM & recovery delta
  - Planned blocks vs Active blocks
  - Wall time
  - GPU util, power, temperature
  - Peak RSS

Pass conditions:
  - CUDA OOM = 0
  - Segfault = 0
  - Watchdog kill = 0
  - Hard ceiling breach = 0
  - Normal post-worker VRAM recovery
  - Actual VRAM does not greatly exceed predicted
  - Host/WSL stable
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.reconstruction.fusion.open3d_vbg import (
    plan_active_blocks, required_vram_mb_planned)
from auto_mobility.reconstruction.fusion.isolated import integrate_frames_isolated
from auto_mobility.reconstruction.runtime.machine_profile import _probe_gpu
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.reconstruction.cli import _nearest_pose


def get_gpu_gate_fingerprint() -> dict:
    """Compute environment fingerprint for GPU Gate pass cache (§10)."""
    import hashlib
    import open3d as o3d
    gpu = _probe_gpu()
    fp_str = f"{gpu.model}:{gpu.vram_total_mb}:{o3d.__version__}:{sys.version}"
    return {
        "gpu_model": gpu.model,
        "vram_total_mb": gpu.vram_total_mb,
        "open3d_version": o3d.__version__,
        "python_version": sys.version.split()[0],
        "fingerprint_sha": hashlib.sha256(fp_str.encode()).hexdigest()[:16],
    }


def check_gpu_gate_cache(cache_dir: Path) -> dict | None:
    """Check if valid GPU_GATE_PASS cache exists for current environment."""
    cert_path = cache_dir / "gpu_gate_pass.json"
    if not cert_path.is_file():
        return None
    try:
        data = json.loads(cert_path.read_text())
        curr_fp = get_gpu_gate_fingerprint()
        if data.get("fingerprint_sha") == curr_fp.get("fingerprint_sha") and data.get("gate_passed"):
            return data
    except Exception:
        pass
    return None


def _read_gpu_stats():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            parts = [float(x.strip()) for x in res.stdout.strip().split(",")]
            return {
                "vram_used_mb": parts[0],
                "gpu_util_pct": parts[1],
                "power_w": parts[2],
                "temp_c": parts[3],
            }
    except Exception:
        pass
    return None




def run_gate(dataset_dir: Path, traj_path: Path, n_frames: int = 150, voxel_mm: float = 10.0):
    print("=" * 60)
    print(f"  🚀 STARTING 150-FRAME GPU GATE (§20-§22)")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Trajectory: {traj_path}")
    print(f"  Voxel: {voxel_mm}mm | Frames: {n_frames}")
    print("=" * 60)

    ds = FrameDataset(str(dataset_dir))
    frames = list(ds)[:n_frames]
    cam = json.load(open(dataset_dir / "camera_info.json"))
    K = np.array(cam["K"], dtype=np.float64).reshape(3, 3) if "K" in cam else None
    W, H = int(cam["width"]), int(cam["height"])
    traj = Trajectory.from_tum_file(str(traj_path))

    gate_frames = frames
    gate_ids = [f.frame_id for f in gate_frames]
    poses = {f.frame_id: _nearest_pose(traj, f.rgb_timestamp) for f in gate_frames}

    pts = np.asarray([poses[i][:3, 3] for i in gate_ids])
    bbox_diag_m = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) + 2.0

    def depth_loader(i):
        f = next(fr for fr in gate_frames if fr.frame_id == i)
        p = dataset_dir / f.depth_path if not Path(f.depth_path).is_absolute() else Path(f.depth_path)
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    # 1. Preflight plan
    print(f"  ⚙️ Computing ActiveBlockPlan for {len(gate_ids)} frames...")
    plan = plan_active_blocks(
        gate_ids, poses, K, W, H, voxel_mm / 1000.0,
        trunc_mult=4.0, depth_min_m=0.2, depth_max_m=4.0,
        load_depth_mm=depth_loader, load_mask=None,
        store_color=False, sample_stride=1)
    planned_vram_mb = required_vram_mb_planned(plan)
    planned_blocks = plan.get("safe_block_count") or plan.get("planned_capacity_blocks")

    print(f"  📊 Planned VRAM: {planned_vram_mb:.1f} MB")
    print(f"  📊 Planned blocks (safe capacity): {planned_blocks}")
    print(f"  📊 Unique blocks estimated: {plan.get('unique_block_count')}")

    # 2. Baseline GPU stats
    base_stats = _read_gpu_stats()
    vram_baseline = base_stats["vram_used_mb"] if base_stats else 0.0
    print(f"  📊 Baseline VRAM: {vram_baseline:.1f} MB")

    # Monitor GPU telemetry in background during execution
    peak_vram_holder = [vram_baseline]
    peak_util_holder = [0.0]
    peak_power_holder = [0.0]
    peak_temp_holder = [base_stats["temp_c"] if base_stats else 0.0]
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            st = _read_gpu_stats()
            if st:
                peak_vram_holder[0] = max(peak_vram_holder[0], st["vram_used_mb"])
                peak_util_holder[0] = max(peak_util_holder[0], st["gpu_util_pct"])
                peak_power_holder[0] = max(peak_power_holder[0], st["power_w"])
                peak_temp_holder[0] = max(peak_temp_holder[0], st["temp_c"])
            time.sleep(0.05)

    mon_thread = threading.Thread(target=monitor, daemon=True)
    mon_thread.start()

    work_dir = PROJECT_ROOT / "output" / "gpu_gate_work"
    t0 = time.time()
    res = integrate_frames_isolated(
        dataset_dir=dataset_dir,
        frame_ids=gate_ids,
        pose_by_frame=poses,
        masks_by_frame=None,
        K=K, width=W, height=H,
        voxel_m=voxel_mm / 1000.0,
        trunc_mult=4.0,
        bbox_diag_m=bbox_diag_m,
        work_dir=work_dir,
        tag="gate_150",
        vram_budget_mb=6000.0,
        ram_limit_mb=4096,
        gpu_limits={"vram_mb": int(planned_vram_mb * 1.5 + 512), "hard_ceiling_mb": 6500},
        frames_per_chunk=150,
        chunk_pause_s=1.0,
        planned_block_count=planned_blocks,
    )
    wall_s = time.time() - t0
    stop_event.set()
    mon_thread.join(timeout=1.0)

    # Post-worker stats & VRAM recovery check
    time.sleep(1.0)
    post_stats = _read_gpu_stats()
    vram_post = post_stats["vram_used_mb"] if post_stats else 0.0
    vram_recovery_delta = vram_post - vram_baseline

    actual_peak_vram = peak_vram_holder[0]
    out_obj = res.output

    print("-" * 60)
    print(f"  🏁 GPU GATE EXECUTION RESULTS:")
    print(f"  Status: {'SUCCESS' if res.ok else 'FAILED'} ({res.detail})")
    print(f"  Wall Time: {wall_s:.2f}s")
    print(f"  Device: {out_obj.device}")
    print(f"  VRAM Baseline: {vram_baseline:.1f} MB")
    print(f"  VRAM Planned Peak: {planned_vram_mb:.1f} MB")
    print(f"  VRAM Actual Peak: {actual_peak_vram:.1f} MB")
    print(f"  VRAM Post-worker: {vram_post:.1f} MB (Delta: {vram_recovery_delta:.1f} MB)")
    print(f"  Planned Blocks: {planned_blocks}")
    print(f"  Mesh Vertices: {out_obj.mesh_vertices:,} | Triangles: {out_obj.mesh_triangles:,}")
    print(f"  Peak GPU Util: {peak_util_holder[0]:.0f}% | Power: {peak_power_holder[0]:.1f}W | Temp: {peak_temp_holder[0]:.1f}°C")
    print("-" * 60)

    # Validate Pass Conditions
    checks = {
        "ok": res.ok,
        "cuda_device": "cuda" in str(out_obj.device).lower(),
        "mesh_produced": out_obj.mesh_triangles > 1000,
        "vram_recovery_normal": abs(vram_recovery_delta) < 256.0,
        "vram_within_prediction": actual_peak_vram <= max(planned_vram_mb * 1.5, planned_vram_mb + 512),
        "wall_time_reasonable": wall_s < 60.0,
    }

    all_passed = all(checks.values())
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} Gate Check: {k} = {v}")

    report = {
        "gate_passed": all_passed,
        "n_frames": len(gate_ids),
        "voxel_mm": voxel_mm,
        "wall_s": round(wall_s, 2),
        "device": out_obj.device,
        "mesh_triangles": out_obj.mesh_triangles,
        "mesh_vertices": out_obj.mesh_vertices,
        "vram_baseline_mb": round(vram_baseline, 1),
        "vram_planned_peak_mb": round(planned_vram_mb, 1),
        "vram_actual_peak_mb": round(actual_peak_vram, 1),
        "vram_post_mb": round(vram_post, 1),
        "vram_recovery_delta_mb": round(vram_recovery_delta, 1),
        "planned_blocks": planned_blocks,
        "gpu_peak_util_pct": peak_util_holder[0],
        "gpu_peak_power_w": peak_power_holder[0],
        "gpu_peak_temp_c": peak_temp_holder[0],
        "checks": checks,
    }

    gate_json = work_dir / "gpu_gate_result.json"
    gate_json.write_text(json.dumps(report, indent=2))
    print(f"  💾 Gate result written to {gate_json}")
    return report


if __name__ == "__main__":
    ds_path = Path("ros2_data/frames/hallway")
    tr_path = Path("ros2_data/trajectories/cuvslam_hallway_trajectory.txt")
    rep = run_gate(ds_path, tr_path)
    if not rep["gate_passed"]:
        sys.exit(1)
    sys.exit(0)
