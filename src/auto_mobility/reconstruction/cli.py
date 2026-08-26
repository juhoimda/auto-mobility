"""V2 CLI entry: python3 -m auto_mobility.reconstruction.cli BAG --standard

Thin orchestration driver:
  MachineProfile -> budgets -> DatasetAudit -> FrameSplit -> TrajectoryJudge
  -> Standard pipeline -> run_manifest.json / decision_trace.json.

Legacy compare.sh flags (--quick/--full/--phase/--top-k/--no-cache/
--no-resume) are accepted for compatibility and mapped conservatively;
--run-slam generates missing trajectories via run_slam.sh (isolated ROS env).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="auto-mobility-reconstruction")
    p.add_argument("bag", nargs="?", default="", help="bag name or frames dir")
    p.add_argument("--standard", action="store_true", help="standard pipeline mode")
    p.add_argument("--quick", action="store_true", help="compat: reduced time budget")
    p.add_argument("--full", action="store_true", help="compat: accepted, full search")
    p.add_argument("--phase", default="all", help="compat: accepted (single-phase search removed in V2)")
    p.add_argument("--top-k", type=int, default=2, help="finalists to deliver (cap 2)")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="frames dataset dir (default ros2_data/frames/<bag>)")
    p.add_argument("--trajectory", type=Path, default=None, action="append",
                   help="TUM trajectory file to judge (repeatable)")
    p.add_argument("--output", type=Path, default=Path("output"))
    p.add_argument("--budget-min", type=float, default=None,
                   help="wall-time budget in minutes (default 30; quick 12)")
    p.add_argument("--run-slam", action="store_true",
                   help="generate missing trajectories via run_slam.sh")
    p.add_argument("--no-cache", action="store_true", help="compat: accepted (cache is content-verified)")
    p.add_argument("--no-resume", action="store_true", help="compat: accepted")
    return p


def _check_trajectory_cache(tp: Path) -> bool:
    """Format-level check: valid TUM (>=10 lines, 8 cols)."""
    if not tp.is_file() or tp.stat().st_size < 64:
        return False
    try:
        # at least 10 lines with 8 cols
        with open(tp) as fh:
            lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        if len(lines) < 10:
            return False
        for l in lines[:5]:
            if len(l.split()) != 8:
                return False
        return True
    except Exception:
        return False


def _dataset_fingerprint(dataset_dir: Path | None) -> str | None:
    """sha256 over frames.csv + camera_info.json (cache invalidation key)."""
    import hashlib

    if dataset_dir is None:
        return None
    frames_csv = Path(dataset_dir) / "frames.csv"
    if not frames_csv.is_file():
        return None
    h = hashlib.sha256()
    h.update(frames_csv.read_bytes())
    cam = Path(dataset_dir) / "camera_info.json"
    if cam.is_file():
        h.update(cam.read_bytes())
    return h.hexdigest()[:16]


def _verify_trajectory_cache(tp: Path, dataset_dir: Path | None = None) -> bool:
    """Content-verified reuse (#50): TUM format + sidecar integrity.

    A cached trajectory is reusable only when ALL hold:
      - valid TUM format;
      - sidecar <tp>.meta.json exists and its trajectory_sha256 matches the
        file bytes (catches truncation/corruption/overwrites);
      - sidecar dataset_fingerprint matches the current frames.csv+intrinsics
        hash when a dataset is known (catches re-extracted frames, changed
        sync/intrinsics -> stale pose timestamps).
    Missing sidecar or fingerprint mismatch => regenerate (fail-closed).
    """
    if not _check_trajectory_cache(tp):
        return False
    meta_path = Path(str(tp) + ".meta.json")
    if not meta_path.is_file():
        print(f"[v2] trajectory cache rejected (no sidecar): {tp}")
        return False
    try:
        import hashlib

        meta = json.loads(meta_path.read_text())
        sha = hashlib.sha256(Path(tp).read_bytes()).hexdigest()
        if str(meta.get("trajectory_sha256", "")) != sha:
            print(f"[v2] trajectory cache rejected (sha mismatch): {tp}")
            return False
        fp = _dataset_fingerprint(dataset_dir)
        want_fp = meta.get("dataset_fingerprint")
        if fp is not None and want_fp is not None and want_fp != fp:
            print(f"[v2] trajectory cache rejected (dataset changed): {tp}")
            return False
        if fp is not None and want_fp is None:
            # legacy sidecar without provenance: fail-closed
            print(f"[v2] trajectory cache rejected (sidecar lacks dataset "
                  f"fingerprint): {tp}")
            return False
        return True
    except Exception as exc:
        print(f"[v2] trajectory cache rejected ({exc}): {tp}")
        return False


def _wait_gpu_recovery(pre_used_mb: float | None, timeout_s: float = 5.0,
                       tolerance_mb: float = 256.0) -> float | None:
    """§4/§6: wait until VRAM returns near the pre-SLAM baseline.

    Returns recovered delta (post - pre) once recovered, or None when
    nvidia-smi is unavailable.  Never blocks the pipeline beyond timeout_s.
    """
    try:
        from auto_mobility.reconstruction.runtime.machine_profile import _probe_gpu

        deadline = time.time() + timeout_s
        post = None
        while time.time() < deadline:
            g = _probe_gpu()
            post = int(g.vram_total_mb) - int(g.vram_free_mb) \
                if g.present else None
            if post is None:
                break
            if pre_used_mb is None or post <= pre_used_mb + tolerance_mb:
                delta = (post - pre_used_mb) if pre_used_mb is not None else 0.0
                print(f"[v2] VRAM recovery ok: baseline {pre_used_mb}MB -> "
                      f"{post}MB (delta {delta:.0f}MB)")
                return delta
            time.sleep(1.0)
        if post is not None:
            print(f"[v2] WARNING: VRAM not fully recovered after SLAM exit: "
                  f"baseline {pre_used_mb}MB -> {post}MB")
        return None
    except Exception:
        return None


def _gpu_used_mb() -> float | None:
    try:
        from auto_mobility.reconstruction.runtime.machine_profile import _probe_gpu

        g = _probe_gpu()
        return float(int(g.vram_total_mb) - int(g.vram_free_mb)) if g.present else None
    except Exception:
        return None


def _run_slam_subprocess(bag: str, project_dir: Path, backend: str,
                         dataset_dir: Path | None = None) -> Path | None:
    """Run one SLAM backend in isolated subprocess (§4 sequential).

    - RTAB: via run_slam.sh (ROS-isolated)
    - cuVSLAM: via python -m auto_mobility.reconstruction.pose.backends.cuvslam_worker
      (CUDA context stays in child; parent VRAM recovers after exit).
    Returns trajectory path if cache-valid.
    """
    traj_dir = project_dir / "ros2_data" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    if backend == "rtab":
        script = project_dir / "scripts" / "pipeline" / "run_slam.sh"
        # content-verified reuse (#50)
        cand = sorted(traj_dir.glob(f"*{bag}*rtab*trajectory*.txt"))
        for c in cand:
            if _verify_trajectory_cache(c, dataset_dir):
                print(f"[v2] reusing cached RTAB trajectory: {c}")
                return c
        if not script.is_file():
            print(f"[v2] run_slam.sh not found: {script}")
            return None
        # §7 phase-specific: RTAB is CPU-bound, allow 6-8 threads; parent is idle
        env = dict(__import__("os").environ)
        env["OMP_NUM_THREADS"] = env.get("RTAB_OMP_THREADS", "6")
        print(f"[v2] generating RTAB trajectory via run_slam.sh {bag} (isolated)...")
        try:
            subprocess.run(["bash", str(script), bag, "--slam=rtab"],
                           check=False, env=env)
        except OSError as exc:
            print(f"[v2] run_slam.sh failed: {exc}")
            return None
        # verify content after child exit (#50)
        cand = sorted(traj_dir.glob(f"*{bag}*rtab*trajectory*.txt"))
        for c in cand:
            if _verify_trajectory_cache(c, dataset_dir):
                return c
        return None
    elif backend == "cuvslam":
        # §3/§4 cuVSLAM RGB-D via isolated worker subprocess
        # dataset_dir is canonical RGB-D frames
        ds = dataset_dir or (project_dir / "ros2_data" / "frames" / bag)
        if not ds.is_dir():
            print(f"[v2] cuvslam: dataset not found {ds}")
            return None
        out_traj = traj_dir / f"cuvslam_{bag}_trajectory.txt"
        if _verify_trajectory_cache(out_traj, dataset_dir):
            print(f"[v2] reusing cached cuVSLAM trajectory: {out_traj}")
            return out_traj
        # run worker subprocess with GPU-bound thread cap 2-4 (§7)
        cmd = [sys.executable, "-m",
               "auto_mobility.reconstruction.pose.backends.cuvslam_worker",
               "--dataset", str(ds), "--out", str(out_traj)]
        env = {k: v for k, v in __import__("os").environ.items()
               if not k.startswith(("ROS_", "RMW_"))}
        env["OMP_NUM_THREADS"] = env.get("CUVSLAM_OMP_THREADS", "3")
        env["OPENBLAS_NUM_THREADS"] = "1"
        # §6 pre-run baseline for VRAM recovery measurement
        pre_used = _gpu_used_mb()
        print(f"[v2] generating cuVSLAM trajectory via isolated worker ({ds}) ...")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
            print(res.stdout[-2000:] if res.stdout else "")
            if res.stderr:
                print(res.stderr[-2000:])
            if res.returncode == 0 and _verify_trajectory_cache(out_traj, dataset_dir):
                # §4/§6 verify VRAM returns to near-baseline after child exit
                _wait_gpu_recovery(pre_used, timeout_s=5.0)
                return out_traj
            else:
                print(f"[v2] cuvslam worker failed rc={res.returncode}")
        except Exception as exc:
            print(f"[v2] cuvslam worker launch failed: {exc}")
        return None
    return None


def _generate_trajectories(bag: str, project_dir: Path,
                           dataset_dir: Path | None = None) -> list:
    """Generate both RTAB and cuVSLAM trajectories sequentially (§5)."""
    out = []
    # §5 sequential default: cuVSLAM (GPU) then RTAB (CPU) to avoid combined overload
    for backend in ("cuvslam", "rtab"):
        tp = _run_slam_subprocess(bag, project_dir, backend, dataset_dir)
        if tp is not None:
            out.append(tp)
        # small cooldown between GPU-heavy and CPU-heavy stages
        if backend == "cuvslam" and tp is not None:
            time.sleep(2.0)
    return out


def _register_final_artifacts(out_dir: Path, rank_dir: Path, dataset_spec: dict,
                              fusion_spec: dict, decisions_log) -> None:
    """Content-addressed registration of the delivered artifacts (#21/#23)."""
    try:
        from auto_mobility.reconstruction.artifacts import ArtifactStore, make_identity

        store = ArtifactStore(out_dir / "cache" / "artifacts")
        identity = make_identity(
            dataset_spec=dataset_spec,
            trajectory_spec={"winner": rank_dir.name},
            fusion_spec=fusion_spec,
            surface_spec={"method": "tsdf_direct+optional_poisson"},
        )
        registered = {}
        for kind, fname in (("mesh", "model.obj"), ("mesh_raw", "model_raw.obj"),
                            ("mtl", "model.mtl")):
            src = rank_dir / fname
            if src.is_file():
                art = store.put(identity, kind, fname, src,
                                extra_meta={"rank": rank_dir.name})
                registered[kind] = art.to_dict()
        atlas_dir = rank_dir / "textures"
        if atlas_dir.is_dir():
            for img in sorted(atlas_dir.iterdir()):
                if img.is_file():
                    art = store.put(identity, "atlas", img.name, img,
                                    extra_meta={"rank": rank_dir.name})
                    registered[f"atlas:{img.name}"] = art.to_dict()
        cfg_path = rank_dir / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        cfg["artifact_identity"] = identity.to_dict()
        cfg["artifact_store"] = registered
        cfg_path.write_text(json.dumps(cfg, indent=2))
        decisions_log("artifact_store", "REGISTERED",
                      "content-addressed artifacts verified via sha256 sidecars",
                      **identity.to_dict())
    except Exception as exc:
        decisions_log("artifact_store", "FAILED", f"{exc}")


def run(args: argparse.Namespace) -> int:
    from auto_mobility.reconstruction.data import audit_dataset
    from auto_mobility.reconstruction.runtime import (
        BudgetManager, Scheduler, compute_resource_budgets, load_or_probe_profile,
    )
    from auto_mobility.reconstruction.config import default_config
    from auto_mobility.reconstruction.runtime.run_state import (
        detect_previous_host_reset, mark_completed, write_run_state)

    t_start = time.monotonic()
    cfg = default_config()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # §25 unclean-run detection before any heavy work
    # single lightweight profile probe; reused for detection AND budgets (#13)
    profile = load_or_probe_profile(out_dir / "cache", measure_overhead=False)
    reset_info = detect_previous_host_reset(out_dir)
    safe_mode = bool(reset_info.get("previous_host_reset"))
    if safe_mode:
        print(f"[v2] SAFE MODE activated: {reset_info} — sequential, coarser, more reserve")
    # §9 optional overhead measurement in separate call (subprocess-only)
    # we measure overhead lazily once per machine and cache it; here we keep 0 until calibration
    if safe_mode:
        # §26 SAFE MODE profile adjustments (§26): more reserve, smaller block target, no fine
        from dataclasses import replace as _rep
        cfg = _rep(cfg, resources=_rep(cfg.resources, vram_reserve_gb=2.0,
                                        vram_free_fraction=0.45))
    budgets = compute_resource_budgets(profile, cfg.resources)
    # write RUNNING state atomically
    try:
        write_run_state(out_dir, profile, status="RUNNING")
    except Exception:
        pass
    budget_min = args.budget_min if args.budget_min is not None else (12.0 if args.quick else 30.0)
    budget = BudgetManager(budget_min * 60.0, cfg.budget)
    scheduler = Scheduler(
        cpu_threads=budgets.cpu_threads,
        ram_mb=budgets.ram_budget_mb,
        gpu_slots=budgets.gpu_heavy_slots,
        vram_mb=budgets.vram_budget_mb,
    ).start()
    print(f"[v2] gpu={profile.gpu.model} vram_budget={budgets.vram_budget_mb}MB "
          f"ram_budget={budgets.ram_budget_mb}MB time_budget={budget_min:.0f}min")

    dataset_dir = args.dataset_dir or Path("ros2_data/frames") / args.bag
    manifest = {
        "schema_version": "recon-v2",
        "dataset_dir": str(dataset_dir),
        "mode": "standard" if args.standard else "default",
        "machine_profile": profile.to_dict(),
        "resource_budgets": budgets.to_dict(),
        "budget_plan": budget.to_dict(),
    }

    audit_ok = True
    frame_ts = None
    ds = None
    if dataset_dir.is_dir() and (dataset_dir / "frames.csv").is_file():
        from auto_mobility.dataset.frame_dataset import FrameDataset

        ds = FrameDataset(str(dataset_dir))
        result = audit_dataset(
            list(ds), lambda f: _load_rgb(ds, f), lambda f: _load_depth(ds, f),
            probe_count=24,
        )
        manifest["dataset_audit"] = result.to_dict()
        audit_ok = result.ok
        print(f"[v2] audit ok={result.ok} frames={result.n_frames} "
              f"sync_p95={result.sync_dt_ms_p95}ms")
        frame_ts = [f.rgb_timestamp for f in ds]
    else:
        print(f"[v2] dataset dir not found or missing frames.csv: {dataset_dir}")

    trajs = {}
    if args.standard and ds is not None:
        from auto_mobility.trajectory.io import Trajectory

        traj_files = list(args.trajectory or [])
        if not traj_files:
            traj_root = Path("ros2_data/trajectories")
            if traj_root.is_dir():
                traj_files = sorted(traj_root.glob(f"*{args.bag}*trajectory*.txt"))
        if args.run_slam:
            # §5/#50: per-backend cache verification decides reuse vs
            # regeneration — an unrelated stale file must NOT block
            # generation of a missing backend.
            generated = _generate_trajectories(
                args.bag, Path(__file__).resolve().parents[2], dataset_dir)
            existing = {str(Path(t).resolve()) for t in traj_files}
            for g in generated:
                if str(Path(g).resolve()) not in existing:
                    traj_files.append(g)
            print(f"[v2] available/generated {len(traj_files)} "
                  f"trajectory file(s) after sequential isolation")
        for tp in traj_files:
            tp = Path(tp)
            if tp.is_file():
                try:
                    name = tp.stem.replace(f"_{args.bag}", "")
                    trajs[name] = Trajectory.from_tum_file(str(tp))
                except Exception as exc:
                    print(f"[v2] skip trajectory {tp}: {exc}")
        if trajs:
            print(f"[v2] running standard pipeline with {len(trajs)} trajectory candidate(s)")
            from auto_mobility.reconstruction.pipeline.standard import run_standard

            std_result = run_standard(
                dataset_dir, trajs, out_dir,
                vram_budget_mb=float(budgets.vram_budget_mb),
                ram_budget_mb=float(budgets.ram_budget_mb),
                scheduler=scheduler,
                budget=budget,
                top_k=min(2, max(1, args.top_k)),
            )
            manifest["standard_result"] = {
                k: v for k, v in std_result.items()
                if k not in ("holdout", "trajectory_scores")
            }
            manifest["trajectory_scores"] = std_result.get("trajectory_scores", [])
            manifest["holdout_split"] = std_result.get("holdout", manifest.get("holdout_split"))
            print(f"[v2] standard ok={std_result.get('ok')} winner={std_result.get('winner')} "
                  f"wall={std_result.get('wall_s')}s -> {out_dir}/final/rank_01")

            winner_tag = "rank_01" if std_result.get("ok") else None
            if winner_tag:
                _register_final_artifacts(
                    out_dir, out_dir / "final" / winner_tag,
                    dataset_spec={"dir": dataset_dir.name,
                                  "n_frames": len(ds),
                                  "audit_ok": audit_ok},
                    fusion_spec={"backend": "open3d_vbg_cuda",
                                 "voxel_mm_effective": std_result.get(
                                     "rank_01", {}).get("voxel_mm_effective")},
                    decisions_log=lambda *a, **k: std_result.setdefault(
                        "extra_decisions", []).append(
                        {"stage": a[0], "decision": a[1], "reason": a[2],
                         "evidence": k}),
                )
                manifest["standard_result"]["decisions"] += manifest[
                    "standard_result"].get("extra_decisions", [])
        else:
            print("[v2] no trajectory candidates found. Run scripts/pipeline/run_slam.sh "
                  f"{args.bag} --slam=rtab (or pass --run-slam).")

    scheduler.shutdown()
    elapsed = time.monotonic() - t_start
    try:
        budget.spend("pose_exploration", min(elapsed, budget.phase_allocated("pose_exploration")))
    except Exception:
        pass
    manifest["wall_s"] = round(elapsed, 1)
    manifest["budget_actual"] = budget.to_dict()
    manifest["safe_mode"] = safe_mode
    manifest["host_reset_info"] = reset_info

    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    decisions = manifest.get("standard_result", {}).get("decisions", [])
    (out_dir / "decision_trace.json").write_text(json.dumps(
        {"winner": manifest.get("standard_result", {}).get("winner"),
         "decisions": decisions}, indent=2))
    try:
        # §25 mark completed atomically — only when ok to avoid false reset masking
        ok = manifest.get("standard_result", {}).get("ok", audit_ok)
        if ok:
            mark_completed(out_dir)
        # else keep RUNNING so next boot can decide SAFE MODE conservatively
    except Exception:
        pass
    print(f"[v2] manifest written: {out_dir}/run_manifest.json ({elapsed:.1f}s)")
    return 0 if audit_ok else 1


def np_diff_rot(traj):
    import numpy as np

    ori = np.asarray(traj.orientations)
    n = len(ori)
    rot = np.zeros(n)
    for i in range(1, n):
        dot = np.clip(np.abs(np.dot(ori[i - 1], ori[i])), 0.0, 1.0)
        rot[i] = float(2.0 * np.arccos(dot))
    return rot


def _nearest_pose(traj, ts):
    import numpy as np

    idx = int(np.searchsorted(traj.timestamps, ts))
    idx = min(max(idx, 1), len(traj.timestamps) - 1)
    if abs(traj.timestamps[idx - 1] - ts) < abs(traj.timestamps[idx] - ts):
        idx -= 1
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(traj.orientations[idx])
    T[:3, 3] = traj.positions[idx]
    return T


def _quat_to_R(q):
    import numpy as np

    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _load_rgb(ds, frame):
    import cv2

    return cv2.imread(str(frame.rgb_path))


def _load_depth(ds, frame):
    import cv2

    return cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
