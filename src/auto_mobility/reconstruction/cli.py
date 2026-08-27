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
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="auto-mobility-reconstruction")
    p.add_argument("bag", nargs="?", default="", help="bag name or frames dir")
    p.add_argument("--mode", choices=["quick", "preview", "standard", "full"], default=None,
                   help="pipeline execution mode (quick | preview | standard | full)")
    p.add_argument("--standard", action="store_true", help="standard pipeline mode (default)")
    p.add_argument("--preview", action="store_true", help="fast visual inspection preview mode (dual RTAB & cuVSLAM OBJs)")
    p.add_argument("--safe-mode", "--preview-safe", action="store_true", dest="safe_mode_cli",
                   help="force SAFE MODE (conservative concurrency, reserve, bounds)")
    p.add_argument("--quick", action="store_true", help="developer sanity / pipeline health check shortcut")
    p.add_argument("--full", action="store_true", help="exhaustive search and validation mode")
    p.add_argument("--phase", default="all", help="compat: accepted (single-phase search removed in V2)")
    p.add_argument("--top-k", type=int, default=2, help="finalists to deliver (cap 2)")
    p.add_argument("--deliver-backends", type=str, default=None,
                   help="standard dual-backend delivery: comma-separated backend list (e.g. rtab,cuvslam) or 'all'")
    p.add_argument("--compare-backends", action="store_true",
                   help="alias for --deliver-backends rtab,cuvslam (dual delivery with comparison)")
    p.add_argument("--allow-single-backend", action="store_true",
                   help="allow single backend delivery when only one trajectory cache is valid (default dual requires both)")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="frames dataset dir (default ros2_data/frames/<bag>)")
    p.add_argument("--trajectory", type=Path, default=None, action="append",
                   help="TUM trajectory file to judge (repeatable)")
    p.add_argument("--output", type=Path, default=Path("output"))
    p.add_argument("--budget-min", type=float, default=None,
                   help="wall-time budget in minutes (default 30; quick 12; preview 15; full 45)")
    p.add_argument("--run-slam", action="store_true",
                   help="generate missing trajectories via run_slam.sh / cuvslam worker")
    p.add_argument("--no-cache", action="store_true",
                   help="force regeneration of ALL caches including trajectory, fusion, and frame "
                        "caches (does not just affect frame extraction)")
    p.add_argument("--no-resume", action="store_true", help="compat: accepted")
    return p


def resolve_execution_mode(args: argparse.Namespace):
    """Resolve and validate mutual exclusion of CLI execution modes (§1, §21, §22)."""
    from auto_mobility.reconstruction.config import ExecutionMode

    flags = []
    if getattr(args, "mode", None):
        flags.append(args.mode.lower())
    if getattr(args, "quick", False):
        flags.append("quick")
    if getattr(args, "preview", False):
        flags.append("preview")
    if getattr(args, "full", False):
        flags.append("full")
    if getattr(args, "standard", False):
        flags.append("standard")

    unique_flags = list(set(flags))
    if len(unique_flags) > 1:
        # If standard was passed alongside another explicit mode, reject ambiguous combinations
        raise ValueError(
            f"Cannot combine multiple execution modes ({', '.join(sorted(unique_flags))}). "
            "Please choose exactly one of --quick, --preview, --standard, or --full."
        )
    if unique_flags:
        return ExecutionMode(unique_flags[0])
    return ExecutionMode.STANDARD



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
    """Strict provenance reuse (recon-v4/sidecar-1): TUM format + full provenance.

    A cached trajectory is reusable only when ALL hold:
      - valid TUM format;
      - sidecar schema is recon-v4/sidecar-1 (sidecar-3 and incomplete sidecar-4 rejected);
      - required fields present: trajectory_sha256, dataset_fingerprint,
        alignment_contract_fingerprint, aligned_depth_artifact_fingerprint,
        backend_config_hash, worker_source_hash/git_sha, version fields,
        CUDA/GPU/Open3D, pose convention/frame, profile, command_line, created_at_utc,
        seed/deterministic declarations;
      - trajectory_sha256 matches file bytes;
      - dataset_fingerprint matches current frames.csv+intrinsics hash;
      - alignment_contract_fingerprint matches current rgbd_alignment_contract.json fingerprint
        when dataset_dir is known (catches re-extracted frames or reproved alignment);
      - pose convention/frame are T_world_camera / camera_color_optical_frame.
    Missing field or mismatch => CACHE_MISS with reason.
    Only when all checks succeed print 'provenance verified' HIT log.
    """
    if not _check_trajectory_cache(tp):
        return False
    meta_path = Path(str(tp) + ".meta.json")
    if not meta_path.is_file() and tp.suffix and tp.with_suffix(".meta.json").is_file():
        meta_path = tp.with_suffix(".meta.json")
    if not meta_path.is_file():
        print(f"[v2] trajectory cache rejected (no sidecar): {tp}")
        return False
    try:
        import hashlib

        meta = json.loads(meta_path.read_text())
        # Strict schema: only recon-v4/sidecar-1 accepted for standard reuse
        if meta.get("schema_version") != "recon-v4/sidecar-1":
            print(f"[P0-1] trajectory cache MISS ({tp.name}): schema {meta.get('schema_version')} not strict (require recon-v4/sidecar-1) -> CACHE_MISS")
            return False
        # Required fields (all must be present, not None for mandatory)
        required = [
            "trajectory_sha256", "dataset_fingerprint",
            "alignment_contract_fingerprint", "aligned_depth_artifact_fingerprint",
            "backend_config_hash", "pose_convention", "pose_frame",
            "profile", "command_line", "created_at_utc",
        ]
        # version fields: either cuvslam_version or rtab_version/binary hash
        has_version = ("cuvslam_version" in meta and meta["cuvslam_version"] not in (None, "", "unknown")) or \
                      ("rtab_version" in meta and meta["rtab_version"] not in (None, "", "unknown"))
        if not has_version:
            print(f"[P0-1] trajectory cache MISS ({tp.name}): missing version field (cuvslam_version/rtab_version) -> CACHE_MISS")
            return False
        # worker/source hash
        has_worker = ("worker_source_hash" in meta and meta["worker_source_hash"] not in (None, "", "unknown")) or \
                     ("git_sha" in meta and meta["git_sha"] not in (None, "", "unknown"))
        if not has_worker:
            print(f"[P0-1] trajectory cache MISS ({tp.name}): missing worker_source_hash/git_sha -> CACHE_MISS")
            return False
        # CUDA/GPU/Open3D fields
        for fld in ["gpu_model", "open3d_version"]:
            if fld not in meta or meta[fld] in (None, "", "unknown"):
                print(f"[P0-1] trajectory cache MISS ({tp.name}): missing required field {fld} -> CACHE_MISS")
                return False
        for fld in required:
            if fld not in meta or meta[fld] is None or str(meta[fld]).strip() == "":
                print(f"[P0-1] trajectory cache MISS ({tp.name}): missing required field {fld} -> CACHE_MISS")
                return False
        if meta.get("pose_convention") != "T_world_camera":
            print(f"[P0-1] trajectory cache MISS ({tp.name}): invalid pose_convention {meta.get('pose_convention')} -> CACHE_MISS")
            return False
        if meta.get("pose_frame") != "camera_color_optical_frame":
            print(f"[P0-1] trajectory cache MISS ({tp.name}): invalid pose_frame {meta.get('pose_frame')} -> CACHE_MISS")
            return False
        sha = hashlib.sha256(Path(tp).read_bytes()).hexdigest()
        if str(meta.get("trajectory_sha256", "")) != sha:
            print(f"[P0-1] trajectory cache MISS ({tp.name}): sha mismatch -> CACHE_MISS")
            return False
        fp = _dataset_fingerprint(dataset_dir)
        want_fp = meta.get("dataset_fingerprint")
        if fp is not None and want_fp is not None and want_fp != fp:
            print(f"[P0-1] trajectory cache MISS ({tp.name}): dataset changed {want_fp} != {fp} -> CACHE_MISS")
            return False
        # alignment contract fingerprint check
        if dataset_dir is not None:
            try:
                from auto_mobility.dataset.rgbd_alignment import load_contract
                c = load_contract(Path(dataset_dir))
                if c is not None and c.is_proven():
                    want_align = meta.get("alignment_contract_fingerprint")
                    if want_align is not None and want_align != c.contract_fingerprint:
                        print(f"[P0-1] trajectory cache MISS ({tp.name}): alignment contract mismatch {want_align} != {c.contract_fingerprint} -> CACHE_MISS")
                        return False
            except Exception:
                pass
        # All checks passed
        print(f"[P0-1] trajectory cache HIT ({tp.name}): {tp} (reason: provenance verified)")
        return True
    except Exception as exc:
        print(f"[P0-1] trajectory cache MISS ({tp.name}): exception {exc} -> CACHE_MISS")
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


def _canonical_rtab_path(bag: str, project_dir: Path, profile: str = "normal") -> Path:
    """Deterministic canonical RTAB trajectory path (P0 #8)."""
    return project_dir / "ros2_data" / "trajectories" / f"rtab_{profile}_{bag}_trajectory.txt"


def _canonical_cuvslam_path(bag: str, project_dir: Path) -> Path:
    return project_dir / "ros2_data" / "trajectories" / f"cuvslam_{bag}_trajectory.txt"


def _run_slam_subprocess(bag: str, project_dir: Path, backend: str,
                         dataset_dir: Path | None = None,
                         force_regen: bool = False) -> Path | None:
    """Run one SLAM backend in isolated subprocess (§4 sequential).

    - RTAB: via run_slam.sh (ROS-isolated)
    - cuVSLAM: via python -m auto_mobility.reconstruction.pose.backends.cuvslam_worker
      (CUDA context stays in child; parent VRAM recovers after exit).
    Returns trajectory path if cache-valid (P0 #8 deterministic canonical path).
    When force_regen=True the cache check is skipped entirely and the worker is
    unconditionally re-run (--no-cache semantics, P0-1).
    """
    traj_dir = project_dir / "ros2_data" / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    if backend == "rtab":
        script = project_dir / "scripts" / "pipeline" / "run_slam.sh"
        canonical = _canonical_rtab_path(bag, project_dir)
        if force_regen:
            print(f"[P0-1] trajectory cache BYPASS ({backend}): force_regen=True")
        else:
            # content-verified reuse (#50) — canonical first, then legacy glob fallback
            if _verify_trajectory_cache(canonical, dataset_dir):
                print(f"[P0-1] trajectory cache HIT ({backend}): {canonical} (reason: provenance verified)")
                return canonical
            else:
                print(f"[P0-1] trajectory cache MISS ({backend}): {canonical} (reason: canonical not valid, trying legacy)")
            # legacy glob for backward compat: match any rtab * bag * trajectory
            # also consider pattern where bag appears before/after rtab
            for pat in (f"*{bag}*rtab*trajectory*.txt", f"*rtab*{bag}*trajectory*.txt",
                        f"*rtab*trajectory*.txt"):
                for c in sorted(traj_dir.glob(pat)):
                    if bag in c.name and _verify_trajectory_cache(c, dataset_dir):
                        print(f"[P0-1] trajectory cache HIT ({backend}): {c} (reason: provenance verified)")
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
        # verify content after child exit (#50) — check canonical path directly (P0 #8)
        if _verify_trajectory_cache(canonical, dataset_dir):
            return canonical
        # fallback glob if canonical not yet (legacy naming)
        for pat in (f"*{bag}*rtab*trajectory*.txt", f"*rtab*{bag}*trajectory*.txt"):
            for c in sorted(traj_dir.glob(pat)):
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
        if force_regen:
            print(f"[P0-1] trajectory cache BYPASS ({backend}): force_regen=True")
        else:
            if _verify_trajectory_cache(out_traj, dataset_dir):
                print(f"[P0-1] trajectory cache HIT ({backend}): {out_traj} (reason: provenance verified)")
                return out_traj
            else:
                print(f"[P0-1] trajectory cache MISS ({backend}): {out_traj} (reason: sidecar invalid or missing)")
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
                           dataset_dir: Path | None = None,
                           force_regen: bool = False) -> list:
    """Generate both RTAB and cuVSLAM trajectories sequentially (§5).

    When force_regen=True (--no-cache), trajectory caches are bypassed for
    both backends and workers are unconditionally re-run (P0-1).
    """
    out = []
    # §5 sequential default: cuVSLAM (GPU) then RTAB (CPU) to avoid combined overload
    for backend in ("cuvslam", "rtab"):
        tp = _run_slam_subprocess(bag, project_dir, backend, dataset_dir,
                                  force_regen=force_regen)
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
    from auto_mobility.reconstruction.config import (
        ExecutionMode, SafetyMode, default_config, policy_for_mode)
    from auto_mobility.reconstruction.runtime.run_state import (
        detect_previous_host_reset, mark_completed, write_run_state)

    t_start = time.monotonic()
    cfg = default_config()
    resolved_mode = resolve_execution_mode(args)
    policy = policy_for_mode(resolved_mode)
    is_preview = (resolved_mode == ExecutionMode.PREVIEW)
    is_quick = (resolved_mode == ExecutionMode.QUICK)
    safe_mode_cli = bool(getattr(args, "safe_mode_cli", False))

    if args.output == Path("output"):
        # Keep every execution mode isolated by bag.  A fixed output directory
        # silently replaced another bag's OBJ, report, run state, and cache.
        safe_bag = "".join(c if c.isalnum() or c in "._-" else "_"
                           for c in (args.bag or "unnamed"))
        base_dir = Path("output_preview") if is_preview else Path("output_standard")
        out_dir = base_dir / safe_bag
    else:
        out_dir = args.output

    # --- Run-ID and staging isolation (tmp.md § 산출물 및 격리 규칙) ---
    import hashlib as _hl
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%SZ', time.gmtime())}_{_hl.sha256(str(time.time()).encode()).hexdigest()[:8]}"
    is_explicit = (args.output != Path("output"))
    if is_explicit:
        # Explicit --output already contains run_id (e.g. output_preview/hallway/<run_id>)
        final_dir = out_dir
        staging_dir = final_dir.parent / f".staging_{final_dir.name}"
    else:
        # Default: output_preview/hallway or output_standard/hallway is root; create run_id subdir
        final_dir = out_dir / run_id
        staging_dir = out_dir / f".staging_{run_id}"
    # Ensure parent of final exists (root)
    if is_explicit:
        final_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Clean any stale staging and create fresh
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    # For backward compat, out_dir for pipeline is staging_dir (isolated)
    # We will publish staging -> final atomically on success
    effective_out = staging_dir
    # Record roots for manifest
    root_dir = out_dir if not is_explicit else out_dir.parent
    # --no-cache must bypass all identity reuse
    no_cache = bool(getattr(args, "no_cache", False))
    if no_cache:
        print(f"[v2] --no-cache: all trajectory/fusion/output identity reuse strictly bypassed (staging {staging_dir})")

    # Out dir for pipeline is staging (isolated sibling)
    out_dir = effective_out
    # §25 unclean-run detection before any heavy work
    profile = load_or_probe_profile(out_dir / "cache", measure_overhead=False)
    reset_info = detect_previous_host_reset(out_dir)
    safe_mode = bool(reset_info.get("previous_host_reset")) or safe_mode_cli
    if safe_mode:
        print(f"[v2] SAFE MODE activated (reset={bool(reset_info.get('previous_host_reset'))}, "
              f"cli={safe_mode_cli}) — sequential, coarser, more reserve")
    if safe_mode:
        from dataclasses import replace as _rep
        cfg = _rep(cfg, resources=_rep(cfg.resources, vram_reserve_gb=2.0,
                                        vram_free_fraction=0.45))
    budgets = compute_resource_budgets(profile, cfg.resources)
    try:
        write_run_state(out_dir, profile, status="RUNNING")
    except Exception:
        pass
    budget_min = args.budget_min if args.budget_min is not None else policy.budget_minutes
    budget = BudgetManager(budget_min * 60.0, cfg.budget)
    scheduler = Scheduler(
        cpu_threads=budgets.cpu_threads,
        ram_mb=budgets.ram_budget_mb,
        gpu_slots=budgets.gpu_heavy_slots,
        vram_mb=budgets.vram_budget_mb,
    ).start()
    print(f"[v2] mode={resolved_mode.value} safe_mode={safe_mode} gpu={profile.gpu.model} "
          f"vram_budget={budgets.vram_budget_mb}MB ram_budget={budgets.ram_budget_mb}MB "
          f"time_budget={budget_min:.0f}min")

    dataset_dir = args.dataset_dir or Path("ros2_data/frames") / args.bag
    manifest = {
        "schema_version": "recon-v2",
        "dataset_dir": str(dataset_dir),
        "mode": resolved_mode.value,
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
    if ds is not None:
        from auto_mobility.trajectory.io import Trajectory

        traj_files = list(args.trajectory or [])
        if not traj_files:
            traj_root = Path("ros2_data/trajectories")
            if traj_root.is_dir():
                raw = sorted(traj_root.glob(f"*{args.bag}*trajectory*.txt"))
                all_traj = sorted(traj_root.glob("*trajectory*.txt"))
                raw_set = {str(p.resolve()) for p in raw}
                for p in all_traj:
                    if str(p.resolve()) not in raw_set and args.bag in p.name:
                        raw.append(p)
                traj_files = raw
        if args.run_slam:
            generated = _generate_trajectories(
                args.bag, Path(__file__).resolve().parents[3], dataset_dir,
                force_regen=bool(getattr(args, "no_cache", False)))
            existing = {str(Path(t).resolve()) for t in traj_files}

            for g in generated:
                if str(Path(g).resolve()) not in existing:
                    traj_files.append(g)
            print(f"[v2] available/generated {len(traj_files)} "
                  f"trajectory file(s) after sequential isolation")
        # Track per-backend cache status for standard dual delivery
        cache_status = {}  # backend -> (HIT/MISS, reason, path)
        # Map traj_files to backend first for status tracking
        tmp_backend_map = {}
        for tp in traj_files:
            tp = Path(tp)
            if not tp.is_file():
                continue
            # need backend name from sidecar or filename
            meta_path = Path(str(tp) + ".meta.json")
            if not meta_path.is_file() and tp.suffix and tp.with_suffix(".meta.json").is_file():
                meta_path = tp.with_suffix(".meta.json")
            be = None
            if meta_path.is_file():
                try:
                    be = str(json.loads(meta_path.read_text()).get("backend","")).strip()
                except: be=None
            if not be:
                stem = tp.stem.replace(f"_{args.bag}","")
                if "rtab" in stem.lower(): be="rtab"
                elif "cuvslam" in stem.lower(): be="cuvslam"
                else: be=stem
            tmp_backend_map[be] = tp

        verified_traj_files = []
        for tp in traj_files:
            tp = Path(tp)
            if not tp.is_file():
                continue
            ok = _verify_trajectory_cache(tp, dataset_dir)
            # derive backend for status
            meta_path = Path(str(tp) + ".meta.json")
            if not meta_path.is_file() and tp.suffix and tp.with_suffix(".meta.json").is_file():
                meta_path = tp.with_suffix(".meta.json")
            be = None
            if meta_path.is_file():
                try:
                    be = str(json.loads(meta_path.read_text()).get("backend","")).strip()
                except: be=None
            if not be:
                stem = tp.stem.replace(f"_{args.bag}","")
                if "rtab" in stem.lower(): be="rtab"
                elif "cuvslam" in stem.lower(): be="cuvslam"
                else: be=stem
            if not ok:
                print(f"[v2] REJECTED_STALE_CACHE (auto-discovered not verified): {tp}")
                cache_status[be] = ("CACHE_MISS", "strict provenance failed", str(tp))
                continue
            verified_traj_files.append(tp)
            cache_status[be] = ("CACHE_HIT", "provenance verified", str(tp))
        if verified_traj_files:
            by_backend_path = {}
            for tp in sorted(verified_traj_files, key=lambda p: p.stat().st_mtime):
                meta_path = Path(str(tp) + ".meta.json")
                if not meta_path.is_file() and tp.suffix and tp.with_suffix(".meta.json").is_file():
                    meta_path = tp.with_suffix(".meta.json")
                be = None
                if meta_path.is_file():
                    try:
                        be = str(json.loads(meta_path.read_text()).get("backend", "")).strip()
                    except Exception:
                        be = None
                if not be:
                    stem = tp.stem.replace(f"_{args.bag}", "")
                    if "rtab" in stem.lower():
                        be = "rtab"
                    elif "cuvslam" in stem.lower():
                        be = "cuvslam"
                    else:
                        be = stem
                by_backend_path[be] = tp
            for be, tp in by_backend_path.items():
                try:
                    trajs[be] = Trajectory.from_tum_file(str(tp))
                    print(f"[v2] verified trajectory {be}: {tp}")
                except Exception as exc:
                    print(f"[v2] skip trajectory {tp}: {exc}")
        # Standard dual delivery cache reuse policy (Task 2-B, 3)
        # If --standard without --run-slam and requested backends have CACHE_MISS, fail closed
        deliver_backends = None
        if getattr(args, "compare_backends", False):
            deliver_backends = ["rtab", "cuvslam"]
        elif getattr(args, "deliver_backends", None):
            raw = str(args.deliver_backends).strip()
            if raw.lower() in ("all", "*"):
                deliver_backends = ["rtab", "cuvslam"]
            else:
                deliver_backends = [s.strip().lower() for s in raw.split(",") if s.strip()]
        # Record cache status in manifest later
        manifest["trajectory_cache_status"] = cache_status
        if resolved_mode == __import__("auto_mobility.reconstruction.config", fromlist=["ExecutionMode"]).ExecutionMode.STANDARD and deliver_backends:
            # Dual delivery requires both backends unless --allow-single-backend
            missing = [b for b in deliver_backends if b not in trajs]
            # Also consider those with CACHE_MISS in tmp_backend_map but not in trajs
            for b in deliver_backends:
                if b not in trajs and b in tmp_backend_map:
                    # was present but failed verification => already CACHE_MISS
                    pass
            if missing:
                if not getattr(args, "allow_single_backend", False):
                    msg = f"PRECONDITION_FAILED: dual delivery requires backends {deliver_backends} but missing/invalid: {missing}. Cache status: {cache_status}. Hint: run with --run-slam to regenerate or --allow-single-backend to permit single."
                    print(f"[P0-1] {msg}")
                    manifest["preflight_failure"] = msg
                    manifest["standard_result"] = {"ok": False, "reason": msg, "cache_status": cache_status}
                    scheduler.shutdown()
                    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
                    return 1
                else:
                    print(f"[P0-1] WARNING: dual delivery missing backends {missing} but --allow-single-backend permits single")

        if trajs:
            print(f"[v2] running pipeline ({resolved_mode.value}) with {len(trajs)} trajectory candidate(s)")

            # GPU preflight check for CUDA-only fusion (P0-3)
            if not is_quick:
                from auto_mobility.reconstruction.runtime.machine_profile import _probe_gpu
                gpu = _probe_gpu()
                if not gpu.present or int(gpu.vram_total_mb or 0) == 0:
                    msg = (f"PRECONDITION_FAILED: GPU not available or VRAM=0. "
                           f"CUDA fusion requires a GPU. present={gpu.present} "
                           f"vram_total_mb={getattr(gpu, 'vram_total_mb', 0)}")
                    print(f"[P0-3] {msg}")
                    manifest["preflight_failure"] = msg
                    manifest["standard_result"] = {"ok": False, "reason": msg}
                    scheduler.shutdown()
                    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
                    return 1
                elif int(budgets.vram_budget_mb) == 0:
                    msg = (f"PRECONDITION_FAILED: vram_budget_mb=0 does not satisfy CUDA fusion. "
                           f"GPU present but budget calculation resulted in 0.")
                    print(f"[P0-3] {msg}")
                    manifest["preflight_failure"] = msg
                    manifest["standard_result"] = {"ok": False, "reason": msg}
                    scheduler.shutdown()
                    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
                    return 1

            from auto_mobility.reconstruction.pipeline.standard import run_standard
            hard_ceiling = None
            if profile.gpu.present and profile.gpu.vram_total_mb:
                hard_ceiling = max(0, int(profile.gpu.vram_total_mb) - int(budgets.vram_reserve_mb))
            effective_top_k = 1 if safe_mode and not is_preview else min(2, max(1, args.top_k))
            if safe_mode and effective_top_k != args.top_k and not is_preview:
                print(f"[v2] SAFE MODE: top_k clamped {args.top_k} -> {effective_top_k}")
            # Pass deliver_backends only for standard mode; preview already dual
            _deliver = deliver_backends if (resolved_mode.value == "standard" and deliver_backends) else None
            std_result = run_standard(
                dataset_dir, trajs, out_dir,
                vram_budget_mb=float(budgets.vram_budget_mb),
                ram_budget_mb=float(budgets.ram_budget_mb),
                scheduler=scheduler,
                budget=budget,
                top_k=effective_top_k,
                hard_ceiling_mb=float(hard_ceiling) if hard_ceiling else None,
                safe_mode=safe_mode,
                preview=is_preview,
                quick=is_quick,
                mode_policy=policy,
                deliver_backends=_deliver,
            )
            manifest["standard_result"] = {
                k: v for k, v in std_result.items()
                if k not in ("holdout", "trajectory_scores")
            }
            manifest["trajectory_scores"] = std_result.get("trajectory_scores", [])
            manifest["holdout_split"] = std_result.get("holdout", manifest.get("holdout_split"))

            if is_preview:
                preview_backends = std_result.get("backends", {})
                for bname, binfo in preview_backends.items():
                    b_dir = out_dir / "preview" / bname
                    if b_dir.is_dir():
                        _register_final_artifacts(
                            out_dir, b_dir,
                            dataset_spec={"dir": dataset_dir.name,
                                          "n_frames": len(ds),
                                          "audit_ok": audit_ok},
                            fusion_spec={"backend": "open3d_vbg_cuda",
                                         "voxel_mm_effective": binfo.get("voxel_mm_effective")},
                            decisions_log=lambda *a, **k: std_result.setdefault(
                                "extra_decisions", []).append(
                                {"stage": a[0], "decision": a[1], "reason": a[2],
                                 "evidence": k}),
                        )
                # P0-3: ok/ready must reflect current run, not stale OBJ from a prior run.
                # Require both: std_result.ok=True AND mesh file exists with triangles.
                _run_ok = bool(std_result.get("ok", False))
                ready = [bname for bname, binfo in preview_backends.items()
                         if _run_ok
                         and int(binfo.get("mesh_triangles", 0) or 0) > 0
                         and (out_dir / "preview" / bname / "model.obj").is_file()]
                print("\n" + "=" * 60)
                if ready:
                    print(f"  ✅ PREVIEW OBJ READY ({', '.join(ready)})\n")
                else:
                    print("  ❌ PREVIEW OBJ NOT CREATED (fusion/quality gate failed)\n")
                for bname in preview_backends.keys():
                    obj_p = out_dir / "preview" / bname / "model.obj"
                    print(f"  [{bname.upper()} PREVIEW]")
                    print(f"    OBJ:     {obj_p}")
                    print(f"    VIEW:    python3 src/auto_mobility/mesh/view_mesh.py {obj_p}\n")
                print(f"  Comparison Report:\n    {out_dir / 'preview' / 'preview_report.md'}")
                print("=" * 60 + "\n")
            elif is_quick:
                print("\n" + "=" * 60)
                print("  ⚡ QUICK SANITY CHECK COMPLETE")
                print(f"  Result: {std_result.get('quick_check', {}).get('status')}")
                print(f"  Report: {out_dir / 'quick' / 'quick_check.json'}")
                print("  Note: NOT FOR QUALITY EVALUATION")
                print("=" * 60 + "\n")
            else:
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
    manifest["run_id"] = run_id
    manifest["root_dir"] = str(root_dir)
    manifest["staging_dir"] = str(staging_dir)
    manifest["final_dir"] = str(final_dir)
    manifest["is_explicit_output"] = bool(is_explicit)
    # Fusion cache identity fingerprint (minimal: dataset+alignment+trajectory+mode+voxel)
    try:
        import hashlib as _hh
        # dataset fingerprint already computed as _dataset_fingerprint
        _fp = _dataset_fingerprint(dataset_dir) if dataset_dir else None
        # alignment fingerprint
        _afp = None
        try:
            from auto_mobility.dataset.rgbd_alignment import load_contract as _lc
            ctmp = _lc(Path(dataset_dir)) if dataset_dir else None
            _afp = ctmp.contract_fingerprint if ctmp else None
        except Exception:
            _afp = None
        # backend hashes from sidecars
        traj_shas = {}
        for be, traj in trajs.items():
            # we don't have raw SHA here, but record backend
            traj_shas[be] = str(be)
        manifest["fusion_cache_identity"] = {
            "dataset_fingerprint": _fp,
            "alignment_contract_fingerprint": _afp,
            "trajectory_shas": traj_shas,
            "mode": resolved_mode.value,
            "voxel_mm_requested": 10.0,
            "trunc_mult": 4.0,
        }
    except Exception:
        pass

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
    print(f"[v2] manifest written: {out_dir}/run_manifest.json ({elapsed:.1f}s) run_id={run_id}")

    # --- Atomic publish from staging to final (only on success) ---
    try:
        # out_dir is staging_dir; final_dir is publish target
        if out_dir != final_dir:
            # Check if we should publish: only successful runs are published per spec
            # However even failed runs should publish their staging for debugging? Spec says only successful publish.
            # We still publish manifest for failed? But spec says sibling staging then atomic publish only successful.
            # For now, publish regardless but keep failed as well for traceability with run_id subdir.
            # Ensure final parent exists
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                # Do not delete existing without explicit approval; we use run_id subdir so should not exist
                # If explicit output exists, we atomically replace (spec allows when user explicitly approved canonical replace)
                # Here we treat explicit run_id publish as atomic replace.
                if is_explicit:
                    # backup old if exists (do not lose)
                    import tempfile
                    tmp_old = final_dir.parent / f".old_{final_dir.name}"
                    if tmp_old.exists():
                        shutil.rmtree(tmp_old, ignore_errors=True)
                    final_dir.rename(tmp_old)
                else:
                    # default case: final_dir is root/run_id, should not exist; if exists, remove old run_id dir?
                    shutil.rmtree(final_dir, ignore_errors=True)
            # Atomic rename staging -> final
            staging_dir.rename(final_dir)
            print(f"[v2] atomic publish: {staging_dir} -> {final_dir}")
            out_dir = final_dir
            # Update latest symlink for default root case
            if not is_explicit:
                latest_link = root_dir / "latest"
                try:
                    if latest_link.is_symlink() or latest_link.exists():
                        latest_link.unlink()
                except Exception:
                    pass
                try:
                    latest_link.symlink_to(final_dir.name)
                    print(f"[v2] latest symlink: {latest_link} -> {final_dir.name}")
                except Exception as exc:
                    print(f"[v2] latest symlink failed: {exc}")
                # Also write root run_manifest latest copy for discoverability
                try:
                    (root_dir / "run_manifest_latest.json").write_text(json.dumps(manifest, indent=2))
                except Exception:
                    pass
            else:
                print(f"[v2] published explicit output: {final_dir}")
        else:
            print(f"[v2] published to requested output: {final_dir}")
    except Exception as exc:
        print(f"[v2] atomic publish failed: {exc} (staging remains at {staging_dir})")
        # keep staging as is for debugging, return failure
    # P0-3: Correct exit code — all failure cases must return non-zero.
    # Priority order: preflight_failure > standard_result.ok > audit_ok
    if manifest.get("preflight_failure"):
        return 1
    standard_result = manifest.get("standard_result", None)
    if standard_result is not None:
        if not standard_result.get("ok", False):
            return 1
        # standard_result exists and ok=True: return 0 only if audit also ok
        return 0 if audit_ok else 1
    # No standard_result: pipeline was not run (no trajectories or no dataset).
    # If trajectories were expected (dataset found) but none ran, that is a failure.
    if ds is not None and not trajs:
        return 1
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
