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


def _generate_trajectories(bag: str, project_dir: Path) -> list:
    """Run rtabmap via run_slam.sh (ROS-isolated subprocess) for a missing bag."""
    script = project_dir / "scripts" / "pipeline" / "run_slam.sh"
    if not script.is_file():
        print(f"[v2] run_slam.sh not found: {script}")
        return []
    print(f"[v2] generating trajectory via run_slam.sh {bag} --slam=rtab ...")
    try:
        subprocess.run(["bash", str(script), bag, "--slam=rtab"], check=False)
    except OSError as exc:
        print(f"[v2] run_slam.sh failed to launch: {exc}")
        return []
    traj_dir = project_dir / "ros2_data" / "trajectories"
    return sorted(traj_dir.glob(f"*{bag}*trajectory*.txt"))


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

    t_start = time.monotonic()
    cfg = default_config()
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = load_or_probe_profile(out_dir / "cache")
    budgets = compute_resource_budgets(profile, cfg.resources)
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
        if not traj_files and args.run_slam:
            generated = _generate_trajectories(args.bag,
                                               Path(__file__).resolve().parents[2])
            traj_files.extend(generated)
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

    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    decisions = manifest.get("standard_result", {}).get("decisions", [])
    (out_dir / "decision_trace.json").write_text(json.dumps(
        {"winner": manifest.get("standard_result", {}).get("winner"),
         "decisions": decisions}, indent=2))
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
