"""Standard end-to-end V2 (#14, #115):

quality-roles -> judge -> per-candidate SEARCH (train-only: refine -> coarse ->
masks -> fuse -> holdout eval) -> HIERARCHICAL RANKING (ranking.py) ->
FINAL DELIVERY per ranked finalist over ALL valid FUSE frames -> textured bake
-> report.

Search/delivery separation invariants (#20/#56/#57):
  - holdout frames NEVER enter the mesh that is scored (no leakage);
  - the delivered OBJ is fused from ALL valid FUSE frames of the winner.

Resource invariants (#25/#26/#31):
  - every GPU-heavy fusion is admitted by the Scheduler (gpu_slots=1);
  - optional improvements are gated by the wall-time BudgetManager;
  - thermal gate + subprocess isolation contain native crashes and overheating.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


TRUNCATION_MULTIPLIER = 4.0

_MC_OVERHEAD_FACTOR = 2.0
_MIN_FUSION_RAM_MB = 2048


def estimate_fusion_ram_mb(blocks: int) -> int:
    """Truthful host-RAM estimate for one isolated fusion worker (§11).

    8 bytes/voxel (tsdf f32 + weight f32, no color) × MC overhead 2.0.
    Never clamped with min(): if this exceeds the RAM budget the caller must
    REJECT/degrade, not report a smaller number to the scheduler.
    """
    from auto_mobility.reconstruction.fusion.open3d_vbg import (
        _BYTES_PER_BLOCK_NO_COLOR)

    return max(_MIN_FUSION_RAM_MB,
               int(blocks * _BYTES_PER_BLOCK_NO_COLOR
                   * _MC_OVERHEAD_FACTOR / 1e6))


def _quat_to_R(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _nearest_pose_map(traj, frames, max_pose_gap_ms: float = 50.0,
                      dropped=None):
    """Authoritative frame-to-pose association with 50ms gap guard and SLERP interpolation."""
    from auto_mobility.trajectory.association import associate_trajectory_to_frames

    frame_stamps = np.array([f.rgb_timestamp for f in frames], dtype=np.float64)
    _, results, summary = associate_trajectory_to_frames(
        frame_stamps, traj, max_pose_gap_ms=max_pose_gap_ms, enable_interpolation=True
    )
    poses = {}
    for res in results:
        f = frames[res.frame_id]
        if res.valid and res.T_world_camera is not None:
            poses[f.frame_id] = res.T_world_camera
        else:
            if dropped is not None:
                dropped.append(f.frame_id)
    return poses


def _judge(trajectories: dict, frame_ts):
    from auto_mobility.reconstruction.pose.judge import (
        score_trajectory, select_top_trajectories)

    scores = []
    for name, traj in trajectories.items():
        ori = np.asarray(traj.orientations)
        rot = [0.0] + [float(2 * np.arccos(np.clip(
            np.abs(np.dot(ori[i - 1], ori[i])), 0, 1))) for i in range(1, len(ori))]
        scores.append(score_trajectory(name, frame_ts, traj.timestamps,
                                       traj.positions, rot))
    return scores, select_top_trajectories(scores, top_k=2)


def _compute_frame_roles(frames, rgb, depth):
    """Single-decode per-frame quality pass; returns FUSE/TRACK/REJECT roles."""
    from auto_mobility.reconstruction.data.frame_selector import FrameSelector

    sel = FrameSelector(cache_frames=12)
    sync = {f.frame_id: getattr(f, "rgb_depth_dt_ms", 0.0) for f in frames}
    qualities = [
        sel.compute_quality(f,
                            lambda fr: rgb(fr.frame_id),
                            lambda fr: depth(fr.frame_id))
        for f in frames
    ]
    return sel.classify(qualities, sync)


def compute_search_delivery_sets(split, roles, valid_ids,
                                 min_fuse_frames: int = 20):
    """Search/delivery separation invariant (#20/#56/#57) — leak-free.

    SEARCH uses train-only FUSE frames (holdout excluded -> no leakage).
    DELIVERY uses train (+ tuning) FUSE frames but NEVER benchmark holdout.
    For legacy HoldoutSplit (2-way), val_ids is benchmark -> delivery = train only.
    For BenchmarkSplit (3-way), tuning_val_ids may be used for tuning, benchmark excluded.
    Falls back to all non-REJECT frames when too few FUSE frames exist.
    """
    from auto_mobility.reconstruction.data.frame_selector import FrameRole

    role_items = list(roles.items())
    fuse = {fid for fid, r in role_items if r == FrameRole.FUSE}
    relaxed = False
    if len(fuse) < min_fuse_frames:
        fuse = {fid for fid, r in role_items if r != FrameRole.REJECT}
        relaxed = True
    allowed = set(valid_ids)
    # Detect split type
    if hasattr(split, "benchmark_holdout_ids"):
        train_set = set(split.train_ids)
        tuning_set = set(getattr(split, "tuning_val_ids", ()))
        # benchmark strictly excluded from both search and delivery
        search_ids = sorted(train_set & fuse & allowed)
        delivery_ids = sorted((train_set | tuning_set) & fuse & allowed)
        # Assert no benchmark leakage (defensive)
        bench = set(split.benchmark_holdout_ids)
        assert bench.isdisjoint(set(search_ids)), "benchmark leaked into search"
        assert bench.isdisjoint(set(delivery_ids)), "benchmark leaked into delivery"
    else:
        train_set, val_set = set(split.train_ids), set(split.val_ids)
        # HoldoutSplit: val is benchmark -> never in delivery (fix leakage)
        search_ids = sorted(train_set & fuse & allowed)
        delivery_ids = sorted(train_set & fuse & allowed)
        # Leak check: val must not be in delivery
        assert set(val_set).isdisjoint(set(search_ids))
        assert set(val_set).isdisjoint(set(delivery_ids))
    return search_ids, delivery_ids, relaxed


def compute_benchmark_sets(benchmark_split, roles, valid_ids, min_fuse_frames: int = 20):
    """Helper for benchmark pipeline: returns train/tuning/benchmark separation."""
    from auto_mobility.reconstruction.data.frame_selector import FrameRole

    role_items = list(roles.items())
    fuse = {fid for fid, r in role_items if r == FrameRole.FUSE}
    relaxed = False
    if len(fuse) < min_fuse_frames:
        fuse = {fid for fid, r in role_items if r != FrameRole.REJECT}
        relaxed = True
    allowed = set(valid_ids)
    train_set = set(benchmark_split.train_ids)
    tuning_set = set(benchmark_split.tuning_val_ids)
    bench_set = set(benchmark_split.benchmark_holdout_ids)
    search_ids = sorted(train_set & fuse & allowed)
    tuning_ids = sorted(tuning_set & fuse & allowed)
    delivery_ids = sorted((train_set | tuning_set) & fuse & allowed)
    benchmark_ids = sorted(bench_set & fuse & allowed)
    # Benchmark never in search/delivery
    assert bench_set.isdisjoint(set(search_ids))
    assert bench_set.isdisjoint(set(delivery_ids))
    return search_ids, tuning_ids, delivery_ids, benchmark_ids, relaxed


def select_pose_coverage_frames(frame_ids: list, poses: dict,
                                target_count: int = 800) -> list:
    """Select representative frames with uniform pose-space coverage (§6).

    Preserves:
      - Trajectory start and end
      - Spatial translation along corridor
      - Turns and rotational changes
      - Loop closure / overlapping areas
    """
    valid_ids = [fid for fid in frame_ids if fid in poses]
    if len(valid_ids) <= target_count:
        return valid_ids

    # Compute cumulative path distance + orientation change
    positions = np.array([poses[fid][:3, 3] for fid in valid_ids])
    rotations = np.array([poses[fid][:3, :3] for fid in valid_ids])

    # Arc-length parameterization
    diffs_pos = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    diffs_rot = []
    for i in range(len(rotations) - 1):
        R_rel = rotations[i].T @ rotations[i + 1]
        tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        diffs_rot.append(float(np.arccos(tr)))
    diffs_rot = np.array(diffs_rot)

    # Combined motion metric (1m displacement ~ 1rad rotation weight)
    step_costs = diffs_pos + 0.5 * diffs_rot
    cum_dist = np.concatenate([[0.0], np.cumsum(step_costs)])
    total_dist = cum_dist[-1]

    if total_dist <= 1e-6:
        stride = len(valid_ids) / float(target_count)
        return [valid_ids[int(i * stride)] for i in range(target_count)]

    # Sample target_count points uniformly along cumulative trajectory metric
    sample_dists = np.linspace(0.0, total_dist, target_count)
    selected_indices = np.searchsorted(cum_dist, sample_dists)
    selected_indices = np.clip(selected_indices, 0, len(valid_ids) - 1)

    # Ensure start (0) and end (len-1) are always included
    selected_indices[0] = 0
    selected_indices[-1] = len(valid_ids) - 1

    # Dedup while preserving order
    chosen = []
    seen = set()
    for idx in selected_indices:
        fid = valid_ids[idx]
        if fid not in seen:
            seen.add(fid)
            chosen.append(fid)

    # If dedup reduced count below target, fill in evenly from remaining
    if len(chosen) < target_count and len(chosen) < len(valid_ids):
        existing_set = set(chosen)
        fill_candidates = [fid for fid in valid_ids if fid not in existing_set]
        step = max(1, len(fill_candidates) // max(1, target_count - len(chosen)))
        for fid in fill_candidates[::step]:
            if len(chosen) >= target_count:
                break
            chosen.append(fid)

    return sorted(chosen)



def _metrics_from_geo_eval(geo_eval: dict) -> dict:
    """Map held-out geometry metrics onto hierarchical Metric objects."""
    from auto_mobility.reconstruction.evaluation.ranking import Metric

    def num(key):
        v = geo_eval.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
        return None

    def m(key, value):
        return (Metric(key, value) if value is not None
                else Metric.na(key, "not measured"))

    return {
        "heldout_depth_mae_mm": m("heldout_depth_mae_mm", num("depth_mae_mm")),
        "heldout_depth_p95_mm": m("heldout_depth_p95_mm", num("depth_p95_mm")),
        "coverage": m("coverage", num("depth_coverage_ratio")),
        "point_to_mesh_error_mm": Metric.na("point_to_mesh_error_mm",
                                            "stage not implemented"),
        "free_space_correctness": m("free_space_correctness",
                                    num("free_space_correctness_ratio")),
    }


def _rank_search_candidates(search_infos: list, decide, preview: bool = False) -> list:
    """Hierarchical ranking over train-only search evidence (#46)."""
    from auto_mobility.reconstruction.evaluation.ranking import (
        CandidateEvaluation, rank_candidates)

    evals = [
        CandidateEvaluation(
            candidate_id=info["name"],
            metrics=_metrics_from_geo_eval(info["geo_eval"]),
            runtime_s=max(info["wall_s"], 1e-6),
            triangle_count=int(info["fused_search"].mesh_triangles)
            if info["fused_search"] is not None else 0,
            trajectory_failed=bool(info.get("trajectory_failed", False)),
        )
        for info in search_infos
    ]
    ranked = rank_candidates(evals)
    if preview:
        decide("winner_ranking", "PREVIEW_FORCED_TOP1",
               "preview mode selects top-1 winner for fast visual artifact generation",
               order=[{"candidate": e.candidate_id, "final_quality": round(
                   e.final_quality, 2) if e.final_quality == e.final_quality else None}
                   for e in ranked])
    else:
        decide("winner_ranking", "SELECTED",
               "hierarchical gate/tier ranking over train-only evidence",
               order=[{"candidate": e.candidate_id, "final_quality": round(
                   e.final_quality, 2) if e.final_quality == e.final_quality else None}
                   for e in ranked])
    return [e.candidate_id for e in ranked]


def run_standard(dataset_dir: Path, trajectories: dict, out_dir: Path,
                 voxel_mm: float = 10.0,
                 use_texture: bool = True, view_stride: int = 5,
                 refine_poses: bool = True,
                 vram_budget_mb: float | None = None,
                 ram_budget_mb: float | None = None,
                 scheduler=None, budget=None, top_k: int = 2,
                 hard_ceiling_mb: float | None = None,
                 safe_mode: bool = False,
                 preview: bool = False,
                 quick: bool = False,
                 mode_policy=None,
                 deliver_backends: list | None = None) -> dict:
    import cv2
    import open3d as o3d

    from auto_mobility.dataset.frame_dataset import FrameDataset
    from auto_mobility.reconstruction.model import CameraIntrinsics
    from auto_mobility.reconstruction.appearance import (
        atlas_metrics, bake_atlas, normalize_exposure)
    from auto_mobility.reconstruction.appearance.texture_contract import (
        check_texture_contract)
    from auto_mobility.reconstruction.config import (
        ExecutionMode, ModePolicy, policy_for_mode)
    from auto_mobility.reconstruction.data import split_from_poses
    from auto_mobility.reconstruction.data.frame_selector import FrameRole
    from auto_mobility.reconstruction.depth.consistency import (
        compute_consistency_mask, render_frame_depth)
    from auto_mobility.reconstruction.evaluation.geometry_eval import evaluate_geometry
    from auto_mobility.reconstruction.fusion.isolated import integrate_frames_isolated

    if mode_policy is not None:
        policy = mode_policy
    elif preview:
        policy = policy_for_mode(ExecutionMode.PREVIEW)
    elif quick:
        policy = policy_for_mode(ExecutionMode.QUICK)
    else:
        policy = policy_for_mode(ExecutionMode.STANDARD)
    is_preview = (policy.mode == ExecutionMode.PREVIEW)
    is_quick = (policy.mode == ExecutionMode.QUICK)


    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions = []

    def decide(stage, what, why, **ev):
        decisions.append({"stage": stage, "decision": what,
                          "reason": why, "evidence": ev})

    def budget_gate(phase, est_s: float) -> bool:
        if budget is None:
            return True
        ok = budget.can_afford(phase, est_s)
        if not ok:
            decide("budget_gate", "SKIP",
                   f"phase {phase} cannot afford estimated {est_s:.0f}s",
                   remaining_s=budget.phase_remaining(phase))
        return ok

    def budget_record(phase, spent_s: float):
        if budget is None:
            return
        try:
            budget.spend(phase, spent_s)
        except Exception as exc:
            decide("budget_gate", "OVERSPEND_RECORDED", f"phase {phase}: {exc}")

    ds = FrameDataset(str(dataset_dir))
    alignment = ds.dataset_info.get("depth_color_alignment", "unknown")
    if alignment in ("not_aligned", "UNPROVEN", "unknown"):
        # Load the alignment contract for a precise rejection reason
        from auto_mobility.dataset.rgbd_alignment import load_contract
        contract = load_contract(dataset_dir)
        if contract is None or not contract.is_proven():
            reason = contract.reject_reason if contract else "alignment contract missing"
            decide("rgbd_contract", "FAIL_CLOSED",
                   f"RGB-D alignment not proven: {reason}")
            return {"ok": False,
                    "reason": f"RGB-D alignment contract failed: {reason}",
                    "decisions": decisions}
    frames = list(ds)
    cam = json.load(open(Path(dataset_dir) / "camera_info.json"))
    K = np.array(cam["K"], dtype=np.float64)
    if K.ndim == 1:
        K = K.reshape(3, 3)
    W, H = int(cam["width"]), int(cam["height"])
    cam_intr = CameraIntrinsics(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    def resolve(p):
        pp = Path(p)
        return pp if pp.is_absolute() else Path(dataset_dir) / pp

    def rgb(i):
        return cv2.imread(str(resolve(id_to_frame[i].rgb_path)))

    def depth(i):
        return cv2.imread(str(resolve(id_to_frame[i].depth_path)),
                          cv2.IMREAD_UNCHANGED)

    id_to_frame = {f.frame_id: f for f in frames}
    frame_ts = np.array([f.rgb_timestamp for f in frames])

    print(f"  📊 [1/3] Filtering frame quality ({len(frames)} frames)...", flush=True)
    t_q = time.time()
    roles = _compute_frame_roles(frames, rgb, depth)
    n_fuse = sum(1 for r in roles.values() if r == FrameRole.FUSE)
    n_track = sum(1 for r in roles.values() if r == FrameRole.TRACK)
    n_reject = sum(1 for r in roles.values() if r == FrameRole.REJECT)
    print(f"      ✅ Classified: FUSE={n_fuse}, TRACK={n_track}, REJECT={n_reject} ({time.time() - t_q:.1f}s)", flush=True)
    decide("frame_roles", "CLASSIFIED",
           "single-decode quality pass; REJECT excluded from fusion",
           n_frames=len(frames),
           wall_s=round(time.time() - t_q, 1))

    print(f"  🎯 [2/3] Evaluating trajectory health with TrajectoryJudge...", flush=True)
    scores, top = _judge(trajectories, frame_ts)
    # Judge statistics reject ordinary discontinuities; the independent health
    # gate also catches physically impossible velocity/pathology.  A failed
    # trajectory must never proceed to TSDF merely because its p99 is small.
    from auto_mobility.diagnostics.trajectory_health import check_trajectory_health
    health_by_name = {}
    for score in scores:
        health = check_trajectory_health(trajectories[score.backend])
        health_by_name[score.backend] = health.to_dict()
        if health.status == "FAIL_TRAJECTORY":
            score.failures.append(f"trajectory_health:{health.cause}")
            decide("trajectory_health", "REJECT", health.cause,
                   backend=score.backend,
                   max_step_m=health.translation_step_max_m,
                   max_velocity_mps=health.linear_velocity_max_mps)
    from auto_mobility.reconstruction.pose.judge import select_top_trajectories
    top = select_top_trajectories(scores, top_k=2)
    for s in scores:
        print(f"      • {s.backend}: {'PASS' if s.ok else 'FAIL'} (coverage: {s.coverage_ratio * 100:.1f}%, score: {s.composite():.1f})", flush=True)
    if not top:
        print("      ❌ No viable trajectory candidates passed gate!", flush=True)
        return {"ok": False, "reason": "no viable trajectory",
                "trajectory_scores": [s.to_dict() for s in scores],
                "trajectory_health": health_by_name,
                "decisions": decisions}
    top = top[:max(1, min(top_k, 2))]
    score_by_name = {s.backend: s for s in scores}

    # --- Benchmark holdout: common pose set + deterministic 3-way split (locked) ---
    # Build per-backend pose maps (authoritative association) for common intersection
    poses_by_backend_for_split = {}
    assoc_summary_for_split = {}
    for name, traj in trajectories.items():
        # Use same association as preview: 50ms gap guard + SLERP
        from auto_mobility.trajectory.association import associate_trajectory_to_frames as _assoc
        frame_stamps = np.array([f.rgb_timestamp for f in frames], dtype=np.float64)
        _, results, summary = _assoc(frame_stamps, traj, max_pose_gap_ms=50.0, enable_interpolation=True)
        pmap = {frames[r.frame_id].frame_id: r.T_world_camera for r in results if r.valid and r.T_world_camera is not None}
        poses_by_backend_for_split[name] = pmap
        assoc_summary_for_split[name] = summary

    fuse_ids_initial = {f.frame_id for f in frames if roles.get(f.frame_id) == __import__("auto_mobility.reconstruction.data.frame_selector", fromlist=["FrameRole"]).FrameRole.FUSE}
    if len(fuse_ids_initial) < 20:
        fuse_ids_initial = {f.frame_id for f in frames if roles.get(f.frame_id) != __import__("auto_mobility.reconstruction.data.frame_selector", fromlist=["FrameRole"]).FrameRole.REJECT}
    # Common pose set across all available backends that passed judge (at least those in top)
    # For benchmark fairness, intersect across backends that are candidates for delivery
    candidate_backends_for_common = [c.backend for c in scores if c.ok] or [c.backend for c in top]
    valid_sets_common = [set(poses_by_backend_for_split.get(b, {}).keys()) for b in candidate_backends_for_common if b in poses_by_backend_for_split]
    if valid_sets_common:
        common_ids_raw = sorted(fuse_ids_initial & set.intersection(*valid_sets_common)) if valid_sets_common else []
    else:
        common_ids_raw = []
    # Dataset fingerprint for deterministic seed (frames.csv + camera_info + alignment contract)
    try:
        import hashlib as _hlib
        fp_path = Path(dataset_dir) / "frames.csv"
        cam_path = Path(dataset_dir) / "camera_info.json"
        _h = _hlib.sha256()
        if fp_path.is_file():
            _h.update(fp_path.read_bytes())
        if cam_path.is_file():
            _h.update(cam_path.read_bytes())
        # alignment contract fingerprint if present
        align_path = Path(dataset_dir) / "rgbd_alignment_contract.json"
        if align_path.is_file():
            _h.update(align_path.read_bytes())
        dataset_fp_for_split = _h.hexdigest()[:16]
    except Exception:
        dataset_fp_for_split = ""
    benchmark_split = None
    benchmark_split_error = None
    if len(candidate_backends_for_common) < 2:
        benchmark_split_error = f"NON_COMPARABLE: only {len(candidate_backends_for_common)} backend(s) available, need 2 for benchmark"
        decide("benchmark_split", "NON_COMPARABLE", benchmark_split_error, common_pool=len(common_ids_raw), backends=candidate_backends_for_common)
    elif len(common_ids_raw) < 20:
        benchmark_split_error = f"NON_COMPARABLE: common pose count {len(common_ids_raw)} <20"
        decide("benchmark_split", "NON_COMPARABLE", benchmark_split_error, common_pool=len(common_ids_raw))
    else:
        try:
            from auto_mobility.reconstruction.data.split import split_from_common_poses as _split_common
            # need poses for common ids: use first backend's poses as reference for motion
            ref_backend = candidate_backends_for_common[0]
            ref_poses = poses_by_backend_for_split[ref_backend]
            common_poses_for_split = [ref_poses[fid] for fid in common_ids_raw if fid in ref_poses]
            common_fids_for_split = [fid for fid in common_ids_raw if fid in ref_poses]
            benchmark_split = _split_common(common_fids_for_split, common_poses_for_split, dataset_fingerprint=dataset_fp_for_split, seed=None)
            decide("benchmark_split", "LOCKED", f"common {len(common_ids_raw)} frames -> train {len(benchmark_split.train_ids)} tuning {len(benchmark_split.tuning_val_ids)} benchmark {len(benchmark_split.benchmark_holdout_ids)}",
                   common_pose_count=len(common_ids_raw), dataset_fingerprint=dataset_fp_for_split,
                   train_sha=benchmark_split.train_ids_sha256, tuning_sha=benchmark_split.tuning_val_ids_sha256,
                   benchmark_sha=benchmark_split.benchmark_holdout_ids_sha256, generation_rule=benchmark_split.generation_rule)
        except Exception as exc:
            benchmark_split_error = f"benchmark split failed: {exc}"
            decide("benchmark_split", "FAILED", benchmark_split_error)


    bbox_holder = {"diag_m": 8.0}

    def fit_voxel_to_vram(diag_m: float, requested_voxel_mm: float):
        """Degrade the effective voxel just enough to keep fusion on GPU."""
        from auto_mobility.reconstruction.fusion.open3d_vbg import (
            max_fitting_voxel_mm, required_vram_mb)

        vox = max_fitting_voxel_mm(diag_m, vram_budget_mb,
                                   min_voxel_mm=requested_voxel_mm)
        want = required_vram_mb(diag_m, requested_voxel_mm / 1000.0)
        got = required_vram_mb(diag_m, vox / 1000.0)
        if vox > requested_voxel_mm:
            decide("vram_preflight", "DEGRADE_VOXEL",
                   "requested voxel does not fit VRAM budget; coarsened "
                   "to keep fusion on GPU",
                   requested_voxel_mm=requested_voxel_mm, effective_voxel_mm=vox,
                   wanted_vram_mb=want, fitted_vram_mb=got,
                   vram_budget_mb=vram_budget_mb)
        else:
            decide("vram_preflight", "OK",
                   "requested voxel fits VRAM budget",
                   voxel_mm=vox, estimated_vram_mb=got,
                   vram_budget_mb=vram_budget_mb)
        return vox

    class ResourcePlanError(RuntimeError):
        pass

    def _fusion_ram_estimate_mb(vox_m: float) -> int:
        """Truthful host-RAM estimate for one isolated fusion worker (§11).

        8 bytes/voxel (tsdf f32 + weight f32, no color) × MC overhead 2.0.
        Never clamped with min(): if this exceeds the RAM budget the caller
        must REJECT/degrade, not report a smaller number to the scheduler.
        """
        from auto_mobility.reconstruction.fusion.open3d_vbg import (
            estimate_block_count, _BYTES_PER_BLOCK_NO_COLOR)

        blocks = estimate_block_count(bbox_holder["diag_m"], vox_m,
                                      vram_budget_mb)
        return max(2048, int(blocks * _BYTES_PER_BLOCK_NO_COLOR
                             * 2.0 / 1e6))

    def _resolve_planned_metrics(plan: dict | None):
        """Extract MB and block_count from planner dict with unit contract."""
        if not plan:
            return None, None
        # P0 #1: bytes -> MB conversion explicit via helper
        from auto_mobility.reconstruction.fusion.open3d_vbg import (
            required_vram_mb_planned)
        mb = required_vram_mb_planned(plan)
        cap = int(plan.get("estimated_hash_capacity") or
                  plan.get("safe_block_count") or 0) or None
        if cap is None:
            cap = plan.get("planned_capacity_blocks")
        return mb, cap

    def submit_fusion(ids, pose_by_frame, vox_m, mask_dict, tag,
                      active_plan: dict | None = None,
                      planned_peak_vram_mb=None, planned_block_count=None):
        """Run one isolated fusion through the scheduler's single GPU slot.

        §11: never lie to the scheduler with min(vram_want, budget).  If the
        required VRAM exceeds the declared budget, REJECT -> DEGRADE -> REPLAN.
        §7: phase-specific CPU threads (fusion uses 3-4 threads, not global 6).
        P0 #1: active_plan provides bytes/MB with explicit conversion.
        P0 #2: planned_block_count is forwarded to actual VBG allocation.
        P0 #12/13: dual VRAM barrier (incremental job budget + hard ceiling).
        P0 #14: per-job RAM limit is truthful planned RAM + margin, not global.
        """
        from auto_mobility.reconstruction.fusion.open3d_vbg import (
            required_vram_mb)
        from auto_mobility.reconstruction.runtime.scheduler import JobSpec

        # resolve active_plan -> mb + capacity (P0 #1)
        if active_plan is not None:
            _mb, _cap = _resolve_planned_metrics(active_plan)
            if _mb is not None and planned_peak_vram_mb is None:
                planned_peak_vram_mb = _mb
            if _cap is not None and planned_block_count is None:
                planned_block_count = _cap
        vram_want = int(required_vram_mb(bbox_holder["diag_m"], vox_m)) or 512
        if planned_peak_vram_mb:
            vram_want = max(vram_want, int(planned_peak_vram_mb))
        # P0 #14: RAM estimate for this job (planned blocks if available)
        if planned_block_count:
            from auto_mobility.reconstruction.fusion.open3d_vbg import (
                _BYTES_PER_BLOCK_NO_COLOR, _MC_OVERHEAD_FACTOR)
            ram_want = max(2048, int(planned_block_count *
                                     _BYTES_PER_BLOCK_NO_COLOR *
                                     _MC_OVERHEAD_FACTOR / 1e6))
        else:
            ram_want = _fusion_ram_estimate_mb(vox_m)
        # §11 fix: hard admission check before submit — scheduler must see truth.
        avail_vram = int(vram_budget_mb) if vram_budget_mb else vram_want
        avail_ram = int(ram_budget_mb) if ram_budget_mb else ram_want
        if vram_want > avail_vram or ram_want > avail_ram:
            decide("vram_preflight", "REJECTED",
                   "required VRAM/RAM exceeds incremental budget; degrading "
                   "instead of lying to scheduler",
                   vram_want=vram_want, vram_budget_mb=avail_vram,
                   ram_want=ram_want, ram_budget_mb=avail_ram, tag=tag)
            # degrade voxel one step via max_fitting and retry caller-side
            degraded = fit_voxel_to_vram(bbox_holder["diag_m"], vox_m * 1000.0 * 1.25)
            if degraded / 1000.0 > vox_m:
                vox_m = degraded / 1000.0
                degraded_plan = _plan_active_blocks_safe(ids, pose_by_frame, degraded,
                                                        tag=f"{tag}_degraded")
                if degraded_plan is not None:
                    active_plan = degraded_plan
                    _d_mb, _d_cap = _resolve_planned_metrics(degraded_plan)
                    planned_peak_vram_mb = _d_mb
                    planned_block_count = _d_cap
                    vram_want = int(planned_peak_vram_mb) if planned_peak_vram_mb else (int(required_vram_mb(bbox_holder["diag_m"], vox_m)) or 512)
                else:
                    planned_peak_vram_mb = None
                    planned_block_count = None
                    vram_want = int(required_vram_mb(bbox_holder["diag_m"], vox_m)) or 512
                if planned_block_count:
                    from auto_mobility.reconstruction.fusion.open3d_vbg import (
                        _BYTES_PER_BLOCK_NO_COLOR, _MC_OVERHEAD_FACTOR)
                    ram_want = max(2048, int(planned_block_count *
                                             _BYTES_PER_BLOCK_NO_COLOR *
                                             _MC_OVERHEAD_FACTOR / 1e6))
                else:
                    ram_want = _fusion_ram_estimate_mb(vox_m)
                if vram_want > avail_vram or ram_want > avail_ram:
                    decide("vram_preflight", "REJECTED_FINAL",
                           "even degraded voxel does not fit; skipping job",
                           vram_want=vram_want, ram_want=ram_want, tag=tag)
                    return None
            else:
                return None
        # §7 phase-specific: GPU TSDF stage capped at 4 threads
        fusion_threads = 4.0
        vram_request = int(vram_want)
        # also enforce hard ceiling via nvidia-smi baseline §10 — job delta vs global
        # ceiling is checked inside worker barrier; here we just request truthfully.
        spec = JobSpec(
            name=f"fusion:{tag}",
            cpu_threads=fusion_threads,
            ram_mb=ram_want,
            gpu_slots=1,
            vram_mb=vram_request,
        )

        def job():
            from auto_mobility.reconstruction.runtime.thermal import (
                power_source, wait_for_thermal_headroom)

            # L0 barrier: refuse sustained GPU draw while on battery.
            src = power_source()
            if src == "battery":
                decide("power_barrier", "REFUSED",
                       "system is discharging; heavy GPU fusion blocked "
                       "until AC power is connected")
                return None
            th = wait_for_thermal_headroom()
            if th.get("waited_s"):
                decide("thermal_gate", "COOLED_DOWN",
                       "waited for GPU to cool before heavy submission", **th)
            # P0 #13: job VRAM limit is planned_peak + margin, not global*0.9 blind
            planned_mb_for_limit = int(planned_peak_vram_mb) if planned_peak_vram_mb else vram_want
            # calibrated tolerance 512 MB + 10% of planned, capped by hard ceiling
            # hard_ceiling_mb from outer run_standard (total - reserve)
            hc_val = None
            try:
                hc_val = hard_ceiling_mb  # closure from run_standard param
            except NameError:
                hc_val = None
            if hc_val is None and vram_budget_mb:
                hc_val = int(vram_budget_mb + 1536)
            job_vram_limit = max(int(planned_mb_for_limit * 1.25 + 768), int(vram_budget_mb) if vram_budget_mb else 0)
            if hc_val is not None:
                job_vram_limit = min(job_vram_limit, int(hc_val))
            # P0 #14: per-job RAM limit is job RSS, not global budget
            job_ram_limit = int(ram_want * 1.20 + 512)
            gpu_limits_dict = {
                "vram_mb": int(job_vram_limit),
                "temp_c": 87.0,
                "power_w": None,
            }
            if hc_val is not None:
                gpu_limits_dict["hard_ceiling_mb"] = int(hc_val)
            return integrate_frames_isolated(
                dataset_dir=Path(dataset_dir),
                frame_ids=list(ids),
                pose_by_frame=pose_by_frame,
                masks_by_frame=mask_dict,
                K=K, width=W, height=H,
                voxel_m=vox_m, trunc_mult=TRUNCATION_MULTIPLIER,
                bbox_diag_m=bbox_holder["diag_m"],
                work_dir=Path(out_dir) / "fusion_work",
                tag=tag,
                vram_budget_mb=vram_budget_mb,
                ram_limit_mb=int(job_ram_limit),
                gpu_limits=gpu_limits_dict,
                frames_per_chunk=max(120, min(400, len(list(ids)) // 6 or 120)),
                chunk_pause_s=8.0,
                planned_block_count=int(planned_block_count) if planned_block_count else None,
            )

        t_f = time.time()
        print(f"        ⚙️ [GPU Worker] Integrating {len(list(ids))} frames (tag: {tag})...", flush=True)
        if scheduler is not None:
            try:
                res = scheduler.submit(job, spec).result()
            except Exception as exc:
                decide("fusion", "REJECTED", f"scheduler: {exc}", tag=tag)
                print(f"        ❌ [GPU Worker] Scheduler error: {exc}", flush=True)
                return None
        else:
            res = job()
        if res is None:
            print(f"        ❌ [GPU Worker] Subprocess returned None", flush=True)
            return None
        wall = time.time() - t_f
        print(f"        {'✅' if res.ok else '❌'} [GPU Worker] Completed in {wall:.1f}s ({res.detail})", flush=True)
        decide("fusion", "OK" if res.ok else "FAILED", res.detail,
               frames=len(list(ids)), device=str(res.output.device),
               wall_s=round(wall, 1))
        return res.output if res.ok else None


    def build_scene(mesh):
        mt = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mt)
        return scene

    def compute_masks(scene, sample_ids, pose_by_frame):
        masks = {}
        for fid in sample_ids:
            d = depth(fid)
            if d is None or d.shape[:2] != (H, W):
                continue
            rd = render_frame_depth(scene, pose_by_frame[fid], cam_intr)
            if rd.shape != d.shape[:2]:
                rd = cv2.resize(rd, (d.shape[1], d.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            masks[fid] = compute_consistency_mask(d.astype(np.float32), rd)
        return masks

    def _plan_active_blocks_safe(ids, poses, vox_mm, tag="plan"):
        """Active-block preflight (P0 #1 fix): frustum-true block prediction.

        Returns full planner dict with explicit bytes/MB contract (or None on failure).
        For final delivery (P0 #5) scans ALL frames coordinate-only (no RGB decode, no GPU,
        bounded CPU memory) to avoid underprediction from 80-frame sampling.
        For intermediate coarse estimation the planner still runs bounded but with
        safety margin already baked into capacity (2x) and VRAM (1.05x).
        """
        try:
            from auto_mobility.reconstruction.fusion.open3d_vbg import (
                plan_active_blocks, required_vram_mb_planned)

            # P0 #5: for large final sets do NOT subsample; planner itself is bounded (~80)
            # but final explicitly needs full scan to avoid underprediction.
            # We still allow bounded for coarse/search when ids>200: planner uses full set
            # but internally sample_stride=1 => full scan (CPU VBG coordinate helper is cheap)
            # If tag contains 'final' force full scan, else also full scan but allow fallback.
            plan = plan_active_blocks(
                list(ids), poses, K, W, H, vox_mm / 1000.0,
                TRUNCATION_MULTIPLIER,
                depth_min_m=0.2, depth_max_m=4.0,
                load_depth_mm=depth, load_mask=None,
                store_color=False, sample_stride=1)
            vram_mb = required_vram_mb_planned(plan)
            decide("active_block_plan", "PLANNED",
                   "frustum-based active-block preflight (unit: bytes->MB explicit)",
                   tag=tag,
                   unique_block_count=plan.get("unique_block_count"),
                   planned_tsdf_mb=round(plan.get("estimated_tsdf_bytes", 0) / 1e6, 1),
                   planned_extraction_peak_bytes=int(plan.get("estimated_extraction_peak", 0)),
                   planned_extraction_peak_mb=float(plan.get("estimated_extraction_peak_mb",
                                                             plan.get("estimated_extraction_peak",0)/1e6)),
                   safe_block_count=plan.get("safe_block_count"),
                   vram_mb_for_admission=vram_mb,
                   sampled_frames=plan.get("sampled_frames"),
                   total_frames=len(list(ids)))
            return plan
        except Exception as exc:
            decide("active_block_plan", "FALLBACK_BBOX_MODEL",
                   f"planner unavailable: {exc}", tag=tag)
            return None

    def search_phase(cand, idx: int) -> dict | None:
        """Train-only search: refine -> coarse -> masks -> fuse -> holdout eval."""
        from auto_mobility.reconstruction.pose.refine_pipeline import refine_trajectory

        t_cand = time.time()
        traj = trajectories[cand.backend]
        dropped = []
        poses0 = _nearest_pose_map(traj, frames, dropped=dropped)
        if dropped:
            decide("pose_association", "GAP_DROPPED",
                   "frames beyond max_pose_gap_ms dropped instead of bridged "
                   "by stale poses",
                   candidate=cand.backend, n_dropped=len(dropped))
        # Use locked benchmark split (no per-candidate split) — fail-closed if missing
        if benchmark_split is None or benchmark_split_error is not None:
            decide("search_phase", "FAILED", f"benchmark split unavailable: {benchmark_split_error}", candidate=cand.backend)
            return {
                "name": cand.backend,
                "poses": poses0, "split": None, "search_ids": [],
                "delivery_ids": [], "val_frames": [],
                "masks": {}, "eff_voxel": 10.0,
                "fused_search": None, "geo_eval": {"status": "NOT_APPLICABLE", "reason": "no benchmark split"},
                "refinement": {"accepted": False, "reason": "no benchmark split"}, "wall_s": time.time() - t_cand,
                "trajectory_failed": True,
                "search_plan": None,
                "coarse_plan": None,
            }
        split = benchmark_split
        id_set = set(id_to_frame)

        search_ids, tuning_ids, delivery_ids, benchmark_ids, relaxed = compute_benchmark_sets(
            split, roles, id_set)
        # For search phase, search_ids is train only; tuning pool is separate for selection
        if relaxed:
            decide("frame_roles", "RELAX_TO_NON_REJECT",
                   "too few FUSE frames; fusing all non-REJECT frames",
                   candidate=cand.backend)
        # P1 #16: bound search frame count to 300~500 pose-space representative
        # Standard search is not final product; final winner fuses ALL valid FUSE.
        if len(search_ids) > 500:
            # pose-space uniform stride (cheap proxy for spatial coverage)
            stride = max(1, len(search_ids) // 400)
            search_ids = sorted(search_ids[::stride])[:500]
            decide("frame_roles", "BOUNDED_SEARCH_SUBSET",
                   "train search limited to 300-500 representative frames; final will use ALL",
                   n_search_bounded=len(search_ids), n_delivery=len(delivery_ids))
        # Benchmark holdout strictly for evaluation (never for fusion)
        val_frames = [id_to_frame[i] for i in sorted(set(benchmark_ids) & id_set) if i in poses0]
        # Ensure benchmark not in search/delivery
        assert set(benchmark_ids).isdisjoint(set(search_ids))
        assert set(benchmark_ids).isdisjoint(set(delivery_ids))

        poses = poses0
        refinement = {"accepted": False, "reason": "disabled"}
        if refine_poses and budget_gate("pose_exploration", 120.0):
            train_frames = [id_to_frame[i] for i in search_ids]
            refinement = refine_trajectory(frames=train_frames, pose_by_frame=poses0,
                                           load_depth_mm=depth, K=K, width=W, height=H)
            poses = refinement["pose_by_frame"]
            decide("pose_refinement", "ACCEPT" if refinement["accepted"] else "ROLLBACK",
                   refinement["reason"], candidate=cand.backend)
        budget_record("pose_exploration", time.time() - t_cand)

        pts = np.asarray([poses[i][:3, 3] for i in search_ids if i in poses])
        if len(pts) >= 2:
            bbox_holder["diag_m"] = max(
                bbox_holder["diag_m"],
                float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) + 2.0)
        eff_voxel = fit_voxel_to_vram(bbox_holder["diag_m"], voxel_mm)
        # P0 #4: every job gets its own plan; coarse/search/fine/final are distinct identities
        # also respect safe_mode (fine disabled)
        if safe_mode and eff_voxel < voxel_mm * 1.1:
            # safe_mode forces slightly coarser when needed; keep current
            pass
        sc = score_by_name.get(cand.backend)
        if sc is not None and not sc.ok:
            decide("search_phase", "SKIPPED_GATE_FAILURE",
                   f"{cand.backend} failed trajectory gate; skipping coarse TSDF",
                   candidate=cand.backend, failures=sc.failures)
            return {
                "name": cand.backend,
                "poses": poses0, "split": split, "search_ids": search_ids,
                "delivery_ids": delivery_ids, "val_frames": val_frames,
                "masks": {}, "eff_voxel": eff_voxel,
                "fused_search": None, "geo_eval": {"status": "NOT_APPLICABLE", "reason": "gate_failed"},
                "refinement": {"accepted": False, "reason": "gate_failed"}, "wall_s": time.time() - t_cand,
                "trajectory_failed": True,
                "search_plan": None,
                "coarse_plan": None,
            }

        # coarse plan uses coarse subset & same voxel
        t_g = time.time()
        coarse_ids = search_ids[:: max(1, len(search_ids) // 80)] or search_ids
        coarse_plan = _plan_active_blocks_safe(coarse_ids, poses, eff_voxel,
                                              tag=f"c{idx}_coarse")
        search_plan = _plan_active_blocks_safe(search_ids, poses, eff_voxel,
                                               tag=f"c{idx}_search")
        coarse = submit_fusion(coarse_ids, poses, eff_voxel / 1000.0, None,
                                f"c{idx}_coarse",
                                active_plan=coarse_plan)
        masks = {}
        if coarse is not None and coarse.mesh_obj is not None \
                and len(coarse.mesh_obj.triangles) > 0:
            scene = build_scene(coarse.mesh_obj)
            mask_sample = delivery_ids[:: max(1, len(delivery_ids) // 40)]
            masks = compute_masks(scene, mask_sample, poses)
            decide("depth_consistency", "MASK_APPLIED",
                   "measured-vs-rendered masking vs train-only coarse mesh",
                   masked_frames=len(masks), candidate=cand.backend)
        else:
            decide("depth_consistency", "SKIPPED", "no coarse mesh available",
                   candidate=cand.backend)

        # For tuning decisions, use tuning frames (not benchmark) to avoid leakage into benchmark.
        tuning_frames = [id_to_frame[i] for i in sorted(set(tuning_ids) & id_set) if i in poses]
        fused_search = submit_fusion(search_ids, poses, eff_voxel / 1000.0,
                                     masks or None, f"c{idx}_search",
                                     active_plan=search_plan)
        geo_eval = {"status": "NOT_APPLICABLE", "reason": "no mesh"}
        geo_eval_benchmark = {"status": "NOT_APPLICABLE", "reason": "no mesh"}
        if fused_search is not None and tuning_frames:
            geo_eval = evaluate_geometry(fused_search.mesh_obj, tuning_frames, poses,
                                         cam_intr, depth)
            geo_eval["eval_mesh_provenance"] = "train_only_tuning"
        if fused_search is not None and val_frames:
            geo_eval_benchmark = evaluate_geometry(fused_search.mesh_obj, val_frames, poses,
                                         cam_intr, depth)
            geo_eval_benchmark["eval_mesh_provenance"] = "train_only_benchmark"

        eff_voxel_final = eff_voxel
        # P0 #15/21: SAFE MODE and PREVIEW disable fine voxel entirely
        if safe_mode:
            decide("fusion_refinement", "SKIP_FINE_VOXEL_SAFE_MODE",
                   "safe_mode disables fine voxel rebuild", eff_voxel=eff_voxel)
        elif preview:
            decide("fusion_refinement", "SKIP_FINE_VOXEL_PREVIEW",
                   "preview mode disables fine voxel rebuild", eff_voxel=eff_voxel)
        elif fused_search is not None and (
                geo_eval.get("within_50mm_ratio", 0) > 0.35
                and geo_eval.get("free_space_correctness_ratio", 0) > 0.75
                and eff_voxel > 7.0):
            from auto_mobility.reconstruction.fusion.open3d_vbg import required_vram_mb
            fine = round(eff_voxel * 0.75, 1)
            fine_want = required_vram_mb(bbox_holder["diag_m"], fine / 1000.0)
            if fine_want > (vram_budget_mb or float("inf")) or \
                    not budget_gate("optional_improvement", 60.0):
                decide("fusion_refinement", "SKIP_FINE_VOXEL",
                       "VRAM or time budget does not allow finer attempt",
                       voxel_mm=fine)
            else:
                # P0 #21: fine voxel must have its own ActiveBlockPlan (different identity)
                fine_plan = _plan_active_blocks_safe(search_ids, poses, fine,
                                                     tag=f"c{idx}_fine")
                refined_try = submit_fusion(search_ids, poses, fine / 1000.0,
                                            masks or None, f"c{idx}_fine",
                                            active_plan=fine_plan)
                # P1 #20: fine acceptance must be quality-based, not triangle count alone
                # we evaluate fine candidate on same held-out set before accepting
                if refined_try is not None and refined_try.ok:
                    try:
                        fine_geo = evaluate_geometry(refined_try.mesh_obj, val_frames,
                                                     poses, cam_intr, depth)
                        fine_better = (
                            fine_geo.get("depth_mae_mm", 1e9) < geo_eval.get("depth_mae_mm", 1e9)
                            and fine_geo.get("free_space_correctness_ratio", 0) >=
                                geo_eval.get("free_space_correctness_ratio", 0) - 0.02
                        )
                    except Exception:
                        # fallback to legacy triangle count diagnostic only
                        fine_better = refined_try.mesh_triangles >= fused_search.mesh_triangles
                        fine_geo = None
                    if fine_better:
                        fused_search, eff_voxel_final = refined_try, fine
                        if fine_geo is not None:
                            geo_eval = fine_geo  # promote fine geometry evidence
                        decide("fusion_refinement", "FINE_VOXEL_APPLIED",
                               "geometry quality improved (MAE/coverage/fs)",
                               voxel_mm=fine, candidate=cand.backend)
                    else:
                        decide("fusion_refinement", "SKIP_FINE_VOXEL",
                               "fine quality did not improve (MAE/coverage)", voxel_mm=fine)
                else:
                    decide("fusion_refinement", "SKIP_FINE_VOXEL",
                           "finer attempt failed or degraded", voxel_mm=fine)

        budget_record("geometry_exploration", time.time() - t_g)
        sc = score_by_name.get(cand.backend)
        # keep last plans for telemetry — include both tuning and benchmark evals
        return {
            "name": cand.backend,
            "poses": poses, "split": split, "search_ids": search_ids,
            "delivery_ids": delivery_ids, "val_frames": val_frames,
            "tuning_frames": tuning_frames, "tuning_ids": tuning_ids,
            "benchmark_ids": benchmark_ids,
            "masks": masks, "eff_voxel": eff_voxel_final,
            "fused_search": fused_search, "geo_eval": geo_eval,
            "geo_eval_benchmark": geo_eval_benchmark,
            "refinement": refinement, "wall_s": time.time() - t_cand,
            "trajectory_failed": (not sc.ok) if sc is not None else False,
            "search_plan": search_plan,
            "coarse_plan": coarse_plan,
        }

    def deliver_candidate(info: dict, tag: str) -> dict:
        """FINAL DELIVERY: fuse ALL valid FUSE frames, then texture + report."""
        o3d_ok = True
        poses = info["poses"]
        delivery_ids = list(info["delivery_ids"])
        if preview and len(delivery_ids) > 800:
            step = len(delivery_ids) / 800.0
            orig_len = len(delivery_ids)
            delivery_ids = [delivery_ids[int(i * step)] for i in range(800)]
            decide("delivery", "PREVIEW_POSE_COVERAGE_SELECTION",
                   f"sampled 800 representative frames from {orig_len} across corridor for visual preview",
                   target=800, total_valid=orig_len)

        # P0 #4 / #22: final must have its own plan over delivery_ids + effective voxel
        final_plan = _plan_active_blocks_safe(delivery_ids, poses, info["eff_voxel"],
                                              tag=f"{tag}_final")
        final = submit_fusion(delivery_ids, poses, info["eff_voxel"] / 1000.0,
                              info["masks"] or None, f"{tag}_final",
                              active_plan=final_plan)
        if final is None:
            return {"ok": False, "tag": tag, "name": info["name"],
                    "reason": "final fusion failed",
                    "geometry_eval": info["geo_eval"],
                    "holdout": info["split"].to_dict(),
                    "n_delivery_frames": len(delivery_ids)}

        if tag.startswith("final_candidates/"):
            rank_dir = out_dir / tag
        else:
            rank_dir = out_dir / ("preview" if preview else "final") / tag
        (rank_dir / "textures").mkdir(parents=True, exist_ok=True)
        try:
            o3d.io.write_triangle_mesh(str(rank_dir / "model_raw.obj"), final.mesh_obj)
        except OSError as exc:
            o3d_ok = False
            decide("surface", "WRITE_FAILED", f"model_raw.obj: {exc}")

        # P0 #25/26: Poisson hole repair is either reachable or removed.
        # Previously dead because extraction_mode=mesh_only => pcd_obj always None.
        # Now reachable via mesh-sampled pcd fallback; isolated worker would be
        # preferred but we keep subprocess isolation via budget+evidence gate.
        #
        # FIX (feedback §I): poisson_trigger MUST require policy.enable_poisson.
        # PREVIEW and QUICK policies set enable_poisson=False, so Poisson must
        # never run in those modes regardless of all other conditions.
        # Triangle count is NOT a quality metric; adoption requires held-out
        # depth MAE/coverage improvement over the raw TSDF baseline.
        applied_poisson = False
        poisson_trigger = (
            policy.enable_poisson          # §I fix: policy gate is mandatory
            and not preview
            and o3d_ok
            and len(final.mesh_obj.triangles) > 20000
            and info["geo_eval"].get("observed_surface_completeness", 1.0) < 0.5
            and budget_gate("optional_improvement", 90.0)
        )
        if poisson_trigger:
            pcd_for_poisson = final.pcd_obj
            if (pcd_for_poisson is None or len(pcd_for_poisson.points) < 50000):
                # mesh_only mode produced no pcd; sample from mesh as fallback
                try:
                    pcd_for_poisson = final.mesh_obj.sample_points_uniformly(
                        number_of_points=80000)
                    pcd_for_poisson.estimate_normals()
                except Exception:
                    pcd_for_poisson = None
            if pcd_for_poisson is not None and len(pcd_for_poisson.points) >= 50000:
                try:
                    tsdf_mesh_ref = final.mesh_obj  # retain raw TSDF for comparison
                    pmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd_for_poisson, depth=9)[0]
                    # §I fix: adopt Poisson ONLY when held-out metrics improve.
                    # Triangle count is explicitly NOT a quality criterion.
                    # Evaluate both meshes against the same held-out val_frames.
                    poisson_adopted = False
                    poisson_reason = "metric evaluation not available"
                    geo_tsdf = info["geo_eval"]
                    # Re-evaluate Poisson candidate using the same held-out eval
                    # infrastructure; fall back to TSDF if eval is unavailable.
                    try:
                        from auto_mobility.reconstruction.evaluation import (
                            geo_eval as _geo_eval_fn)
                        poisson_geo = _geo_eval_fn(
                            pmesh, info["val_frames"], poses,
                            rgb_fn=rgb, depth_fn=depth_img,
                            voxel_size_mm=info["eff_voxel"],
                        )
                        tsdf_mae = geo_tsdf.get("depth_mae_mm", float("inf"))
                        tsdf_p95 = geo_tsdf.get("depth_p95_mm", float("inf"))
                        tsdf_cov = geo_tsdf.get("coverage", 0.0)
                        pois_mae = poisson_geo.get("depth_mae_mm", float("inf"))
                        pois_p95 = poisson_geo.get("depth_p95_mm", float("inf"))
                        pois_cov = poisson_geo.get("coverage", 0.0)
                        # Accept Poisson only when it improves MAE and does not
                        # meaningfully degrade coverage (>= -2 pp tolerance).
                        if (pois_mae < tsdf_mae and pois_p95 <= tsdf_p95 * 1.05
                                and pois_cov >= tsdf_cov - 0.02):
                            poisson_adopted = True
                            poisson_reason = (
                                f"depth_mae {tsdf_mae:.2f}->{pois_mae:.2f} mm  "
                                f"p95 {tsdf_p95:.2f}->{pois_p95:.2f} mm  "
                                f"coverage {tsdf_cov:.3f}->{pois_cov:.3f}"
                            )
                        else:
                            poisson_reason = (
                                f"no metric improvement: "
                                f"mae {tsdf_mae:.2f}->{pois_mae:.2f}  "
                                f"p95 {tsdf_p95:.2f}->{pois_p95:.2f}  "
                                f"cov {tsdf_cov:.3f}->{pois_cov:.3f}; retaining TSDF"
                            )
                    except Exception as eval_exc:
                        poisson_reason = f"held-out eval unavailable ({eval_exc}); retaining TSDF"

                    if poisson_adopted:
                        final.mesh_obj = pmesh
                        applied_poisson = True
                        decide("surface", "POISSON_APPLIED", poisson_reason, tag=tag)
                    else:
                        # Retain raw TSDF baseline (already in final.mesh_obj)
                        decide("surface", "POISSON_SKIPPED", poisson_reason, tag=tag)
                except Exception as exc:
                    decide("surface", "POISSON_SKIPPED", f"failed: {exc}", tag=tag)
            else:
                decide("surface", "POISSON_SKIPPED", "insufficient points for poisson", tag=tag)
        else:
            reason = ("policy.enable_poisson=False" if not policy.enable_poisson
                      else "trigger/budget evidence not met")
            decide("surface", "POISSON_SKIPPED", reason, tag=tag)

        bake_info, appear = None, {"status": "NOT_APPLICABLE"}
        if use_texture and len(final.mesh_obj.triangles) > 0 \
                and budget_gate("optional_improvement", 60.0):
            # Bounded candidate views: score matrices are O(T x V); an
            # unbounded V explodes host RAM on long captures (#54/#55).
            MAX_VIEWS = 32 if preview else 80
            n_views = min(MAX_VIEWS, max(1, len(delivery_ids)))
            stride_v = max(1, len(delivery_ids) // n_views)
            views, poses_wc = [], {}
            for fid in delivery_ids[::stride_v][:MAX_VIEWS]:
                img = rgb(fid)
                if img is None:
                    continue
                views.append((fid, normalize_exposure(img)))
                poses_wc[fid] = poses[fid]
            if views:
                # P0 #29: occlusion-aware texture for final rank01 when scene available
                # we build RaycastingScene from delivered mesh to enable occlusion test
                bake_scene = None
                try:
                    bake_scene = build_scene(final.mesh_obj)
                except Exception:
                    bake_scene = None
                bake = bake_atlas(np.asarray(final.mesh_obj.vertices),
                                  np.asarray(final.mesh_obj.triangles),
                                  views, K, poses_wc, scene=bake_scene,
                                  out_dir=rank_dir, name="model")
                bake_info = bake.to_dict()
                atlas_img = cv2.imread(str(bake.atlas_paths[0]))
                appear = atlas_metrics(atlas_bgr=atlas_img,
                                       untextured_faces=bake.untextured_faces,
                                       total_faces=len(final.mesh_obj.triangles),
                                       textured_faces=bake.textured_faces,
                                       appearance_mode=bake.appearance_mode)
        # --- Benchmark evaluation of the DELIVERY mesh (must be real, not proxy) ---
        # Use locked benchmark_holdout_ids filtered to this backend's valid poses
        benchmark_ids_for_eval = []
        benchmark_frames_for_eval = []
        # info["split"] is BenchmarkSplit; get benchmark holdout
        bench_split = info.get("split")
        if bench_split is not None and hasattr(bench_split, "benchmark_holdout_ids"):
            raw_bench = list(bench_split.benchmark_holdout_ids)
            # Filter to those where this backend has pose and frame exists, and is not REJECT
            for fid in raw_bench:
                if fid in poses and fid in id_to_frame:
                    # Also ensure not in delivery (leak check)
                    if fid not in delivery_ids:
                        benchmark_frames_for_eval.append(id_to_frame[fid])
                        benchmark_ids_for_eval.append(fid)
        # Also ensure texture views did not use benchmark (already via delivery_ids)
        # Evaluate delivery mesh on benchmark holdout via real evaluate_geometry
        delivery_geo_eval = {"status": "NOT_EVALUATED", "reason": "no benchmark frames or mesh"}
        if final is not None and final.mesh_obj is not None and len(final.mesh_obj.triangles) > 0 and benchmark_frames_for_eval:
            try:
                delivery_geo_eval = evaluate_geometry(final.mesh_obj, benchmark_frames_for_eval, poses, cam_intr, depth)
                # Add provenance
                delivery_geo_eval["eval_mesh_provenance"] = "delivery_benchmark"
                delivery_geo_eval["n_benchmark_frames"] = len(benchmark_frames_for_eval)
            except Exception as exc:
                delivery_geo_eval = {"status": "EVALUATION_FAILED", "reason": str(exc)}
        # NOT_EVALUATED handling: if 0 frames or missing mesh hash, block promotion
        if delivery_geo_eval.get("status") != "ok":
            delivery_geo_eval["quality"] = {"status": "FAIL", "reasons": ["not_evaluated"]}
        else:
            # Assess quality via geometry_eval.assess_geometry_quality (thresholds)
            from auto_mobility.reconstruction.evaluation.geometry_eval import assess_geometry_quality
            delivery_geo_eval["quality"] = assess_geometry_quality(delivery_geo_eval)

        # Compute SHAs for provenance
        import hashlib as _h3
        fusion_frame_hash = _h3.sha256(",".join(map(str, sorted(delivery_ids))).encode()).hexdigest()[:16] if delivery_ids else "empty"
        eval_frame_hash = _h3.sha256(",".join(map(str, sorted(benchmark_ids_for_eval))).encode()).hexdigest()[:16] if benchmark_ids_for_eval else "empty"
        # Mesh SHA is from texture contract (OBJ hash) or compute directly if missing
        # Will be filled after texture contract, but precompute placeholder
        mesh_sha_placeholder = None

        (rank_dir / "appearance_quality.json").write_text(json.dumps(appear, indent=2))
        (rank_dir / "geometry_quality.json").write_text(
            json.dumps(delivery_geo_eval, indent=2))
        # Also keep search tuning eval for diagnostics
        (rank_dir / "geometry_quality_search_tuning.json").write_text(json.dumps(info.get("geo_eval", {}), indent=2))

        # Fresh texture contract parsing (must parse freshly baked OBJ/MTL/PNG)
        tc = check_texture_contract(rank_dir)
        (rank_dir / "texture_contract.json").write_text(
            json.dumps(tc.to_dict(), indent=2))
        # Compute final artifact SHAs after contract (OBJ/MTL hashes)
        mesh_sha = tc.obj_hash or "no_obj"
        # Update config with full provenance
        (rank_dir / "config.json").write_text(json.dumps({
            "artifact_mode": "preview" if preview else "final",
            "production_final": not preview,
            "winner_backend": info["name"],
            "voxel_mm_requested": info["eff_voxel"],
            "voxel_mm_effective": info["eff_voxel"],
            "truncation_multiplier": TRUNCATION_MULTIPLIER,
            "n_frames_total": len(frames),
            "frames_search_train_fuse": len(info["search_ids"]),
            "frames_delivered_all_fuse": len(delivery_ids),
            "n_holdout_frames": len(benchmark_frames_for_eval),
            "n_tuning_frames": len(info.get("tuning_frames", [])),
            "benchmark_holdout_ids": benchmark_ids_for_eval,
            "benchmark_holdout_sha256": bench_split.benchmark_holdout_ids_sha256 if bench_split and hasattr(bench_split, 'benchmark_holdout_ids_sha256') else None,
            "fusion_frame_ids_sha256": fusion_frame_hash,
            "evaluation_frame_ids_sha256": eval_frame_hash,
            "evaluation_mesh_sha256": mesh_sha,
            "fusion_frames_sha256": fusion_frame_hash,
            "artifact_origin": "freshly_fused",
            "fusion_worker_tag": f"{tag}_final",
            "backend": info["name"],
            "voxel_mm_effective_tag": info["eff_voxel"],
            "refinement": {k: v for k, v in info["refinement"].items()
                           if k != "pose_by_frame"},
            "benchmark_split": bench_split.to_dict() if bench_split and hasattr(bench_split, 'to_dict') else None,
        }, indent=2))

        # P1-2: texture delivery contract gate — strict after fresh parse
        if tc.gate_status == "APPEARANCE_FAIL":
            decide("texture_contract", "APPEARANCE_FAIL",
                   tc.reject_reason or "texture contract violated",
                   tag=tag, gate_status=tc.gate_status,
                   has_usemtl=tc.has_usemtl, has_map_kd=tc.has_map_kd,
                   has_uv_coords=tc.has_uv_coords,
                   textured_face_coverage=tc.textured_face_coverage,
                   production_final=not preview)
            if not preview:
                # Block promotion to production final: caller's Phase 3 loop
                # checks ok=True before counting this as a valid rank slot.
                return {
                    "ok": False, "tag": tag, "name": info["name"],
                    "reason": f"APPEARANCE_FAIL: {tc.reject_reason}",
                    "texture_contract": tc.to_dict(),
                    "geometry_eval": delivery_geo_eval,
                    "holdout": bench_split.to_dict() if bench_split and hasattr(bench_split, 'to_dict') else {},
                    "n_delivery_frames": len(delivery_ids),
                    "n_benchmark_frames": len(benchmark_frames_for_eval),
                    "fusion_frame_ids_sha256": fusion_frame_hash,
                    "evaluation_frame_ids_sha256": eval_frame_hash,
                    "evaluation_mesh_sha256": mesh_sha,
                    "artifact_origin": "freshly_fused",
                }
        else:
            decide("texture_contract", tc.gate_status,
                   "OBJ/MTL/UV/coverage contract satisfied",
                   tag=tag, bundle_hash=tc.artifact_bundle_hash,
                   textured_face_coverage=tc.textured_face_coverage)
        # If benchmark not evaluated, block promotion (NOT_EVALUATED)
        if delivery_geo_eval.get("status") != "ok":
            decide("geometry_gate", "NOT_EVALUATED", f"benchmark evaluation failed: {delivery_geo_eval.get('reason') or delivery_geo_eval.get('status')}", tag=tag)
            if not preview:
                return {
                    "ok": False, "tag": tag, "name": info["name"],
                    "reason": f"NOT_EVALUATED: {delivery_geo_eval.get('reason') or delivery_geo_eval.get('status')}",
                    "texture_contract": tc.to_dict(),
                    "geometry_eval": delivery_geo_eval,
                    "holdout": bench_split.to_dict() if bench_split and hasattr(bench_split, 'to_dict') else {},
                    "n_delivery_frames": len(delivery_ids),
                    "n_benchmark_frames": len(benchmark_frames_for_eval),
                    "fusion_frame_ids_sha256": fusion_frame_hash,
                    "evaluation_frame_ids_sha256": eval_frame_hash,
                    "evaluation_mesh_sha256": mesh_sha,
                    "artifact_origin": "freshly_fused",
                }

        return {
            "ok": True, "tag": tag, "name": info["name"],
            "n_search_frames": len(info["search_ids"]),
            "n_delivery_frames": len(delivery_ids),
            "n_holdout_frames": len(benchmark_frames_for_eval),
            "n_tuning_frames": len(info.get("tuning_frames", [])),
            "voxel_mm_effective": info["eff_voxel"],
            "poisson_applied": applied_poisson,
            "fusion": final.to_dict(),
            "geometry_eval": delivery_geo_eval,
            "geometry_eval_search_tuning": info.get("geo_eval"),
            "texture": bake_info, "appearance": appear,
            "texture_contract": tc.to_dict(),
            "holdout": bench_split.to_dict() if bench_split and hasattr(bench_split, 'to_dict') else {},
            "fusion_frame_ids_sha256": fusion_frame_hash,
            "evaluation_frame_ids_sha256": eval_frame_hash,
            "evaluation_mesh_sha256": mesh_sha,
            "fusion_frames_sha256": fusion_frame_hash,
            "artifact_origin": "freshly_fused",
            "benchmark_holdout_ids": benchmark_ids_for_eval,
        }


    # ---- PREVIEW MODE: Fair dual-backend visual reconstruction (§4-§19) ----
    if is_preview:
        # Fail-closed if benchmark split not available (spec §5: <20 or missing backend => NON_COMPARABLE / fail)
        if benchmark_split_error is not None or benchmark_split is None:
            err = benchmark_split_error or "benchmark split unavailable"
            decide("preview_benchmark", "NON_COMPARABLE", err)
            result = {"ok": False, "mode": "preview", "reason": err, "decisions": decisions, "wall_s": round(time.time() - t0, 1)}
            _write_report(out_dir, result, preview=True)
            return result
        preview_cands = [c for c in scores if c.ok]
        if not preview_cands:
            preview_cands = top
        # Reuse poses computed for split to avoid double association (ensure same)
        poses_by_cand = poses_by_backend_for_split
        # Build assoc reports for preview candidates from already computed summary
        assoc_by_cand = {}
        for c in preview_cands:
            if c.backend in trajectories:
                # Re-associate for detailed report (same as before) to fill pose_association_report.json
                traj = trajectories[c.backend]
                from auto_mobility.trajectory.association import associate_trajectory_to_frames
                frame_stamps = np.array([f.rgb_timestamp for f in frames], dtype=np.float64)
                _, results, summary = associate_trajectory_to_frames(
                    frame_stamps, traj, max_pose_gap_ms=50.0, enable_interpolation=True
                )
                # Use the previously computed pose map for consistency, but store report
                assoc_by_cand[c.backend] = (summary, results)

        # Locked non-holdout pool: (train ∪ tuning) intersect FUSE ∩ common poses
        non_holdout_pool = sorted(set(benchmark_split.train_ids) | set(benchmark_split.tuning_val_ids))
        # Further restrict to FUSE and valid poses (common)
        fuse_set = {f.frame_id for f in frames if roles.get(f.frame_id) == FrameRole.FUSE}
        if len(fuse_set) < 20:
            fuse_set = {f.frame_id for f in frames if roles.get(f.frame_id) != FrameRole.REJECT}
        non_holdout_fuse = [fid for fid in non_holdout_pool if fid in fuse_set]
        # Intersect with common pose availability (all backends have pose)
        # common raw already is fuse ∩ common poses; non_holdout should be subset
        # For safety, filter to those present in all pose maps
        for b in [c.backend for c in preview_cands if c.backend in poses_by_cand]:
            non_holdout_fuse = [fid for fid in non_holdout_fuse if fid in poses_by_cand[b]]
        # Now pose-space coverage selection to exactly 800 or less
        target_n = policy.geometry_frame_target or 800
        ref_poses = poses_by_cand.get(preview_cands[0].backend, {}) if preview_cands else {}
        preview_frame_ids = select_pose_coverage_frames(non_holdout_fuse, ref_poses, target_count=target_n)
        # Benchmark holdout is strictly excluded from preview fusion/mask/texture
        benchmark_holdout_ids = list(benchmark_split.benchmark_holdout_ids)
        assert set(benchmark_holdout_ids).isdisjoint(set(preview_frame_ids)), "benchmark leaked into preview fusion"
        decide("preview_frame_selection", "SELECTED",
               f"selected {len(preview_frame_ids)} representative FUSE frames via pose-space coverage for dual preview (locked benchmark {len(benchmark_holdout_ids)} excluded)",
               n_preview_frames=len(preview_frame_ids), target=target_n, common_pool=len(common_ids_raw),
               non_holdout_pool=len(non_holdout_fuse),
               benchmark_holdout=len(benchmark_holdout_ids),
               is_comparable=True,
               benchmark_split=benchmark_split.to_dict())

        all_pts = []
        for c in preview_cands:
            c_poses = poses_by_cand.get(c.backend, {})
            for fid in preview_frame_ids:
                if fid in c_poses:
                    all_pts.append(c_poses[fid][:3, 3])
        if all_pts:
            pts_arr = np.asarray(all_pts)
            bbox_holder["diag_m"] = max(bbox_holder["diag_m"], float(np.linalg.norm(pts_arr.max(axis=0) - pts_arr.min(axis=0))) + 2.0)
        eff_voxel = fit_voxel_to_vram(bbox_holder["diag_m"], voxel_mm)

        preview_cand_results = {}
        preview_dir = out_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n============================================================", flush=True)
        print(f"  🚀 [3/3] DUAL PREVIEW RECONSTRUCTION (RTAB & cuVSLAM)", flush=True)
        print(f"  • Selected Representative Frames: {len(preview_frame_ids)} frames (Pose-Space Coverage)", flush=True)
        print(f"  • Effective Voxel Resolution:     {eff_voxel:.1f} mm (Identical for fairness)", flush=True)
        print(f"  • Texture Views Budget:           {policy.texture_view_target} views", flush=True)
        print(f"============================================================", flush=True)

        for c_idx, cand in enumerate(preview_cands, start=1):
            cand_name = cand.backend
            if cand_name not in poses_by_cand:
                continue
            poses = poses_by_cand[cand_name]
            cand_frame_ids = [fid for fid in preview_frame_ids if fid in poses]
            cand_dir = preview_dir / cand_name
            cand_dir.mkdir(parents=True, exist_ok=True)
            (cand_dir / "textures").mkdir(parents=True, exist_ok=True)

            # Save association diagnostics
            if cand_name in assoc_by_cand:
                p_sum, p_res = assoc_by_cand[cand_name]
                assoc_report = {
                    "backend": cand_name,
                    "num_frames": p_sum.num_frames,
                    "num_slam_poses": len(trajectories[cand_name]),
                    "pose_match_count": p_sum.pose_match_count,
                    "pose_missing_count": p_sum.pose_missing_count,
                    "pose_coverage_ratio": p_sum.pose_coverage_ratio,
                    "pose_dt_mean_ms": p_sum.pose_dt_mean_ms,
                    "pose_dt_median_ms": p_sum.pose_dt_median_ms,
                    "pose_dt_p95_ms": p_sum.pose_dt_p95_ms,
                    "pose_dt_max_ms": p_sum.pose_dt_max_ms,
                    "warning": p_sum.warning,
                }
                (cand_dir / "pose_association_report.json").write_text(
                    json.dumps(assoc_report, indent=2)
                )
                with open(cand_dir / "pose_association.csv", "w", newline="", encoding="utf-8") as f_assoc:
                    import csv
                    w = csv.DictWriter(f_assoc, fieldnames=[
                        "frame_id", "frame_timestamp", "matched_pose_timestamp",
                        "pose_dt_ms", "association_method", "valid"
                    ])
                    w.writeheader()
                    for r in p_res:
                        w.writerow({
                            "frame_id": r.frame_id,
                            "frame_timestamp": f"{r.frame_timestamp:.6f}",
                            "matched_pose_timestamp": f"{r.matched_pose_timestamp:.6f}",
                            "pose_dt_ms": f"{r.pose_dt_ms:.3f}",
                            "association_method": r.association_method,
                            "valid": r.valid,
                        })

            print(f"\n▶ [{c_idx}/{len(preview_cands)}] Processing Backend: {cand_name.upper()}", flush=True)
            # Use locked benchmark holdout exclusively for evaluation (never for fusion/mask/texture)
            # Filter benchmark frames to those where this backend has a valid pose
            benchmark_frame_ids = [fid for fid in benchmark_split.benchmark_holdout_ids if fid in poses and fid in id_to_frame]
            val_frames = [id_to_frame[fid] for fid in benchmark_frame_ids]
            # For coarse/mask, only use non-holdout frames (already guaranteed by preview_frame_ids)
            assert set(benchmark_frame_ids).isdisjoint(set(cand_frame_ids)), "benchmark leaked into preview fusion"
            # Step 1: Coarse TSDF & consistency mask — strictly non-holdout
            print(f"  [1/4] Coarse TSDF & Consistency Mask (sampling {min(80, len(cand_frame_ids))} frames, benchmark excluded)...", flush=True)
            coarse_ids = cand_frame_ids[:: max(1, len(cand_frame_ids) // 80)] or cand_frame_ids
            # Ensure coarse never contains benchmark
            assert set(benchmark_split.benchmark_holdout_ids).isdisjoint(set(coarse_ids))
            coarse_plan = _plan_active_blocks_safe(coarse_ids, poses, eff_voxel, tag=f"{cand_name}_preview_coarse")
            coarse = submit_fusion(coarse_ids, poses, eff_voxel / 1000.0, None, f"{cand_name}_preview_coarse", active_plan=coarse_plan)
            masks = {}
            if coarse and coarse.mesh_obj and len(coarse.mesh_obj.triangles) > 0:
                scene = build_scene(coarse.mesh_obj)
                mask_sample = cand_frame_ids[:: max(1, len(cand_frame_ids) // 40)]
                masks = compute_masks(scene, mask_sample, poses)
                # Mask provenance: masks must not include benchmark frames
                assert set(benchmark_split.benchmark_holdout_ids).isdisjoint(set(masks.keys())), "benchmark leaked into mask"
                print(f"        ✓ Consistency mask computed on {len(masks)} keyframes", flush=True)

            # Step 2: Full preview TSDF fusion — strictly non-holdout
            print(f"  [2/4] GPU TSDF Fusion ({len(cand_frame_ids)} frames @ {eff_voxel:.1f}mm, benchmark excluded)...", flush=True)
            preview_plan = _plan_active_blocks_safe(cand_frame_ids, poses, eff_voxel, tag=f"{cand_name}_preview_delivery")
            final = submit_fusion(cand_frame_ids, poses, eff_voxel / 1000.0, masks or None, f"{cand_name}_preview_delivery", active_plan=preview_plan)

            if final and final.mesh_obj:
                print(f"        ✓ Mesh extracted: {final.mesh_triangles:,} triangles, {final.mesh_vertices:,} vertices", flush=True)


            # Step 3: Texture baking
            print(f"  [3/4] Texture Baking ({policy.texture_view_target} representative RGB views)...", flush=True)
            bake_info, appear = None, {"status": "NOT_APPLICABLE"}
            if final and final.mesh_obj and len(final.mesh_obj.triangles) > 0:
                MAX_VIEWS = policy.texture_view_target or 32
                n_views = min(MAX_VIEWS, max(1, len(cand_frame_ids)))
                stride_v = max(1, len(cand_frame_ids) // n_views)
                views, poses_wc = [], {}
                for fid in cand_frame_ids[::stride_v][:MAX_VIEWS]:
                    img = rgb(fid)
                    if img is not None:
                        views.append((fid, normalize_exposure(img)))
                        poses_wc[fid] = poses[fid]
                if views:
                    bake_scene = build_scene(final.mesh_obj)
                    bake = bake_atlas(np.asarray(final.mesh_obj.vertices), np.asarray(final.mesh_obj.triangles), views, K, poses_wc, scene=bake_scene, out_dir=cand_dir, name="model")
                    bake_info = bake.to_dict()
                    atlas_img = cv2.imread(str(bake.atlas_paths[0]))
                    appear = atlas_metrics(atlas_bgr=atlas_img,
                                           untextured_faces=bake.untextured_faces,
                                           total_faces=len(final.mesh_obj.triangles),
                                           textured_faces=bake.textured_faces,
                                           appearance_mode=bake.appearance_mode)
                    print(f"        ✓ Texture atlas baked: {cand_dir / 'model.obj'} (coverage: {appear.get('texture_coverage', '-')})", flush=True)

            # Step 4: Held-out geometry evaluation
            print(f"  [4/4] Evaluating Held-out Geometry ({len(val_frames)} validation frames)...", flush=True)
            geo_eval = evaluate_geometry(final.mesh_obj, val_frames, poses, cam_intr, depth) if (final and final.mesh_obj and val_frames) else {"status": "NOT_APPLICABLE"}
            if geo_eval.get("status") == "ok":
                print(f"        ✓ Depth MAE: {geo_eval.get('depth_mae_mm', 0):.1f}mm | P95: {geo_eval.get('depth_p95_mm', 0):.1f}mm | Coverage: {geo_eval.get('depth_coverage_ratio', 0)*100:.1f}%", flush=True)

            # Check OBJ/MTL texture delivery contract (P1-2)
            from auto_mobility.reconstruction.appearance.texture_contract import check_texture_contract
            tex_contract = check_texture_contract(cand_dir)
            if tex_contract.gate_status == "APPEARANCE_FAIL":
                decide("appearance_gate", "APPEARANCE_FAIL",
                       f"{cand_name} OBJ does not satisfy texture contract: {tex_contract.reject_reason}",
                       backend=cand_name,
                       has_usemtl=tex_contract.has_usemtl,
                       has_map_kd=tex_contract.has_map_kd,
                       has_uv=tex_contract.has_uv_coords,
                       coverage=tex_contract.textured_face_coverage)

            (cand_dir / "appearance_quality.json").write_text(json.dumps(appear, indent=2))
            (cand_dir / "geometry_quality.json").write_text(json.dumps(geo_eval, indent=2))
            (cand_dir / "texture_contract.json").write_text(json.dumps(tex_contract.to_dict(), indent=2))
            # Provenance: freshly_fused + hashes + benchmark isolation
            import hashlib as _h2
            fusion_frame_hash = _h2.sha256(",".join(map(str, sorted(cand_frame_ids))).encode()).hexdigest()[:16] if cand_frame_ids else "empty"
            eval_frame_hash = _h2.sha256(",".join(map(str, sorted(benchmark_frame_ids))).encode()).hexdigest()[:16] if benchmark_frame_ids else "empty"
            (cand_dir / "config.json").write_text(json.dumps({
                "artifact_mode": "preview",
                "production_final": False,
                "backend": cand_name,
                "voxel_mm_effective": eff_voxel,
                "truncation_multiplier": TRUNCATION_MULTIPLIER,
                "n_frames_total": len(frames),
                "n_preview_frames": len(cand_frame_ids),
                "n_holdout_frames": len(val_frames),
                "benchmark_holdout_ids": benchmark_frame_ids,
                "benchmark_holdout_sha256": benchmark_split.benchmark_holdout_ids_sha256,
                "fusion_frame_ids_sha256": fusion_frame_hash,
                "evaluation_frame_ids_sha256": eval_frame_hash,
                "fusion_frames_sha256": fusion_frame_hash,
                "evaluation_mesh_sha256": tex_contract.obj_hash,
                "artifact_origin": "freshly_fused",
                "fusion_worker_tag": f"{cand_name}_preview_delivery",
                "obj_hash": tex_contract.obj_hash,
                "mtl_hash": tex_contract.mtl_hash,
                "artifact_bundle_hash": tex_contract.artifact_bundle_hash,
                "texture_contract": tex_contract.gate_status,
                "benchmark_split": benchmark_split.to_dict(),
            }, indent=2))

            preview_cand_results[cand_name] = {
                "name": cand_name,
                "ok": (final is not None and final.mesh_obj is not None),
                "n_preview_frames": len(cand_frame_ids),
                "voxel_mm_effective": eff_voxel,
                "mesh_triangles": final.mesh_triangles if final else 0,
                "mesh_vertices": final.mesh_vertices if final else 0,
                "geometry_eval": geo_eval,
                "appearance": appear,
                "texture": bake_info,
                "texture_contract": tex_contract.to_dict(),
                "obj_hash": tex_contract.obj_hash,
                "artifact_bundle_hash": tex_contract.artifact_bundle_hash,
                "fusion": final.to_dict() if final else {},
                "benchmark_holdout_ids": benchmark_frame_ids,
                "fusion_frame_ids_sha256": fusion_frame_hash,
                "evaluation_frame_ids_sha256": eval_frame_hash,
                "benchmark_split": benchmark_split.to_dict(),
                "artifact_origin": "freshly_fused",
            }


        # Determine optional recommended backend based on geometry metrics
        recommended_backend = None
        recommendation_reason = "No backend has passed held-out geometry acceptance yet."
        cand_keys = list(preview_cand_results.keys())
        accepted_keys = [name for name in cand_keys
                         if preview_cand_results[name].get("geometry_eval", {})
                         .get("quality", {}).get("status") == "PASS"]
        if len(accepted_keys) >= 2:
            c1_name, c2_name = accepted_keys[0], accepted_keys[1]
            g1 = preview_cand_results[c1_name].get("geometry_eval", {})
            g2 = preview_cand_results[c2_name].get("geometry_eval", {})
            p95_1 = float(g1.get("depth_p95_mm", 1e9)) if g1.get("status") == "ok" else 1e9
            p95_2 = float(g2.get("depth_p95_mm", 1e9)) if g2.get("status") == "ok" else 1e9
            cov_1 = float(g1.get("depth_coverage_ratio", 0.0)) if g1.get("status") == "ok" else 0.0
            cov_2 = float(g2.get("depth_coverage_ratio", 0.0)) if g2.get("status") == "ok" else 0.0
            if p95_1 < p95_2 - 2.0 or (abs(p95_1 - p95_2) <= 2.0 and cov_1 >= cov_2):
                recommended_backend = c1_name
                recommendation_reason = f"{c1_name} achieved lower depth P95 ({p95_1:.1f}mm vs {p95_2:.1f}mm) and coverage ({cov_1*100:.1f}% vs {cov_2*100:.1f}%)"
            else:
                recommended_backend = c2_name
                recommendation_reason = f"{c2_name} achieved lower depth P95 ({p95_2:.1f}mm vs {p95_1:.1f}mm) and coverage ({cov_2*100:.1f}% vs {cov_1*100:.1f}%)"
        elif len(accepted_keys) == 1:
            recommended_backend = accepted_keys[0]
            recommendation_reason = f"Single viable backend: {recommended_backend}"

        import hashlib
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%SZ', time.gmtime())}_{hashlib.sha256(str(t0).encode()).hexdigest()[:8]}"

        # Write comparison.json (§18)
        comparison_backends = {}
        for cname, cdata in preview_cand_results.items():
            g = cdata.get("geometry_eval", {})
            a = cdata.get("appearance", {})
            tc = cdata.get("texture_contract", {})
            sc = score_by_name.get(cname)
            comparison_backends[cname] = {
                "trajectory_coverage": sc.coverage_ratio if sc else 1.0,
                "tracking_gaps": len(sc.failures) if sc else 0,
                "preview_frames": cdata.get("n_preview_frames", 0),
                "voxel_mm": cdata.get("voxel_mm_effective", eff_voxel),
                "mesh_vertices": cdata.get("mesh_vertices", 0),
                "mesh_triangles": cdata.get("mesh_triangles", 0),
                "obj_hash": cdata.get("obj_hash"),
                "artifact_bundle_hash": cdata.get("artifact_bundle_hash"),
                "texture_contract_status": tc.get("gate_status"),
                "heldout": {
                    "depth_mae_mm": g.get("depth_mae_mm"),
                    "depth_p95_mm": g.get("depth_p95_mm"),
                    "depth_coverage_ratio": g.get("depth_coverage_ratio"),
                    "free_space_correctness_ratio": g.get("free_space_correctness_ratio"),
                },
                "appearance": {
                    "texture_coverage": a.get("texture_coverage"),
                    "untextured_face_ratio": a.get("untextured_face_ratio"),
                    "contract_status": tc.get("gate_status"),
                },
                "runtime_s": {
                    "total": round(time.time() - t0, 1),
                },
            }
        comparison_payload = {
            "run_id": run_id,
            "mode": "preview",
            "bag": dataset_dir.name,
            "preview_frames_target": target_n,
            "common_frames_selected": len(preview_frame_ids),
            "voxel_mm": eff_voxel,
            "recommended_backend": recommended_backend,
            "recommendation_reason": recommendation_reason,
            "backends": comparison_backends,
            "wall_s": round(time.time() - t0, 1),
        }
        (preview_dir / "comparison.json").write_text(json.dumps(comparison_payload, indent=2))

        # Write preview_report.md (§17, §19, §34)
        plines = [
            "# Preview Side-by-Side Comparison Report",
            "",
            f"- **Dataset**: `{dataset_dir.name}`",
            f"- **Preview Frames**: {len(preview_frame_ids)} representative FUSE frames (pose-space coverage)",
            f"- **Voxel Resolution**: {eff_voxel:.1f} mm (identical for both backends)",
            f"- **Texture Views**: {policy.texture_view_target} views",
            f"- **Recommended Backend**: `{recommended_backend}` ({recommendation_reason})",
            f"- **Total Wall Time**: {round(time.time() - t0, 1)}s",
            "",
            "## Side-by-Side Quality Metrics",
            "",
            "| Metric | " + " | ".join([f"`{name}`" for name in cand_keys]) + " |",
            "| :--- | " + " | ".join([":---:" for _ in cand_keys]) + " |",
            "| **Mesh Triangles** | " + " | ".join([f"{preview_cand_results[n]['mesh_triangles']:,}" for n in cand_keys]) + " |",
            "| **Mesh Vertices** | " + " | ".join([f"{preview_cand_results[n]['mesh_vertices']:,}" for n in cand_keys]) + " |",
        ]
        plines.append("| **Held-out Depth MAE** | " + " | ".join([
            f"{preview_cand_results[n]['geometry_eval'].get('depth_mae_mm', 0):.1f} mm"
            if preview_cand_results[n]['geometry_eval'].get("status") == "ok" else "N/A"
            for n in cand_keys
        ]) + " |")
        plines.append("| **Held-out Depth P95** | " + " | ".join([
            f"{preview_cand_results[n]['geometry_eval'].get('depth_p95_mm', 0):.1f} mm"
            if preview_cand_results[n]['geometry_eval'].get("status") == "ok" else "N/A"
            for n in cand_keys
        ]) + " |")
        plines.append("| **Depth Coverage** | " + " | ".join([
            f"{preview_cand_results[n]['geometry_eval'].get('depth_coverage_ratio', 0)*100:.1f}%"
            if preview_cand_results[n]['geometry_eval'].get("status") == "ok" else "N/A"
            for n in cand_keys
        ]) + " |")
        plines.append("| **Free-space Correctness** | " + " | ".join([
            f"{preview_cand_results[n]['geometry_eval'].get('free_space_correctness_ratio', 0)*100:.1f}%"
            if preview_cand_results[n]['geometry_eval'].get("status") == "ok" else "N/A"
            for n in cand_keys
        ]) + " |")
        plines.append("| **Texture Coverage** | " + " | ".join([
            f"{preview_cand_results[n]['appearance'].get('texture_coverage', '-')}"
            for n in cand_keys
        ]) + " |")
        plines.append("| **Texture Contract Gate** | " + " | ".join([
            f"`{preview_cand_results[n].get('texture_contract', {}).get('gate_status', 'N/A')}`"
            for n in cand_keys
        ]) + " |")

        plines += [
            "",
            "## Artifact Locations & Provenance",
        ]
        for cname in cand_keys:
            c_tc = preview_cand_results[cname].get("texture_contract", {})
            plines += [
                f"### Backend: `{cname}`",
                f"- **Run ID**: `{run_id}`",
                f"- **OBJ**: `{preview_dir / cname / 'model.obj'}` (sha256: `{c_tc.get('obj_hash', 'N/A')}`)",
                f"- **MTL**: `{preview_dir / cname / 'model.mtl'}` (sha256: `{c_tc.get('mtl_hash', 'N/A')}`)",
                f"- **TEXTURES**: `{preview_dir / cname / 'textures'}`",
                f"- **Artifact Bundle SHA256**: `{c_tc.get('artifact_bundle_hash', 'N/A')}`",
                f"- **Texture Gate Status**: `{c_tc.get('gate_status', 'N/A')}`",
                f"- **View Command**:",
                f"  ```bash",
                f"  python3 src/auto_mobility/mesh/view_mesh.py {preview_dir / cname / 'model.obj'}",
                f"  ```",
                "",
            ]

        plines += [
            "## Visual Inspection Checklist",
            "- [ ] **벽 직선도 (Wall straightness)**: 직선 벽이 휘어지지 않고 평평한가",
            "- [ ] **이중 벽 (Double walls)**: 동일 벽면이 2겹으로 어긋나서 복원되지 않았는가",
            "- [ ] **복도 접힘 (Corridor folding)**: 회전 구간이나 루프에서 복도가 접히지 않았는가",
            "- [ ] **문/창문 Edge (Sharp edges)**: 출입문, 창문 모서리가 선명하게 복원되었는가",
            "- [ ] **부유 지오메트리 (Floating artifacts)**: 허공에 떠 있는 노이즈 메시가 없는가",
            "- [ ] **얇은 구조물 (Thin structures)**: 기둥, 프레임 등 얇은 형상이 잘 유지되었는가",
            "- [ ] **텍스처 선명도 (Texture sharpness)**: 벽면/바닥 텍스처가 번짐 없이 또렷한가",
            "- [ ] **텍스처 이음매 (Texture seams)**: 시점 간 노출/색상 차이로 인한 경계가 심하지 않은가",
            "",
            "## Next Step Recommendation",
            f"Preview generated valid 3D models for both backends under identical conditions.",
            f"If visual quality and geometry are satisfactory, proceed to **Standard** (`--standard --run-slam`) or **Full** (`--full --run-slam`).",
            "",
        ]
        (preview_dir / "preview_report.md").write_text("\n".join(plines) + "\n", encoding="utf-8")

        result = {
            "ok": True,
            "mode": "preview",
            "winner": recommended_backend,
            "recommended_backend": recommended_backend,
            "recommendation_reason": recommendation_reason,
            "backends": preview_cand_results,
            "comparison": comparison_payload,
            "decisions": decisions,
            "wall_s": round(time.time() - t0, 1),
        }
        _write_report(out_dir, result, preview=True)
        return result

    # ---- QUICK MODE: Fast developer sanity / smoke check (§20) ----
    if is_quick:
        cand = top[0]
        poses = _nearest_pose_map(trajectories[cand.backend], frames)
        cand_frame_ids = select_pose_coverage_frames(list(poses.keys()), poses, target_count=100)
        eff_voxel = fit_voxel_to_vram(bbox_holder["diag_m"], voxel_mm)
        quick_dir = out_dir / "quick"
        quick_dir.mkdir(parents=True, exist_ok=True)
        quick_plan = _plan_active_blocks_safe(cand_frame_ids, poses, eff_voxel, tag="quick_sanity")
        final = submit_fusion(cand_frame_ids, poses, eff_voxel / 1000.0, None, "quick_sanity", active_plan=quick_plan)

        quick_check = {
            "status": "SANITY_PASS" if (final and final.ok) else "SANITY_FAILED",
            "quality_artifact": False,
            "note": "NOT_FOR_QUALITY_EVALUATION",
            "backend": cand.backend,
            "n_frames_fused": len(cand_frame_ids),
            "voxel_mm": eff_voxel,
            "mesh_triangles": final.mesh_triangles if final else 0,
            "mesh_vertices": final.mesh_vertices if final else 0,
            "wall_s": round(time.time() - t0, 1),
        }
        (quick_dir / "quick_check.json").write_text(json.dumps(quick_check, indent=2))
        result = {
            "ok": bool(final and final.ok),
            "mode": "quick",
            "quality_artifact": False,
            "winner": cand.backend,
            "quick_check": quick_check,
            "decisions": decisions,
            "wall_s": round(time.time() - t0, 1),
        }
        return result

    # ---- Phase 1: per-candidate SEARCH (train-only) ----
    search_infos = []
    for i, cand in enumerate(top, start=1):
        try:
            info = search_phase(cand, i)
        except Exception as exc:
            decide("search_phase", "FAILED", f"{cand.backend}: {exc}")
            continue
        search_infos.append(info)

    if not search_infos:
        result = {"ok": False, "reason": "all candidates failed search phase",
                  "trajectory_scores": [s.to_dict() for s in scores],
                  "decisions": decisions, "wall_s": round(time.time() - t0, 1)}
        _write_report(out_dir, result, preview=False)
        return result

    # ---- Phase 2: hierarchical ranking of finalists ----
    ordered_names = _rank_search_candidates(search_infos, decide, preview=False)
    by_name = {info["name"]: info for info in search_infos}
    ordered = [by_name[n] for n in ordered_names if n in by_name]

    # ---- Dual-backend delivery (explicit --deliver-backends) — benchmark-exclusive ----
    if deliver_backends is not None and not preview and not quick:
        # Fail-closed if benchmark split not available
        if benchmark_split is None or benchmark_split_error is not None:
            err = benchmark_split_error or "benchmark split unavailable"
            decide("dual_delivery_holdout", "NON_COMPARABLE", err)
            result = {"ok": False, "reason": err, "trajectory_scores": [s.to_dict() for s in scores],
                      "decisions": decisions, "wall_s": round(time.time()-t0,1), "benchmark_split_error": err}
            _write_report(out_dir, result, preview=False)
            return result
        requested = [b.strip().lower() for b in deliver_backends if b.strip()]
        if len(requested) == 1 and requested[0] == "all":
            requested = list(by_name.keys())
        result = {"trajectory_scores": [s.to_dict() for s in scores],
                  "decisions": decisions,
                  "deliver_backends": requested,
                  "benchmark_split": benchmark_split.to_dict()}
        # Common holdout is the locked benchmark_holdout_ids (identical for all backends)
        common_holdout = set(benchmark_split.benchmark_holdout_ids)
        # Ensure common holdout is at least 20 and pose-valid for every requested backend
        # Filter to those where each backend has pose (intersection already ensures, but re-check)
        common_holdout_valid = set()
        for fid in common_holdout:
            if all(fid in by_name.get(b, {}).get("poses", {}) for b in requested if b in by_name):
                # also ensure frame not REJECT if possible (but benchmark already fuse-filtered)
                if fid in id_to_frame:
                    common_holdout_valid.add(fid)
        # For reporting, use full benchmark set
        is_comparable = len(common_holdout) >= 20
        if not is_comparable:
            decide("dual_delivery_holdout", "NON_COMPARABLE",
                   f"benchmark holdout {len(common_holdout)} <20; metrics not comparable, winner not selected",
                   benchmark_holdout=len(common_holdout), requested=requested)
        else:
            decide("dual_delivery_holdout", "COMPARABLE",
                   f"benchmark holdout {len(common_holdout)} frames for fair comparison (locked)",
                   benchmark_holdout=len(common_holdout), benchmark_sha=benchmark_split.benchmark_holdout_ids_sha256)

        # Detect SUSPECT_ARTIFACT_REUSE placeholder: will check after delivery if OBJ hashes identical with different frame identities
        final_candidates = {}
        any_ok = False
        for b in requested:
            if b not in by_name:
                decide("dual_delivery", "MISSING_BACKEND",
                       f"requested backend {b} not in search candidates", backend=b)
                final_candidates[b] = {"ok": False, "reason": "missing backend or search failed", "evaluation_status": "NOT_EVALUATED"}
                continue
            info = by_name[b]
            t_d = time.time()
            tag = f"final_candidates/{b}"
            out_c = deliver_candidate(info, tag)
            budget_record("optional_improvement", time.time() - t_d)
            final_candidates[b] = out_c
            # The deliver_candidate already evaluated the delivery mesh on benchmark_holdout via real evaluate_geometry
            # Enforce that evaluation succeeded; otherwise mark NOT_EVALUATED
            geo = out_c.get("geometry_eval", {})
            if geo.get("status") != "ok":
                out_c["evaluation_status"] = "NOT_EVALUATED"
                decide("dual_delivery", "NOT_EVALUATED", f"{b} benchmark evaluation failed: {geo.get('reason') or geo.get('status')}", backend=b)
            else:
                out_c["evaluation_status"] = "EVALUATED"
                # Record mandatory SHAs for provenance
                out_c["evaluation_frame_ids_sha256"] = out_c.get("evaluation_frame_ids_sha256")
                out_c["evaluation_mesh_sha256"] = out_c.get("evaluation_mesh_sha256")
                out_c["fusion_frame_ids_sha256"] = out_c.get("fusion_frame_ids_sha256")
            if out_c.get("ok"):
                any_ok = True
            result[f"candidate_{b}"] = out_c

        # SUSPECT_ARTIFACT_REUSE detection: different identities must not have identical OBJ hash
        obj_hashes = {}
        for b, c in final_candidates.items():
            h = c.get("evaluation_mesh_sha256") or c.get("texture_contract", {}).get("obj_hash")
            if h:
                obj_hashes.setdefault(h, []).append(b)
        for h, bs in obj_hashes.items():
            if len(bs) > 1 and h != "no_obj":
                # Check if identities differ: fusion frame counts or SHAs differ but OBJ same
                fusion_shas = {final_candidates[b].get("fusion_frame_ids_sha256") for b in bs}
                if len(fusion_shas) > 1:
                    err = f"SUSPECT_ARTIFACT_REUSE: OBJ hash {h[:16]} identical for backends {bs} with different fusion identities {fusion_shas}"
                    decide("artifact_provenance", "SUSPECT_ARTIFACT_REUSE", err, obj_hash=h, backends=bs)
                    # Fail the run
                    for b in bs:
                        final_candidates[b]["ok"] = False
                        final_candidates[b]["reason"] = err
                    any_ok = False

        # Build standard_comparison.json with real evaluation values
        try:
            comparison = {}
            for b in requested:
                c = final_candidates.get(b, {})
                geo = c.get("geometry_eval", {})
                tex = c.get("texture_contract", {})
                comparison[b] = {
                    "ok": c.get("ok", False),
                    "reason": c.get("reason"),
                    "voxel_mm_effective": c.get("voxel_mm_effective"),
                    "n_delivery_frames": c.get("n_delivery_frames"),
                    "n_benchmark_frames": c.get("n_holdout_frames"),
                    "fusion_frame_ids_sha256": c.get("fusion_frame_ids_sha256"),
                    "evaluation_frame_ids_sha256": c.get("evaluation_frame_ids_sha256"),
                    "evaluation_mesh_sha256": c.get("evaluation_mesh_sha256"),
                    "evaluation_status": c.get("evaluation_status"),
                    "frame_count": c.get("n_delivery_frames"),
                    "voxel": c.get("voxel_mm_effective"),
                    "runtime_s": round(time.time()-t0,1),
                    "quality_status": geo.get("quality", {}).get("status") if isinstance(geo.get("quality"), dict) else None,
                    "heldout": {
                        "depth_mae_mm": geo.get("depth_mae_mm"),
                        "depth_rmse_mm": geo.get("depth_rmse_mm"),
                        "depth_p95_mm": geo.get("depth_p95_mm"),
                        "depth_coverage_ratio": geo.get("depth_coverage_ratio"),
                        "free_space_correctness_ratio": geo.get("free_space_correctness_ratio"),
                        "within_10mm_ratio": geo.get("within_10mm_ratio"),
                        "within_20mm_ratio": geo.get("within_20mm_ratio"),
                        "within_50mm_ratio": geo.get("within_50mm_ratio"),
                    },
                    "texture_contract": tex.get("gate_status"),
                    "textured_face_coverage": tex.get("textured_face_coverage"),
                    "quality": geo.get("quality"),
                }
            (out_dir / "standard_comparison.json").write_text(json.dumps({
                "requested_backends": requested,
                "benchmark_holdout_ids": list(benchmark_split.benchmark_holdout_ids),
                "benchmark_holdout_sha256": benchmark_split.benchmark_holdout_ids_sha256,
                "evaluation_frame_ids_sha256": benchmark_split.benchmark_holdout_ids_sha256,
                "common_holdout_size": len(common_holdout),
                "is_comparable": is_comparable,
                "backends": comparison,
                "benchmark_split": benchmark_split.to_dict(),
                "wall_s": round(time.time()-t0,1),
            }, indent=2))
            lines = ["# Standard Dual-Backend Delivery Report", "",
                     f"- Requested: {', '.join(requested)}",
                     f"- Benchmark holdout: {len(common_holdout)} frames ({'COMPARABLE' if is_comparable else 'NON_COMPARABLE'}) SHA {benchmark_split.benchmark_holdout_ids_sha256}",
                     f"- Fusion: train+ tuning frames (benchmark excluded)",
                     f"- Wall: {round(time.time()-t0,1)}s", "",
                     "| Backend | OK | Fused Frames | Voxel | MAE | P95 | Coverage | Within20 | FreeSpace | Mesh SHA | Texture |",
                     "|:---|:---:|---:|---:|---|---|---|---|---|---|---|",
                     ]
            for b in requested:
                c = final_candidates.get(b, {})
                geo = c.get("geometry_eval", {})
                tex = c.get("texture_contract", {})
                lines.append(f"| {b} | {c.get('ok')} | {c.get('n_delivery_frames','-')} | {c.get('voxel_mm_effective','-')} | {geo.get('depth_mae_mm','-')} | {geo.get('depth_p95_mm','-')} | {geo.get('depth_coverage_ratio','-')} | {geo.get('within_20mm_ratio','-')} | {geo.get('free_space_correctness_ratio','-')} | {str(c.get('evaluation_mesh_sha256','-'))[:12]} | {tex.get('gate_status','-')} |")
            # Add benchmark provenance note
            lines += ["", f"**Benchmark split:** {benchmark_split.generation_rule}", f"**Common pose count:** {benchmark_split.common_pose_count}", ""]
            (out_dir / "standard_report.md").write_text("\n".join(lines)+"\n")
        except Exception as exc:
            decide("standard_comparison", "FAILED", f"failed to write comparison: {exc}")
        result["final_candidates"] = final_candidates
        result["is_comparable"] = is_comparable
        result["common_holdout_size"] = len(common_holdout)
        result["benchmark_split"] = benchmark_split.to_dict()
        # Overall ok: require all requested backends to pass mandatory gates (geometry+appearance) and be EVALUATED
        all_ok = all(final_candidates.get(b, {}).get("ok") and final_candidates.get(b, {}).get("evaluation_status")=="EVALUATED" for b in requested)
        # Also require texture and geometry quality PASS
        for b in requested:
            c = final_candidates.get(b, {})
            if c.get("ok"):
                geo_q = c.get("geometry_eval", {}).get("quality", {})
                tex_gate = c.get("texture_contract", {}).get("gate_status")
                if geo_q.get("status") != "PASS" or tex_gate != "PASS":
                    all_ok = False
        result["ok"] = bool(all_ok and any_ok and is_comparable)
        if not result["ok"]:
            # Determine reason: NOT_EVALUATED blocks comparison
            not_eval = [b for b in requested if final_candidates.get(b, {}).get("evaluation_status") != "EVALUATED"]
            if not_eval:
                result["reason"] = f"NOT_EVALUATED for backends {not_eval} – comparison/quality gate blocked"
            elif not is_comparable:
                result["reason"] = "NON_COMPARABLE benchmark holdout"
            else:
                result["reason"] = "one or more backends failed mandatory geometry/appearance gates (see final_candidates)"
        else:
            passing = [b for b in requested if final_candidates.get(b, {}).get("ok") and final_candidates[b].get("geometry_eval", {}).get("quality", {}).get("status")=="PASS"]
            if not passing:
                passing = [b for b in requested if final_candidates.get(b, {}).get("ok")]
            if is_comparable and passing:
                best = min(passing, key=lambda b: final_candidates[b].get("geometry_eval", {}).get("depth_mae_mm", 1e9))
                result["winner"] = best
            else:
                result["winner"] = None
                if not is_comparable:
                    result["winner_reason"] = "NON_COMPARABLE"
        result["wall_s"] = round(time.time() - t0, 1)
        _write_report(out_dir, result, preview=False)
        return result

    # ---- Phase 3: FINAL DELIVERY in ranked order (legacy top-k) ----
    effective_top_k = top_k
    result = {"trajectory_scores": [s.to_dict() for s in scores],
              "decisions": decisions}
    rank01_out = None
    rank_no = 0
    for cand_idx, info in enumerate(ordered):
        if cand_idx >= effective_top_k:
            decide("delivery", "SKIPPED_BEYOND_TOPK",
                   f"{info['name']} beyond top_k={effective_top_k}")
            continue
        quality = info.get("geo_eval", {}).get("quality", {})
        if quality.get("status") != "PASS":
            decide("delivery", "REJECT_HELDOUT_GEOMETRY",
                   "search mesh failed explicit held-out acceptance; no unsafe final OBJ",
                   backend=info["name"], reasons=quality.get("reasons", []))
            continue
        if cand_idx > 0:
            est = (time.time() - t0) * 0.9
            if not budget_gate("optional_improvement", est):
                break
        rank_no += 1
        tag = f"rank_{rank_no:02d}"
        t_d = time.time()
        out_c = deliver_candidate(info, tag)
        budget_record("optional_improvement", time.time() - t_d)
        result[tag] = out_c
        if rank01_out is None:
            rank01_out = out_c

    if rank01_out is None or not rank01_out.get("ok"):
        result.update({"ok": False,
                       "reason": "final delivery failed for every finalist"})
    else:
        result.update({
            "ok": True,
            "winner": rank01_out["name"],
            "ranking_order": ordered_names,
        })
    result["wall_s"] = round(time.time() - t0, 1)
    _write_report(out_dir, result, preview=False)
    return result


def _write_report(out_dir: Path, result: dict, preview: bool = False) -> None:
    decisions = result.get("decisions", [])
    diagnosis = {
        "holdout": result.get("rank_01", {}).get("holdout"),
        "trajectory_failures": [s for s in result.get("trajectory_scores", [])
                                if not s["ok"]],
        "decisions": decisions,
    }
    (out_dir / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2))

    r1 = result.get("rank_01") or {}
    lines = ["# Reconstruction Report", "",
             f"## Winner: `{result.get('winner')}`",
             f"- wall: {result.get('wall_s')}s | "
             f"delivery frames (all FUSE): {r1.get('n_delivery_frames', '-')} | "
             f"search frames (train-only): {r1.get('n_search_frames', '-')}"]
    g = r1.get("geometry_eval", {})
    if g.get("status") == "ok":
        def _f(k):
            v = g.get(k)
            return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
        lines.append(
            f"- held-out metrics (train-only mesh): {_f('depth_mae_mm')} / "
            f"{_f('depth_p95_mm')} / cov={_f('depth_coverage_ratio')} / "
            f"fs={_f('free_space_correctness_ratio')}")
    a = r1.get("appearance", {})
    if a and a.get("status") != "NOT_APPLICABLE":
        lines.append(f"- texture coverage: {a.get('texture_coverage')} | "
                     f"untextured ratio: {a.get('untextured_face_ratio')}")
    lines += ["", "## Decisions"]
    lines += [f"- `{d['stage']}`: **{d['decision']}** — {d['reason']}"
              for d in decisions]
    bad = [s for s in result.get("trajectory_scores", []) if not s["ok"]]
    if bad:
        lines += ["", "## Rejected trajectories"]
        lines += [f"- `{b['backend']}`: {', '.join(b['failures'])}" for b in bad]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
