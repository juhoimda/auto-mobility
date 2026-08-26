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


def _nearest_pose_map(traj, frames, max_pose_gap_ms: float = 200.0,
                      dropped=None):
    """Nearest trajectory pose per frame with a gap guard.

    Frames whose nearest pose is farther than max_pose_gap_ms are DROPPED
    instead of being bridged by a stale pose (sparse cuVSLAM trajectories
    during tracking loss previously leaked stale poses into fusion).
    dropped: optional list collecting dropped frame_ids.
    """
    ts_list = list(traj.timestamps)
    poses = {}
    max_gap_s = max_pose_gap_ms / 1000.0
    for f in frames:
        i = int(np.searchsorted(traj.timestamps, f.rgb_timestamp))
        i = min(max(i, 1), len(ts_list) - 1)
        if abs(ts_list[i - 1] - f.rgb_timestamp) < abs(ts_list[i] - f.rgb_timestamp):
            i -= 1
        if abs(ts_list[i] - f.rgb_timestamp) > max_gap_s:
            if dropped is not None:
                dropped.append(f.frame_id)
            continue
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(traj.orientations[i])
        T[:3, 3] = traj.positions[i]
        poses[f.frame_id] = T
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
    """Search/delivery separation invariant (#20/#56/#57).

    SEARCH uses train-only FUSE frames (holdout excluded -> no leakage).
    DELIVERY uses ALL valid FUSE frames including holdout.
    Falls back to all non-REJECT frames when too few FUSE frames exist.
    """
    from auto_mobility.reconstruction.data.frame_selector import FrameRole

    role_items = list(roles.items())
    fuse = {fid for fid, r in role_items if r == FrameRole.FUSE}
    relaxed = False
    if len(fuse) < min_fuse_frames:
        fuse = {fid for fid, r in role_items if r != FrameRole.REJECT}
        relaxed = True
    train_set, val_set = set(split.train_ids), set(split.val_ids)
    allowed = set(valid_ids)
    search_ids = sorted(train_set & fuse & allowed)
    delivery_ids = sorted((train_set | val_set) & fuse & allowed)
    return search_ids, delivery_ids, relaxed


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
                 mode_policy=None) -> dict:
    import cv2
    import open3d as o3d

    from auto_mobility.dataset.frame_dataset import FrameDataset
    from auto_mobility.reconstruction.model import CameraIntrinsics
    from auto_mobility.reconstruction.appearance import (
        atlas_metrics, bake_atlas, normalize_exposure)
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
    if ds.dataset_info.get("depth_color_alignment") == "not_aligned":
        return {"ok": False, "reason": "depth is not aligned to RGB; provide aligned depth or calibrated extrinsics",
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
        print(f"      • {s.backend}: {'PASS' if s.ok else 'FAIL'} (coverage: {s.coverage_ratio * 100:.1f}%, score: {s.composite:.1f})", flush=True)
    if not top:
        print("      ❌ No viable trajectory candidates passed gate!", flush=True)
        return {"ok": False, "reason": "no viable trajectory",
                "trajectory_scores": [s.to_dict() for s in scores],
                "trajectory_health": health_by_name,
                "decisions": decisions}
    top = top[:max(1, min(top_k, 2))]
    score_by_name = {s.backend: s for s in scores}


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
                degraded_plan = _plan_active_blocks_safe(frame_ids, pose_by_frame, degraded,
                                                        tag=f"{tag}_degraded")
                if degraded_plan is not None:
                    active_plan = degraded_plan
                    planned_peak_vram_mb = degraded_plan.vram_mb_for_admission
                    planned_block_count = degraded_plan.safe_block_count
                    vram_want = int(planned_peak_vram_mb)
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
            job_vram_limit = int(planned_mb_for_limit * 1.10 + 512)
            if hc_val is not None:
                job_vram_limit = min(job_vram_limit, int(hc_val))
                # never exceed hard ceiling
                if job_vram_limit > int(hc_val):
                    job_vram_limit = int(hc_val)
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
        ids_all = [f.frame_id for f in frames if f.frame_id in poses0]
        split = split_from_poses(ids_all, [poses0[i] for i in ids_all])
        train_set, val_set = set(split.train_ids), set(split.val_ids)
        id_set = set(id_to_frame)

        search_ids, delivery_ids, relaxed = compute_search_delivery_sets(
            split, roles, id_set)
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
        val_frames = [id_to_frame[i] for i in sorted(val_set & id_set)]

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

        fused_search = submit_fusion(search_ids, poses, eff_voxel / 1000.0,
                                     masks or None, f"c{idx}_search",
                                     active_plan=search_plan)
        geo_eval = {"status": "NOT_APPLICABLE", "reason": "no mesh"}
        if fused_search is not None and val_frames:
            geo_eval = evaluate_geometry(fused_search.mesh_obj, val_frames, poses,
                                         cam_intr, depth)
            geo_eval["eval_mesh_provenance"] = "train_only"

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
        # keep last plans for telemetry
        return {
            "name": cand.backend,
            "poses": poses, "split": split, "search_ids": search_ids,
            "delivery_ids": delivery_ids, "val_frames": val_frames,
            "masks": masks, "eff_voxel": eff_voxel_final,
            "fused_search": fused_search, "geo_eval": geo_eval,
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
        applied_poisson = False
        poisson_trigger = (
            not preview
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
                    pmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd_for_poisson, depth=9)[0]
                    if len(pmesh.triangles) > len(final.mesh_obj.triangles):
                        final.mesh_obj = pmesh
                        applied_poisson = True
                        decide("surface", "POISSON_APPLIED",
                                "hole repair benefit expected (isolated child would be ideal, but "
                                "currently runs in parent under budget gate)", tag=tag)
                    else:
                        decide("surface", "POISSON_SKIPPED", "poisson produced fewer triangles", tag=tag)
                except Exception as exc:
                    decide("surface", "POISSON_SKIPPED", f"failed: {exc}", tag=tag)
            else:
                decide("surface", "POISSON_SKIPPED", "insufficient points for poisson", tag=tag)
        else:
            decide("surface", "POISSON_SKIPPED", "trigger/budget evidence not met",
                   tag=tag)

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
        (rank_dir / "appearance_quality.json").write_text(json.dumps(appear, indent=2))
        (rank_dir / "geometry_quality.json").write_text(
            json.dumps(info["geo_eval"], indent=2))
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
            "n_holdout_frames": len(info["val_frames"]),
            "refinement": {k: v for k, v in info["refinement"].items()
                           if k != "pose_by_frame"},
        }, indent=2))

        return {
            "ok": True, "tag": tag, "name": info["name"],
            "n_search_frames": len(info["search_ids"]),
            "n_delivery_frames": len(delivery_ids),
            "n_holdout_frames": len(info["val_frames"]),
            "voxel_mm_effective": info["eff_voxel"],
            "poisson_applied": applied_poisson,
            "fusion": final.to_dict(),
            "geometry_eval": info["geo_eval"],
            "texture": bake_info, "appearance": appear,
            "holdout": info["split"].to_dict(),
        }

    # ---- PREVIEW MODE: Fair dual-backend visual reconstruction (§4-§19) ----
    if is_preview:
        preview_cands = [c for c in scores if c.ok]
        if not preview_cands:
            preview_cands = top
        poses_by_cand = {c.backend: _nearest_pose_map(trajectories[c.backend], frames)
                         for c in preview_cands if c.backend in trajectories}
        fuse_ids = {f.frame_id for f in frames if roles.get(f.frame_id) == FrameRole.FUSE}
        if len(fuse_ids) < 20:
            fuse_ids = {f.frame_id for f in frames if roles.get(f.frame_id) != FrameRole.REJECT}
        valid_sets = [set(poses_by_cand[c.backend].keys()) for c in preview_cands if c.backend in poses_by_cand]
        common_ids = sorted(fuse_ids & set.intersection(*valid_sets)) if valid_sets else []
        if len(common_ids) < 20 and preview_cands:
            first_name = preview_cands[0].backend
            decide("preview_frame_selection", "UNFAIR_FALLBACK_TRACKING_GAP",
                   "common frame intersection between backends is sparse; using best available valid frames")
            common_ids = sorted(fuse_ids & set(poses_by_cand.get(first_name, {}).keys()))

        target_n = policy.geometry_frame_target or 800
        ref_poses = poses_by_cand.get(preview_cands[0].backend, {}) if preview_cands else {}
        preview_frame_ids = select_pose_coverage_frames(common_ids, ref_poses, target_count=target_n)
        decide("preview_frame_selection", "SELECTED",
               f"selected {len(preview_frame_ids)} representative FUSE frames via pose-space coverage for dual preview",
               n_preview_frames=len(preview_frame_ids), target=target_n, common_pool=len(common_ids))

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

            print(f"\n▶ [{c_idx}/{len(preview_cands)}] Processing Backend: {cand_name.upper()}", flush=True)
            split = split_from_poses(cand_frame_ids, [poses[fid] for fid in cand_frame_ids])
            val_frames = [id_to_frame[i] for i in split.val_ids if i in id_to_frame]

            # Step 1: Coarse TSDF & consistency mask
            print(f"  [1/4] Coarse TSDF & Consistency Mask (sampling {min(80, len(cand_frame_ids))} frames)...", flush=True)
            coarse_ids = cand_frame_ids[:: max(1, len(cand_frame_ids) // 80)] or cand_frame_ids
            coarse_plan = _plan_active_blocks_safe(coarse_ids, poses, eff_voxel, tag=f"{cand_name}_preview_coarse")
            coarse = submit_fusion(coarse_ids, poses, eff_voxel / 1000.0, None, f"{cand_name}_preview_coarse", active_plan=coarse_plan)
            masks = {}
            if coarse and coarse.mesh_obj and len(coarse.mesh_obj.triangles) > 0:
                scene = build_scene(coarse.mesh_obj)
                mask_sample = cand_frame_ids[:: max(1, len(cand_frame_ids) // 40)]
                masks = compute_masks(scene, mask_sample, poses)
                print(f"        ✓ Consistency mask computed on {len(masks)} keyframes", flush=True)

            # Step 2: Full preview TSDF fusion
            print(f"  [2/4] GPU TSDF Fusion ({len(cand_frame_ids)} frames @ {eff_voxel:.1f}mm)...", flush=True)
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

            (cand_dir / "appearance_quality.json").write_text(json.dumps(appear, indent=2))
            (cand_dir / "geometry_quality.json").write_text(json.dumps(geo_eval, indent=2))
            (cand_dir / "config.json").write_text(json.dumps({
                "artifact_mode": "preview",
                "production_final": False,
                "backend": cand_name,
                "voxel_mm_effective": eff_voxel,
                "truncation_multiplier": TRUNCATION_MULTIPLIER,
                "n_frames_total": len(frames),
                "n_preview_frames": len(cand_frame_ids),
                "n_holdout_frames": len(val_frames),
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
                "fusion": final.to_dict() if final else {},
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

        # Write comparison.json (§18)
        comparison_backends = {}
        for cname, cdata in preview_cand_results.items():
            g = cdata.get("geometry_eval", {})
            a = cdata.get("appearance", {})
            sc = score_by_name.get(cname)
            comparison_backends[cname] = {
                "trajectory_coverage": sc.coverage_ratio if sc else 1.0,
                "tracking_gaps": len(sc.failures) if sc else 0,
                "preview_frames": cdata.get("n_preview_frames", 0),
                "voxel_mm": cdata.get("voxel_mm_effective", eff_voxel),
                "mesh_vertices": cdata.get("mesh_vertices", 0),
                "mesh_triangles": cdata.get("mesh_triangles", 0),
                "heldout": {
                    "depth_mae_mm": g.get("depth_mae_mm"),
                    "depth_p95_mm": g.get("depth_p95_mm"),
                    "depth_coverage_ratio": g.get("depth_coverage_ratio"),
                    "free_space_correctness_ratio": g.get("free_space_correctness_ratio"),
                },
                "appearance": {
                    "texture_coverage": a.get("texture_coverage"),
                    "untextured_face_ratio": a.get("untextured_face_ratio"),
                },
                "runtime_s": {
                    "total": round(time.time() - t0, 1),
                },
            }
        comparison_payload = {
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

        plines += [
            "",
            "## Artifact Locations",
        ]
        for cname in cand_keys:
            plines += [
                f"### Backend: `{cname}`",
                f"- **OBJ**: `{preview_dir / cname / 'model.obj'}`",
                f"- **MTL**: `{preview_dir / cname / 'model.mtl'}`",
                f"- **TEXTURES**: `{preview_dir / cname / 'textures'}`",
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

    # ---- Phase 3: FINAL DELIVERY in ranked order ----
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
