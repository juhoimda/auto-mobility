"""
search.py — Multi-Axis Candidate Search Engine with Beam Search, Trajectory Health Gate, and Full Rebuild.

Implements Coarse-to-Fine Exploration:
  - Phase A0 (Trajectory Health Gate): Cheap sanity checks to eliminate broken trajectories before reconstruction.
  - Phase A1 (SLAM Profile Calibration): Internal profile selection per backend -> Selects Backend Champions.
  - Phase A2 (SLAM Backend Screening): Evaluates Backend Champions on fixed 10mm TSDF -> Selects Top-K SLAMs.
  - Phase B1 (Fusion Screening): Explores TSDF (20/10/8mm, adaptive 5mm) and Direct Point Cloud with common surface adapter.
  - Phase B2 (Fusion Refinement): Small refinement on truncation multiplier for top TSDF candidates.
  - Phase C (Surface Screening): Explores surface algorithms with fair no-simplification policy -> Selects Top-3 Finalists.
  - Phase D (Full Rebuild): Full reconstruction (stride=1, all train frames) & Full-Fidelity evaluation on Top-3 Finalists.
"""

from __future__ import annotations

import os
import sys
import time
import copy
import json
import hashlib
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from auto_mobility.config import MESH_DIR, POINTCLOUD_DIR, EVALUATION_DIR, get_evaluation_config
from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.evaluation.evaluator import evaluate_reconstruction
from auto_mobility.benchmark.candidate import (
    CandidateSpec,
    SlamProfileSpec,
    SlamChampion,
    get_slam_profile_spec
)
from auto_mobility.benchmark.artifacts import (
    ArtifactManager,
    is_artifact_valid,
    compute_file_sha256
)
from auto_mobility.diagnostics.trajectory_health import check_trajectory_health, TrajectoryHealthResult
from auto_mobility.benchmark.workers import (
    run_tsdf_worker,
    run_surface_worker,
    run_direct_fusion_worker,
    WorkerStatus
)
from auto_mobility.benchmark.scoring import rank_candidate_summaries, HardGateFilter, compute_absolute_scores
from auto_mobility.mesh.reconstruct_tsdf import estimate_vbg_memory_gb


class SearchEngine:
    """Executes multi-axis exploration DAG with Beam Search, Health Gate, Caching, and Full Rebuild."""

    def __init__(
        self,
        bag_name: str,
        dataset: FrameDataset,
        split_file: Path,
        artifact_mgr: ArtifactManager,
        quick: bool = False,
        full: bool = False,
        mode: str = "standard",
        force: bool = False,
        beam_width_slam: int = 2,
        beam_width_fusion: int = 3,
        finalists_count: int = 2
    ):
        self.bag_name = bag_name
        self.dataset = dataset
        self.split_file = split_file
        self.artifact_mgr = artifact_mgr
        self.force = force

        # Mode normalization: quick / standard / full
        if quick or mode.lower() == "quick":
            self.mode = "quick"
        elif full or mode.lower() == "full":
            self.mode = "full"
        else:
            self.mode = "standard"

        self.quick = (self.mode == "quick")
        self.full = (self.mode == "full")

        # Beam search widths
        if self.mode == "quick":
            self.beam_width_slam = 1
            self.beam_width_fusion = 2
            self.finalists_count = 2
        elif self.mode == "full":
            self.beam_width_slam = max(beam_width_slam, 3)
            self.beam_width_fusion = max(beam_width_fusion, 4)
            self.finalists_count = max(finalists_count, 3)
        else:
            # Standard mode
            self.beam_width_slam = beam_width_slam
            self.beam_width_fusion = beam_width_fusion
            self.finalists_count = finalists_count

        # Load unified evaluation config
        cfg = get_evaluation_config()
        eval_cfg = cfg.get("evaluation", {})
        adaptive_cfg = eval_cfg.get("adaptive_search", {})

        self.depth_min = float(eval_cfg.get("raycasting", {}).get("depth_min_m", 0.3))
        self.depth_max = float(eval_cfg.get("raycasting", {}).get("depth_max_m", 3.0))
        self.trunc_mult = 4.0

        fine_trigger_cfg = adaptive_cfg.get("fine_voxel_trigger", {})
        self.min_quality_gain_5mm = float(fine_trigger_cfg.get("min_quality_gain", 1.0))
        self.max_memory_gb_5mm = float(fine_trigger_cfg.get("max_estimated_memory_gb", 12.0))

        eval_adaptive = adaptive_cfg.get("evaluation", {})
        self.stage1_samples = int(eval_adaptive.get("stage1_holdout_samples", 12))
        self.stage2_samples = int(eval_adaptive.get("stage2_expanded_samples", 30))
        self.close_candidate_delta = float(eval_adaptive.get("close_candidate_delta_threshold", 2.0))

        # Dataset & split fingerprints for cache validation
        self.dataset_fingerprint = compute_file_sha256(self.dataset.dataset_dir / "frames.csv")[:16]
        self.split_hash = compute_file_sha256(self.split_file)[:16]

        # Search trace & execution statistics
        self.decision_trace: List[Dict[str, str]] = []
        self.stats: Dict[str, Any] = {
            "total_candidates": 0,
            "evaluated_count": 0,
            "cached_count": 0,
            "failed_count": 0,
            "pruned_count": 0,
            "rebuilt_count": 0,
            "total_runtime_sec": 0.0
        }

    def _log_decision(self, phase: str, candidate: str, decision: str, reason: str) -> None:
        self.decision_trace.append({
            "phase": phase,
            "candidate": candidate,
            "decision": decision,
            "reason": reason
        })

    def _fail_summary(self, candidate_name: str, error: str, status: str = "FAIL") -> dict:
        self.stats["failed_count"] += 1
        return {
            "candidate_name": candidate_name,
            "status": status,
            "overall_status": status,
            "error": str(error),
            "geometry": {},
            "mesh": {},
            "trajectory_metrics": {},
            "runtime_sec": None,
        }

    def _screening_stride(self) -> int:
        """Screening stride during fast search phases."""
        if self.mode == "full":
            return 1
        return max(3, len(self.dataset) // 300) if self.quick else max(2, len(self.dataset) // 1000)

    # ───────────────────────────────────────────────────────────
    # PHASE A0: Trajectory Health Gate (Cheap Preflight Diagnostics)
    # ───────────────────────────────────────────────────────────
    def run_phase_a0(
        self,
        trajectories: Dict[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, dict]]:
        """Diagnoses trajectory sanity before Open3D reconstruction and filters out broken trajectories."""
        print("\n==========================================================")
        print(" 🩺 [PHASE A0] Trajectory Health Gate (Pre-Reconstruction Sanity)")
        print("==========================================================")

        healthy_trajs: Dict[str, str] = {}
        health_diagnostics: Dict[str, dict] = {}

        for cand_key, traj_path in trajectories.items():
            diag = check_trajectory_health(traj_path)
            health_diagnostics[cand_key] = diag.to_dict()

            if diag.status == "FAIL_TRAJECTORY":
                print(f"❌ [Phase A0 Gate FAIL] `{cand_key}`: {diag.cause} ({', '.join(diag.warnings)})")
                self._log_decision("Phase A0 (Health)", cand_key, "FAIL_TRAJECTORY", f"Health check failed: {diag.cause}")
            elif diag.status == "WARN":
                print(f"⚠️ [Phase A0 Gate WARN] `{cand_key}`: {diag.cause} ({', '.join(diag.warnings)})")
                self._log_decision("Phase A0 (Health)", cand_key, "WARN_HEALTH", f"Health check warning: {diag.cause}")
                healthy_trajs[cand_key] = traj_path
            else:
                print(f"✅ [Phase A0 Gate PASS] `{cand_key}`: {diag.pose_count} poses, path: {diag.total_path_length_m:.1f}m, bbox diag: {diag.bbox_diagonal_m:.1f}m")
                self._log_decision("Phase A0 (Health)", cand_key, "PASS_HEALTH", "Passed trajectory health criteria")
                healthy_trajs[cand_key] = traj_path

        return healthy_trajs, health_diagnostics

    # ───────────────────────────────────────────────────────────
    # PHASE A1 & A2: SLAM Profile Calibration & Backend Screening
    # ───────────────────────────────────────────────────────────
    def run_phase_a(
        self,
        trajectories: Dict[str, str],
        traj_metrics: Dict[str, dict],
        pose_diagnostics: Optional[Dict[str, dict]] = None,
        health_diagnostics: Optional[Dict[str, dict]] = None
    ) -> Tuple[List[dict], List[SlamChampion]]:
        """Runs Phase A1 profile calibration and Phase A2 backend screening."""
        print("\n==========================================================")
        print(f" 🚀 [PHASE A] SLAM Profile Calibration & Backend Screening (Beam Width: Top {self.beam_width_slam}, Mode: {self.mode.upper()})")
        print("==========================================================")

        slam_eval_results: List[dict] = []
        traj_sha_map: Dict[str, str] = {}

        # 1. Group trajectories by backend family
        backend_families: Dict[str, List[str]] = {}
        for cand_key in trajectories.keys():
            spec_prof = get_slam_profile_spec(cand_key)
            backend_families.setdefault(spec_prof.backend, []).append(cand_key)

        # 2. Phase A1: Evaluate and select Backend Champions
        backend_champions: List[SlamChampion] = []

        # ── Job planning: cache/reuse decisions for every profile up-front ──
        profile_jobs: List[dict] = []
        for backend_name, profile_keys in backend_families.items():
            for prof_key in profile_keys:
                traj_file = trajectories[prof_key]
                traj_sha = compute_file_sha256(traj_file)
                traj_sha_map[prof_key] = traj_sha
                prof_spec = get_slam_profile_spec(prof_key)

                self.stats["total_candidates"] += 1
                cand_name = f"{prof_key}_tsdf10mm"
                mesh_out = self.artifact_mgr.get_mesh_path(prof_key, 10)
                pcd_out = self.artifact_mgr.get_pcd_path(prof_key, 10)
                meta_out = self.artifact_mgr.get_artifact_meta_path(mesh_out)

                spec = CandidateSpec(
                    dataset_name=self.bag_name,
                    slam_backend=prof_spec.backend,
                    slam_profile=prof_spec.profile,
                    replay_rate=prof_spec.replay_rate,
                    fusion_method="tsdf",
                    fusion_params={"voxel_size_m": 0.010, "depth_min_m": self.depth_min, "depth_max_m": self.depth_max, "trunc_mult": self.trunc_mult},
                    surface_method="tsdf_direct",
                    frame_stride=self._screening_stride()
                )
                spec_hash = spec.compute_spec_hash()

                job = {
                    "backend_name": backend_name,
                    "prof_key": prof_key,
                    "traj_file": traj_file,
                    "traj_sha": traj_sha,
                    "spec": spec,
                    "spec_hash": spec_hash,
                    "cand_name": cand_name,
                    "mesh_out": mesh_out,
                    "pcd_out": pcd_out,
                    "meta_out": meta_out,
                    "action": "worker",
                    "cached_summary": None,
                    "w_res": None,
                }

                cached_summary = self.artifact_mgr.should_reuse_evaluation(
                    cand_name, self.force, expected_spec_hash=spec_hash,
                    dataset_fingerprint=self.dataset_fingerprint, split_hash=self.split_hash
                )
                if cached_summary:
                    print(f"⏭️ Evaluation 재사용: {cand_name}")
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase A (SLAM)", cand_name, "REUSED_CACHE", "Reused evaluation from cache")
                    job["action"] = "reuse_eval"
                    job["cached_summary"] = cached_summary
                elif self.artifact_mgr.should_reuse_reconstruction(
                    mesh_out, pcd_out, candidate_spec=spec, fusion_hash=spec.compute_fusion_hash(),
                    dataset_fingerprint=self.dataset_fingerprint,
                    trajectory_sha256=traj_sha, split_hash=self.split_hash, meta_path=meta_out, force=self.force
                ):
                    print(f"⏭️ Mesh & PCD 재사용: {mesh_out.name}")
                    self._log_decision("Phase A (SLAM)", cand_name, "REUSED_MESH", "Reused existing mesh/pcd")
                    job["action"] = "reuse_mesh"
                else:
                    job["stride"] = self._screening_stride()

                profile_jobs.append(job)

        # ── Parallel pre-pass: TSDF screening workers 2-way 동시 실행 ──
        run_jobs = [j for j in profile_jobs if j["action"] == "worker"]
        if len(run_jobs) > 1:
            print(f"⚡ [Phase A] {len(run_jobs)} TSDF screening workers 병렬 실행 (2-way)")

        def _exec_phase_a_tsdf(job: dict):
            return run_tsdf_worker(
                dataset_dir=str(self.dataset.dataset_dir),
                traj_file=job["traj_file"],
                mesh_path=str(job["mesh_out"]),
                pcd_path=str(job["pcd_out"]),
                voxel=0.010,
                depth_max=self.depth_max,
                trunc_mult=self.trunc_mult,
                stride=job["stride"],
                split_file=str(self.split_file),
                quick=self.quick,
                no_color=True
            )

        if run_jobs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_map = {executor.submit(_exec_phase_a_tsdf, j): j for j in run_jobs}
                for fut in concurrent.futures.as_completed(future_map):
                    j = future_map[fut]
                    try:
                        j["w_res"] = fut.result()
                    except Exception as exc:
                        j["w_res"] = None
                        j["worker_exception"] = str(exc)
                    status = j["w_res"].status if j["w_res"] else "EXCEPTION"
                    print(f"  ⚙️ TSDF screening worker done: {j['cand_name']} ({status})")

        # ── Sequential deterministic evaluation pass ──
        family_results_map: Dict[str, List[dict]] = {}
        for job in profile_jobs:
            backend_name = job["backend_name"]
            prof_key = job["prof_key"]
            cand_name = job["cand_name"]
            traj_file = job["traj_file"]
            recon_t = 0.0
            family_results = family_results_map.setdefault(backend_name, [])

            if job["action"] == "reuse_eval":
                cached_summary = job["cached_summary"]
                cached_summary["trajectory_metrics"] = traj_metrics.get(prof_key, {})
                cached_summary["trajectory_path"] = traj_file
                cached_summary["mesh_path"] = str(job["mesh_out"])
                cached_summary["pcd_path"] = str(job["pcd_out"])
                cached_summary["spec"] = job["spec"].to_metadata_dict()
                slam_eval_results.append(cached_summary)
                family_results.append(cached_summary)
                continue

            if job["action"] == "worker":
                w_res = job.get("w_res")
                if w_res is None:
                    err = job.get("worker_exception", "tsdf worker exception")
                    print(f"  ❌ SLAM {prof_key} reconstruct 실패: EXCEPTION ({err})")
                    self._log_decision("Phase A (SLAM)", cand_name, "FAILED", f"TSDF worker exception: {err}")
                    fail_rec = self._fail_summary(cand_name, err, status="FAIL_EXCEPTION")
                    fail_rec["spec"] = job["spec"].to_metadata_dict()
                    fail_rec["trajectory_path"] = traj_file
                    slam_eval_results.append(fail_rec)
                    family_results.append(fail_rec)
                    continue

                recon_t = w_res.runtime_sec
                self.stats["total_runtime_sec"] += recon_t
                if not w_res.is_success:
                    print(f"  ❌ SLAM {prof_key} reconstruct 실패: {w_res.status} ({w_res.error_message})")
                    self._log_decision("Phase A (SLAM)", cand_name, "FAILED", f"TSDF worker failed: {w_res.status}")
                    fail_rec = self._fail_summary(cand_name, w_res.error_message or "worker crash", status=w_res.status)
                    fail_rec["spec"] = job["spec"].to_metadata_dict()
                    fail_rec["trajectory_path"] = traj_file
                    if w_res.resources:
                        fail_rec["resources"] = w_res.resources.to_dict()
                    slam_eval_results.append(fail_rec)
                    family_results.append(fail_rec)
                    continue

                # Save artifact metadata on successful worker generation
                self.artifact_mgr.save_artifact_metadata(
                    job["meta_out"], job["spec"], self.dataset_fingerprint, job["traj_sha"], self.split_hash
                )

            try:
                summary = evaluate_reconstruction(
                    dataset_input=self.dataset,
                    trajectory_input=traj_file,
                    mesh_input=str(job["mesh_out"]),
                    output_dir=self.artifact_mgr.get_candidate_eval_dir(cand_name),
                    candidate_name=cand_name,
                    split_json=str(self.split_file),
                    runtime_sec=recon_t,
                    cheap=True,
                    max_holdout_samples=self.stage1_samples
                )
                summary["trajectory_metrics"] = traj_metrics.get(prof_key, {})
                summary["trajectory_path"] = traj_file
                summary["mesh_path"] = str(job["mesh_out"])
                summary["pcd_path"] = str(job["pcd_out"])
                summary["spec"] = job["spec"].to_metadata_dict()
                summary["spec_hash"] = job["spec_hash"]
                summary["dataset_fingerprint"] = self.dataset_fingerprint
                summary["split_hash"] = self.split_hash
                w_res = job.get("w_res")
                if w_res is not None and w_res.resources:
                    summary["resources"] = w_res.resources.to_dict()
                self.stats["evaluated_count"] += 1
                self._log_decision("Phase A (SLAM)", cand_name, "EXECUTED", "Evaluated with cheap screening")
                slam_eval_results.append(summary)
                family_results.append(summary)
            except Exception as e:
                print(f"❌ Phase A {prof_key} 평가 실패: {e}")
                self._log_decision("Phase A (SLAM)", cand_name, "FAILED", f"Evaluation exception: {str(e)}")
                fail_rec = self._fail_summary(cand_name, str(e), status="FAIL_EXCEPTION")
                fail_rec["spec"] = job["spec"].to_metadata_dict()
                fail_rec["trajectory_path"] = traj_file
                slam_eval_results.append(fail_rec)
                family_results.append(fail_rec)

        # Rank family results and pick champion
        for backend_name in backend_families.keys():
            family_results = family_results_map.get(backend_name, [])
            ranked_fam = rank_candidate_summaries(family_results)
            valid_fam = [r for r in ranked_fam if r.get("hard_gate_pass", False)]
            if valid_fam:
                champ_summary = valid_fam[0]["summary_data"]
                champ_cand_name = champ_summary.get("candidate_name")
                champ_key = champ_cand_name.replace("_tsdf10mm", "").replace("_voxel10mm", "")
                champ_spec = get_slam_profile_spec(champ_key)
                champ_traj = trajectories.get(champ_key, "")
                champ_sha = traj_sha_map.get(champ_key, compute_file_sha256(champ_traj))
                champ_obj = SlamChampion(
                    profile_spec=champ_spec,
                    trajectory_path=champ_traj,
                    trajectory_sha256=champ_sha,
                    phase_a_summary=champ_summary
                )
                backend_champions.append(champ_obj)
                self._log_decision("Phase A1 (SLAM Profile)", champ_key, "SELECTED_BACKEND_CHAMPION", f"Selected as {backend_name} champion (Quality: {valid_fam[0].get('quality_score', 0):.1f})")
                print(f"👑 [{backend_name} Champion] `{champ_key}` (Quality: {valid_fam[0].get('quality_score', 0):.1f})")
            elif family_results:
                print(f"⚠️ No valid candidate passed for backend `{backend_name}`")

        # 3. Phase A2: Compare Backend Champions & Adaptive Evaluation
        champ_keys = [c.profile_spec.candidate_key for c in backend_champions]
        champ_names = [f"{k}_tsdf10mm" for k in champ_keys]
        champ_summaries = [s for s in slam_eval_results if s.get("candidate_name") in champ_names]
        ranked_champs = rank_candidate_summaries(champ_summaries)
        valid_champs = [r for r in ranked_champs if r.get("hard_gate_pass", False)]

        # Adaptive evaluation if close candidates in Top 2
        if len(valid_champs) >= 2:
            q1 = valid_champs[0].get("quality_score", 0.0)
            q2 = valid_champs[1].get("quality_score", 0.0)
            if abs(q1 - q2) <= self.close_candidate_delta and not self.quick:
                print(f"\n🔍 [Adaptive Evaluation Triggered] Quality difference ({abs(q1 - q2):.2f} pts <= {self.close_candidate_delta}) is close.")
                print(f"   Re-evaluating Top 2 champions with {self.stage2_samples} holdout samples for definitive comparison...")
                for item in valid_champs[:2]:
                    c_sum = item["summary_data"]
                    c_name = c_sum["candidate_name"]
                    c_key = c_name.replace("_tsdf10mm", "").replace("_voxel10mm", "")
                    t_path = trajectories.get(c_key, "")
                    m_path = self.artifact_mgr.get_mesh_path(c_key, 10)
                    e_dir = self.artifact_mgr.get_candidate_eval_dir(c_name)
                    try:
                        exp_summary = evaluate_reconstruction(
                            dataset_input=self.dataset,
                            trajectory_input=t_path,
                            mesh_input=str(m_path),
                            output_dir=e_dir,
                            candidate_name=c_name,
                            split_json=str(self.split_file),
                            cheap=True,
                            max_holdout_samples=self.stage2_samples
                        )
                        exp_summary["trajectory_metrics"] = traj_metrics.get(c_key, {})
                        exp_summary["trajectory_path"] = t_path
                        exp_summary["mesh_path"] = str(m_path)
                        exp_summary["spec"] = c_sum.get("spec", {})
                        self._log_decision("Phase A2 (Backend)", c_name, "ADAPTIVE_EXPANDED_EVAL", f"Expanded evaluation with {self.stage2_samples} samples")
                        for idx, el in enumerate(slam_eval_results):
                            if el.get("candidate_name") == c_name:
                                slam_eval_results[idx] = exp_summary
                    except Exception as exc:
                        print(f"⚠️ Adaptive re-eval warning: {exc}")

                # Re-rank after expanded evaluation
                champ_summaries = [s for s in slam_eval_results if s.get("candidate_name") in champ_names]
                ranked_champs = rank_candidate_summaries(champ_summaries)
                valid_champs = [r for r in ranked_champs if r.get("hard_gate_pass", False)]

        # Select Top-K SLAMs
        top_slams: List[SlamChampion] = []
        for item in valid_champs[:self.beam_width_slam]:
            cand_n = item["candidate_name"]
            slam_name = cand_n.replace("_tsdf10mm", "").replace("_voxel10mm", "")
            champ_obj = next((c for c in backend_champions if c.profile_spec.candidate_key == slam_name), None)
            if champ_obj and champ_obj not in top_slams:
                top_slams.append(champ_obj)
                self._log_decision("Phase A (SLAM)", slam_name, "SELECTED_BEAM", f"Selected in Top {self.beam_width_slam} SLAM beam (Quality: {item.get('quality_score', 0):.1f})")
                print(f"🌟 [Phase A Winner] SLAM: `{slam_name}` (Quality: {item.get('quality_score', 0):.1f}, Score: {item.get('composite_score', 0):.1f})")

        return slam_eval_results, top_slams

    # ───────────────────────────────────────────────────────────
    # PHASE B: Fusion Screening (TSDF vs DirectCloud, Common Adapter)
    # ───────────────────────────────────────────────────────────
    def run_phase_b(
        self,
        top_slams: Union[List[SlamChampion], List[Tuple[str, str]]],
        phase_a_results: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        """Runs Phase B Fusion exploration with common surface adapter to eliminate surface confound."""
        # Normalize top_slams to SlamChampion objects
        normalized_champions: List[SlamChampion] = []
        for elem in top_slams:
            if isinstance(elem, SlamChampion):
                normalized_champions.append(elem)
            elif isinstance(elem, (tuple, list)) and len(elem) >= 2:
                s_key, t_path = elem[0], elem[1]
                prof_spec = get_slam_profile_spec(s_key)
                t_sha = compute_file_sha256(t_path)
                normalized_champions.append(SlamChampion(
                    profile_spec=prof_spec,
                    trajectory_path=t_path,
                    trajectory_sha256=t_sha,
                    phase_a_summary={}
                ))

        print("\n==========================================================")
        print(f" 🚀 [PHASE B] Fusion Exploration (TSDF vs DirectCloud across {len(normalized_champions)} SLAMs, Mode: {self.mode.upper()})")
        print("==========================================================")

        fusion_eval_results: List[dict] = []

        # TSDF resolutions to evaluate
        if self.mode == "quick":
            voxel_options = [0.020, 0.010]
        elif self.mode == "full":
            voxel_options = [0.020, 0.015, 0.010, 0.008, 0.006]
        else:
            # Standard mode
            voxel_options = [0.020, 0.010, 0.008]

        for champ in normalized_champions:
            prof_spec = champ.profile_spec
            slam_name = prof_spec.candidate_key
            traj_path = champ.trajectory_path
            traj_sha = champ.trajectory_sha256

            # 1. TSDF Resolution Search — plan all voxel jobs up-front
            voxel_jobs: List[dict] = []
            for v in voxel_options:
                v_mm = int(round(v * 1000))
                cand_name = f"{slam_name}_tsdf{v_mm}mm"
                mesh_tsdf = self.artifact_mgr.get_mesh_path(slam_name, v_mm)
                pcd_tsdf = self.artifact_mgr.get_pcd_path(slam_name, v_mm)
                meta_tsdf = self.artifact_mgr.get_artifact_meta_path(mesh_tsdf)

                # Common surface adapter mesh (Poisson depth=8, no simplification) for fair fusion comparison
                adapter_mesh = self.artifact_mgr.get_mesh_path(f"{slam_name}_tsdf", v_mm, method="poisson")
                adapter_meta = self.artifact_mgr.get_artifact_meta_path(adapter_mesh)

                spec = CandidateSpec(
                    dataset_name=self.bag_name,
                    slam_backend=prof_spec.backend,
                    slam_profile=prof_spec.profile,
                    replay_rate=prof_spec.replay_rate,
                    fusion_method="tsdf",
                    fusion_params={"voxel_size_m": v, "depth_min_m": self.depth_min, "depth_max_m": self.depth_max, "trunc_mult": self.trunc_mult, "weight_threshold": 1.5},
                    surface_method="poisson",
                    surface_params={"depth": 8},
                    postprocess_params={"clean_density": True, "simplify_target": 0.0},
                    frame_stride=self._screening_stride()
                )
                spec_hash = spec.compute_spec_hash()

                print(f"▶️ Evaluating TSDF Voxel: {cand_name} ({v_mm}mm) with common Poisson adapter")
                self.stats["total_candidates"] += 1

                job = {
                    "v": v,
                    "v_mm": v_mm,
                    "cand_name": cand_name,
                    "mesh_tsdf": mesh_tsdf,
                    "pcd_tsdf": pcd_tsdf,
                    "meta_tsdf": meta_tsdf,
                    "adapter_mesh": adapter_mesh,
                    "adapter_meta": adapter_meta,
                    "spec": spec,
                    "spec_hash": spec_hash,
                    "action": "worker_tsdf",
                    "cached_summary": None,
                    "w_res": None,
                    "w_surf": None,
                    "needs_adapter": False,
                }

                cached_summary = self.artifact_mgr.should_reuse_evaluation(
                    cand_name, self.force, expected_spec_hash=spec_hash,
                    dataset_fingerprint=self.dataset_fingerprint, split_hash=self.split_hash
                )
                if cached_summary:
                    print(f"⏭️ Evaluation 재사용: {cand_name}")
                    job["action"] = "reuse_eval"
                    job["cached_summary"] = cached_summary
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase B (Fusion)", cand_name, "REUSED_CACHE", "Reused evaluation from cache")
                elif self.artifact_mgr.should_reuse_reconstruction(
                    None, pcd_tsdf, candidate_spec=spec, fusion_hash=spec.compute_fusion_hash(),
                    dataset_fingerprint=self.dataset_fingerprint,
                    trajectory_sha256=traj_sha, split_hash=self.split_hash, meta_path=meta_tsdf, force=self.force
                ):
                    print(f"⏭️ TSDF PCD 재사용: {pcd_tsdf.name}")
                    self._log_decision("Phase B (Fusion)", cand_name, "REUSED_PCD", "Reused existing TSDF PCD")
                    job["action"] = "pcd_ready"
                else:
                    job["stride"] = self._screening_stride()

                voxel_jobs.append(job)

            # ── Stage 1: TSDF workers 2-way 병렬 실행 ──
            tsdf_run_jobs = [j for j in voxel_jobs if j["action"] == "worker_tsdf"]
            if len(tsdf_run_jobs) > 1:
                print(f"⚡ [Phase B] {len(tsdf_run_jobs)} TSDF workers 병렬 실행 (2-way)")

            def _exec_phase_b_tsdf(job: dict):
                return run_tsdf_worker(
                    dataset_dir=str(self.dataset.dataset_dir),
                    traj_file=traj_path,
                    mesh_path=str(job["mesh_tsdf"]),
                    pcd_path=str(job["pcd_tsdf"]),
                    voxel=job["v"],
                    depth_max=self.depth_max,
                    trunc_mult=self.trunc_mult,
                    stride=job["stride"],
                    split_file=str(self.split_file),
                    quick=self.quick,
                    no_color=True
                )

            if tsdf_run_jobs:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_map = {executor.submit(_exec_phase_b_tsdf, j): j for j in tsdf_run_jobs}
                    for fut in concurrent.futures.as_completed(future_map):
                        j = future_map[fut]
                        try:
                            j["w_res"] = fut.result()
                        except Exception as exc:
                            j["w_res"] = None
                            j["worker_exception"] = str(exc)
                        status = j["w_res"].status if j["w_res"] else "EXCEPTION"
                        print(f"  ⚙️ TSDF worker done: {j['cand_name']} ({status})")

                for j in tsdf_run_jobs:
                    if j["w_res"] is not None and j["w_res"].is_success:
                        self.artifact_mgr.save_artifact_metadata(
                            j["meta_tsdf"], j["spec"], self.dataset_fingerprint, traj_sha, self.split_hash
                        )

            # ── Stage 2: Common Surface Adapter (Poisson depth=8) 2-way 병렬 실행 ──
            surf_run_jobs = []
            for j in voxel_jobs:
                if j["action"] == "reuse_eval":
                    continue
                if j["action"] == "worker_tsdf" and (j["w_res"] is None or not j["w_res"].is_success):
                    continue
                # PCD must exist by now (reused or freshly generated)
                if is_artifact_valid(j["pcd_tsdf"]) and (not is_artifact_valid(j["adapter_mesh"]) or self.force):
                    j["needs_adapter"] = True
                    surf_run_jobs.append(j)

            if len(surf_run_jobs) > 1:
                print(f"⚡ [Phase B] {len(surf_run_jobs)} Poisson adapter workers 병렬 실행 (2-way)")

            def _exec_phase_b_adapter(job: dict):
                return run_surface_worker(
                    input_ply=str(job["pcd_tsdf"]),
                    output_mesh=str(job["adapter_mesh"]),
                    method="poisson",
                    voxel=job["v"],
                    depth=8,
                    simplify=0.0,
                    no_simplify=True,
                    no_color_transfer=True
                )

            if surf_run_jobs:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_map = {executor.submit(_exec_phase_b_adapter, j): j for j in surf_run_jobs}
                    for fut in concurrent.futures.as_completed(future_map):
                        j = future_map[fut]
                        try:
                            j["w_surf"] = fut.result()
                        except Exception as exc:
                            j["w_surf"] = None
                            j["surf_exception"] = str(exc)
                        status = j["w_surf"].status if j["w_surf"] else "EXCEPTION"
                        print(f"  ⚙️ Adapter worker done: {j['cand_name']} ({status})")

                for j in surf_run_jobs:
                    if j["w_surf"] is not None and j["w_surf"].is_success:
                        self.artifact_mgr.save_artifact_metadata(
                            j["adapter_meta"], j["spec"], self.dataset_fingerprint, traj_sha, self.split_hash
                        )

            # ── Stage 3: Sequential deterministic evaluation ──
            for job in voxel_jobs:
                v = job["v"]
                v_mm = job["v_mm"]
                cand_name = job["cand_name"]
                recon_t = 0.0

                if job["action"] == "reuse_eval":
                    cached_summary = job["cached_summary"]
                    cached_summary["fusion_method"] = "tsdf"
                    cached_summary["voxel_size_m"] = v
                    cached_summary["pcd_path"] = str(job["pcd_tsdf"])
                    cached_summary["mesh_path"] = str(job["adapter_mesh"])
                    cached_summary["direct_tsdf_mesh_path"] = str(job["mesh_tsdf"])
                    cached_summary["trajectory_path"] = traj_path
                    cached_summary["spec"] = job["spec"].to_metadata_dict()
                    fusion_eval_results.append(cached_summary)
                    continue

                w_res = job["w_res"]
                if job["action"] == "worker_tsdf":
                    if w_res is None:
                        err = job.get("worker_exception", "tsdf worker exception")
                        print(f"  ❌ TSDF {v_mm}mm reconstruct 실패: EXCEPTION ({err})")
                        self._log_decision("Phase B (Fusion)", cand_name, "FAILED", f"Worker exception: {err}")
                        fail_rec = self._fail_summary(cand_name, err, status="FAIL_EXCEPTION")
                        fail_rec["fusion_method"] = "tsdf"
                        fail_rec["voxel_size_m"] = v
                        fail_rec["spec"] = job["spec"].to_metadata_dict()
                        fail_rec["trajectory_path"] = traj_path
                        fusion_eval_results.append(fail_rec)
                        continue

                    recon_t += w_res.runtime_sec
                    self.stats["total_runtime_sec"] += w_res.runtime_sec
                    if not w_res.is_success:
                        print(f"  ❌ TSDF {v_mm}mm reconstruct 실패: {w_res.status} ({w_res.error_message})")
                        self._log_decision("Phase B (Fusion)", cand_name, "FAILED", f"Worker failure: {w_res.status}")
                        fail_rec = self._fail_summary(cand_name, w_res.error_message or "worker crash", status=w_res.status)
                        fail_rec["fusion_method"] = "tsdf"
                        fail_rec["voxel_size_m"] = v
                        fail_rec["spec"] = job["spec"].to_metadata_dict()
                        fail_rec["trajectory_path"] = traj_path
                        if w_res.resources:
                            fail_rec["resources"] = w_res.resources.to_dict()
                        fusion_eval_results.append(fail_rec)
                        continue

                # Common surface adapter result check
                w_surf = job["w_surf"]
                if job["needs_adapter"]:
                    if w_surf is None:
                        err = job.get("surf_exception", "surface worker exception")
                        print(f"  ❌ TSDF common adapter failed: EXCEPTION ({err})")
                        fail_rec = self._fail_summary(cand_name, err, status="FAIL_EXCEPTION")
                        fail_rec["fusion_method"] = "tsdf"
                        fail_rec["voxel_size_m"] = v
                        fail_rec["spec"] = job["spec"].to_metadata_dict()
                        fail_rec["trajectory_path"] = traj_path
                        fusion_eval_results.append(fail_rec)
                        continue

                    recon_t += w_surf.runtime_sec
                    self.stats["total_runtime_sec"] += w_surf.runtime_sec
                    if not w_surf.is_success:
                        print(f"  ❌ TSDF common adapter failed: {w_surf.status}")
                        fail_rec = self._fail_summary(cand_name, w_surf.error_message or "adapter failure", status=w_surf.status)
                        fail_rec["fusion_method"] = "tsdf"
                        fail_rec["voxel_size_m"] = v
                        fail_rec["spec"] = job["spec"].to_metadata_dict()
                        fail_rec["trajectory_path"] = traj_path
                        fusion_eval_results.append(fail_rec)
                        continue

                try:
                    summary = evaluate_reconstruction(
                        dataset_input=self.dataset,
                        trajectory_input=traj_path,
                        mesh_input=str(job["adapter_mesh"]),
                        output_dir=self.artifact_mgr.get_candidate_eval_dir(cand_name),
                        candidate_name=cand_name,
                        split_json=str(self.split_file),
                        runtime_sec=recon_t,
                        cheap=True,
                        max_holdout_samples=self.stage1_samples
                    )
                    summary["fusion_method"] = "tsdf"
                    summary["voxel_size_m"] = v
                    summary["pcd_path"] = str(job["pcd_tsdf"])
                    summary["mesh_path"] = str(job["adapter_mesh"])
                    summary["direct_tsdf_mesh_path"] = str(job["mesh_tsdf"])
                    summary["trajectory_path"] = traj_path
                    summary["spec"] = job["spec"].to_metadata_dict()
                    summary["spec_hash"] = job["spec_hash"]
                    if w_res is not None and w_res.resources:
                        summary["resources"] = w_res.resources.to_dict()
                    summary["dataset_fingerprint"] = self.dataset_fingerprint
                    summary["split_hash"] = self.split_hash
                    self.stats["evaluated_count"] += 1
                    self._log_decision("Phase B (Fusion)", cand_name, "EXECUTED", "Evaluated TSDF candidate with common adapter")
                    fusion_eval_results.append(summary)
                except Exception as e:
                    print(f"⚠️ TSDF eval failure: {e}")
                    self._log_decision("Phase B (Fusion)", cand_name, "FAILED", f"Evaluation exception: {str(e)}")
                    fail_rec = self._fail_summary(cand_name, str(e), status="FAIL_EXCEPTION")
                    fail_rec["fusion_method"] = "tsdf"
                    fail_rec["voxel_size_m"] = v
                    fail_rec["spec"] = job["spec"].to_metadata_dict()
                    fail_rec["trajectory_path"] = traj_path
                    fusion_eval_results.append(fail_rec)

            # 2. Evaluate Direct Point Cloud Fusion Baseline
            s_champ_key = slam_name
            s_champ_obj = champ
            direct_cand_name = f"{s_champ_key}_direct_cloud_poisson"
            direct_pcd_out = self.artifact_mgr.get_pcd_path(s_champ_key, 10)  # Standard 10mm
            direct_pcd_meta = self.artifact_mgr.get_artifact_meta_path(direct_pcd_out)
            direct_mesh_out = self.artifact_mgr.get_candidate_artifact_dir(direct_cand_name) / "mesh.obj"
            direct_mesh_meta = self.artifact_mgr.get_artifact_meta_path(direct_mesh_out)
            direct_eval_dir = self.artifact_mgr.get_candidate_eval_dir(direct_cand_name)

            spec_direct = CandidateSpec(
                dataset_name=self.bag_name,
                slam_backend=s_champ_obj.profile_spec.backend,
                slam_profile=s_champ_obj.profile_spec.profile,
                replay_rate=s_champ_obj.profile_spec.replay_rate,
                fusion_method="direct_pointcloud",
                fusion_params={"voxel_size_m": 0.010, "depth_min_m": self.depth_min, "depth_max_m": self.depth_max, "weight_threshold": 1.5},
                surface_method="poisson",
                surface_params={"depth": 8},
                postprocess_params={"clean_density": True, "simplify_target": 0.0},
                frame_stride=self._screening_stride()
            )
            spec_dir_hash = spec_direct.compute_spec_hash()

            print(f"▶️ Evaluating Direct Point Cloud Fusion Baseline: {direct_cand_name}")
            self.stats["total_candidates"] += 1

            dir_recon_t = 0.0
            if not is_artifact_valid(direct_pcd_out) or self.force:
                w_res = run_direct_fusion_worker(
                    dataset_dir=str(self.dataset.dataset_dir),
                    traj_file=traj_path,
                    pcd_path=str(direct_pcd_out),
                    voxel=0.010,
                    depth_min=self.depth_min,
                    depth_max=self.depth_max,
                    stride=self._screening_stride(),
                    split_file=str(self.split_file),
                    no_color=True
                )
                dir_recon_t += w_res.runtime_sec
                self.stats["total_runtime_sec"] += w_res.runtime_sec
                if not w_res.is_success:
                    print(f"  ❌ Direct fusion failed: {w_res.status} ({w_res.error_message})")
                    self._log_decision("Phase B (Fusion)", direct_cand_name, "FAILED", f"Direct fusion failed: {w_res.status}")
                else:
                    self.artifact_mgr.save_artifact_metadata(
                        direct_pcd_meta, spec_direct, self.dataset_fingerprint, traj_sha, self.split_hash
                    )
                    self._log_decision("Phase B (Fusion)", direct_cand_name, "EXECUTED", "Generated direct point cloud")

            # Common surface adapter (Poisson depth=8, no simplification)
            if is_artifact_valid(direct_pcd_out):
                cached_dir_eval = self.artifact_mgr.should_reuse_evaluation(
                    direct_cand_name, self.force, expected_spec_hash=spec_dir_hash,
                    dataset_fingerprint=self.dataset_fingerprint, split_hash=self.split_hash
                )
                if cached_dir_eval:
                    cached_dir_eval["fusion_method"] = "direct_pointcloud"
                    cached_dir_eval["voxel_size_m"] = 0.010
                    cached_dir_eval["pcd_path"] = str(direct_pcd_out)
                    cached_dir_eval["mesh_path"] = str(direct_mesh_out)
                    cached_dir_eval["trajectory_path"] = traj_path
                    cached_dir_eval["spec"] = spec_direct.to_metadata_dict()
                    fusion_eval_results.append(cached_dir_eval)
                else:
                    w_surf = run_surface_worker(
                        input_ply=str(direct_pcd_out),
                        output_mesh=str(direct_mesh_out),
                        method="poisson",
                        voxel=0.010,
                        depth=8,
                        simplify=0.0,
                        no_simplify=True,
                        no_color_transfer=True
                    )
                    dir_recon_t += w_surf.runtime_sec
                    self.stats["total_runtime_sec"] += w_surf.runtime_sec
                    if w_surf.is_success:
                        self.artifact_mgr.save_artifact_metadata(
                            direct_mesh_meta, spec_direct, self.dataset_fingerprint, traj_sha, self.split_hash
                        )
                        try:
                            dir_summary = evaluate_reconstruction(
                                dataset_input=self.dataset,
                                trajectory_input=traj_path,
                                mesh_input=str(direct_mesh_out),
                                output_dir=direct_eval_dir,
                                candidate_name=direct_cand_name,
                                split_json=str(self.split_file),
                                runtime_sec=dir_recon_t,
                                cheap=True,
                                max_holdout_samples=self.stage1_samples
                            )
                            dir_summary["fusion_method"] = "direct_pointcloud"
                            dir_summary["voxel_size_m"] = 0.010
                            dir_summary["pcd_path"] = str(direct_pcd_out)
                            dir_summary["mesh_path"] = str(direct_mesh_out)
                            dir_summary["trajectory_path"] = traj_path
                            dir_summary["spec"] = spec_direct.to_metadata_dict()
                            dir_summary["spec_hash"] = spec_dir_hash
                            if 'w_res' in locals() and w_res.resources:
                                dir_summary["resources"] = w_res.resources.to_dict()
                            self.stats["evaluated_count"] += 1
                            fusion_eval_results.append(dir_summary)
                        except Exception as e:
                            print(f"⚠️ Direct fusion eval notice: {e}")

        # 3. Adaptive Fine Voxel (5mm) Check with Strict Memory Gate
        if self.mode != "quick":
            valid_b_curr = [r for r in rank_candidate_summaries(fusion_eval_results) if r.get("hard_gate_pass", False)]
            if valid_b_curr:
                best_item = valid_b_curr[0]
                best_spec = best_item["summary_data"].get("spec", {}).get("requested_params", {})
                best_backend = best_spec.get("slam_backend") or normalized_champions[0].profile_spec.backend
                best_profile = best_spec.get("slam_profile") or normalized_champions[0].profile_spec.profile
                best_rate = float(best_spec.get("replay_rate", 1.0))
                best_traj = best_item["summary_data"].get("trajectory_path") or normalized_champions[0].trajectory_path
                best_slam_key = next((c.profile_spec.candidate_key for c in normalized_champions if c.trajectory_path == best_traj), f"{best_backend}_{best_profile}_rate{best_rate:g}")

                # Check 10mm vs 8mm quality gain
                q_10mm = None
                q_8mm = None
                for r in valid_b_curr:
                    c_n = r["candidate_name"]
                    if f"{best_slam_key}_tsdf10mm" in c_n:
                        q_10mm = r.get("quality_score", 0.0)
                    elif f"{best_slam_key}_tsdf8mm" in c_n:
                        q_8mm = r.get("quality_score", 0.0)

                quality_gain = (q_8mm - q_10mm) if (q_8mm is not None and q_10mm is not None) else 0.0
                est_mem_5mm = estimate_vbg_memory_gb(best_traj, voxel_size=0.005, depth_max=self.depth_max)

                cand_5mm_name = f"{best_slam_key}_tsdf5mm"
                if q_8mm is not None and quality_gain < self.min_quality_gain_5mm:
                    print(f"⏭️ [Adaptive 5mm Skipped] Quality gain from 10mm to 8mm is {quality_gain:+.2f} pts (< {self.min_quality_gain_5mm} threshold) -> Plateau reached.")
                    self._log_decision("Phase B (Fusion)", cand_5mm_name, "SKIPPED_PLATEAU", f"8mm quality gain vs 10mm = +{quality_gain:.2f} < threshold {self.min_quality_gain_5mm}")
                elif est_mem_5mm > self.max_memory_gb_5mm:
                    print(f"⏭️ [Adaptive 5mm Skipped] Estimated memory {est_mem_5mm:.1f}GB exceeds limit {self.max_memory_gb_5mm:.1f}GB.")
                    self._log_decision("Phase B (Fusion)", cand_5mm_name, "SKIPPED_RESOURCE", f"Estimated memory {est_mem_5mm:.1f}GB > {self.max_memory_gb_5mm:.1f}GB")
                else:
                    mesh_5mm = self.artifact_mgr.get_mesh_path(best_slam_key, 5)
                    pcd_5mm = self.artifact_mgr.get_pcd_path(best_slam_key, 5)
                    meta_5mm = self.artifact_mgr.get_artifact_meta_path(mesh_5mm)
                    adapter_5mm = self.artifact_mgr.get_mesh_path(f"{best_slam_key}_tsdf", 5, method="poisson")
                    adapter_5mm_meta = self.artifact_mgr.get_artifact_meta_path(adapter_5mm)
                    eval_5mm_dir = self.artifact_mgr.get_candidate_eval_dir(cand_5mm_name)

                    spec_5mm = CandidateSpec(
                        dataset_name=self.bag_name,
                        slam_backend=best_backend,
                        slam_profile=best_profile,
                        replay_rate=best_rate,
                        fusion_method="tsdf",
                        fusion_params={"voxel_size_m": 0.005, "depth_min_m": self.depth_min, "depth_max_m": self.depth_max, "trunc_mult": self.trunc_mult, "weight_threshold": 1.5},
                        surface_method="poisson",
                        surface_params={"depth": 8},
                        postprocess_params={"clean_density": True, "simplify_target": 0.0},
                        frame_stride=self._screening_stride()
                    )

                    print(f"▶️ Adaptive Fine Voxel (5mm) Execution: {cand_5mm_name} (Gain was {quality_gain:+.2f} >= {self.min_quality_gain_5mm}, EstMem: {est_mem_5mm:.1f}GB)")
                    self.stats["total_candidates"] += 1
                    w_res = run_tsdf_worker(
                        dataset_dir=str(self.dataset.dataset_dir),
                        traj_file=best_traj,
                        mesh_path=str(mesh_5mm),
                        pcd_path=str(pcd_5mm),
                        voxel=0.005,
                        depth_max=self.depth_max,
                        trunc_mult=self.trunc_mult,
                        stride=self._screening_stride(),
                        split_file=str(self.split_file),
                        quick=self.quick,
                        no_color=True
                    )
                    if w_res.is_success:
                        self.artifact_mgr.save_artifact_metadata(
                            meta_5mm, spec_5mm, self.dataset_fingerprint, compute_file_sha256(best_traj), self.split_hash
                        )
                        w_surf = run_surface_worker(
                            input_ply=str(pcd_5mm),
                            output_mesh=str(adapter_5mm),
                            method="poisson",
                            voxel=0.005,
                            depth=8,
                            simplify=0.0,
                            no_simplify=True,
                            no_color_transfer=True
                        )
                        if w_surf.is_success:
                            self.artifact_mgr.save_artifact_metadata(
                                adapter_5mm_meta, spec_5mm, self.dataset_fingerprint, compute_file_sha256(best_traj), self.split_hash
                            )
                            try:
                                sum_5mm = evaluate_reconstruction(
                                    dataset_input=self.dataset,
                                    trajectory_input=best_traj,
                                    mesh_input=str(adapter_5mm),
                                    output_dir=eval_5mm_dir,
                                    candidate_name=cand_5mm_name,
                                    split_json=str(self.split_file),
                                    runtime_sec=w_res.runtime_sec + w_surf.runtime_sec,
                                    cheap=True,
                                    max_holdout_samples=self.stage1_samples
                                )
                                sum_5mm["fusion_method"] = "tsdf"
                                sum_5mm["voxel_size_m"] = 0.005
                                sum_5mm["pcd_path"] = str(pcd_5mm)
                                sum_5mm["mesh_path"] = str(adapter_5mm)
                                sum_5mm["direct_tsdf_mesh_path"] = str(mesh_5mm)
                                sum_5mm["trajectory_path"] = best_traj
                                sum_5mm["spec"] = spec_5mm.to_metadata_dict()
                                sum_5mm["spec_hash"] = spec_5mm.compute_spec_hash()
                                if w_res.resources:
                                    sum_5mm["resources"] = w_res.resources.to_dict()
                                fusion_eval_results.append(sum_5mm)
                                self._log_decision("Phase B (Fusion)", cand_5mm_name, "EXECUTED", "Evaluated 5mm TSDF candidate")
                            except Exception as e:
                                print(f"⚠️ 5mm eval notice: {e}")
                    else:
                        self._log_decision("Phase B (Fusion)", cand_5mm_name, "FAILED", f"5mm worker failed: {w_res.status}")

        # Rank Phase B candidates to select Top-K Fusion pipelines with diversity retention
        ranked_b = rank_candidate_summaries(fusion_eval_results)
        valid_b = [r for r in ranked_b if r.get("hard_gate_pass", False)]

        top_pipelines: List[dict] = []
        seen_slams = set()
        for item in valid_b:
            cand_slam = item["summary_data"].get("spec", {}).get("requested_params", {}).get("slam_backend")
            if cand_slam not in seen_slams and len(top_pipelines) < self.beam_width_fusion:
                seen_slams.add(cand_slam)
                top_pipelines.append(item["summary_data"])
                self._log_decision("Phase B (Fusion)", item["candidate_name"], "SELECTED_BEAM_DIVERSITY", f"Selected in fusion beam for diversity retention (Quality: {item.get('quality_score', 0):.1f})")

        # Fill remaining beam slots by global score
        for item in valid_b:
            if item["summary_data"] not in top_pipelines and len(top_pipelines) < self.beam_width_fusion:
                top_pipelines.append(item["summary_data"])
                self._log_decision("Phase B (Fusion)", item["candidate_name"], "SELECTED_BEAM", f"Selected in fusion beam (Quality: {item.get('quality_score', 0):.1f})")

        for p in top_pipelines:
            print(f"🌟 [Phase B Beam Winner] Pipeline: `{p.get('candidate_name')}`")

        return fusion_eval_results, top_pipelines

    # ───────────────────────────────────────────────────────────
    # PHASE C: Surface Reconstruction Screening (Fair Baseline)
    # ───────────────────────────────────────────────────────────
    def run_phase_c(
        self,
        top_fusion_pipelines: List[dict],
        trajectories: Dict[str, str]
    ) -> Tuple[List[dict], List[dict]]:
        """Runs Phase C Surface exploration with unsimplified fair baseline and selects Top Finalists.

        Surface worker(서브프로세스)는 파이프라인당 최대 2개까지 병렬 선실행한다.
        각 워커는 CPU 스레드 상한을 유지하며, 메모리 추정도 process budget 내에 있다.
        평가(evaluate_reconstruction)는 결정적 순서 보장을 위해 순차 실행한다.
        """
        print("\n==========================================================")
        print(f" 🚀 [PHASE C] Surface Reconstruction Exploration (across {len(top_fusion_pipelines)} Fusion Pipelines, Mode: {self.mode.upper()}, Workers: 2-way)")
        print("==========================================================")

        surface_eval_results: List[dict] = []

        tier_1 = ["tsdf_direct", "poisson"]
        tier_2 = ["alpha_shape", "bpa", "cgal_polygonal"] if (self.mode == "full") else ["alpha_shape"]
        all_methods = tier_1 + tier_2

        for pipe_summary in top_fusion_pipelines:
            cand_base = pipe_summary.get("candidate_name", "candidate")
            spec_info = pipe_summary.get("spec", {}).get("requested_params", {})
            slam_backend = spec_info.get("slam_backend")
            slam_profile = spec_info.get("slam_profile", "normal")
            replay_rate = float(spec_info.get("replay_rate", 1.0))
            traj_path = pipe_summary.get("trajectory_path") or trajectories.get(slam_backend, "")
            fusion_method = spec_info.get("fusion_method") or pipe_summary.get("fusion_method", "tsdf")
            fusion_params = spec_info.get("fusion_params", {})
            voxel_m = float(fusion_params.get("voxel_size_m", pipe_summary.get("voxel_size_m", 0.010)))
            v_mm = int(round(voxel_m * 1000))
            pcd_file = pipe_summary.get("pcd_path") or str(self.artifact_mgr.get_pcd_path(slam_backend, v_mm))

            # ── Prepare: 후보별 컨텍스트/캐시 판정을 먼저 확정한다 ──
            jobs: List[dict] = []
            for sm in all_methods:
                if fusion_method == "direct_pointcloud" and sm == "tsdf_direct":
                    continue

                cand_name = f"{cand_base}_{sm}"
                self.stats["total_candidates"] += 1
                mesh_out = self.artifact_mgr.get_mesh_path(f"{slam_backend}_{slam_profile}_rate{replay_rate:g}_{fusion_method}", v_mm, method=sm)
                meta_out = self.artifact_mgr.get_artifact_meta_path(mesh_out)
                eval_dir = self.artifact_mgr.get_candidate_eval_dir(cand_name)

                spec = CandidateSpec(
                    dataset_name=self.bag_name,
                    slam_backend=slam_backend,
                    slam_profile=slam_profile,
                    replay_rate=replay_rate,
                    fusion_method=fusion_method,
                    fusion_params=copy.deepcopy(fusion_params),
                    surface_method=sm,
                    surface_params={"depth": 8, "alpha_factor": 3.0, "orient": "centroid"},
                    postprocess_params={"clean_density": True, "simplify_target": 0.0},
                    frame_stride=self._screening_stride()
                )
                spec_hash = spec.compute_spec_hash()

                job = {
                    "sm": sm, "cand_name": cand_name, "mesh_out": mesh_out, "meta_out": meta_out,
                    "eval_dir": eval_dir, "spec": spec, "spec_hash": spec_hash,
                    "action": "worker", "w_res": None, "pcd_missing": False,
                }

                # If TSDF direct, use TSDF direct mesh directly
                if sm == "tsdf_direct" and pipe_summary.get("direct_tsdf_mesh_path") and Path(pipe_summary["direct_tsdf_mesh_path"]).exists():
                    job["action"] = "reuse_tsdf_direct"
                elif self.artifact_mgr.should_reuse_evaluation(
                    cand_name, self.force, expected_spec_hash=spec_hash,
                    dataset_fingerprint=self.dataset_fingerprint, split_hash=self.split_hash
                ):
                    job["action"] = "reuse_cache"
                elif self.artifact_mgr.should_reuse_reconstruction(
                    mesh_out, None, candidate_spec=spec, dataset_fingerprint=self.dataset_fingerprint,
                    trajectory_sha256=compute_file_sha256(traj_path), split_hash=self.split_hash, meta_path=meta_out, force=self.force
                ):
                    job["action"] = "reuse_mesh"
                elif not is_artifact_valid(pcd_file):
                    job["action"] = "skip_missing_pcd"

                jobs.append(job)

            # ── Parallel pre-pass: worker가 필요한 후보를 2-way로 동시 생성 ──
            run_jobs = [j for j in jobs if j["action"] == "worker"]
            if len(run_jobs) > 1:
                print(f"⚡ [Phase C] {len(run_jobs)} surface workers 병렬 실행 (2-way)")

            def _exec_surface_worker(job: dict):
                return run_surface_worker(
                    input_ply=str(pcd_file),
                    output_mesh=str(job["mesh_out"]),
                    method=job["sm"],
                    voxel=voxel_m,
                    depth=8,
                    simplify=0.0,
                    no_simplify=True,
                    no_color_transfer=True
                )

            if run_jobs:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    future_map = {executor.submit(_exec_surface_worker, j): j for j in run_jobs}
                    for fut in concurrent.futures.as_completed(future_map):
                        j = future_map[fut]
                        try:
                            j["w_res"] = fut.result()
                        except Exception as exc:
                            j["w_res"] = None
                            j["worker_exception"] = str(exc)
                        name = j["w_res"].status if j["w_res"] else "EXCEPTION"
                        print(f"  ⚙️ surface worker done: {j['cand_name']} ({name})")

            # ── Evaluate (결정적 순서, 순차) ──
            for job in jobs:
                sm = job["sm"]
                cand_name = job["cand_name"]
                mesh_out = job["mesh_out"]
                spec = job["spec"]
                surf_t = 0.0
                w_res = job.get("w_res")

                if job["action"] == "reuse_tsdf_direct":
                    tsdf_dir_summary = copy.deepcopy(pipe_summary)
                    tsdf_dir_summary["candidate_name"] = cand_name
                    tsdf_dir_summary["surface_method"] = "tsdf_direct"
                    tsdf_dir_summary["mesh_path"] = pipe_summary["direct_tsdf_mesh_path"]
                    tsdf_dir_summary["spec"] = spec.to_metadata_dict()
                    tsdf_dir_summary["spec_hash"] = job["spec_hash"]
                    tsdf_dir_summary["dataset_fingerprint"] = self.dataset_fingerprint
                    tsdf_dir_summary["split_hash"] = self.split_hash
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase C (Surface)", cand_name, "REUSED_TSDF_DIRECT", "Reused direct TSDF mesh")
                    surface_eval_results.append(tsdf_dir_summary)
                    continue

                if job["action"] == "reuse_cache":
                    cached_summary = self.artifact_mgr.should_reuse_evaluation(
                        cand_name, self.force, expected_spec_hash=job["spec_hash"],
                        dataset_fingerprint=self.dataset_fingerprint, split_hash=self.split_hash
                    )
                    if cached_summary:
                        print(f"⏭️ Evaluation 재사용: {cand_name}")
                        cached_summary["surface_method"] = sm
                        cached_summary["fusion_method"] = fusion_method
                        cached_summary["voxel_size_m"] = voxel_m
                        cached_summary["spec"] = spec.to_metadata_dict()
                        cached_summary["trajectory_path"] = traj_path
                        self.stats["cached_count"] += 1
                        self._log_decision("Phase C (Surface)", cand_name, "REUSED_CACHE", "Reused evaluation from cache")
                        surface_eval_results.append(cached_summary)
                        continue

                if job["action"] in ("reuse_mesh", "worker"):
                    if job["action"] == "reuse_mesh":
                        print(f"⏭️ Mesh 재사용: {mesh_out.name}")
                        self._log_decision("Phase C (Surface)", cand_name, "REUSED_MESH", "Reused existing surface mesh")
                    else:
                        if w_res is None:
                            err = job.get("worker_exception", "surface worker exception")
                            print(f"  ❌ Surface {sm} 생성 실패: EXCEPTION ({err})")
                            self._log_decision("Phase C (Surface)", cand_name, "FAILED", f"Surface worker exception: {err}")
                            fail_rec = self._fail_summary(cand_name, err, status="FAIL_EXCEPTION")
                            fail_rec["surface_method"] = sm
                            fail_rec["fusion_method"] = fusion_method
                            fail_rec["voxel_size_m"] = voxel_m
                            fail_rec["spec"] = spec.to_metadata_dict()
                            fail_rec["trajectory_path"] = traj_path
                            surface_eval_results.append(fail_rec)
                            continue

                        surf_t = w_res.runtime_sec
                        self.stats["total_runtime_sec"] += surf_t
                        if not w_res.is_success:
                            print(f"  ❌ Surface {sm} 생성 실패: {w_res.status} ({w_res.error_message})")
                            self._log_decision("Phase C (Surface)", cand_name, "FAILED", f"Surface worker failed: {w_res.status}")
                            fail_rec = self._fail_summary(cand_name, w_res.error_message or "surface failure", status=w_res.status)
                            fail_rec["surface_method"] = sm
                            fail_rec["fusion_method"] = fusion_method
                            fail_rec["voxel_size_m"] = voxel_m
                            fail_rec["spec"] = spec.to_metadata_dict()
                            fail_rec["trajectory_path"] = traj_path
                            if w_res.resources:
                                fail_rec["resources"] = w_res.resources.to_dict()
                            surface_eval_results.append(fail_rec)
                            continue

                        self.artifact_mgr.save_artifact_metadata(
                            job["meta_out"], spec, self.dataset_fingerprint, compute_file_sha256(traj_path), self.split_hash
                        )

                    try:
                        summary = evaluate_reconstruction(
                            dataset_input=self.dataset,
                            trajectory_input=traj_path,
                            mesh_input=str(mesh_out),
                            output_dir=job["eval_dir"],
                            candidate_name=cand_name,
                            split_json=str(self.split_file),
                            runtime_sec=surf_t,
                            cheap=True,
                            max_holdout_samples=self.stage1_samples
                        )
                        summary["surface_method"] = sm
                        summary["fusion_method"] = fusion_method
                        summary["voxel_size_m"] = voxel_m
                        summary["pcd_path"] = str(pcd_file)
                        summary["mesh_path"] = str(mesh_out)
                        summary["trajectory_path"] = traj_path
                        summary["spec"] = spec.to_metadata_dict()
                        summary["spec_hash"] = job["spec_hash"]
                        summary["dataset_fingerprint"] = self.dataset_fingerprint
                        summary["split_hash"] = self.split_hash
                        if w_res is not None and w_res.resources:
                            summary["resources"] = w_res.resources.to_dict()
                        self.stats["evaluated_count"] += 1
                        self._log_decision("Phase C (Surface)", cand_name, "EXECUTED", "Evaluated surface candidate")
                        surface_eval_results.append(summary)
                    except Exception as e:
                        print(f"❌ Phase C {cand_name} 평가 실패: {e}")
                        self._log_decision("Phase C (Surface)", cand_name, "FAILED", f"Surface eval exception: {str(e)}")
                        fail_rec = self._fail_summary(cand_name, str(e), status="FAIL_EXCEPTION")
                        fail_rec["surface_method"] = sm
                        fail_rec["fusion_method"] = fusion_method
                        fail_rec["voxel_size_m"] = voxel_m
                        fail_rec["spec"] = spec.to_metadata_dict()
                        fail_rec["trajectory_path"] = traj_path
                        surface_eval_results.append(fail_rec)

                elif job["action"] == "skip_missing_pcd":
                    print(f"  ❌ PCD missing for surface worker: {pcd_file}")
                    self.stats["total_candidates"] -= 1

        # Rank Phase C candidates to select Top-3 Finalists for Full Rebuild
        ranked_c = rank_candidate_summaries(surface_eval_results)
        valid_c = [r for r in ranked_c if r.get("hard_gate_pass", False)]

        finalists: List[dict] = []
        for item in valid_c[:self.finalists_count]:
            finalists.append(item["summary_data"])
            self._log_decision("Phase C (Surface)", item["candidate_name"], "FINALIST", f"Selected as Top {self.finalists_count} Finalist for Full Rebuild (Quality: {item.get('quality_score', 0):.1f})")
            print(f"🏅 [Finalist Selected] `{item['candidate_name']}` (Quality: {item.get('quality_score', 0):.1f}, Score: {item.get('composite_score', 0):.1f})")

        return surface_eval_results, finalists

    # ───────────────────────────────────────────────────────────
    # PHASE D: FULL REBUILD (stride=1, ALL TRAIN FRAMES, Unique Paths)
    # ───────────────────────────────────────────────────────────
    def run_full_rebuild(
        self,
        finalists: List[dict],
        trajectories: Dict[str, str]
    ) -> Tuple[List[dict], Optional[dict]]:
        """Reconstructs Top 3 Finalists with FULL train frames (stride=1) and runs Full-Fidelity Evaluation."""
        print("\n==========================================================")
        print(f" 🔬 [PHASE D] FULL REBUILD on Top {len(finalists)} Finalists (stride=1, ALL TRAIN FRAMES)")
        print("==========================================================")

        rebuilt_eval_results: List[dict] = []

        for rank_idx, f_summary in enumerate(finalists, 1):
            cand_name = f_summary.get("candidate_name", f"finalist_{rank_idx}")
            rebuild_cand_name = f"{cand_name}_fullrebuild"

            # Clone finalist CandidateSpec preserving all exact effective params
            spec_info = f_summary.get("spec", {}).get("requested_params", {})
            slam_backend = spec_info.get("slam_backend") or cand_name.split("_")[0]
            slam_profile = spec_info.get("slam_profile", "normal")
            replay_rate = float(spec_info.get("replay_rate", 1.0))
            fusion_method = spec_info.get("fusion_method") or f_summary.get("fusion_method", "tsdf")
            surface_method = spec_info.get("surface_method") or f_summary.get("surface_method", "tsdf_direct")
            fusion_params = copy.deepcopy(spec_info.get("fusion_params") or {"voxel_size_m": f_summary.get("voxel_size_m", 0.010), "depth_min_m": self.depth_min, "depth_max_m": self.depth_max, "trunc_mult": self.trunc_mult})
            surface_params = copy.deepcopy(spec_info.get("surface_params") or {"depth": 8, "alpha_factor": 3.0, "orient": "centroid"})
            postprocess_params = copy.deepcopy(spec_info.get("postprocess_params") or {"clean_density": True, "simplify_target": 0.0})
            voxel_m = float(fusion_params.get("voxel_size_m", 0.010))
            v_mm = int(round(voxel_m * 1000))

            traj_path = f_summary.get("trajectory_path") or trajectories.get(slam_backend, "")

            full_spec = CandidateSpec(
                dataset_name=self.bag_name,
                slam_backend=slam_backend,
                slam_profile=slam_profile,
                replay_rate=replay_rate,
                fusion_method=fusion_method,
                fusion_params=fusion_params,
                surface_method=surface_method,
                surface_params=surface_params,
                postprocess_params=postprocess_params,
                frame_stride=1,
                is_full_rebuild=True,
                evaluation_profile="full"
            )
            full_spec_hash = full_spec.compute_spec_hash()
            full_candidate_id = full_spec.compute_candidate_id(include_hash=True)

            # Unique isolated directory per finalist including hash to prevent collision
            cand_artifact_dir = self.artifact_mgr.get_candidate_artifact_dir(full_candidate_id)
            rebuild_pcd = cand_artifact_dir / f"{self.bag_name}_{full_candidate_id}_cloud.ply"
            rebuild_mesh = cand_artifact_dir / f"{self.bag_name}_{full_candidate_id}.obj"
            rebuild_meta = self.artifact_mgr.get_artifact_meta_path(rebuild_mesh)
            rebuild_eval_dir = self.artifact_mgr.get_candidate_eval_dir(rebuild_cand_name)

            print(f"\n🔨 [Full Rebuild #{rank_idx}] Candidate: `{rebuild_cand_name}`")
            print(f"   SLAM: {slam_backend} (profile={slam_profile}, rate={replay_rate}), Fusion: {fusion_method} ({v_mm}mm), Surface: {surface_method}, Stride: 1 (FULL)")
            print(f"   Isolated Output Mesh: {rebuild_mesh.name}")

            # Pre-flight memory gate: stride=1 full rebuild은 screening보다 훨씬 큰
            # VBG를 할당한다. 워치독이 죽을 만한 후보는 실행 전에 걸러낸다.
            if fusion_method != "direct_pointcloud":
                est_mem_gb = estimate_vbg_memory_gb(traj_path, voxel_size=voxel_m, depth_max=self.depth_max)
                if est_mem_gb > self.max_memory_gb_5mm:
                    print(f"⏭️ [Full Rebuild Skipped] Estimated memory {est_mem_gb:.1f}GB exceeds limit {self.max_memory_gb_5mm:.1f}GB.")
                    self._log_decision("Phase D (Full Rebuild)", rebuild_cand_name, "SKIPPED_RESOURCE", f"Estimated memory {est_mem_gb:.1f}GB > {self.max_memory_gb_5mm:.1f}GB")
                    continue

            # Cache check for Phase D Full Rebuild
            cached_rebuild = self.artifact_mgr.should_reuse_evaluation(
                rebuild_cand_name, self.force, expected_spec_hash=full_spec_hash,
                dataset_fingerprint=self.dataset_fingerprint, split_hash=self.split_hash
            )
            if cached_rebuild:
                print(f"⏭️ [Full Rebuild 캐시 재사용] {rebuild_cand_name}")
                cached_rebuild["is_full_rebuild"] = True
                cached_rebuild["trajectory_metrics"] = f_summary.get("trajectory_metrics", {})
                cached_rebuild["spec"] = full_spec.to_metadata_dict()
                cached_rebuild["spec_hash"] = full_spec_hash
                self.stats["cached_count"] += 1
                self._log_decision("Phase D (Full Rebuild)", rebuild_cand_name, "REUSED_CACHE", "Reused full rebuild from cache")
                rebuilt_eval_results.append(cached_rebuild)
                continue

            self.stats["total_candidates"] += 1
            t_rb_start = time.time()

            # 1. Full Rebuild Fusion (stride=1, Train frames only)
            if fusion_method == "direct_pointcloud":
                w_res = run_direct_fusion_worker(
                    dataset_dir=str(self.dataset.dataset_dir),
                    traj_file=traj_path,
                    pcd_path=str(rebuild_pcd),
                    voxel=voxel_m,
                    depth_min=self.depth_min,
                    depth_max=self.depth_max,
                    stride=1,
                    split_file=str(self.split_file)
                )
                if not w_res.is_success:
                    print(f"❌ Rebuild direct fusion failed: {w_res.status}")
                    continue
                w_surf = run_surface_worker(
                    input_ply=str(rebuild_pcd),
                    output_mesh=str(rebuild_mesh),
                    method=surface_method,
                    voxel=voxel_m,
                    depth=surface_params.get("depth", 8),
                    simplify=0.0,
                    no_simplify=True
                )
                if not w_surf.is_success:
                    print(f"❌ Rebuild surface failed: {w_surf.status}")
                    continue
            else:
                # TSDF Full Rebuild (stride=1, Train frames only)
                t_mult = float(fusion_params.get("trunc_mult", self.trunc_mult))
                w_res = run_tsdf_worker(
                    dataset_dir=str(self.dataset.dataset_dir),
                    traj_file=traj_path,
                    mesh_path=str(rebuild_mesh) if surface_method == "tsdf_direct" else None,
                    pcd_path=None if surface_method == "tsdf_direct" else str(rebuild_pcd),
                    voxel=voxel_m,
                    depth_max=self.depth_max,
                    trunc_mult=t_mult,
                    stride=1,
                    split_file=str(self.split_file),
                    quick=False
                )
                if not w_res.is_success:
                    print(f"❌ Rebuild TSDF failed: {w_res.status}")
                    continue

                if surface_method != "tsdf_direct":
                    w_surf = run_surface_worker(
                        input_ply=str(rebuild_pcd),
                        output_mesh=str(rebuild_mesh),
                        method=surface_method,
                        voxel=voxel_m,
                        depth=surface_params.get("depth", 8),
                        simplify=0.0,
                        no_simplify=True
                    )
                    if not w_surf.is_success:
                        print(f"❌ Rebuild surface failed: {w_surf.status}")
                        continue

            rb_runtime = time.time() - t_rb_start
            self.stats["total_runtime_sec"] += rb_runtime
            self.stats["rebuilt_count"] += 1

            self.artifact_mgr.save_artifact_metadata(
                rebuild_meta, full_spec, self.dataset_fingerprint, compute_file_sha256(traj_path), self.split_hash
            )

            # 2. Full-Fidelity Evaluation
            try:
                full_summary = evaluate_reconstruction(
                    dataset_input=self.dataset,
                    trajectory_input=traj_path,
                    mesh_input=str(rebuild_mesh),
                    output_dir=rebuild_eval_dir,
                    candidate_name=rebuild_cand_name,
                    split_json=str(self.split_file),
                    runtime_sec=rb_runtime,
                    cheap=False,
                    render_samples=10
                )
                full_summary["fusion_method"] = fusion_method
                full_summary["surface_method"] = surface_method
                full_summary["voxel_size_m"] = voxel_m
                full_summary["pcd_path"] = str(rebuild_pcd)
                full_summary["mesh_path"] = str(rebuild_mesh)
                full_summary["trajectory_path"] = traj_path
                full_summary["is_full_rebuild"] = True
                full_summary["trajectory_metrics"] = f_summary.get("trajectory_metrics", {})
                full_summary["spec"] = full_spec.to_metadata_dict()
                full_summary["spec_hash"] = full_spec_hash
                full_summary["dataset_fingerprint"] = self.dataset_fingerprint
                full_summary["split_hash"] = self.split_hash
                if 'w_res' in locals() and w_res.resources:
                    full_summary["resources"] = w_res.resources.to_dict()
                self.stats["evaluated_count"] += 1
                self._log_decision("Phase D (Full Rebuild)", rebuild_cand_name, "FULL_REBUILT", f"Full rebuild complete (stride=1, Quality: {full_summary.get('geometry', {}).get('depth_mae_mm', 0):.1f}mm MAE)")
                rebuilt_eval_results.append(full_summary)
            except Exception as e:
                print(f"❌ Rebuild evaluation failed: {e}")
                self._log_decision("Phase D (Full Rebuild)", rebuild_cand_name, "FAILED", f"Rebuild eval exception: {str(e)}")

        # Rank rebuilt finalists
        ranked_final = rank_candidate_summaries(rebuilt_eval_results)
        valid_final = [r for r in ranked_final if r.get("hard_gate_pass", False)]
        winner = valid_final[0] if valid_final else None

        if winner:
            self._log_decision("Final Ranking", winner["candidate_name"], "WINNER", f"Selected as overall winner after Full Rebuild (Quality: {winner.get('quality_score', 0):.1f}, Composite: {winner.get('composite_score', 0):.1f})")
            print("\n==========================================================")
            print(f" 🏆 [Definitive Benchmark Winner] `{winner['candidate_name']}`")
            print(f"    Quality Score  : {winner.get('quality_score', 0):.2f} / 100")
            print(f"    Cost Score     : {winner.get('cost_score', 0):.2f} / 100")
            print(f"    Composite Score: {winner.get('composite_score', 0):.2f} / 100")
            print("==========================================================")

        return ranked_final, winner
