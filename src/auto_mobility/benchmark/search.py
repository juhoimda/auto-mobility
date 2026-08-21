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
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from auto_mobility.config import MESH_DIR, POINTCLOUD_DIR, EVALUATION_DIR, get_evaluation_config
from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.evaluation.evaluator import evaluate_reconstruction
from auto_mobility.benchmark.candidate import CandidateSpec, SlamProfileSpec, get_slam_profile_spec
from auto_mobility.benchmark.artifacts import (
    ArtifactManager,
    is_artifact_valid,
    compute_file_sha256,
    save_trajectory_metadata,
    verify_trajectory_provenance
)
from auto_mobility.diagnostics.trajectory_health import check_trajectory_health, TrajectoryHealthResult
from auto_mobility.benchmark.workers import (
    run_tsdf_worker,
    run_surface_worker,
    run_direct_fusion_worker,
    WorkerStatus
)
from auto_mobility.benchmark.scoring import rank_candidate_summaries, HardGateFilter, compute_absolute_scores


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
        finalists_count: int = 3
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
    ) -> Tuple[List[dict], List[Tuple[str, str]]]:
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
        backend_champions: List[Tuple[str, str]] = []

        for backend_name, profile_keys in backend_families.items():
            print(f"\n🔬 [Phase A1: Profile Calibration] Backend `{backend_name}` ({len(profile_keys)} profiles)")
            family_results: List[dict] = []

            for prof_key in profile_keys:
                traj_file = trajectories[prof_key]
                traj_sha = compute_file_sha256(traj_file)
                traj_sha_map[prof_key] = traj_sha

                prof_spec = get_slam_profile_spec(prof_key)
                save_trajectory_metadata(
                    traj_file, prof_spec, bag_fingerprint=self.dataset_fingerprint, pose_count=len(self.dataset)
                )

                self.stats["total_candidates"] += 1
                cand_name = f"{prof_key}_tsdf10mm"
                mesh_out = self.artifact_mgr.get_mesh_path(prof_key, 10)
                pcd_out = self.artifact_mgr.get_pcd_path(prof_key, 10)
                eval_dir = self.artifact_mgr.get_candidate_eval_dir(cand_name)

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

                # Reuse evaluation if valid
                cached_summary = self.artifact_mgr.should_reuse_evaluation(
                    cand_name, self.force, expected_spec_hash=spec_hash,
                    dataset_fingerprint=self.dataset_fingerprint, split_hash=self.split_hash
                )
                if cached_summary:
                    print(f"⏭️ Evaluation 재사용: {cand_name}")
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase A (SLAM)", cand_name, "REUSED_CACHE", "Reused evaluation from cache")
                    cached_summary["trajectory_metrics"] = traj_metrics.get(prof_key, {})
                    cached_summary["spec"] = spec.to_metadata_dict()
                    slam_eval_results.append(cached_summary)
                    family_results.append(cached_summary)
                    continue

                recon_t = 0.0
                if self.artifact_mgr.should_reuse_reconstruction(
                    mesh_out, pcd_out, candidate_spec=spec, dataset_fingerprint=self.dataset_fingerprint,
                    trajectory_sha256=traj_sha, split_hash=self.split_hash, force=self.force
                ):
                    print(f"⏭️ Mesh & PCD 재사용: {mesh_out.name}")
                    self._log_decision("Phase A (SLAM)", cand_name, "REUSED_MESH", "Reused existing mesh/pcd")
                else:
                    stride = self._screening_stride()
                    w_res = run_tsdf_worker(
                        dataset_dir=str(self.dataset.dataset_dir),
                        traj_file=traj_file,
                        mesh_path=str(mesh_out),
                        pcd_path=str(pcd_out),
                        voxel=0.010,
                        depth_max=self.depth_max,
                        trunc_mult=self.trunc_mult,
                        stride=stride,
                        split_file=str(self.split_file),
                        quick=self.quick
                    )
                    recon_t = w_res.runtime_sec
                    self.stats["total_runtime_sec"] += recon_t
                    if not w_res.is_success:
                        print(f"  ❌ SLAM {prof_key} reconstruct 실패: {w_res.status} ({w_res.error_message})")
                        self._log_decision("Phase A (SLAM)", cand_name, "FAILED", f"TSDF worker failed: {w_res.status}")
                        fail_rec = self._fail_summary(cand_name, w_res.error_message or "worker crash", status=w_res.status)
                        fail_rec["spec"] = spec.to_metadata_dict()
                        slam_eval_results.append(fail_rec)
                        family_results.append(fail_rec)
                        continue

                try:
                    summary = evaluate_reconstruction(
                        dataset_input=self.dataset,
                        trajectory_input=traj_file,
                        mesh_input=str(mesh_out),
                        output_dir=eval_dir,
                        candidate_name=cand_name,
                        split_json=str(self.split_file),
                        runtime_sec=recon_t,
                        cheap=True,
                        max_holdout_samples=self.stage1_samples
                    )
                    summary["trajectory_metrics"] = traj_metrics.get(prof_key, {})
                    summary["spec"] = spec.to_metadata_dict()
                    summary["spec_hash"] = spec_hash
                    summary["dataset_fingerprint"] = self.dataset_fingerprint
                    summary["split_hash"] = self.split_hash
                    self.stats["evaluated_count"] += 1
                    self._log_decision("Phase A (SLAM)", cand_name, "EXECUTED", "Evaluated with cheap screening")
                    slam_eval_results.append(summary)
                    family_results.append(summary)
                except Exception as e:
                    print(f"❌ Phase A {prof_key} 평가 실패: {e}")
                    self._log_decision("Phase A (SLAM)", cand_name, "FAILED", f"Evaluation exception: {str(e)}")
                    fail_rec = self._fail_summary(cand_name, str(e), status="FAIL_EXCEPTION")
                    fail_rec["spec"] = spec.to_metadata_dict()
                    slam_eval_results.append(fail_rec)
                    family_results.append(fail_rec)

            # Rank family results and pick champion
            ranked_fam = rank_candidate_summaries(family_results)
            valid_fam = [r for r in ranked_fam if r.get("hard_gate_pass", False)]
            if valid_fam:
                champ_summary = valid_fam[0]["summary_data"]
                champ_cand_name = champ_summary.get("candidate_name")
                champ_key = champ_cand_name.replace("_tsdf10mm", "").replace("_voxel10mm", "")
                champ_traj = trajectories.get(champ_key, "")
                backend_champions.append((champ_key, champ_traj))
                self._log_decision("Phase A1 (SLAM Profile)", champ_key, "SELECTED_BACKEND_CHAMPION", f"Selected as {backend_name} champion (Quality: {valid_fam[0].get('quality_score', 0):.1f})")
                print(f"👑 [{backend_name} Champion] `{champ_key}` (Quality: {valid_fam[0].get('quality_score', 0):.1f})")
            elif family_results:
                # No passing candidate for this backend
                print(f"⚠️ No valid candidate passed for backend `{backend_name}`")

        # 3. Phase A2: Compare Backend Champions & Adaptive Evaluation
        champ_names = [f"{c[0]}_tsdf10mm" for c in backend_champions]
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
                        exp_summary["spec"] = c_sum.get("spec", {})
                        self._log_decision("Phase A2 (Backend)", c_name, "ADAPTIVE_EXPANDED_EVAL", f"Expanded evaluation with {self.stage2_samples} samples")
                        # Update summary in results
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
        top_slams: List[Tuple[str, str]] = []
        for item in valid_champs[:self.beam_width_slam]:
            cand_n = item["candidate_name"]
            slam_name = cand_n.replace("_tsdf10mm", "").replace("_voxel10mm", "")
            t_file = trajectories.get(slam_name, "")
            if t_file and (slam_name, t_file) not in top_slams:
                top_slams.append((slam_name, t_file))
                self._log_decision("Phase A (SLAM)", slam_name, "SELECTED_BEAM", f"Selected in Top {self.beam_width_slam} SLAM beam (Quality: {item.get('quality_score', 0):.1f})")
                print(f"🌟 [Phase A Winner] SLAM: `{slam_name}` (Quality: {item.get('quality_score', 0):.1f}, Score: {item.get('composite_score', 0):.1f})")

        return slam_eval_results, top_slams

    # ───────────────────────────────────────────────────────────
    # PHASE B: Fusion Screening (TSDF vs DirectCloud, Common Adapter)
    # ───────────────────────────────────────────────────────────
    def run_phase_b(
        self,
        top_slams: List[Tuple[str, str]],
        phase_a_results: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        """Runs Phase B Fusion exploration with common surface adapter to eliminate surface confound."""
        print("\n==========================================================")
        print(f" 🚀 [PHASE B] Fusion Exploration (TSDF vs DirectCloud across {len(top_slams)} SLAMs, Mode: {self.mode.upper()})")
        print("==========================================================")

        fusion_eval_results: List[dict] = []
        phase_a_by_cand = {s.get("candidate_name"): s for s in phase_a_results}

        # TSDF resolutions to evaluate
        if self.mode == "quick":
            voxel_options = [0.020, 0.010]
        elif self.mode == "full":
            voxel_options = [0.020, 0.015, 0.010, 0.008, 0.006]
        else:
            # Standard mode
            voxel_options = [0.020, 0.010, 0.008]

        for slam_name, traj_path in top_slams:
            traj_sha = compute_file_sha256(traj_path)

            # 1. TSDF Resolution Search
            for v in voxel_options:
                v_mm = int(round(v * 1000))
                cand_name = f"{slam_name}_tsdf{v_mm}mm"
                mesh_out = self.artifact_mgr.get_mesh_path(slam_name, v_mm)
                pcd_out = self.artifact_mgr.get_pcd_path(slam_name, v_mm)
                eval_dir = self.artifact_mgr.get_candidate_eval_dir(cand_name)

                spec = CandidateSpec(
                    dataset_name=self.bag_name,
                    slam_backend=slam_name,
                    fusion_method="tsdf",
                    fusion_params={"voxel_size_m": v, "depth_min_m": self.depth_min, "depth_max_m": self.depth_max, "trunc_mult": self.trunc_mult},
                    surface_method="tsdf_direct",
                    frame_stride=self._screening_stride()
                )
                spec_hash = spec.compute_spec_hash()

                print(f"▶️ Evaluating TSDF Voxel: {cand_name} ({v_mm}mm)")
                self.stats["total_candidates"] += 1

                # Check if 10mm result exists from Phase A
                cand_10mm_a = f"{slam_name}_tsdf10mm"
                if v_mm == 10 and cand_10mm_a in phase_a_by_cand:
                    existing = phase_a_by_cand.get(cand_10mm_a)
                    if existing and existing.get("status") not in ("FAIL", "FAIL_CRASH", "FAIL_SEGFAULT", "FAIL_OOM"):
                        print(f"♻️ Phase A 10mm 결과 직접 재사용: {cand_name}")
                        existing_copy = copy.deepcopy(existing)
                        existing_copy["candidate_name"] = cand_name
                        existing_copy["fusion_method"] = "tsdf"
                        existing_copy["voxel_size_m"] = v
                        existing_copy["pcd_path"] = str(pcd_out)
                        existing_copy["mesh_path"] = str(mesh_out)
                        existing_copy["spec"] = spec.to_metadata_dict()
                        self.stats["cached_count"] += 1
                        self._log_decision("Phase B (Fusion)", cand_name, "REUSED_PHASE_A", "Reused Phase A 10mm TSDF directly")
                        fusion_eval_results.append(existing_copy)
                        continue

                cached_summary = self.artifact_mgr.should_reuse_evaluation(
                    cand_name, self.force, expected_spec_hash=spec_hash,
                    dataset_fingerprint=self.dataset_fingerprint, split_hash=self.split_hash
                )
                if cached_summary:
                    print(f"⏭️ Evaluation 재사용: {cand_name}")
                    cached_summary["fusion_method"] = "tsdf"
                    cached_summary["voxel_size_m"] = v
                    cached_summary["pcd_path"] = str(pcd_out)
                    cached_summary["mesh_path"] = str(mesh_out)
                    cached_summary["spec"] = spec.to_metadata_dict()
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase B (Fusion)", cand_name, "REUSED_CACHE", "Reused evaluation from cache")
                    fusion_eval_results.append(cached_summary)
                    continue

                recon_t = 0.0
                if self.artifact_mgr.should_reuse_reconstruction(
                    mesh_out, pcd_out, candidate_spec=spec, dataset_fingerprint=self.dataset_fingerprint,
                    trajectory_sha256=traj_sha, split_hash=self.split_hash, force=self.force
                ):
                    print(f"⏭️ Mesh & PCD 재사용: {mesh_out.name}")
                    self._log_decision("Phase B (Fusion)", cand_name, "REUSED_MESH", "Reused existing mesh & PCD files")
                else:
                    stride = self._screening_stride()
                    w_res = run_tsdf_worker(
                        dataset_dir=str(self.dataset.dataset_dir),
                        traj_file=traj_path,
                        mesh_path=str(mesh_out),
                        pcd_path=str(pcd_out),
                        voxel=v,
                        depth_max=self.depth_max,
                        trunc_mult=self.trunc_mult,
                        stride=stride,
                        split_file=str(self.split_file),
                        quick=self.quick
                    )
                    recon_t = w_res.runtime_sec
                    self.stats["total_runtime_sec"] += recon_t
                    if not w_res.is_success:
                        print(f"  ❌ TSDF {v_mm}mm reconstruct 실패: {w_res.status} ({w_res.error_message})")
                        self._log_decision("Phase B (Fusion)", cand_name, "FAILED", f"Worker failure: {w_res.status}")
                        fail_rec = self._fail_summary(cand_name, w_res.error_message or "worker crash", status=w_res.status)
                        fail_rec["fusion_method"] = "tsdf"
                        fail_rec["voxel_size_m"] = v
                        fail_rec["spec"] = spec.to_metadata_dict()
                        fusion_eval_results.append(fail_rec)
                        continue

                try:
                    summary = evaluate_reconstruction(
                        dataset_input=self.dataset,
                        trajectory_input=traj_path,
                        mesh_input=str(mesh_out),
                        output_dir=eval_dir,
                        candidate_name=cand_name,
                        split_json=str(self.split_file),
                        runtime_sec=recon_t,
                        cheap=True,
                        max_holdout_samples=self.stage1_samples
                    )
                    summary["fusion_method"] = "tsdf"
                    summary["voxel_size_m"] = v
                    summary["pcd_path"] = str(pcd_out)
                    summary["mesh_path"] = str(mesh_out)
                    summary["spec"] = spec.to_metadata_dict()
                    summary["spec_hash"] = spec_hash
                    summary["dataset_fingerprint"] = self.dataset_fingerprint
                    summary["split_hash"] = self.split_hash
                    self.stats["evaluated_count"] += 1
                    self._log_decision("Phase B (Fusion)", cand_name, "EXECUTED", "Evaluated TSDF voxel candidate")
                    fusion_eval_results.append(summary)
                except Exception as e:
                    print(f"❌ Phase B {v_mm}mm 평가 실패: {e}")
                    self._log_decision("Phase B (Fusion)", cand_name, "FAILED", f"Evaluation exception: {str(e)}")
                    fail_rec = self._fail_summary(cand_name, str(e), status="FAIL_EXCEPTION")
                    fail_rec["fusion_method"] = "tsdf"
                    fail_rec["voxel_size_m"] = v
                    fail_rec["spec"] = spec.to_metadata_dict()
                    fusion_eval_results.append(fail_rec)

            # 2. Direct Point Cloud Fusion (with common surface adapter for fair comparison)
            direct_v_mm = 10
            direct_cand_name = f"{slam_name}_direct{direct_v_mm}mm"
            direct_pcd_out = self.artifact_mgr.get_direct_pcd_path(slam_name, direct_v_mm)
            direct_mesh_out = self.artifact_mgr.get_mesh_path(f"{slam_name}_direct", direct_v_mm, method="poisson")
            direct_eval_dir = self.artifact_mgr.get_candidate_eval_dir(direct_cand_name)

            spec_direct = CandidateSpec(
                dataset_name=self.bag_name,
                slam_backend=slam_name,
                fusion_method="direct_pointcloud",
                fusion_params={"voxel_size_m": 0.010, "depth_min_m": self.depth_min, "depth_max_m": self.depth_max},
                surface_method="poisson",
                frame_stride=self._screening_stride()
            )
            spec_dir_hash = spec_direct.compute_spec_hash()

            print(f"▶️ Evaluating Direct Point Cloud Fusion Baseline: {direct_cand_name}")
            self.stats["total_candidates"] += 1

            if not is_artifact_valid(direct_pcd_out) or self.force:
                w_res = run_direct_fusion_worker(
                    dataset_dir=str(self.dataset.dataset_dir),
                    traj_file=traj_path,
                    pcd_path=str(direct_pcd_out),
                    voxel=0.010,
                    depth_min=self.depth_min,
                    depth_max=self.depth_max,
                    stride=self._screening_stride(),
                    split_file=str(self.split_file)
                )
                self.stats["total_runtime_sec"] += w_res.runtime_sec
                if not w_res.is_success:
                    print(f"  ❌ Direct fusion failed: {w_res.status} ({w_res.error_message})")
                    self._log_decision("Phase B (Fusion)", direct_cand_name, "FAILED", f"Direct fusion failed: {w_res.status}")
                else:
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
                        no_simplify=True
                    )
                    if w_surf.is_success:
                        try:
                            dir_summary = evaluate_reconstruction(
                                dataset_input=self.dataset,
                                trajectory_input=traj_path,
                                mesh_input=str(direct_mesh_out),
                                output_dir=direct_eval_dir,
                                candidate_name=direct_cand_name,
                                split_json=str(self.split_file),
                                runtime_sec=w_surf.runtime_sec,
                                cheap=True,
                                max_holdout_samples=self.stage1_samples
                            )
                            dir_summary["fusion_method"] = "direct_pointcloud"
                            dir_summary["voxel_size_m"] = 0.010
                            dir_summary["pcd_path"] = str(direct_pcd_out)
                            dir_summary["mesh_path"] = str(direct_mesh_out)
                            dir_summary["spec"] = spec_direct.to_metadata_dict()
                            dir_summary["spec_hash"] = spec_dir_hash
                            dir_summary["dataset_fingerprint"] = self.dataset_fingerprint
                            dir_summary["split_hash"] = self.split_hash
                            self.stats["evaluated_count"] += 1
                            fusion_eval_results.append(dir_summary)
                        except Exception as e:
                            print(f"⚠️ Direct fusion eval notice: {e}")

        # 3. Adaptive Fine Voxel (5mm) Check on Quality Gain & Plateau Detection (Section 19)
        if self.mode != "quick":
            valid_b_curr = [r for r in rank_candidate_summaries(fusion_eval_results) if r.get("hard_gate_pass", False)]
            if valid_b_curr:
                best_item = valid_b_curr[0]
                best_slam = best_item["summary_data"].get("spec", {}).get("requested_params", {}).get("slam_backend") or top_slams[0][0]
                best_traj = next((t[1] for t in top_slams if t[0] == best_slam), top_slams[0][1])

                # Check 10mm vs 8mm quality gain
                q_10mm = None
                q_8mm = None
                for r in valid_b_curr:
                    c_n = r["candidate_name"]
                    if f"{best_slam}_tsdf10mm" in c_n:
                        q_10mm = r.get("quality_score", 0.0)
                    elif f"{best_slam}_tsdf8mm" in c_n:
                        q_8mm = r.get("quality_score", 0.0)

                quality_gain = (q_8mm - q_10mm) if (q_8mm is not None and q_10mm is not None) else 0.0

                if q_8mm is not None and quality_gain < self.min_quality_gain_5mm:
                    print(f"⏭️ [Adaptive 5mm Skipped] Quality gain from 10mm to 8mm is {quality_gain:+.2f} pts (< {self.min_quality_gain_5mm} threshold) -> Plateau reached.")
                    self._log_decision("Phase B (Fusion)", f"{best_slam}_tsdf5mm", "SKIPPED_PLATEAU", f"8mm quality gain vs 10mm = +{quality_gain:.2f} < threshold {self.min_quality_gain_5mm}")
                else:
                    cand_5mm_name = f"{best_slam}_tsdf5mm"
                    mesh_5mm = self.artifact_mgr.get_mesh_path(best_slam, 5)
                    pcd_5mm = self.artifact_mgr.get_pcd_path(best_slam, 5)
                    eval_5mm_dir = self.artifact_mgr.get_candidate_eval_dir(cand_5mm_name)

                    spec_5mm = CandidateSpec(
                        dataset_name=self.bag_name,
                        slam_backend=best_slam,
                        fusion_method="tsdf",
                        fusion_params={"voxel_size_m": 0.005, "depth_min_m": self.depth_min, "depth_max_m": self.depth_max, "trunc_mult": self.trunc_mult},
                        surface_method="tsdf_direct",
                        frame_stride=self._screening_stride()
                    )

                    print(f"▶️ Adaptive Fine Voxel (5mm) Execution: {cand_5mm_name} (Gain was {quality_gain:+.2f} >= {self.min_quality_gain_5mm})")
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
                        quick=self.quick
                    )
                    if w_res.is_success:
                        try:
                            sum_5mm = evaluate_reconstruction(
                                dataset_input=self.dataset,
                                trajectory_input=best_traj,
                                mesh_input=str(mesh_5mm),
                                output_dir=eval_5mm_dir,
                                candidate_name=cand_5mm_name,
                                split_json=str(self.split_file),
                                runtime_sec=w_res.runtime_sec,
                                cheap=True,
                                max_holdout_samples=self.stage1_samples
                            )
                            sum_5mm["fusion_method"] = "tsdf"
                            sum_5mm["voxel_size_m"] = 0.005
                            sum_5mm["pcd_path"] = str(pcd_5mm)
                            sum_5mm["mesh_path"] = str(mesh_5mm)
                            sum_5mm["spec"] = spec_5mm.to_metadata_dict()
                            sum_5mm["spec_hash"] = spec_5mm.compute_spec_hash()
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
        # Diversity retention: ensure each surviving SLAM has at least 1 fusion pipeline in beam if valid
        seen_slams = set()
        for item in valid_b:
            cand_slam = item["summary_data"].get("spec", {}).get("requested_params", {}).get("slam_backend") or item["candidate_name"].split("_")[0]
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
        """Runs Phase C Surface exploration with unsimplified fair baseline and selects Top-3 Finalists."""
        print("\n==========================================================")
        print(f" 🚀 [PHASE C] Surface Reconstruction Exploration (across {len(top_fusion_pipelines)} Fusion Pipelines, Mode: {self.mode.upper()})")
        print("==========================================================")

        surface_eval_results: List[dict] = []

        tier_1 = ["tsdf_direct", "poisson"]
        tier_2 = ["alpha_shape", "bpa", "cgal_polygonal"] if (self.mode != "quick") else ["alpha_shape"]
        all_methods = tier_1 + tier_2

        for pipe_summary in top_fusion_pipelines:
            cand_base = pipe_summary.get("candidate_name", "candidate")
            spec_info = pipe_summary.get("spec", {}).get("requested_params", {})
            slam_name = spec_info.get("slam_backend") or cand_base.split("_")[0]
            traj_path = pipe_summary.get("trajectory_path") or trajectories.get(slam_name, "")
            fusion_method = pipe_summary.get("fusion_method", "tsdf")
            voxel_m = pipe_summary.get("voxel_size_m", 0.010)
            v_mm = int(round(voxel_m * 1000))
            pcd_file = pipe_summary.get("pcd_path") or str(self.artifact_mgr.get_pcd_path(slam_name, v_mm))

            for sm in all_methods:
                if fusion_method == "direct_pointcloud" and sm == "tsdf_direct":
                    continue

                cand_name = f"{cand_base}_{sm}"
                self.stats["total_candidates"] += 1
                mesh_out = self.artifact_mgr.get_mesh_path(f"{slam_name}_{fusion_method}", v_mm, method=sm)
                eval_dir = self.artifact_mgr.get_candidate_eval_dir(cand_name)

                spec = CandidateSpec(
                    dataset_name=self.bag_name,
                    slam_backend=slam_name,
                    slam_profile=spec_info.get("slam_profile", "normal"),
                    replay_rate=spec_info.get("replay_rate", 1.0),
                    fusion_method=fusion_method,
                    fusion_params=spec_info.get("fusion_params", {"voxel_size_m": voxel_m, "depth_min_m": self.depth_min, "depth_max_m": self.depth_max}),
                    surface_method=sm,
                    postprocess_params={"clean_density": True, "simplify_target": 0.0},
                    frame_stride=self._screening_stride()
                )
                spec_hash = spec.compute_spec_hash()

                print(f"▶️ Evaluating Surface Method: {cand_name}")

                # If TSDF direct, reuse Phase B mesh directly
                if sm == "tsdf_direct" and pipe_summary.get("mesh_path") and Path(pipe_summary["mesh_path"]).exists():
                    tsdf_dir_summary = copy.deepcopy(pipe_summary)
                    tsdf_dir_summary["candidate_name"] = cand_name
                    tsdf_dir_summary["surface_method"] = "tsdf_direct"
                    tsdf_dir_summary["spec"] = spec.to_metadata_dict()
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase C (Surface)", cand_name, "REUSED_PHASE_B", "Reused TSDF Direct mesh from Phase B")
                    surface_eval_results.append(tsdf_dir_summary)
                    continue

                cached_summary = self.artifact_mgr.should_reuse_evaluation(
                    cand_name, self.force, expected_spec_hash=spec_hash,
                    dataset_fingerprint=self.dataset_fingerprint, split_hash=self.split_hash
                )
                if cached_summary:
                    print(f"⏭️ Evaluation 재사용: {cand_name}")
                    cached_summary["surface_method"] = sm
                    cached_summary["fusion_method"] = fusion_method
                    cached_summary["voxel_size_m"] = voxel_m
                    cached_summary["spec"] = spec.to_metadata_dict()
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase C (Surface)", cand_name, "REUSED_CACHE", "Reused evaluation from cache")
                    surface_eval_results.append(cached_summary)
                    continue

                surf_t = 0.0
                if is_artifact_valid(mesh_out) and not self.force:
                    print(f"⏭️ Mesh 재사용: {mesh_out.name}")
                    self._log_decision("Phase C (Surface)", cand_name, "REUSED_MESH", "Reused existing surface mesh")
                else:
                    if not is_artifact_valid(pcd_file):
                        print(f"  ❌ PCD missing for surface worker: {pcd_file}")
                        continue

                    w_res = run_surface_worker(
                        input_ply=str(pcd_file),
                        output_mesh=str(mesh_out),
                        method=sm,
                        voxel=voxel_m,
                        depth=8,
                        simplify=0.0,
                        no_simplify=True
                    )
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
                        surface_eval_results.append(fail_rec)
                        continue

                try:
                    summary = evaluate_reconstruction(
                        dataset_input=self.dataset,
                        trajectory_input=traj_path,
                        mesh_input=str(mesh_out),
                        output_dir=eval_dir,
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
                    summary["spec"] = spec.to_metadata_dict()
                    summary["spec_hash"] = spec_hash
                    summary["dataset_fingerprint"] = self.dataset_fingerprint
                    summary["split_hash"] = self.split_hash
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
                    surface_eval_results.append(fail_rec)

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

            # Reconstruct CandidateSpec preserving all exact params
            spec_info = f_summary.get("spec", {}).get("requested_params", {})
            slam_backend = spec_info.get("slam_backend") or cand_name.split("_")[0]
            slam_profile = spec_info.get("slam_profile", "normal")
            replay_rate = float(spec_info.get("replay_rate", 1.0))
            fusion_method = spec_info.get("fusion_method") or f_summary.get("fusion_method", "tsdf")
            surface_method = spec_info.get("surface_method") or f_summary.get("surface_method", "tsdf_direct")
            voxel_m = float(spec_info.get("fusion_params", {}).get("voxel_size_m", f_summary.get("voxel_size_m", 0.010)))
            v_mm = int(round(voxel_m * 1000))

            traj_path = f_summary.get("trajectory_path") or trajectories.get(slam_backend, "")

            full_spec = CandidateSpec(
                dataset_name=self.bag_name,
                slam_backend=slam_backend,
                slam_profile=slam_profile,
                replay_rate=replay_rate,
                fusion_method=fusion_method,
                fusion_params={"voxel_size_m": voxel_m, "depth_min_m": self.depth_min, "depth_max_m": self.depth_max, "trunc_mult": self.trunc_mult},
                surface_method=surface_method,
                postprocess_params={"clean_density": True, "simplify_target": 0.0},
                frame_stride=1,
                is_full_rebuild=True,
                evaluation_profile="full"
            )
            full_spec_hash = full_spec.compute_spec_hash()
            full_candidate_id = full_spec.compute_candidate_id()

            # Unique isolated directory per finalist
            cand_artifact_dir = self.artifact_mgr.get_candidate_artifact_dir(full_candidate_id)
            rebuild_pcd = cand_artifact_dir / f"{self.bag_name}_{full_candidate_id}_cloud.ply"
            rebuild_mesh = cand_artifact_dir / f"{self.bag_name}_{full_candidate_id}.obj"
            rebuild_eval_dir = self.artifact_mgr.get_candidate_eval_dir(rebuild_cand_name)

            print(f"\n🔨 [Full Rebuild #{rank_idx}] Candidate: `{rebuild_cand_name}`")
            print(f"   SLAM: {slam_backend} (profile={slam_profile}, rate={replay_rate}), Fusion: {fusion_method} ({v_mm}mm), Surface: {surface_method}, Stride: 1 (FULL)")
            print(f"   Isolated Output Mesh: {rebuild_mesh.name}")

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
                    depth=8,
                    simplify=0.0,
                    no_simplify=True
                )
                if not w_surf.is_success:
                    print(f"❌ Rebuild surface failed: {w_surf.status}")
                    continue
            else:
                # TSDF Full Rebuild (stride=1, Train frames only)
                w_res = run_tsdf_worker(
                    dataset_dir=str(self.dataset.dataset_dir),
                    traj_file=traj_path,
                    mesh_path=str(rebuild_mesh) if surface_method == "tsdf_direct" else None,
                    pcd_path=str(rebuild_pcd),
                    voxel=voxel_m,
                    depth_max=self.depth_max,
                    trunc_mult=self.trunc_mult,
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
                        depth=8,
                        simplify=0.0,
                        no_simplify=True
                    )
                    if not w_surf.is_success:
                        print(f"❌ Rebuild surface failed: {w_surf.status}")
                        continue

            rb_runtime = time.time() - t_rb_start
            self.stats["total_runtime_sec"] += rb_runtime
            self.stats["rebuilt_count"] += 1

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
                full_summary["is_full_rebuild"] = True
                full_summary["trajectory_metrics"] = f_summary.get("trajectory_metrics", {})
                full_summary["spec"] = full_spec.to_metadata_dict()
                full_summary["spec_hash"] = full_spec_hash
                full_summary["dataset_fingerprint"] = self.dataset_fingerprint
                full_summary["split_hash"] = self.split_hash
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
