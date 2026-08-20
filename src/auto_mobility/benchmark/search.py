"""
search.py — Multi-Axis Candidate Search Engine.

Implements sequential exploration with winner propagation:
  - Phase A: Evaluates SLAM trajectories on fixed 10mm TSDF -> Ranks and selects actual best SLAM.
  - Phase B: Uses best SLAM -> Evaluates voxel sizes (reusing 10mm from Phase A) -> Ranks and selects best TSDF voxel & PCD.
  - Phase C: Uses best PCD from Phase B -> Evaluates surface methods (reusing TSDF direct from Phase B) -> Ranks surface methods.
  - Overall Joint Ranking: Combines all valid candidates and determines the overall best reconstruction.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from auto_mobility.config import MESH_DIR, POINTCLOUD_DIR, EVALUATION_DIR
from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.evaluation.evaluator import evaluate_reconstruction
from auto_mobility.benchmark.artifacts import ArtifactManager, is_artifact_valid
from auto_mobility.benchmark.workers import run_tsdf_worker, run_surface_worker, WorkerStatus
from auto_mobility.benchmark.scoring import rank_candidate_summaries, HardGateFilter


class SearchEngine:
    """Executes multi-axis exploration stages with caching, decision trace, and winner propagation."""

    def __init__(
        self,
        bag_name: str,
        dataset: FrameDataset,
        split_file: Path,
        artifact_mgr: ArtifactManager,
        quick: bool = False,
        full: bool = False,
        mode: str = "standard",
        force: bool = False
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

        # Search trace & execution statistics
        self.decision_trace: List[Dict[str, str]] = []
        self.stats: Dict[str, Any] = {
            "total_candidates": 0,
            "evaluated_count": 0,
            "cached_count": 0,
            "failed_count": 0,
            "pruned_count": 0,
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

    # ───────────────────────────────────────────────────────────
    # PHASE A: SLAM Backend Comparison (Fixed 10mm TSDF)
    # ───────────────────────────────────────────────────────────
    def run_phase_a(
        self,
        trajectories: Dict[str, str],
        traj_metrics: Dict[str, dict]
    ) -> Tuple[List[dict], str, str]:
        """Runs Phase A SLAM comparison and returns (results, best_slam_name, best_slam_traj)."""
        print("\n==========================================================")
        print(f" 🚀 [PHASE A] SLAM Backend Comparison (Fixed 10mm TSDF, Mode: {self.mode.upper()})")
        print("==========================================================")
        slam_eval_results: List[dict] = []

        for slam_k, traj_file in trajectories.items():
            self.stats["total_candidates"] += 1
            print(f"▶️ Evaluating SLAM candidate: {slam_k}")
            cand_name = f"{slam_k}_voxel10mm"
            mesh_out = self.artifact_mgr.get_mesh_path(slam_k, 10)
            pcd_out = self.artifact_mgr.get_pcd_path(slam_k, 10)
            eval_dir = self.artifact_mgr.get_candidate_eval_dir(cand_name)

            cached_summary = self.artifact_mgr.should_reuse_evaluation(cand_name, self.force)
            if cached_summary:
                print(f"⏭️ Evaluation 재사용: {cand_name}")
                self.stats["cached_count"] += 1
                self._log_decision("Phase A (SLAM)", cand_name, "REUSED", "Reused existing valid evaluation from cache")
                cached_summary["trajectory_metrics"] = traj_metrics.get(slam_k, {})
                slam_eval_results.append(cached_summary)
                continue

            recon_t = 0.0
            if self.artifact_mgr.should_reuse_reconstruction(mesh_out, pcd_out, cand_name, self.force):
                print(f"⏭️ Mesh & PCD 재사용: {mesh_out.name}")
                self._log_decision("Phase A (SLAM)", cand_name, "REUSED_MESH", "Reused existing valid mesh/pcd artifact")
            else:
                w_res = run_tsdf_worker(
                    dataset_dir=str(self.dataset.dataset_dir),
                    traj_file=traj_file,
                    mesh_path=str(mesh_out),
                    pcd_path=str(pcd_out),
                    voxel=0.010,
                    split_file=str(self.split_file),
                    quick=self.quick
                )
                recon_t = w_res.runtime_sec
                self.stats["total_runtime_sec"] += recon_t
                if not w_res.is_success:
                    print(f"  ❌ SLAM {slam_k} reconstruct 실패: {w_res.status} ({w_res.error_message})")
                    self._log_decision("Phase A (SLAM)", cand_name, "FAILED", f"TSDF worker failed: {w_res.status}")
                    slam_eval_results.append(self._fail_summary(cand_name, w_res.error_message or "worker crash", status=w_res.status))
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
                    cheap=True
                )
                summary["trajectory_metrics"] = traj_metrics.get(slam_k, {})
                self.stats["evaluated_count"] += 1
                self._log_decision("Phase A (SLAM)", cand_name, "EXECUTED", "Successfully evaluated with cheap screening")
                slam_eval_results.append(summary)
            except Exception as e:
                print(f"❌ Phase A {slam_k} 평가 실패: {e}")
                self._log_decision("Phase A (SLAM)", cand_name, "FAILED", f"Evaluation exception: {str(e)}")
                slam_eval_results.append(self._fail_summary(cand_name, str(e), status="FAIL_EXCEPTION"))

        # Rank Phase A candidates to select the ACTUAL best SLAM
        ranked_a = rank_candidate_summaries(slam_eval_results)
        valid_winners = [r for r in ranked_a if r.get("hard_gate_pass", False)]
        
        if valid_winners:
            best_cand = valid_winners[0]["candidate_name"]
            # e.g. "rtab_rgbd_voxel10mm" -> "rtab_rgbd"
            best_slam = best_cand.replace("_voxel10mm", "")
            self._log_decision("Phase A (SLAM)", best_slam, "WINNER", f"Selected as best SLAM backend (Quality: {valid_winners[0].get('quality_score', 0):.1f})")
            print(f"🥇 [Phase A Winner] SLAM: `{best_slam}` (Quality Score: {valid_winners[0].get('quality_score', 0):.1f}, Composite: {valid_winners[0]['composite_score']:.1f})")
        else:
            best_slam = list(trajectories.keys())[0] if trajectories else "rtab_rgbd"
            self._log_decision("Phase A (SLAM)", best_slam, "FALLBACK", "No passing candidates; selected default fallback")
            print(f"⚠️ [Phase A Notice] 유효한 PASS 후보 없음 -> 기본 `{best_slam}` 선택")

        best_traj = trajectories.get(best_slam, list(trajectories.values())[0] if trajectories else "")
        return slam_eval_results, best_slam, best_traj

    # ───────────────────────────────────────────────────────────
    # PHASE B: TSDF Fusion Comparison (Adaptive Search, Fixed: Best SLAM)
    # ───────────────────────────────────────────────────────────
    def run_phase_b(
        self,
        best_slam: str,
        best_traj: str,
        phase_a_results: List[dict]
    ) -> Tuple[List[dict], float, Path, Path, dict]:
        """Runs Phase B Adaptive TSDF comparison and returns (results, best_voxel_m, best_pcd, best_mesh, best_summary)."""
        print("\n==========================================================")
        print(f" 🚀 [PHASE B] Adaptive TSDF Resolution Search (Fixed SLAM: {best_slam})")
        print("==========================================================")

        tsdf_eval_results: List[dict] = []

        # Index Phase A results by candidate_name to reuse 10mm result directly
        phase_a_by_cand = {s.get("candidate_name"): s for s in phase_a_results}
        cand_10mm_name = f"{best_slam}_voxel10mm"

        # 1. Evaluate baseline 10mm & coarse 20mm
        base_voxels = [0.020, 0.010]

        for v in base_voxels:
            v_mm = int(round(v * 1000))
            v_tag = f"tsdf_{v_mm}mm"
            cand_name = f"{best_slam}_voxel{v_mm}mm"
            mesh_out = self.artifact_mgr.get_mesh_path(best_slam, v_mm)
            pcd_out = self.artifact_mgr.get_pcd_path(best_slam, v_mm)
            eval_dir = self.artifact_mgr.get_candidate_eval_dir(cand_name)

            print(f"▶️ Evaluating TSDF Voxel: {v_tag} ({v*1000:.1f}mm)")
            self.stats["total_candidates"] += 1

            # Check if 10mm was already evaluated in Phase A
            if v_mm == 10 and cand_10mm_name in phase_a_by_cand:
                existing = phase_a_by_cand[cand_10mm_name]
                if existing.get("status") not in ("FAIL", "FAIL_CRASH", "FAIL_SEGFAULT", "FAIL_OOM"):
                    print(f"♻️ Phase A 10mm 결과 직접 재사용: {cand_name}")
                    existing_copy = dict(existing)
                    existing_copy["voxel_size_m"] = v
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase B (TSDF)", cand_name, "REUSED_PHASE_A", "Reused Phase A 10mm TSDF directly")
                    tsdf_eval_results.append(existing_copy)
                    continue

            cached_summary = self.artifact_mgr.should_reuse_evaluation(cand_name, self.force)
            if cached_summary:
                print(f"⏭️ Evaluation 재사용: {cand_name}")
                cached_summary["voxel_size_m"] = v
                self.stats["cached_count"] += 1
                self._log_decision("Phase B (TSDF)", cand_name, "REUSED_CACHE", "Reused evaluation from cache")
                tsdf_eval_results.append(cached_summary)
                continue

            recon_t = 0.0
            if self.artifact_mgr.should_reuse_reconstruction(mesh_out, pcd_out, cand_name, self.force):
                print(f"⏭️ Mesh & PCD 재사용: {mesh_out.name}")
                self._log_decision("Phase B (TSDF)", cand_name, "REUSED_MESH", "Reused existing mesh & PCD files")
            else:
                w_res = run_tsdf_worker(
                    dataset_dir=str(self.dataset.dataset_dir),
                    traj_file=best_traj,
                    mesh_path=str(mesh_out),
                    pcd_path=str(pcd_out),
                    voxel=v,
                    split_file=str(self.split_file),
                    quick=self.quick
                )
                recon_t = w_res.runtime_sec
                self.stats["total_runtime_sec"] += recon_t
                if not w_res.is_success:
                    print(f"  ❌ TSDF {v_mm}mm reconstruct 실패: {w_res.status} ({w_res.error_message})")
                    self._log_decision("Phase B (TSDF)", cand_name, "FAILED", f"Worker failure: {w_res.status}")
                    tsdf_eval_results.append({
                        "candidate_name": cand_name,
                        "voxel_size_m": v,
                        "status": w_res.status,
                        "overall_status": w_res.status,
                        "error": w_res.error_message or "reconstruct crashed",
                        "geometry": {}, "mesh": {}
                    })
                    continue

            try:
                summary = evaluate_reconstruction(
                    dataset_input=self.dataset,
                    trajectory_input=best_traj,
                    mesh_input=str(mesh_out),
                    output_dir=eval_dir,
                    candidate_name=cand_name,
                    split_json=str(self.split_file),
                    runtime_sec=recon_t,
                    cheap=True
                )
                summary["voxel_size_m"] = v
                self.stats["evaluated_count"] += 1
                self._log_decision("Phase B (TSDF)", cand_name, "EXECUTED", "Evaluated voxel candidate")
                tsdf_eval_results.append(summary)
            except Exception as e:
                print(f"❌ Phase B {v_mm}mm 평가 실패: {e}")
                self._log_decision("Phase B (TSDF)", cand_name, "FAILED", f"Evaluation exception: {str(e)}")
                tsdf_eval_results.append({
                    "candidate_name": cand_name,
                    "voxel_size_m": v,
                    "status": "FAIL_EXCEPTION",
                    "overall_status": "FAIL_EXCEPTION",
                    "error": str(e),
                    "geometry": {}, "mesh": {}
                })

        # 2. Adaptive 5mm Decision:
        # Compare 10mm vs 20mm quality score. Only run 5mm if 10mm shows substantial gain and resource preflight passes.
        should_run_5mm = (self.mode == "full") or (not self.quick)
        if should_run_5mm and self.mode != "full":
            s_10 = next((s for s in tsdf_eval_results if s.get("voxel_size_m") == 0.010 or "voxel10mm" in s.get("candidate_name", "")), None)
            s_20 = next((s for s in tsdf_eval_results if s.get("voxel_size_m") == 0.020 or "voxel20mm" in s.get("candidate_name", "")), None)
            if s_10 and s_20 and s_10.get("geometry") and s_20.get("geometry"):
                ranked_10_20 = rank_candidate_summaries([s_10, s_20])
                score_10 = next((r["quality_score"] for r in ranked_10_20 if "voxel10mm" in r["candidate_name"]), 0.0)
                score_20 = next((r["quality_score"] for r in ranked_10_20 if "voxel20mm" in r["candidate_name"]), 0.0)
                diff = score_10 - score_20
                if diff < 1.5:
                    print(f"💡 [Adaptive TSDF] 10mm가 20mm 대비 품질 향상 미미함 (+{diff:.1f} pts < +1.5 pts) → 5mm 고비용 탐색 생략 (자원 절약)")
                    self._log_decision("Phase B (TSDF)", f"{best_slam}_voxel5mm", "PRUNED", f"10mm quality gain (+{diff:.1f} pts) < 1.5 pts threshold")
                    self.stats["pruned_count"] += 1
                    should_run_5mm = False
                else:
                    # Resource Preflight Check for 5mm fine voxel grid
                    try:
                        import psutil
                        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
                        if avail_gb < 3.0:
                            print(f"⚠️ [Adaptive TSDF Resource Preflight] 가용 RAM 부족 ({avail_gb:.1f} GB < 3.0 GB) → 5mm 탐색 생략 (OOM 방지)")
                            self._log_decision("Phase B (TSDF)", f"{best_slam}_voxel5mm", "SKIPPED_RESOURCE", f"Available RAM ({avail_gb:.1f} GB) < 3.0 GB required")
                            should_run_5mm = False
                        else:
                            print(f"💡 [Adaptive TSDF] 10mm가 20mm 대비 유의미한 품질 향상 확인 (+{diff:.1f} pts, 가용 RAM {avail_gb:.1f} GB) → 5mm 미세 탐색 진행")
                    except Exception:
                        print(f"💡 [Adaptive TSDF] 10mm가 20mm 대비 유의미한 품질 향상 확인 (+{diff:.1f} pts) → 5mm 미세 탐색 진행")
        elif not should_run_5mm:
            self._log_decision("Phase B (TSDF)", f"{best_slam}_voxel5mm", "PRUNED", "Quick mode active; skipped fine voxel")
            self.stats["pruned_count"] += 1

        if should_run_5mm:
            v = 0.005
            v_mm = 5
            v_tag = f"tsdf_{v_mm}mm"
            cand_name = f"{best_slam}_voxel{v_mm}mm"
            self.stats["total_candidates"] += 1
            mesh_out = self.artifact_mgr.get_mesh_path(best_slam, v_mm)
            pcd_out = self.artifact_mgr.get_pcd_path(best_slam, v_mm)
            eval_dir = self.artifact_mgr.get_candidate_eval_dir(cand_name)

            print(f"▶️ Evaluating TSDF Voxel: {v_tag} ({v*1000:.1f}mm)")
            cached_summary = self.artifact_mgr.should_reuse_evaluation(cand_name, self.force)
            if cached_summary:
                print(f"⏭️ Evaluation 재사용: {cand_name}")
                cached_summary["voxel_size_m"] = v
                self.stats["cached_count"] += 1
                self._log_decision("Phase B (TSDF)", cand_name, "REUSED_CACHE", "Reused 5mm evaluation from cache")
                tsdf_eval_results.append(cached_summary)
            else:
                recon_t = 0.0
                if self.artifact_mgr.should_reuse_reconstruction(mesh_out, pcd_out, cand_name, self.force):
                    print(f"⏭️ Mesh & PCD 재사용: {mesh_out.name}")
                    self._log_decision("Phase B (TSDF)", cand_name, "REUSED_MESH", "Reused existing 5mm mesh & PCD")
                else:
                    w_res = run_tsdf_worker(
                        dataset_dir=str(self.dataset.dataset_dir),
                        traj_file=best_traj,
                        mesh_path=str(mesh_out),
                        pcd_path=str(pcd_out),
                        voxel=v,
                        split_file=str(self.split_file),
                        quick=self.quick
                    )
                    recon_t = w_res.runtime_sec
                    self.stats["total_runtime_sec"] += recon_t
                    if not w_res.is_success:
                        print(f"  ❌ TSDF {v_mm}mm reconstruct 실패: {w_res.status} ({w_res.error_message})")
                        self._log_decision("Phase B (TSDF)", cand_name, "FAILED", f"5mm reconstruct failed: {w_res.status}")
                        tsdf_eval_results.append({
                            "candidate_name": cand_name,
                            "voxel_size_m": v,
                            "status": w_res.status,
                            "overall_status": w_res.status,
                            "error": w_res.error_message or "reconstruct crashed",
                            "geometry": {}, "mesh": {}
                        })
                    else:
                        try:
                            summary = evaluate_reconstruction(
                                dataset_input=self.dataset,
                                trajectory_input=best_traj,
                                mesh_input=str(mesh_out),
                                output_dir=eval_dir,
                                candidate_name=cand_name,
                                split_json=str(self.split_file),
                                runtime_sec=recon_t,
                                cheap=True
                            )
                            summary["voxel_size_m"] = v
                            self.stats["evaluated_count"] += 1
                            self._log_decision("Phase B (TSDF)", cand_name, "EXECUTED", "Evaluated 5mm fine voxel")
                            tsdf_eval_results.append(summary)
                        except Exception as e:
                            print(f"❌ Phase B {v_mm}mm 평가 실패: {e}")
                            self._log_decision("Phase B (TSDF)", cand_name, "FAILED", f"5mm eval exception: {str(e)}")
                            tsdf_eval_results.append({
                                "candidate_name": cand_name,
                                "voxel_size_m": v,
                                "status": "FAIL_EXCEPTION",
                                "overall_status": "FAIL_EXCEPTION",
                                "error": str(e),
                                "geometry": {}, "mesh": {}
                            })

        # Rank Phase B candidates to select the ACTUAL best TSDF voxel
        ranked_b = rank_candidate_summaries(tsdf_eval_results)
        valid_b = [r for r in ranked_b if r.get("hard_gate_pass", False)]

        if valid_b:
            best_cand = valid_b[0]["candidate_name"]
            best_summary = valid_b[0]["summary_data"]
            best_voxel_m = best_summary.get("voxel_size_m")
            if best_voxel_m is None:
                # Fallback: parse from candidate_name e.g. "rtab_rgbd_voxel10mm" -> 0.010
                try:
                    import re
                    m = re.search(r"voxel(\d+)mm", best_cand)
                    best_voxel_m = float(m.group(1)) / 1000.0 if m else 0.010
                except Exception:
                    best_voxel_m = 0.010
            self._log_decision("Phase B (TSDF)", best_cand, "WINNER", f"Selected as best TSDF voxel (Quality: {valid_b[0].get('quality_score', 0):.1f})")
            print(f"🥇 [Phase B Winner] TSDF: `{best_cand}` ({best_voxel_m*1000:.1f}mm, Quality: {valid_b[0].get('quality_score', 0):.1f}, Score: {valid_b[0]['composite_score']:.1f})")
        else:
            best_voxel_m = 0.010
            best_cand = f"{best_slam}_voxel10mm"
            best_summary = {}
            self._log_decision("Phase B (TSDF)", best_cand, "FALLBACK", "No passing voxel candidates; selected default 10mm")
            print(f"⚠️ [Phase B Notice] 유효한 PASS 후보 없음 -> 기본 10mm 선택")

        best_v_mm = int(round(best_voxel_m * 1000))
        best_pcd = self.artifact_mgr.get_pcd_path(best_slam, best_v_mm)
        best_mesh = self.artifact_mgr.get_mesh_path(best_slam, best_v_mm)

        return tsdf_eval_results, best_voxel_m, best_pcd, best_mesh, best_summary

    # ───────────────────────────────────────────────────────────
    # PHASE C: Surface Reconstruction Comparison (Tiered Search)
    # ───────────────────────────────────────────────────────────
    def run_phase_c(
        self,
        best_slam: str,
        best_traj: str,
        best_voxel_m: float,
        best_pcd: Path,
        best_tsdf_mesh: Path,
        best_tsdf_summary: dict,
        all_surfaces: bool = False
    ) -> Tuple[List[dict], dict]:
        """Runs Phase C Tiered Surface comparison reusing the best PCD from Phase B."""
        best_v_mm = int(round(best_voxel_m * 1000))
        print("\n==========================================================")
        print(f" 🚀 [PHASE C] Tiered Surface Reconstruction Search (Upstream: {best_pcd.name}, Mode: {self.mode.upper()})")
        print("==========================================================")

        # Ensure base PCD exists (should already exist from Phase B)
        if not best_pcd.exists() or self.force:
            print(f"⚙️ Phase C 기준 Point Cloud 생성: {best_pcd.name}")
            w_res = run_tsdf_worker(
                dataset_dir=str(self.dataset.dataset_dir),
                traj_file=best_traj,
                mesh_path=None,
                pcd_path=str(best_pcd),
                voxel=best_voxel_m,
                split_file=str(self.split_file),
                quick=self.quick
            )
            if not w_res.is_success:
                print(f"❌ Phase C 기준 Point Cloud 생성 실패: {w_res.error_message}")
        else:
            print(f"♻️ Phase B 우수 Point Cloud 재사용: {best_pcd.name}")

        # Tier 1 Surface Backends: TSDF Direct & Poisson
        tier_1 = ["tsdf_direct", "poisson"]
        # Tier 2 Surface Backends: Alpha Shape, BPA, CGAL Polygonal
        tier_2 = ["alpha_shape", "bpa", "cgal_polygonal"] if (self.mode != "quick") else ["alpha_shape"]

        surface_eval_results: List[dict] = []

        # 1. Run Tier 1 Surface Methods
        for sm in tier_1:
            cand_name = f"{best_slam}_voxel{best_v_mm}mm_{sm}"
            self.stats["total_candidates"] += 1
            print(f"▶️ Evaluating Tier-1 Surface Backend: {sm}")

            if sm == "tsdf_direct":
                if best_tsdf_summary and best_tsdf_mesh.exists():
                    print(f"♻️ Phase B TSDF Direct 메쉬 직접 재사용: {best_tsdf_mesh.name}")
                    tsdf_dir_summary = dict(best_tsdf_summary)
                    tsdf_dir_summary["candidate_name"] = cand_name
                    tsdf_dir_summary["surface_method"] = "tsdf_direct"
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase C (Surface)", cand_name, "REUSED_PHASE_B", "Reused TSDF Direct mesh from Phase B")
                    surface_eval_results.append(tsdf_dir_summary)
                    continue

            mesh_out = self.artifact_mgr.get_mesh_path(best_slam, best_v_mm, method=sm)
            eval_dir = self.artifact_mgr.get_candidate_eval_dir(cand_name)

            cached_summary = self.artifact_mgr.should_reuse_evaluation(cand_name, self.force)
            if cached_summary:
                print(f"⏭️ Evaluation 재사용: {cand_name}")
                cached_summary["surface_method"] = sm
                self.stats["cached_count"] += 1
                self._log_decision("Phase C (Surface)", cand_name, "REUSED_CACHE", "Reused evaluation from cache")
                surface_eval_results.append(cached_summary)
                continue

            surf_t = 0.0
            if is_artifact_valid(mesh_out) and not self.force:
                print(f"⏭️ Mesh 재사용: {mesh_out.name}")
                self._log_decision("Phase C (Surface)", cand_name, "REUSED_MESH", "Reused existing mesh file")
            else:
                if sm == "tsdf_direct":
                    w_res = run_tsdf_worker(
                        dataset_dir=str(self.dataset.dataset_dir),
                        traj_file=best_traj,
                        mesh_path=str(mesh_out),
                        pcd_path=None,
                        voxel=best_voxel_m,
                        split_file=str(self.split_file),
                        quick=self.quick
                    )
                else:
                    w_res = run_surface_worker(
                        input_ply=str(best_pcd),
                        output_mesh=str(mesh_out),
                        method=sm,
                        voxel=best_voxel_m,
                        depth=8,
                    )
                surf_t = w_res.runtime_sec
                self.stats["total_runtime_sec"] += surf_t
                if not w_res.is_success:
                    print(f"  ❌ Surface {sm} 생성 실패: {w_res.status} ({w_res.error_message})")
                    self._log_decision("Phase C (Surface)", cand_name, "FAILED", f"Surface worker failed: {w_res.status}")
                    surface_eval_results.append(self._fail_summary(cand_name, w_res.error_message or "surface generation failed", status=w_res.status))
                    continue

            try:
                summary = evaluate_reconstruction(
                    dataset_input=self.dataset,
                    trajectory_input=best_traj,
                    mesh_input=str(mesh_out),
                    output_dir=eval_dir,
                    candidate_name=cand_name,
                    split_json=str(self.split_file),
                    runtime_sec=surf_t,
                    cheap=True
                )
                summary["surface_method"] = sm
                self.stats["evaluated_count"] += 1
                self._log_decision("Phase C (Surface)", cand_name, "EXECUTED", "Evaluated Tier 1 surface method")
                surface_eval_results.append(summary)
            except Exception as e:
                print(f"❌ Phase C {sm} 평가 실패: {e}")
                self._log_decision("Phase C (Surface)", cand_name, "FAILED", f"Surface eval exception: {str(e)}")
                surface_eval_results.append(self._fail_summary(cand_name, str(e), status="FAIL_EXCEPTION"))

        # 2. Check Tier 1 Results to decide whether to evaluate Tier 2
        ranked_t1 = rank_candidate_summaries(surface_eval_results)
        valid_t1 = [r for r in ranked_t1 if r.get("hard_gate_pass", False)]
        tier1_best_score = valid_t1[0]["quality_score"] if valid_t1 else 0.0

        run_tier_2 = (self.mode == "full") or all_surfaces or (tier1_best_score < 70.0) or (not self.quick and not valid_t1)
        if not run_tier_2 and not all_surfaces:
            print(f"💡 [Adaptive Surface] Tier 1 우수 후보 확인 (Quality Score: {tier1_best_score:.1f} >= 70.0) → 추가 Tier-2 탐색 생략 (시간 단축)")
            for sm in tier_2:
                self._log_decision("Phase C (Surface)", f"{best_slam}_voxel{best_v_mm}mm_{sm}", "PRUNED", f"Tier 1 candidate quality ({tier1_best_score:.1f}) >= 70.0 threshold")
                self.stats["pruned_count"] += 1
        else:
            print(f"💡 [Adaptive Surface] Tier-2 추가 탐색 진행: {tier_2}")
            for sm in tier_2:
                cand_name = f"{best_slam}_voxel{best_v_mm}mm_{sm}"
                self.stats["total_candidates"] += 1
                print(f"▶️ Evaluating Tier-2 Surface Backend: {sm}")

                mesh_out = self.artifact_mgr.get_mesh_path(best_slam, best_v_mm, method=sm)
                eval_dir = self.artifact_mgr.get_candidate_eval_dir(cand_name)

                cached_summary = self.artifact_mgr.should_reuse_evaluation(cand_name, self.force)
                if cached_summary:
                    print(f"⏭️ Evaluation 재사용: {cand_name}")
                    cached_summary["surface_method"] = sm
                    self.stats["cached_count"] += 1
                    self._log_decision("Phase C (Surface)", cand_name, "REUSED_CACHE", "Reused evaluation from cache")
                    surface_eval_results.append(cached_summary)
                    continue

                surf_t = 0.0
                if is_artifact_valid(mesh_out) and not self.force:
                    print(f"⏭️ Mesh 재사용: {mesh_out.name}")
                    self._log_decision("Phase C (Surface)", cand_name, "REUSED_MESH", "Reused existing mesh")
                else:
                    w_res = run_surface_worker(
                        input_ply=str(best_pcd),
                        output_mesh=str(mesh_out),
                        method=sm,
                        voxel=best_voxel_m,
                        depth=7,
                    )
                    surf_t = w_res.runtime_sec
                    self.stats["total_runtime_sec"] += surf_t
                    if not w_res.is_success:
                        print(f"  ❌ Surface {sm} 생성 실패: {w_res.status} ({w_res.error_message})")
                        self._log_decision("Phase C (Surface)", cand_name, "FAILED", f"Surface worker failed: {w_res.status}")
                        surface_eval_results.append(self._fail_summary(cand_name, w_res.error_message or "surface generation failed", status=w_res.status))
                        continue

                try:
                    summary = evaluate_reconstruction(
                        dataset_input=self.dataset,
                        trajectory_input=best_traj,
                        mesh_input=str(mesh_out),
                        output_dir=eval_dir,
                        candidate_name=cand_name,
                        split_json=str(self.split_file),
                        runtime_sec=surf_t,
                        cheap=True
                    )
                    summary["surface_method"] = sm
                    self.stats["evaluated_count"] += 1
                    self._log_decision("Phase C (Surface)", cand_name, "EXECUTED", "Evaluated Tier 2 surface method")
                    surface_eval_results.append(summary)
                except Exception as e:
                    print(f"❌ Phase C {sm} 평가 실패: {e}")
                    self._log_decision("Phase C (Surface)", cand_name, "FAILED", f"Surface eval exception: {str(e)}")
                    surface_eval_results.append(self._fail_summary(cand_name, str(e), status="FAIL_EXCEPTION"))

        # Rank Phase C candidates
        ranked_c = rank_candidate_summaries(surface_eval_results)
        valid_c = [r for r in ranked_c if r.get("hard_gate_pass", False)]
        winner_c = valid_c[0] if valid_c else (ranked_c[0] if ranked_c else {})
        if valid_c:
            self._log_decision("Phase C (Surface)", winner_c["candidate_name"], "WINNER", f"Selected as best surface method (Quality: {winner_c.get('quality_score', 0):.1f})")
            print(f"🥇 [Phase C Winner] Surface: `{winner_c['candidate_name']}` (Quality: {winner_c.get('quality_score', 0):.1f}, Score: {winner_c['composite_score']:.1f})")

        return surface_eval_results, winner_c

    # ───────────────────────────────────────────────────────────
    # Joint Overall Ranking & Full-Fidelity Winner Validation
    # ───────────────────────────────────────────────────────────
    def compute_overall_rankings(
        self,
        phase_a_results: List[dict],
        phase_b_results: List[dict],
        phase_c_results: List[dict],
        run_full_eval_on_winner: bool = True
    ) -> Tuple[List[dict], Optional[dict]]:
        """Collects all evaluated candidates, determines preliminary winner,
        runs Full Fidelity evaluation on Top-1 winner, and produces final rankings."""
        seen_candidates = {}
        for s in (phase_a_results + phase_b_results + phase_c_results):
            c_name = s.get("candidate_name")
            if c_name and c_name not in seen_candidates:
                seen_candidates[c_name] = s

        all_summaries = list(seen_candidates.values())
        overall_ranked = rank_candidate_summaries(all_summaries)

        winner = None
        valid_ranked = [r for r in overall_ranked if r.get("hard_gate_pass", False)]
        if valid_ranked:
            winner = valid_ranked[0]
            winner_cand_name = winner["candidate_name"]
            winner_summary = winner["summary_data"]

            # Run Full-Fidelity Evaluation on the top winner to produce high-res metrics & heatmap renders
            if run_full_eval_on_winner:
                mesh_p = winner_summary.get("mesh_path")
                traj_p = winner_summary.get("trajectory_path")
                if mesh_p and traj_p and Path(mesh_p).exists() and Path(traj_p).exists():
                    print(f"\n🔬 [Full-Fidelity Final Evaluation] Top Candidate: `{winner_cand_name}`")
                    eval_dir = self.artifact_mgr.get_candidate_eval_dir(winner_cand_name)
                    try:
                        full_summary = evaluate_reconstruction(
                            dataset_input=self.dataset,
                            trajectory_input=traj_p,
                            mesh_input=mesh_p,
                            output_dir=eval_dir,
                            candidate_name=winner_cand_name,
                            split_json=str(self.split_file),
                            runtime_sec=winner_summary.get("runtime_sec") or winner_summary.get("performance", {}).get("runtime_sec", 0.0),
                            cheap=False,
                            render_samples=10
                        )
                        # Carry over metadata
                        if "voxel_size_m" in winner_summary:
                            full_summary["voxel_size_m"] = winner_summary["voxel_size_m"]
                        if "surface_method" in winner_summary:
                            full_summary["surface_method"] = winner_summary["surface_method"]
                        if "trajectory_metrics" in winner_summary:
                            full_summary["trajectory_metrics"] = winner_summary["trajectory_metrics"]

                        # Re-score with full fidelity summary
                        full_ranked = rank_candidate_summaries([full_summary])
                        if full_ranked:
                            winner = full_ranked[0]
                            # Update entry in overall_ranked
                            for idx, item in enumerate(overall_ranked):
                                if item.get("candidate_name") == winner_cand_name:
                                    overall_ranked[idx] = winner
                                    break
                    except Exception as e:
                        print(f"⚠️ Full fidelity evaluation warning: {e}")

            print("\n==========================================================")
            print(f" 🏆 [Overall Benchmark Winner] `{winner['candidate_name']}`")
            print(f"    Quality Score  : {winner.get('quality_score', 0):.2f} / 100")
            print(f"    Composite Score: {winner['composite_score']:.2f} / 100")
            print("==========================================================")
            self._log_decision("Final Ranking", winner["candidate_name"], "OVERALL_WINNER", f"Selected as overall winner (Composite Score: {winner['composite_score']:.1f})")
        else:
            winner = None
            print("\n==========================================================")
            print(" ❌ [Overall Benchmark Result] All candidates failed Hard Gate. No valid winner.")
            print("==========================================================")
            self._log_decision("Final Ranking", "NONE", "ALL_FAILED", "All evaluated candidates failed Hard Gate criteria")

        # Compute summary stats
        total_eval_or_cached = self.stats["evaluated_count"] + self.stats["cached_count"]
        self.stats["reuse_rate_pct"] = (self.stats["cached_count"] / max(total_eval_or_cached, 1)) * 100.0

        return overall_ranked, winner
