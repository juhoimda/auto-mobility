"""
orchestrator.py — Benchmark Execution Control, Dependency DAG, and Beam Search Lifecycle Management.
"""

import os
import sys
import time
import json
import random
import numpy as np
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from auto_mobility.config import (
    BAG_DIR, DB_DIR, MESH_DIR, TRAJECTORY_DIR, EVALUATION_DIR, FRAME_DIR, PROJECT_DIR
)
from auto_mobility.dataset.extract_frames import extract_dataset_from_bag
from auto_mobility.dataset.frame_dataset import FrameDataset
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.trajectory.export_trajectory import export_from_db
from auto_mobility.evaluation.split import create_holdout_split, save_split_json, load_split_json
from auto_mobility.diagnostics.pose_alignment import diagnose_pose_alignment
from auto_mobility.diagnostics.pipeline_diagnosis import diagnose_pipeline
from auto_mobility.diagnostics.frame_quality import analyze_frame_quality

from auto_mobility.benchmark.artifacts import ArtifactManager
from auto_mobility.benchmark.search import SearchEngine
from auto_mobility.benchmark.scoring import rank_candidate_summaries
from auto_mobility.benchmark.manifest import (
    BenchmarkManifestExporter,
    get_system_hardware_info,
    get_software_info,
    compute_dataset_fingerprint
)

SLAM_TRAJ_FILES = {
    "rtab_dense_rate0.5": lambda n: TRAJECTORY_DIR / f"rtab_dense_rate0.5_{n}_trajectory.txt",
    "rtab_dense_rate1.0": lambda n: TRAJECTORY_DIR / f"rtab_dense_rate1.0_{n}_trajectory.txt",
    "rtab_normal_rate0.5": lambda n: TRAJECTORY_DIR / f"rtab_rate0.5_{n}_trajectory.txt",
    "rtab_normal_rate1.0": lambda n: TRAJECTORY_DIR / f"rtab_rate1.0_{n}_trajectory.txt",
    "orb_rgbd": lambda n: TRAJECTORY_DIR / f"orb_rgbd_{n}_trajectory.txt",
    "orb_rgbdi": lambda n: TRAJECTORY_DIR / f"orb_rgbdi_{n}_trajectory.txt",
    "stella_rgbd": lambda n: TRAJECTORY_DIR / f"stella_{n}_trajectory.txt",
}


def _trajectory_candidates(key: str, bag_name: str) -> List[Path]:
    """Return accepted trajectory artifacts in preference order."""
    if key == "rtab_dense_rate0.5":
        return [
            TRAJECTORY_DIR / f"rtab_dense_rate0.5_{bag_name}_trajectory.txt",
            TRAJECTORY_DIR / f"rtab_dense_{bag_name}_trajectory.txt",
            TRAJECTORY_DIR / f"rtab_{bag_name}_trajectory.txt",
        ]
    if key == "rtab_dense_rate1.0":
        return [
            TRAJECTORY_DIR / f"rtab_dense_rate1.0_{bag_name}_trajectory.txt",
            TRAJECTORY_DIR / f"rtab_dense_{bag_name}_trajectory.txt",
        ]
    if key == "rtab_normal_rate1.0":
        return [
            TRAJECTORY_DIR / f"rtab_rate1.0_{bag_name}_trajectory.txt",
            TRAJECTORY_DIR / f"rtab_{bag_name}_trajectory.txt",
        ]
    if key == "rtab_normal_rate0.5":
        return [
            TRAJECTORY_DIR / f"rtab_rate0.5_{bag_name}_trajectory.txt",
        ]
    return [SLAM_TRAJ_FILES.get(key, lambda n: TRAJECTORY_DIR / f"{key}_{n}_trajectory.txt")(bag_name)]


SLAM_RUN_ARGS = {
    "rtab_dense_rate0.5": ("--slam=rtab", "--dense", "--rate=0.5"),
    "rtab_dense_rate1.0": ("--slam=rtab", "--dense", "--rate=1.0"),
    "rtab_normal_rate0.5": ("--slam=rtab", "--rate=0.5"),
    "rtab_normal_rate1.0": ("--slam=rtab", "--rate=1.0"),
    "orb_rgbd": ("--slam=orb_rgbd",),
    "orb_rgbdi": ("--slam=orb_rgbdi",),
    "stella_rgbd": ("--slam=stella_rgbd",),
}


class BenchmarkOrchestrator:
    """Orchestrates the modular SLAM & 3D Reconstruction benchmark workflow with Beam Search."""

    def __init__(
        self,
        bag_input: str,
        phase: str = "all",
        quick: bool = False,
        full: bool = False,
        mode: str = "standard",
        run_slam: bool = False,
        force: bool = False,
        resume: bool = True,
        top_k: int = 3,
        random_seed: int = 42,
        output_dir: Optional[Union[str, Path]] = None
    ):
        self.bag_path = self._resolve_bag_path(bag_input)
        self.bag_name = self.bag_path.name
        self.phase = phase.lower()
        self.run_slam = run_slam
        self.force = force
        self.resume = resume and not force
        self.top_k = top_k
        self.random_seed = random_seed

        # Mode normalization: quick / standard / full
        if quick or mode.lower() == "quick":
            self.mode = "quick"
        elif full or mode.lower() == "full":
            self.mode = "full"
        else:
            self.mode = "standard"

        self.quick = (self.mode == "quick")
        self.full = (self.mode == "full")

        # Set deterministic seeds for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.report_dir = Path(output_dir) if output_dir else (EVALUATION_DIR / self.bag_name)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_mgr = ArtifactManager(self.bag_name, self.report_dir)

    def _resolve_bag_path(self, bag_input: str) -> Path:
        p = Path(bag_input)
        if p.is_absolute() and p.exists():
            return p
        if (BAG_DIR / bag_input).exists():
            return BAG_DIR / bag_input
        if p.exists():
            return p.resolve()
        return BAG_DIR / bag_input

    def load_trajectories(self) -> Tuple[Dict[str, str], Dict[str, dict]]:
        """Finds or generates TUM trajectories for available SLAM backends."""
        trajectories: Dict[str, str] = {}
        traj_metrics: Dict[str, dict] = {}

        # Default keys to evaluate in standard mode
        active_keys = list(SLAM_TRAJ_FILES.keys()) if self.full else ["rtab_dense_rate0.5", "orb_rgbd", "stella_rgbd", "orb_rgbdi"]

        for key in active_keys:
            traj_candidates = _trajectory_candidates(key, self.bag_name)
            traj_file = next((p for p in traj_candidates if p.exists() and p.stat().st_size > 0), None)

            if traj_file and not self.force:
                print(f"📍 궤적 재사용: {key} → {traj_file.name}")
                trajectories[key] = str(traj_file)
                try:
                    traj_metrics[key] = Trajectory.from_tum_file(str(traj_file)).compute_metrics()
                    traj_metrics[key]["slam_backend"] = key
                except Exception:
                    traj_metrics[key] = {"slam_backend": key}
                continue

            # RTAB-Map DB fallback export
            if "rtab" in key:
                db = DB_DIR / f"{self.bag_name}.db"
                if db.exists():
                    target_traj = traj_candidates[0]
                    try:
                        print(f"⚙️ RTAB-Map DB 존재 → 궤적 추출: {target_traj.name}")
                        export_from_db(str(db), str(target_traj), opt=0)
                        trajectories[key] = str(target_traj)
                        traj_metrics[key] = Trajectory.from_tum_file(str(target_traj)).compute_metrics()
                        traj_metrics[key]["slam_backend"] = key
                        continue
                    except Exception as e:
                        print(f"⚠️ RTAB-Map 궤적 추출 실패: {e}")

            if self.run_slam and key in SLAM_RUN_ARGS:
                print(f"⚙️ SLAM 실행 (--run-slam): {key} → run_slam.sh {' '.join(SLAM_RUN_ARGS[key])}")
                script = PROJECT_DIR / "scripts" / "pipeline" / "run_slam.sh"
                subprocess.run(["bash", str(script), self.bag_name, *SLAM_RUN_ARGS[key]], check=False)
                generated = next((p for p in traj_candidates if p.exists() and p.stat().st_size > 0), None)
                if generated:
                    trajectories[key] = str(generated)
                    traj_metrics[key] = Trajectory.from_tum_file(str(generated)).compute_metrics()
                    traj_metrics[key]["slam_backend"] = key
                else:
                    print(f"⚠️ SLAM {key} 실행 실패 → 후보 제외")
            else:
                pass

        # Fallback if no specific keys matched: check if legacy rtab trajectory exists
        if not trajectories:
            legacy_rtab = TRAJECTORY_DIR / f"rtab_{self.bag_name}_trajectory.txt"
            if legacy_rtab.exists() and legacy_rtab.stat().st_size > 0:
                print(f"📍 Legacy RTAB 궤적 발견: {legacy_rtab.name}")
                trajectories["rtab_dense_rate0.5"] = str(legacy_rtab)
                traj_metrics["rtab_dense_rate0.5"] = Trajectory.from_tum_file(str(legacy_rtab)).compute_metrics()

        return trajectories, traj_metrics

    def run(self) -> dict:
        """Executes the full multi-axis benchmark pipeline with Beam Search and Full Rebuild."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n==========================================================")
        print(" 🏁 Autonomous Mobility Multi-Axis 3D SLAM & Reconstruction Benchmark")
        print(f" 📦 Dataset: {self.bag_name}")
        print(f" ⚙️ Mode:    {self.mode.upper()} (Quick={self.quick}, Full={self.full})")
        print(f" 📂 Output:  {self.report_dir}")
        print("==========================================================")

        # 0. Frame Extraction & Dataset Preparation
        frame_out_dir = FRAME_DIR / self.bag_name
        if not (frame_out_dir / "frames.csv").exists():
            print(f"⚙️ RGB-D 프레임 추출 중: {self.bag_name}")
            extract_dataset_from_bag(str(self.bag_path), str(frame_out_dir))

        dataset = FrameDataset(frame_out_dir)
        print(f"📊 Dataset Loaded: {len(dataset)} valid RGB-D frames")
        frame_quality = analyze_frame_quality(dataset, sample_count=100)
        print(
            f"🧪 Canonical frame quality: {frame_quality.get('overall_status')} "
            f"invalid_depth={frame_quality.get('mean_invalid_depth_ratio')}"
        )

        # Create or load holdout split
        split_file = self.report_dir / "holdout_split.json"
        if not split_file.exists() or self.force:
            split_data = create_holdout_split(total_frames=len(dataset), policy="every_nth", nth=5, seed=self.random_seed)
            save_split_json(split_data, str(split_file))
        else:
            split_data = load_split_json(str(split_file))

        # Dataset Fingerprint
        dataset_fingerprint = compute_dataset_fingerprint(frame_out_dir)

        # 1. Initialize Search Engine
        search_engine = SearchEngine(
            bag_name=self.bag_name,
            dataset=dataset,
            split_file=split_file,
            artifact_mgr=self.artifact_mgr,
            quick=self.quick,
            full=self.full,
            mode=self.mode,
            force=self.force
        )

        # 2. Trajectories
        trajectories, traj_metrics = self.load_trajectories()

        # Diagnose frame↔trajectory connectivity
        pose_diagnostics: Dict[str, dict] = {}
        frame_timestamps = dataset.get_timestamps(use_rgb=True)
        for slam_name, traj_path in trajectories.items():
            try:
                traj_obj = Trajectory.from_tum_file(traj_path)
                pose_diagnostics[slam_name] = diagnose_pose_alignment(
                    frame_timestamps,
                    traj_obj,
                    trajectory_path=traj_path,
                    max_pose_gap_ms=50.0,
                )
            except Exception as exc:
                pose_diagnostics[slam_name] = {
                    "trajectory_path": str(traj_path),
                    "status": "FAIL",
                    "cause": "TRAJECTORY_PARSE",
                    "error": str(exc),
                    "warnings": ["Trajectory could not be parsed"],
                }

        sensor_diagnostics = {}
        prev_manifest = {}
        if self.resume and (self.report_dir / "experiment_manifest.json").exists():
            try:
                with open(self.report_dir / "experiment_manifest.json", "r", encoding="utf-8") as f:
                    prev_manifest = json.load(f)
            except Exception:
                prev_manifest = {}

        slam_eval_results: List[dict] = prev_manifest.get("phase_a_slam_results", [])
        fusion_eval_results: List[dict] = prev_manifest.get("phase_b_tsdf_results", [])
        surface_eval_results: List[dict] = prev_manifest.get("phase_c_surface_results", [])
        final_rebuilt_rankings: List[dict] = []
        overall_winner: Optional[dict] = None

        # ── PHASE A (SLAM Screening, Beam Width = Top 2~3) ──
        if self.phase in ("all", "a", "slam"):
            slam_eval_results, top_slams = search_engine.run_phase_a(
                trajectories, traj_metrics, pose_diagnostics=pose_diagnostics
            )
        else:
            if slam_eval_results:
                ranked_a = rank_candidate_summaries(slam_eval_results)
                valid_a = [r for r in ranked_a if r.get("hard_gate_pass", False)]
                top_slams = []
                for item in (valid_a if valid_a else ranked_a)[:search_engine.beam_width_slam]:
                    cand_n = item["candidate_name"]
                    slam_name = cand_n.replace("_tsdf10mm", "").replace("_voxel10mm", "")
                    t_file = trajectories.get(slam_name, "")
                    if t_file:
                        top_slams.append((slam_name, t_file))
                if not top_slams and trajectories:
                    top_slams = [(k, v) for k, v in list(trajectories.items())[:search_engine.beam_width_slam]]
            else:
                top_slams = [(k, v) for k, v in list(trajectories.items())[:search_engine.beam_width_slam]]

        # ── PHASE B (Fusion Screening on surviving SLAMs) ──
        pose_gate_failed = (not trajectories) or (
            bool(pose_diagnostics)
            and not any(d.get("status") in ("PASS", "WARN") for d in pose_diagnostics.values())
        )
        if pose_gate_failed and self.phase in ("all", "b", "tsdf", "fusion"):
            fusion_eval_results = [{
                "candidate_name": f"{top_slams[0][0] if top_slams else 'none'}_fusion",
                "status": "BLOCKED",
                "overall_status": "BLOCKED",
                "blocked_by": "POSE_ALIGNMENT",
                "error": "All available trajectories failed pose alignment gate",
                "geometry": {},
                "mesh": {},
            }]
            top_fusion_pipelines = []
            print("⛔ Phase B BLOCKED: all trajectories failed pose alignment gate")
        elif self.phase in ("all", "b", "tsdf", "fusion"):
            fusion_eval_results, top_fusion_pipelines = search_engine.run_phase_b(
                top_slams=top_slams,
                phase_a_results=slam_eval_results
            )
        else:
            if fusion_eval_results:
                ranked_b = rank_candidate_summaries(fusion_eval_results)
                valid_b = [r for r in ranked_b if r.get("hard_gate_pass", False)]
                top_fusion_pipelines = [item["summary_data"] for item in (valid_b if valid_b else ranked_b)[:search_engine.beam_width_fusion]]
            else:
                top_fusion_pipelines = [slam_eval_results[0]] if slam_eval_results else []

        # ── PHASE C (Surface Screening on surviving Fusion Pipelines) ──
        if pose_gate_failed and self.phase in ("all", "c", "surface", "mesh"):
            surface_eval_results = [{
                "candidate_name": f"{top_slams[0][0] if top_slams else 'none'}_surface",
                "status": "BLOCKED",
                "overall_status": "BLOCKED",
                "blocked_by": "POSE_ALIGNMENT",
                "error": "Surface evaluation blocked because pose alignment failed",
                "geometry": {},
                "mesh": {},
            }]
            finalists = []
            print("⛔ Phase C BLOCKED: all trajectories failed pose alignment gate")
        elif self.phase in ("all", "c", "surface", "mesh"):
            surface_eval_results, finalists = search_engine.run_phase_c(
                top_fusion_pipelines=top_fusion_pipelines,
                trajectories=trajectories
            )
        else:
            finalists = [top_fusion_pipelines[0]] if top_fusion_pipelines else []

        # ── PHASE D (FULL REBUILD & Full-Fidelity Evaluation on Top Finalists) ──
        if finalists and self.phase in ("all", "d", "rebuild", "final"):
            final_rebuilt_rankings, overall_winner = search_engine.run_full_rebuild(
                finalists=finalists,
                trajectories=trajectories
            )
            overall_rankings = final_rebuilt_rankings
        else:
            all_summaries = slam_eval_results + fusion_eval_results + surface_eval_results
            overall_rankings = rank_candidate_summaries(all_summaries)
            valid_ranked = [r for r in overall_rankings if r.get("hard_gate_pass", False)]
            overall_winner = valid_ranked[0] if valid_ranked else (overall_rankings[0] if overall_rankings else None)

        sensor_evidence = dict(sensor_diagnostics)
        if frame_quality.get("overall_status") != "PASS" or not sensor_evidence:
            sensor_evidence = frame_quality
        else:
            sensor_evidence["frame_quality"] = frame_quality

        pipeline_diagnosis = diagnose_pipeline(
            sensor_input=sensor_evidence,
            pose_alignment=pose_diagnostics,
            phase_a=slam_eval_results,
            phase_b=fusion_eval_results,
            phase_c=surface_eval_results,
        )

        manifest = {
            "benchmark_id": f"bench_{self.bag_name}",
            "bag_name": self.bag_name,
            "mode": self.mode,
            "random_seed": self.random_seed,
            "dataset_fingerprint": dataset_fingerprint,
            "evaluated_at": timestamp,
            "output_dir": str(self.report_dir),
            "artifacts": {
                "split": str(split_file),
                "trajectories": trajectories,
            },
            "pose_alignment_diagnostics": pose_diagnostics,
            "sensor_diagnostics": sensor_diagnostics,
            "frame_quality": frame_quality,
            "pipeline_diagnosis": pipeline_diagnosis,
            "hardware": get_system_hardware_info(),
            "software": get_software_info(),
            "summary_stats": search_engine.stats,
            "decision_trace": search_engine.decision_trace,
            "phase_a_slam_results": slam_eval_results,
            "phase_b_tsdf_results": fusion_eval_results,
            "phase_c_surface_results": surface_eval_results,
            "phase_d_rebuild_results": [r.get("summary_data", {}) for r in final_rebuilt_rankings],
            "winner": overall_winner.get("candidate_name") if overall_winner else None
        }

        BenchmarkManifestExporter.export_final_artifacts(
            report_dir=self.report_dir,
            manifest_data=manifest,
            overall_rankings=overall_rankings,
            winner_candidate=overall_winner,
            top_k=self.top_k
        )

        print("\n==========================================================")
        print(" 🎉 Multi-Axis Benchmark Complete!")
        print(f" 📄 Manifest JSON   : {self.report_dir / 'experiment_manifest.json'}")
        print(f" 📑 Report MD       : {self.report_dir / 'benchmark_report.md'}")
        print(f" 🥇 Final Best OBJ  : {self.report_dir / 'final' / 'best.obj'}")
        print(f" ⚙️ Best Config JSON : {self.report_dir / 'final' / 'best_config.json'}")
        print(f" 🔍 Review Meshes   : {self.report_dir / 'review'}")
        print("==========================================================")
        return manifest


def run_benchmark(
    bag_input: str,
    phase: str = "all",
    quick: bool = False,
    full: bool = False,
    mode: str = "standard",
    run_slam: bool = False,
    force: bool = False,
    resume: bool = True,
    top_k: int = 3,
    random_seed: int = 42,
    output_dir: Optional[Union[str, Path]] = None
) -> dict:
    orchestrator = BenchmarkOrchestrator(
        bag_input=bag_input,
        phase=phase,
        quick=quick,
        full=full,
        mode=mode,
        run_slam=run_slam,
        force=force,
        resume=resume,
        top_k=top_k,
        random_seed=random_seed,
        output_dir=output_dir
    )
    return orchestrator.run()
