"""
orchestrator.py — Benchmark Execution Control, Dependency DAG, and Beam Search Lifecycle Management.
"""

from __future__ import annotations

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
from auto_mobility.diagnostics.trajectory_health import check_trajectory_health

from auto_mobility.benchmark.candidate import (
    CandidateSpec,
    SlamProfileSpec,
    STANDARD_SLAM_PROFILES,
    get_slam_profile_spec,
    get_trajectory_filename,
    get_rtab_db_filename
)
from auto_mobility.benchmark.artifacts import (
    ArtifactManager,
    compute_file_sha256,
    save_trajectory_metadata,
    verify_trajectory_provenance
)
from auto_mobility.benchmark.search import SearchEngine
from auto_mobility.benchmark.scoring import rank_candidate_summaries
from auto_mobility.benchmark.manifest import (
    BenchmarkManifestExporter,
    get_system_hardware_info,
    get_software_info,
    compute_dataset_fingerprint
)

SLAM_TRAJ_FILES = {
    key: (lambda n, k=key: TRAJECTORY_DIR / get_trajectory_filename(n, k))
    for key in STANDARD_SLAM_PROFILES.keys()
}

SLAM_RUN_ARGS = {
    "rtab_dense_rate0.5": ("--slam=rtab", "--dense", "--rate=0.5"),
    "rtab_dense_rate1.0": ("--slam=rtab", "--dense", "--rate=1.0"),
    "rtab_normal_rate0.5": ("--slam=rtab", "--rate=0.5"),
    "rtab_normal_rate1.0": ("--slam=rtab", "--rate=1.0"),
    "orb_rgbd_rate0.5": ("--slam=orb_rgbd", "--rate=0.5"),
    "orb_rgbd_rate1.0": ("--slam=orb_rgbd", "--rate=1.0"),
    "orb_rgbdi_rate0.5": ("--slam=orb_rgbdi", "--rate=0.5"),
    "orb_rgbdi_rate1.0": ("--slam=orb_rgbdi", "--rate=1.0"),
    "stella_rgbd_rate0.5": ("--slam=stella_rgbd", "--rate=0.5"),
    "stella_rgbd_rate1.0": ("--slam=stella_rgbd", "--rate=1.0"),
}


class BenchmarkOrchestrator:
    """Orchestrates the modular SLAM & 3D Reconstruction benchmark workflow."""

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

        # Mode normalization
        if quick or mode.lower() == "quick":
            self.mode = "quick"
        elif full or mode.lower() == "full":
            self.mode = "full"
        else:
            self.mode = "standard"

        self.quick = (self.mode == "quick")
        self.full = (self.mode == "full")

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

    def load_trajectories(self, dataset_fingerprint: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, dict]]:
        """Finds or generates TUM trajectories for available SLAM backends with provenance verification."""
        trajectories: Dict[str, str] = {}
        traj_metrics: Dict[str, dict] = {}
        expected_fp = dataset_fingerprint or getattr(self, "dataset_fingerprint", None)

        active_keys = list(STANDARD_SLAM_PROFILES.keys()) if (self.mode != "quick") else [
            "rtab_normal_rate1.0", "orb_rgbd_rate1.0", "orb_rgbdi_rate1.0", "stella_rgbd_rate1.0"
        ]

        for key in active_keys:
            spec = get_slam_profile_spec(key)
            traj_fn = SLAM_TRAJ_FILES.get(key, lambda n: TRAJECTORY_DIR / get_trajectory_filename(n, key))
            traj_file = traj_fn(self.bag_name)

            if traj_file.exists() and traj_file.stat().st_size > 0 and not self.force:
                is_valid, status, meta = verify_trajectory_provenance(traj_file, spec, expected_bag_fingerprint=expected_fp, strict=False)
                if is_valid:
                    print(f"📍 궤적 발견 & 검증 통과: {key} → {traj_file.name}")
                    trajectories[key] = str(traj_file)
                    try:
                        traj_metrics[key] = Trajectory.from_tum_file(str(traj_file)).compute_metrics()
                        traj_metrics[key]["slam_backend"] = spec.backend
                        traj_metrics[key]["slam_profile"] = spec.profile
                        traj_metrics[key]["replay_rate"] = spec.replay_rate
                    except Exception:
                        traj_metrics[key] = {"slam_backend": spec.backend}
                    continue
                else:
                    print(f"⚠️ 궤적 검증 경고 ({status}): {traj_file.name}")

            # Check legacy naming conventions if primary file is missing
            legacy_candidates = [
                TRAJECTORY_DIR / f"{spec.backend}_{self.bag_name}_trajectory.txt",
                TRAJECTORY_DIR / f"{key}_{self.bag_name}_trajectory.txt"
            ]
            found_legacy = next((p for p in legacy_candidates if p.exists() and p.stat().st_size > 0), None)
            if found_legacy and not self.force:
                print(f"📍 궤적 발견 (Legacy path): {key} → {found_legacy.name}")
                trajectories[key] = str(found_legacy)
                try:
                    traj_metrics[key] = Trajectory.from_tum_file(str(found_legacy)).compute_metrics()
                    traj_metrics[key]["slam_backend"] = spec.backend
                    traj_metrics[key]["slam_profile"] = spec.profile
                    traj_metrics[key]["replay_rate"] = spec.replay_rate
                except Exception:
                    traj_metrics[key] = {"slam_backend": spec.backend}
                continue

            # RTAB-Map DB export if db exists
            if "rtab" in key:
                db_name = get_rtab_db_filename(self.bag_name, spec.profile, spec.replay_rate)
                db = DB_DIR / db_name
                if not db.exists():
                    db_legacy_name = f"{self.bag_name}_dense.db" if spec.profile == "dense" else f"{self.bag_name}.db"
                    if (DB_DIR / db_legacy_name).exists():
                        db = DB_DIR / db_legacy_name
                if db.exists():
                    try:
                        print(f"⚙️ RTAB-Map DB 존재 → 궤적 추출: {traj_file.name}")
                        export_from_db(str(db), str(traj_file), opt=0)
                        save_trajectory_metadata(traj_file, spec, bag_fingerprint=expected_fp)
                        trajectories[key] = str(traj_file)
                        traj_metrics[key] = Trajectory.from_tum_file(str(traj_file)).compute_metrics()
                        traj_metrics[key]["slam_backend"] = spec.backend
                        traj_metrics[key]["slam_profile"] = spec.profile
                        traj_metrics[key]["replay_rate"] = spec.replay_rate
                        continue
                    except Exception as e:
                        print(f"⚠️ RTAB-Map 궤적 추출 실패: {e}")

        # Collect missing profiles to generate
        missing_keys = [k for k in active_keys if k not in trajectories and k in SLAM_RUN_ARGS]
        if self.run_slam and missing_keys:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(3, len(missing_keys))
            print(f"\n🚀 병렬 SLAM 생성 시작: {len(missing_keys)}개 후보군 (동시 실행 워커 수: {max_workers})")

            def _run_single_slam(worker_idx: int, k: str):
                s = get_slam_profile_spec(k)
                t_fn = SLAM_TRAJ_FILES.get(k, lambda n: TRAJECTORY_DIR / get_trajectory_filename(n, k))
                t_file = t_fn(self.bag_name)
                env = os.environ.copy()
                env["ROS_DOMAIN_ID"] = str(10 + (worker_idx % 80))

                print(f"⚙️ [워커 #{worker_idx+1} | DOMAIN={env['ROS_DOMAIN_ID']}] SLAM 실행: {k} → run_slam.sh {' '.join(SLAM_RUN_ARGS[k])}")
                script = PROJECT_DIR / "scripts" / "pipeline" / "run_slam.sh"
                subprocess.run(["bash", str(script), self.bag_name, *SLAM_RUN_ARGS[k]], env=env, check=False)

                if t_file.exists() and t_file.stat().st_size > 0:
                    save_trajectory_metadata(t_file, s, bag_fingerprint=expected_fp)
                    try:
                        m = Trajectory.from_tum_file(str(t_file)).compute_metrics()
                        m["slam_backend"] = s.backend
                        m["slam_profile"] = s.profile
                        m["replay_rate"] = s.replay_rate
                        return k, str(t_file), m
                    except Exception:
                        return k, str(t_file), {"slam_backend": s.backend}
                return k, None, None

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_single_slam, idx, k): k
                    for idx, k in enumerate(missing_keys)
                }
                for f in as_completed(futures):
                    k_res, t_res, m_res = f.result()
                    if t_res:
                        trajectories[k_res] = t_res
                        if m_res:
                            traj_metrics[k_res] = m_res
                        print(f"✅ SLAM 완료: {k_res} → {Path(t_res).name}")
                    else:
                        print(f"⚠️ SLAM 생성 실패: {k_res}")

        return trajectories, traj_metrics

    def run(self) -> dict:
        """Executes the complete benchmark DAG."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n==========================================================")
        print(" 🏁 Autonomous Mobility Multi-Axis 3D SLAM & Reconstruction Benchmark")
        print(f" 📦 Dataset: {self.bag_name}")
        print(f" ⚙️ Mode:    {self.mode.upper()} (Quick={self.quick}, Full={self.full})")
        print(f" 📂 Output:  {self.report_dir}")
        print("==========================================================")

        # 0. Frame Extraction & Canonical Dataset Preparation
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

        # Create or load single holdout split
        split_file = self.report_dir / "holdout_split.json"
        if not split_file.exists() or self.force:
            split_data = create_holdout_split(total_frames=len(dataset), policy="every_nth", nth=5, seed=self.random_seed)
            save_split_json(split_data, str(split_file))
        else:
            split_data = load_split_json(str(split_file))

        dataset_fingerprint = compute_dataset_fingerprint(frame_out_dir)
        self.dataset_fingerprint = dataset_fingerprint

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

        # 2. Load Trajectories
        raw_trajectories, traj_metrics = self.load_trajectories(dataset_fingerprint=dataset_fingerprint)

        # 3. PHASE A0: Trajectory Health Gate
        healthy_trajectories, health_diagnostics = search_engine.run_phase_a0(raw_trajectories)

        # Diagnose frame↔trajectory pose connectivity
        pose_diagnostics: Dict[str, dict] = {}
        frame_timestamps = dataset.get_timestamps(use_rgb=True)
        for cand_k, traj_path in healthy_trajectories.items():
            try:
                traj_obj = Trajectory.from_tum_file(traj_path)
                pose_diagnostics[cand_k] = diagnose_pose_alignment(
                    frame_timestamps,
                    traj_obj,
                    trajectory_path=traj_path,
                    max_pose_gap_ms=50.0,
                )
            except Exception as exc:
                pose_diagnostics[cand_k] = {
                    "trajectory_path": str(traj_path),
                    "status": "FAIL",
                    "cause": "TRAJECTORY_PARSE",
                    "error": str(exc),
                    "warnings": ["Trajectory could not be parsed"],
                }

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

        # ── PHASE A (SLAM Profile Calibration & Backend Screening) ──
        if not healthy_trajectories:
            print("⛔ Phase A BLOCKED: No healthy trajectories passed Trajectory Health Gate.")
            slam_eval_results = []
            top_slams = []
        elif self.phase in ("all", "a", "slam"):
            slam_eval_results, top_slams = search_engine.run_phase_a(
                healthy_trajectories, traj_metrics, pose_diagnostics=pose_diagnostics, health_diagnostics=health_diagnostics
            )
        else:
            if slam_eval_results:
                ranked_a = rank_candidate_summaries(slam_eval_results)
                valid_a = [r for r in ranked_a if r.get("hard_gate_pass", False)]
                top_slams = []
                for item in valid_a[:search_engine.beam_width_slam]:
                    cand_n = item["candidate_name"]
                    slam_name = cand_n.replace("_tsdf10mm", "").replace("_voxel10mm", "")
                    t_file = healthy_trajectories.get(slam_name) or next((v for k, v in healthy_trajectories.items() if k.startswith(slam_name) or slam_name.startswith(k.split("_rate")[0])), "")
                    if t_file:
                        top_slams.append((slam_name, t_file))
            else:
                top_slams = []

        # ── PHASE B (Fusion Screening on surviving SLAM champions) ──
        if not top_slams and self.phase in ("all", "b", "tsdf", "fusion"):
            print("⛔ Phase B BLOCKED: No SLAM candidates passed Phase A screening.")
            fusion_eval_results = []
            top_fusion_pipelines = []
        elif self.phase in ("all", "b", "tsdf", "fusion"):
            fusion_eval_results, top_fusion_pipelines = search_engine.run_phase_b(
                top_slams=top_slams,
                phase_a_results=slam_eval_results
            )
        else:
            if fusion_eval_results:
                ranked_b = rank_candidate_summaries(fusion_eval_results)
                valid_b = [r for r in ranked_b if r.get("hard_gate_pass", False)]
                top_fusion_pipelines = [item["summary_data"] for item in valid_b[:search_engine.beam_width_fusion]]
            else:
                top_fusion_pipelines = []

        # ── PHASE C (Surface Screening on surviving Fusion Pipelines) ──
        if not top_fusion_pipelines and self.phase in ("all", "c", "surface", "mesh"):
            print("⛔ Phase C BLOCKED: No fusion pipelines passed Phase B screening.")
            surface_eval_results = []
            finalists = []
        elif self.phase in ("all", "c", "surface", "mesh"):
            surface_eval_results, finalists = search_engine.run_phase_c(
                top_fusion_pipelines=top_fusion_pipelines,
                trajectories=healthy_trajectories
            )
        else:
            finalists = []

        # ── PHASE D (FULL REBUILD on Top-3 Finalists) ──
        if finalists and self.phase in ("all", "d", "rebuild", "final"):
            final_rebuilt_rankings, overall_winner = search_engine.run_full_rebuild(
                finalists=finalists,
                trajectories=healthy_trajectories
            )
            overall_rankings = final_rebuilt_rankings
        else:
            all_summaries = slam_eval_results + fusion_eval_results + surface_eval_results
            overall_rankings = rank_candidate_summaries(all_summaries)
            valid_ranked = [r for r in overall_rankings if r.get("hard_gate_pass", False)]
            overall_winner = valid_ranked[0] if valid_ranked else None

        pipeline_diagnosis = diagnose_pipeline(
            sensor_input=frame_quality,
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
                "trajectories": healthy_trajectories,
            },
            "trajectory_health_diagnostics": health_diagnostics,
            "pose_alignment_diagnostics": pose_diagnostics,
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
