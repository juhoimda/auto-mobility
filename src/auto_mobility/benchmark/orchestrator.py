"""
orchestrator.py — Benchmark Execution Control, Dependency DAG, and Lifecycle Management.
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
from auto_mobility.benchmark.manifest import (
    BenchmarkManifestExporter,
    get_system_hardware_info,
    get_software_info,
    compute_dataset_fingerprint
)

SLAM_TRAJ_FILES = {
    "rtab_rgbd": lambda n: TRAJECTORY_DIR / f"rtab_{n}_trajectory.txt",
    "orb_rgbd": lambda n: TRAJECTORY_DIR / f"orb_rgbd_{n}_trajectory.txt",
    "orb_rgbdi": lambda n: TRAJECTORY_DIR / f"orb_rgbdi_{n}_trajectory.txt",
    "stella_rgbd": lambda n: TRAJECTORY_DIR / f"stella_{n}_trajectory.txt",
}

SLAM_RUN_ARGS = {
    "rtab_rgbd": "rtab",
    "orb_rgbd": "orb_rgbd",
    "orb_rgbdi": "orb_rgbdi",
    "stella_rgbd": "stella_rgbd",
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
        # Fallback to BAG_DIR/bag_input even if not yet created (e.g. synthetic test)
        return BAG_DIR / bag_input

    def load_trajectories(self) -> Tuple[Dict[str, str], Dict[str, dict]]:
        """Finds or generates TUM trajectories for available SLAM backends."""
        trajectories: Dict[str, str] = {}
        traj_metrics: Dict[str, dict] = {}

        for key, path_fn in SLAM_TRAJ_FILES.items():
            traj_file = path_fn(self.bag_name)

            if traj_file.exists() and traj_file.stat().st_size > 0:
                print(f"📍 궤적 재사용: {traj_file.name}")
                trajectories[key] = str(traj_file)
                try:
                    traj_metrics[key] = Trajectory.from_tum_file(str(traj_file)).compute_metrics()
                    traj_metrics[key]["slam_backend"] = key
                except Exception:
                    traj_metrics[key] = {"slam_backend": key}
                continue

            # RTAB-Map: DB exist -> export trajectory
            if key == "rtab_rgbd":
                db = DB_DIR / f"{self.bag_name}.db"
                if db.exists():
                    try:
                        print(f"⚙️ RTAB-Map DB 존재 → 궤적 추출: {traj_file.name}")
                        export_from_db(str(db), str(traj_file), opt=0)
                        trajectories[key] = str(traj_file)
                        traj_metrics[key] = Trajectory.from_tum_file(str(traj_file)).compute_metrics()
                        traj_metrics[key]["slam_backend"] = key
                        continue
                    except Exception as e:
                        print(f"⚠️ RTAB-Map 궤적 추출 실패: {e}")

            if self.run_slam:
                print(f"⚙️ SLAM 실행 (--run-slam): {key} → run_slam.sh --slam={SLAM_RUN_ARGS[key]}")
                script = PROJECT_DIR / "scripts" / "pipeline" / "run_slam.sh"
                subprocess.run(["bash", str(script), self.bag_name, f"--slam={SLAM_RUN_ARGS[key]}"], check=False)
                if traj_file.exists() and traj_file.stat().st_size > 0:
                    trajectories[key] = str(traj_file)
                    traj_metrics[key] = Trajectory.from_tum_file(str(traj_file)).compute_metrics()
                    traj_metrics[key]["slam_backend"] = key
                else:
                    print(f"⚠️ SLAM {key} 실행 실패 → 후보 제외")
            else:
                print(f"ℹ️ {key} 궤적 없음 (스킵). 생성 명령: ./scripts/pipeline/run_slam.sh {self.bag_name} --slam={SLAM_RUN_ARGS[key]}")

        return trajectories, traj_metrics

    def run(self) -> dict:
        """Executes the full multi-axis benchmark pipeline with mode and search trace."""
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

        # Diagnose frame↔trajectory connectivity before spending time on TSDF
        # and surface candidates.  A downstream mesh cannot be trusted when
        # the trajectory covers only a small fraction of the RGB-D sequence.
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
        if pose_diagnostics:
            for name, diag in pose_diagnostics.items():
                print(
                    f"📐 Pose alignment [{name}]: {diag.get('status')} "
                    f"coverage={diag.get('pose_coverage_ratio', 0.0) * 100:.1f}% "
                    f"cause={diag.get('cause', 'UNKNOWN')}"
                )

        sensor_diagnostics: dict = {}
        sensor_manifest = self.bag_path / "dataset_manifest.json"
        if sensor_manifest.exists():
            try:
                with open(sensor_manifest, "r", encoding="utf-8") as f:
                    sensor_diagnostics = json.load(f)
                # validate_bag manifests do not have an overall_status field;
                # derive it from required checks and retain all raw evidence.
                if "overall_status" not in sensor_diagnostics:
                    checks = sensor_diagnostics.get("checks", {})
                    sensor_diagnostics["overall_status"] = (
                        "FAIL" if any(not c.get("pass", False) for c in checks.values())
                        else ("WARN" if sensor_diagnostics.get("warnings") else "PASS")
                    )
            except Exception as exc:
                sensor_diagnostics = {
                    "overall_status": "WARN",
                    "warnings": [f"Sensor manifest could not be read: {exc}"],
                }

        # Previous state restoration for Resume
        prev_manifest = {}
        if self.resume:
            manifest_p = self.report_dir / "experiment_manifest.json"
            if manifest_p.exists():
                try:
                    with open(manifest_p, "r", encoding="utf-8") as f:
                        prev_manifest = json.load(f)
                    print("📄 [Resume] 기존 manifest 로드 완료")
                except Exception as e:
                    print(f"⚠️ [Resume] manifest 로드 실패: {e}")

        slam_eval_results = prev_manifest.get("phase_a_slam_results", [])
        tsdf_eval_results = prev_manifest.get("phase_b_tsdf_results", [])
        surface_eval_results = prev_manifest.get("phase_c_surface_results", [])

        best_slam = list(trajectories.keys())[0] if trajectories else "rtab_rgbd"
        best_traj = trajectories.get(best_slam, "")
        best_voxel_m = 0.010
        best_pcd = self.artifact_mgr.get_pcd_path(best_slam, 10)
        best_tsdf_mesh = self.artifact_mgr.get_mesh_path(best_slam, 10)
        best_tsdf_summary = {}

        # ── PHASE A ──
        if self.phase in ("all", "a", "slam"):
            slam_eval_results, best_slam, best_traj = search_engine.run_phase_a(
                trajectories, traj_metrics, pose_diagnostics=pose_diagnostics
            )
        else:
            if slam_eval_results:
                from auto_mobility.benchmark.scoring import rank_candidate_summaries
                ranked_a = rank_candidate_summaries(slam_eval_results)
                valid_a = [r for r in ranked_a if r.get("hard_gate_pass", False)]
                if valid_a:
                    best_slam = valid_a[0]["candidate_name"].replace("_voxel10mm", "")
                    print(f"📄 [Resume] Phase A winner 복원: `{best_slam}` (Score: {valid_a[0]['composite_score']:.1f})")
            best_traj = trajectories.get(best_slam, list(trajectories.values())[0] if trajectories else "")

        # ── PHASE B ──
        # Do not attribute a bad pose stream to TSDF or surface code.  The
        # phase remains represented in the manifest as BLOCKED for auditability.
        pose_gate_failed = (not trajectories) or (
            bool(pose_diagnostics)
            and not any(d.get("status") in ("PASS", "WARN") for d in pose_diagnostics.values())
        )
        if pose_gate_failed and self.phase in ("all", "b", "tsdf", "fusion"):
            tsdf_eval_results = [{
                "candidate_name": f"{best_slam}_tsdf",
                "status": "BLOCKED",
                "overall_status": "BLOCKED",
                "blocked_by": "POSE_ALIGNMENT",
                "error": "All available trajectories failed pose alignment gate",
                "geometry": {},
                "mesh": {},
            }]
            best_tsdf_summary = tsdf_eval_results[0]
            best_pcd = self.artifact_mgr.get_pcd_path(best_slam, 10)
            best_tsdf_mesh = self.artifact_mgr.get_mesh_path(best_slam, 10)
            print("⛔ Phase B BLOCKED: all trajectories failed pose alignment gate")
        elif self.phase in ("all", "b", "tsdf", "fusion"):
            tsdf_eval_results, best_voxel_m, best_pcd, best_tsdf_mesh, best_tsdf_summary = search_engine.run_phase_b(
                best_slam=best_slam,
                best_traj=best_traj,
                phase_a_results=slam_eval_results
            )
        else:
            if tsdf_eval_results:
                from auto_mobility.benchmark.scoring import rank_candidate_summaries
                ranked_b = rank_candidate_summaries(tsdf_eval_results)
                valid_b = [r for r in ranked_b if r.get("hard_gate_pass", False)]
                if valid_b:
                    best_voxel_m = valid_b[0]["summary_data"].get("voxel_size_m", 0.010)
                    best_tsdf_summary = valid_b[0]["summary_data"]
                    print(f"📄 [Resume] Phase B winner 복원: voxel={best_voxel_m*1000:.1f}mm (Score: {valid_b[0]['composite_score']:.1f})")
            best_v_mm = int(round(best_voxel_m * 1000))
            best_pcd = self.artifact_mgr.get_pcd_path(best_slam, best_v_mm)
            best_tsdf_mesh = self.artifact_mgr.get_mesh_path(best_slam, best_v_mm)

        # ── PHASE C ──
        if pose_gate_failed and self.phase in ("all", "c", "surface", "mesh"):
            surface_eval_results = [{
                "candidate_name": f"{best_slam}_surface",
                "status": "BLOCKED",
                "overall_status": "BLOCKED",
                "blocked_by": "POSE_ALIGNMENT",
                "error": "Surface evaluation blocked because pose alignment failed",
                "geometry": {},
                "mesh": {},
            }]
            print("⛔ Phase C BLOCKED: all trajectories failed pose alignment gate")
        elif self.phase in ("all", "c", "surface", "mesh"):
            surface_eval_results, winner_c = search_engine.run_phase_c(
                best_slam=best_slam,
                best_traj=best_traj,
                best_voxel_m=best_voxel_m,
                best_pcd=best_pcd,
                best_tsdf_mesh=best_tsdf_mesh,
                best_tsdf_summary=best_tsdf_summary
            )

        # ── Overall Ranking & Deliverables ──
        overall_rankings, overall_winner = search_engine.compute_overall_rankings(
            slam_eval_results,
            tsdf_eval_results,
            surface_eval_results
        )
        sensor_evidence = dict(sensor_diagnostics)
        if frame_quality.get("overall_status") != "PASS" or not sensor_evidence:
            sensor_evidence = frame_quality
        else:
            sensor_evidence["frame_quality"] = frame_quality
        pipeline_diagnosis = diagnose_pipeline(
            sensor_input=sensor_evidence,
            pose_alignment=pose_diagnostics,
            phase_a=slam_eval_results,
            phase_b=tsdf_eval_results,
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
            "phase_b_tsdf_results": tsdf_eval_results,
            "phase_c_surface_results": surface_eval_results,
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
