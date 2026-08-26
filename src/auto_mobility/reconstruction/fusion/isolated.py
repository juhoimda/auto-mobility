"""Subprocess-isolated fusion: native crash containment for Open3D VBG.

Open3D 0.19 Marching-Cubes extraction can segfault (CUDA OOM / large active
block sets). Running integration+extraction in a monitored subprocess keeps
such crashes from destroying the pipeline (#24/#31/#33): the parent gets a
ProcessOutcome, partial files are never treated as success, and the pipeline
continues with degraded evidence.

Complexity: one subprocess per fusion call. Memory: parent holds only the
returned mesh/point cloud loaded from disk.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from auto_mobility.reconstruction.fusion.open3d_vbg import FusionOutput
from auto_mobility.reconstruction.runtime.process import (
    ProcessStatus, run_monitored_process,
)

_PROJECT_SRC = Path(__file__).resolve().parents[3]


@dataclass
class IsolatedFusionResult:
    output: FusionOutput
    ok: bool
    detail: str


def integrate_frames_isolated(
    dataset_dir: Path,
    frame_ids: list,
    pose_by_frame: dict,
    masks_by_frame: dict | None,
    K,
    width: int,
    height: int,
    voxel_m: float,
    trunc_mult: float,
    bbox_diag_m: float,
    work_dir: Path,
    tag: str,
    vram_budget_mb: float | None = None,
    ram_limit_mb: int | None = None,
    timeout_s: float = 1200.0,
    gpu_limits: dict | None = None,
    frames_per_chunk: int = 400,
    chunk_pause_s: float = 8.0,
) -> IsolatedFusionResult:
    work_dir = Path(work_dir).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset dir missing: {dataset_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    poses_npz = work_dir / f"{tag}_poses.npz"
    np.savez_compressed(
        poses_npz,
        **{str(fid): np.asarray(T) for fid, T in pose_by_frame.items()},
    )

    masks_npz = None
    if masks_by_frame:
        masks_npz = work_dir / f"{tag}_masks.npz"
        np.savez_compressed(
            masks_npz,
            **{str(fid): np.asarray(m, dtype=bool) for fid, m in masks_by_frame.items()},
        )

    mesh_out = work_dir / f"{tag}_mesh.ply"
    pcd_out = work_dir / f"{tag}_pcd.ply"
    stats_out = work_dir / f"{tag}_stats.json"
    spec = {
        "dataset_dir": str(dataset_dir),
        "frame_ids": [int(i) for i in frame_ids],
        "poses_npz": str(poses_npz),
        "masks_npz": str(masks_npz) if masks_npz else None,
        "K": np.asarray(K).tolist(),
        "width": int(width),
        "height": int(height),
        "voxel_m": float(voxel_m),
        "trunc_mult": float(trunc_mult),
        "bbox_diag_m": float(bbox_diag_m),
        "use_cuda": True,
        "vram_budget_mb": vram_budget_mb,
        "frames_per_chunk": int(frames_per_chunk),
        "chunk_pause_s": float(chunk_pause_s),
        "store_color": False,
        "weight_dtype": "float32",
        "extraction_mode": "mesh_only",
        "allow_cpu_migration": False,
        "mesh_out": str(mesh_out),
        "pcd_out": str(pcd_out),
        "stats_out": str(stats_out),
    }
    spec_path = work_dir / f"{tag}_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    cmd = [
        sys.executable, "-m", "auto_mobility.reconstruction.fusion.worker",
        "--spec", str(spec_path),
    ]
    env = {
        k: v for k, v in __import__("os").environ.items()
        if not k.startswith(("ROS_", "RMW_"))
    }
    env["PYTHONPATH"] = f"{_PROJECT_SRC}:{env.get('PYTHONPATH', '')}"
    # §7 phase-specific CPU caps: GPU TSDF stage is CPU-capped (3-4 threads) so
    # that combined CPU+GPU power stays laptop-safe. Previously global OMP=6
    # caused sustained 572% load with GPU 100% (§7).
    # Caller may override via env; default 4 for GPU-bound fusion.
    omp_threads = str(__import__("os").environ.get("FUSION_OMP_THREADS", "4"))
    env["OMP_NUM_THREADS"] = omp_threads
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = omp_threads
    env["OPENCV_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"

    outcome = run_monitored_process(
        cmd,
        log_path=work_dir / f"{tag}_worker.log",
        env=env,
        cwd=str(work_dir),
        timeout_s=timeout_s,
        ram_limit_mb=ram_limit_mb,
        gpu_limits=gpu_limits,
    )
    if outcome.status != ProcessStatus.OK or not stats_out.is_file():
        return IsolatedFusionResult(
            output=FusionOutput(), ok=False,
            detail=f"worker {outcome.status.value} rc={outcome.returncode} "
                   f"elapsed={outcome.elapsed_s:.0f}s peak_rss={outcome.peak_rss_mb:.0f}MB",
        )

    import open3d as o3d

    out = FusionOutput(**json.loads(stats_out.read_text()))
    if mesh_out.is_file():
        out.mesh_obj = o3d.io.read_triangle_mesh(str(mesh_out))
        out.mesh_vertices = len(out.mesh_obj.vertices)
        out.mesh_triangles = len(out.mesh_obj.triangles)
    if pcd_out.is_file():
        out.pcd_obj = o3d.io.read_point_cloud(str(pcd_out))
        out.pcd_points = len(out.pcd_obj.points)
    return IsolatedFusionResult(output=out, ok=out.mesh_triangles > 0,
                                detail=out.device)
