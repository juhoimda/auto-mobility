"""
rebuild.py — Standalone Deterministic Rebuild Entry Point from Best Configuration.

Usage:
  python3 -m auto_mobility.benchmark.rebuild --config path/to/best_config.json [--out path/to/output.obj]
"""

import sys
import json
import argparse
from pathlib import Path

from auto_mobility.config import FRAME_DIR
from auto_mobility.benchmark.workers import run_tsdf_worker, run_direct_fusion_worker, run_surface_worker
from auto_mobility.benchmark.candidate import CandidateSpec


def rebuild_from_config(config_path: Path, output_mesh_path: Path = None) -> Path:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dataset_name = cfg.get("dataset")
    dataset_dir = FRAME_DIR / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    traj_path = cfg.get("trajectory_path")
    if not traj_path or not Path(traj_path).exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    fusion_info = cfg.get("fusion", {})
    fusion_method = fusion_info.get("method", "tsdf")
    fusion_params = fusion_info.get("params", {})

    surface_info = cfg.get("surface", {})
    surface_method = surface_info.get("method", "tsdf_direct")
    surface_params = surface_info.get("params", {})

    recon_info = cfg.get("reconstruction", {})
    stride = recon_info.get("frame_stride", 1)

    voxel_m = float(fusion_params.get("voxel_size_m", 0.010))
    depth_min_m = float(fusion_params.get("depth_min_m", 0.3))
    depth_max_m = float(fusion_params.get("depth_max_m", 3.0))
    trunc_mult = float(fusion_params.get("trunc_mult", 4.0))

    if output_mesh_path is None:
        out_dir = config_path.parent
        output_mesh_path = out_dir / "best_rebuilt.obj"
    else:
        output_mesh_path = Path(output_mesh_path)
        output_mesh_path.parent.mkdir(parents=True, exist_ok=True)

    temp_pcd = output_mesh_path.parent / f"{output_mesh_path.stem}_cloud.ply"

    print(f"🔨 Rebuilding mesh for '{dataset_name}' from {config_path.name}")
    print(f"   Fusion: {fusion_method} (voxel={voxel_m*1000:.1f}mm), Surface: {surface_method}, Stride: {stride}")

    if fusion_method == "direct_pointcloud":
        w_res = run_direct_fusion_worker(
            dataset_dir=str(dataset_dir),
            traj_file=traj_path,
            pcd_path=str(temp_pcd),
            voxel=voxel_m,
            depth_min=depth_min_m,
            depth_max=depth_max_m,
            stride=stride
        )
        if not w_res.is_success:
            raise RuntimeError(f"Direct fusion rebuild failed: {w_res.error_message}")

        w_surf = run_surface_worker(
            input_ply=str(temp_pcd),
            output_mesh=str(output_mesh_path),
            method=surface_method,
            voxel=voxel_m,
            depth=surface_params.get("depth", 8),
            simplify=0.0,
            no_simplify=True
        )
        if not w_surf.is_success:
            raise RuntimeError(f"Surface reconstruction rebuild failed: {w_surf.error_message}")
    else:
        # TSDF
        w_res = run_tsdf_worker(
            dataset_dir=str(dataset_dir),
            traj_file=traj_path,
            mesh_path=str(output_mesh_path) if surface_method == "tsdf_direct" else None,
            pcd_path=str(temp_pcd),
            voxel=voxel_m,
            depth_max=depth_max_m,
            trunc_mult=trunc_mult,
            stride=stride,
            quick=False
        )
        if not w_res.is_success:
            raise RuntimeError(f"TSDF fusion rebuild failed: {w_res.error_message}")

        if surface_method != "tsdf_direct":
            w_surf = run_surface_worker(
                input_ply=str(temp_pcd),
                output_mesh=str(output_mesh_path),
                method=surface_method,
                voxel=voxel_m,
                depth=surface_params.get("depth", 8),
                simplify=0.0,
                no_simplify=True
            )
            if not w_surf.is_success:
                raise RuntimeError(f"Surface reconstruction rebuild failed: {w_surf.error_message}")

    print(f"✅ Rebuild complete! Output mesh: {output_mesh_path}")
    return output_mesh_path


def main():
    parser = argparse.ArgumentParser(description="Standalone Full Rebuild from best_config.json")
    parser.add_argument("--config", required=True, type=Path, help="Path to best_config.json")
    parser.add_argument("--out", required=False, type=Path, default=None, help="Path to output mesh .obj")
    args = parser.parse_args()

    try:
        rebuild_from_config(args.config, args.out)
    except Exception as e:
        print(f"❌ Error during rebuild: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
