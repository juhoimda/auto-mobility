"""Fusion worker: isolated TSDF integration + mesh extraction subprocess.

Heavy Open3D VBG work (integration AND Marching-Cubes extraction) runs here so
that native crashes (CUDA OOM segfaults) cannot kill the main pipeline. The
parent passes a JSON spec + pose/mask npz bundles; this worker writes
mesh/point-cloud files plus a stats JSON.

Run: python -m auto_mobility.reconstruction.fusion.worker --spec spec.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _resolve(dataset_dir: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else dataset_dir / pp


def run_with_spec(spec_path: Path) -> int:
    import cv2
    import open3d as o3d

    from auto_mobility.reconstruction.fusion import FusionInput, integrate_frames

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    dataset_dir = Path(spec["dataset_dir"])
    ds_rows = {}
    import csv

    with open(dataset_dir / "frames.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            ds_rows[int(row["frame_id"])] = row

    def rgb(i: int):
        row = ds_rows.get(i)
        if row is None:
            return None
        img = cv2.imread(str(_resolve(dataset_dir, row["rgb_path"])))
        if img is None:
            img = cv2.imread(str(row["rgb_path"]))
        return img

    def depth(i: int):
        row = ds_rows.get(i)
        if row is None:
            return None
        img = cv2.imread(str(_resolve(dataset_dir, row["depth_path"])),
                         cv2.IMREAD_UNCHANGED)
        if img is None:
            img = cv2.imread(str(row["depth_path"]), cv2.IMREAD_UNCHANGED)
        return img

    bundle = np.load(spec["poses_npz"])
    pose_by_frame = {int(k): bundle[k] for k in bundle.files}

    masks = None
    mask_fn = None
    if spec.get("masks_npz"):
        mb = np.load(spec["masks_npz"])
        masks = {int(k): mb[k] for k in mb.files}
        mask_fn = lambda i: masks.get(i)

    fi = FusionInput(
        frame_ids=list(spec["frame_ids"]),
        load_depth_mm=depth,
        load_rgb=rgb,
        pose_by_frame=pose_by_frame,
        load_mask=mask_fn,
    )
    K = np.asarray(spec["K"], dtype=np.float64)

    out = integrate_frames(
        fi, K, int(spec["width"]), int(spec["height"]),
        voxel_m=float(spec["voxel_m"]),
        trunc_mult=float(spec["trunc_mult"]),
        bbox_diag_m=float(spec["bbox_diag_m"]),
        use_cuda=bool(spec.get("use_cuda", True)),
        vram_budget_mb=spec.get("vram_budget_mb"),
        frames_per_chunk=int(spec.get("frames_per_chunk", 400)),
        chunk_pause_s=float(spec.get("chunk_pause_s", 8.0)),
    )

    if out.mesh_obj is not None and len(out.mesh_obj.triangles) > 0:
        o3d.io.write_triangle_mesh(str(spec["mesh_out"]), out.mesh_obj)
    if out.pcd_obj is not None and len(out.pcd_obj.points) > 0:
        o3d.io.write_point_cloud(str(spec["pcd_out"]), out.pcd_obj)
    Path(spec["stats_out"]).write_text(json.dumps(out.to_dict(), indent=2))
    return 0 if out.mesh_triangles > 0 else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args()
    try:
        return run_with_spec(Path(args.spec))
    except Exception as exc:  # surface errors via non-zero exit; no traceback spam
        print(f"[fusion_worker] failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
