"""Open3D VolumetricBlockGrid fusion backend (V2).

Streaming integrate -> on-device extraction. Resource calculation, frame
loading and artifact export live elsewhere (#40). Any CUDA failure falls back
to a single CPU retry; repeated OOM must be prevented upstream by
ResourceEstimate preflight (#99).

Time  O(F * visible_voxels)
Memory O(active_voxels) on device; bounded prefetch window in host RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class FusionInput:
    frame_ids: list
    load_depth_mm: Callable[[int], np.ndarray]
    load_rgb: Callable[[int], np.ndarray]
    pose_by_frame: dict
    load_mask: Optional[Callable[[int], np.ndarray]] = None


@dataclass
class FusionOutput:
    mesh_vertices: int = 0
    mesh_triangles: int = 0
    pcd_points: int = 0
    device: str = "cpu:0"
    cuda_fell_back: bool = False
    mesh_obj: object = None
    pcd_obj: object = None

    def to_dict(self) -> dict:
        return {
            "mesh_vertices": self.mesh_vertices,
            "mesh_triangles": self.mesh_triangles,
            "pcd_points": self.pcd_points,
            "device": self.device,
            "cuda_fell_back": self.cuda_fell_back,
        }

    @property
    def ok(self) -> bool:
        return self.mesh_triangles > 0


def _make_vbg(voxel_m: float, trunc_m: float, block_count: int, device):
    import open3d as o3d
    import open3d.core as o3c

    return o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel_m,
        block_resolution=16,
        block_count=block_count,
        device=device,
    )


_BYTES_PER_BLOCK = 16**3 * (1 + 1 + 3) * 4  # 4096 voxels * (tsdf+w+rgb f32)
_MC_OVERHEAD_FACTOR = 2.0  # Marching-Cubes assistance structure ~= TSDF size
# Occupancy is scene-dependent: a looping corridor fills space far denser than
# a bbox-cube model predicts. Degrade decisions therefore require the estimate
# to fit with 2x headroom, while the hard block cap stays budget-bounded.
_OCCUPANCY_SAFETY_FACTOR = 2.0
# Calibrated against an observed 62s-corridor capture (500 frames @10mm):
# 51,493 active blocks measured vs 226k predicted by a cube-fill model.
# 0.08 keeps ~3.4x headroom over that measurement while staying far below
# the naive cubic estimate, so corridor scenes are not degraded excessively.
_OCCUPANCY_FACTOR = 0.08


def _estimate_active_blocks(bbox_diag_m: float, voxel_m: float) -> int:
    voxels_along = bbox_diag_m / max(voxel_m, 1e-6)
    return int(_OCCUPANCY_FACTOR * (voxels_along**3) / 4096.0)


def required_vram_mb(bbox_diag_m: float, voxel_m: float,
                     vram_budget_mb: float | None = None) -> int:
    """Total VRAM (TSDF buffer + MC assistance) the fusion would want."""
    return int(_estimate_active_blocks(bbox_diag_m, voxel_m)
               * _BYTES_PER_BLOCK * _MC_OVERHEAD_FACTOR / 1e6)


def max_fitting_voxel_mm(bbox_diag_m: float, vram_budget_mb: float | None,
                         min_voxel_mm: float = 10.0, max_voxel_mm: float = 20.0,
                         step_factor: float = 1.25) -> float:
    """Coarsest-first search for the finest voxel that fits the VRAM budget."""
    if not vram_budget_mb or vram_budget_mb <= 0:
        return min_voxel_mm
    vox = min_voxel_mm
    while vox < max_voxel_mm and (
            required_vram_mb(bbox_diag_m, vox / 1000.0)
            * _OCCUPANCY_SAFETY_FACTOR > vram_budget_mb):
        vox = round(vox * step_factor, 1)
    return min(vox, max_voxel_mm)


def estimate_block_count(bbox_diag_m: float, voxel_m: float,
                         vram_budget_mb: float | None = None,
                         cap_blocks: int = 100000) -> int:
    """Block-count estimate, hard-capped by the usable VRAM budget.

    Open3D VBG allocates its full block buffer up front, so an inflated
    estimate is a guaranteed OOM on small GPUs: the VRAM cap is mandatory
    whenever a budget is supplied.
    """
    voxels_along = bbox_diag_m / max(voxel_m, 1e-6)
    est = _estimate_active_blocks(bbox_diag_m, voxel_m)
    limit = cap_blocks
    if vram_budget_mb is not None and vram_budget_mb > 0:
        usable_mb = vram_budget_mb / _MC_OVERHEAD_FACTOR
        limit = min(limit, int(usable_mb * 1e6 / _BYTES_PER_BLOCK))
    return max(4096, min(est, limit))


def integrate_frames(
    fusion_input: FusionInput,
    intrinsics_matrix: np.ndarray,
    width: int,
    height: int,
    voxel_m: float,
    trunc_mult: float = 4.0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 6.0,
    bbox_diag_m: float = 10.0,
    use_cuda: bool = True,
    prefetch: int = 4,
    depth_scale: float = 1000.0,
    vram_budget_mb: float | None = None,
    frames_per_chunk: int = 400,
    chunk_pause_s: float = 8.0,
) -> FusionOutput:
    """Integrate FUSE frames; returns extracted mesh stats without disk IO.

    Duty-cycle (L3 barrier): after every `frames_per_chunk` integrated frames
    the GPU idles `chunk_pause_s` seconds, capping sustained power draw.
    """
    import open3d as o3d
    import open3d.core as o3c

    try:
        if not (use_cuda and o3c.cuda.is_available()):
            raise RuntimeError("cuda unavailable")
        return _run(o3d, o3c, "CUDA:0", fusion_input, intrinsics_matrix, width, height,
                    voxel_m, trunc_mult, depth_min_m, depth_max_m, bbox_diag_m, prefetch,
                    depth_scale, vram_budget_mb, frames_per_chunk, chunk_pause_s)
    except Exception as cuda_err:
        n_frames = len(fusion_input.frame_ids)
        # L4 barrier: never auto-retry a huge integration on CPU — the
        # resulting minutes-long full-CPU load is its own hazard. The caller
        # degrades (coarser voxel / fewer frames) instead.
        if not use_cuda or n_frames > 600:
            raise
        print(f"[open3d_vbg] CUDA path failed ({cuda_err}); retrying on CPU")
        return _run(o3d, o3c, "CPU:0", fusion_input, intrinsics_matrix, width, height,
                    voxel_m, trunc_mult, depth_min_m, depth_max_m, bbox_diag_m, prefetch,
                    depth_scale, vram_budget_mb, frames_per_chunk, chunk_pause_s)


_CHUNKED_FUSION_THRESHOLD = 900  # frames; beyond this a single VBG cannot be
# trusted to stay within an 8GB-class VRAM budget on large scenes.


def _run(
    o3d, o3c, device_str, fi: FusionInput, K, width, height,
    voxel_m, trunc_mult, dmin, dmax, bbox_diag, prefetch, depth_scale,
    vram_budget_mb=None, frames_per_chunk=400, chunk_pause_s=8.0,
) -> FusionOutput:
    import time as _time

    dev = o3c.Device(device_str)
    K_t = o3c.Tensor(np.ascontiguousarray(K, dtype=np.float64))
    ids = [i for i in fi.frame_ids if i in fi.pose_by_frame]

    def duty_pause(n_done):
        if frames_per_chunk > 0 and n_done >= frames_per_chunk and chunk_pause_s > 0:
            print(f"[open3d_vbg] duty-cycle pause {chunk_pause_s:.0f}s after "
                  f"{n_done} frames", flush=True)
            _time.sleep(chunk_pause_s)
            return 0
        return n_done

    def integrate_into(vbg, frame_ids):
        n = 0
        for start in range(0, len(frame_ids), prefetch):
            batch = frame_ids[start : start + prefetch]
            for fid in batch:
                depth_mm = fi.load_depth_mm(fid)
                bgr = fi.load_rgb(fid)
                if depth_mm is None or bgr is None:
                    continue
                T_wc = fi.pose_by_frame[fid]
                extrinsic = np.linalg.inv(T_wc)
                if fi.load_mask is not None:
                    m = fi.load_mask(fid)
                    if m is not None:
                        depth_mm = np.where(m, depth_mm, 0)
                if not np.any(depth_mm):
                    continue
                color_rgb = bgr[:, :, ::-1] if bgr.ndim == 3 else bgr
                depth_t = o3d.t.geometry.Image(
                    o3c.Tensor(np.ascontiguousarray(depth_mm.astype(np.uint16)), device=dev))
                color_t = o3d.t.geometry.Image(
                    o3c.Tensor(np.ascontiguousarray(color_rgb[:, :, :3].astype(np.uint8)), device=dev))
                extrinsic_t = o3c.Tensor(np.asarray(extrinsic, dtype=np.float64))
                coords = vbg.compute_unique_block_coordinates(
                    depth_t, K_t, extrinsic_t,
                    depth_scale=depth_scale, depth_max=dmax,
                    trunc_voxel_multiplier=float(trunc_mult),
                )
                vbg.integrate(
                    coords, depth_t, color_t,
                    K_t, K_t,
                    extrinsic_t,
                    depth_scale=depth_scale, depth_max=dmax,
                    trunc_voxel_multiplier=float(trunc_mult),
                )
                n += 1
        return n

    def extract(vbg):
        mesh, pcd = _extract(vbg, o3c, dev, device_str)
        return mesh, pcd

    if len(ids) > _CHUNKED_FUSION_THRESHOLD:
        # ---- Chunked fusion: bounded VRAM regardless of scene size (#30) ----
        # Each frame chunk gets its own VBG sized by the chunk-local camera
        # bbox; meshes are extracted per chunk and merged. Peak GPU memory is
        # therefore governed by the chunk, never by the whole capture.
        import gc

        CHUNK = 800
        mesh_acc = None
        pcd_acc = None
        total_stats = {"mesh_vertices": 0, "mesh_triangles": 0, "pcd_points": 0}
        n_integrated = 0
        n_chunks = (len(ids) + CHUNK - 1) // CHUNK
        for ci in range(n_chunks):
            chunk_ids = ids[ci * CHUNK : (ci + 1) * CHUNK]
            pts = np.asarray([fi.pose_by_frame[i][:3, 3] for i in chunk_ids])
            chunk_diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) \
                if len(pts) >= 2 else 6.0
            chunk_diag += 2.0  # observed-surface margin beyond camera path
            vbg = _make_vbg(voxel_m, trunc_mult,
                            estimate_block_count(chunk_diag, voxel_m,
                                                 vram_budget_mb), dev)
            integrate_into(vbg, chunk_ids)
            cmesh, cpcd = extract(vbg)
            del vbg
            gc.collect()
            total_stats["mesh_vertices"] += len(cmesh.vertices)
            total_stats["mesh_triangles"] += len(cmesh.triangles)
            total_stats["pcd_points"] += len(cpcd.points)
            if mesh_acc is None:
                mesh_acc, pcd_acc = cmesh, cpcd
            else:
                mesh_acc += cmesh
                pcd_acc += cpcd
            print(f"[open3d_vbg] chunk {ci+1}/{n_chunks} done "
                  f"(diag {chunk_diag:.1f}m, tris so far "
                  f"{total_stats['mesh_triangles']})", flush=True)
            n_integrated = duty_pause(len(chunk_ids)) or 0
            n_integrated = min(n_integrated + len(chunk_ids), frames_per_chunk + 1)

        out = FusionOutput(device=device_str.lower())
        out.mesh_vertices = total_stats["mesh_vertices"]
        out.mesh_triangles = total_stats["mesh_triangles"]
        out.pcd_points = total_stats["pcd_points"]
        out.mesh_obj = mesh_acc
        out.pcd_obj = pcd_acc
        return out

    # ---- Single-VBG path (small captures) ----
    vbg = _make_vbg(voxel_m, trunc_mult,
                    estimate_block_count(bbox_diag, voxel_m, vram_budget_mb), dev)
    n_integrated = 0
    for start in range(0, len(ids), prefetch):
        batch = ids[start : start + prefetch]
        n_integrated = duty_pause(n_integrated)
        integrate_into(vbg, batch)
        n_integrated += len(batch)

    mesh, pcd = extract(vbg)
    out = FusionOutput(device=device_str.lower())
    out.mesh_vertices = len(mesh.vertices)
    out.mesh_triangles = len(mesh.triangles)
    out.pcd_points = len(pcd.points)
    out.mesh_obj = mesh
    out.pcd_obj = pcd

    import gc

    del vbg
    gc.collect()
    return out


def _extract(vbg, o3c, dev, device_str):
    """Extract mesh/point cloud; on CUDA extraction OOM migrate the VBG to CPU."""
    import open3d as o3d

    try:
        mesh = vbg.extract_triangle_mesh().to_legacy()
        pcd = vbg.extract_point_cloud().to_legacy()
        return mesh, pcd
    except Exception as exc:
        if dev == o3c.Device("CPU:0"):
            raise
        print(f"[open3d_vbg] GPU extraction failed ({exc}); migrating VBG to CPU")
        vbg_cpu = vbg.to(o3c.Device("CPU:0"))
        mesh = vbg_cpu.extract_triangle_mesh().to_legacy()
        pcd = vbg_cpu.extract_point_cloud().to_legacy()
        return mesh, pcd
