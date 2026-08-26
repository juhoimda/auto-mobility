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


from enum import Enum


@dataclass(frozen=True)
class ActiveBlockPlan:
    """Explicit unit contract for VBG allocation planning (P0 #1 fix).

    All byte fields are bytes (SI). MB fields are megabytes (1e6 bytes).
    planned_capacity_blocks is the hash capacity actually allocated via _make_vbg.
    """
    unique_blocks: int
    planned_capacity_blocks: int
    tsdf_bytes: int  # allocated tsdf bytes (capacity * bpb)
    extraction_peak_bytes: int  # allocated * MC overhead + margin
    extraction_peak_mb: float  # extraction_peak_bytes / 1e6

    def to_dict(self) -> dict:
        return {
            "unique_block_count": self.unique_blocks,
            "estimated_hash_capacity": self.planned_capacity_blocks,
            "safe_block_count": self.planned_capacity_blocks,
            "estimated_tsdf_bytes": self.tsdf_bytes,
            "estimated_extraction_peak": self.extraction_peak_bytes,
            "estimated_extraction_peak_mb": self.extraction_peak_mb,
            "estimated_tsdf_bytes_unique": self.unique_blocks * 0,  # telemetry placeholder
            # compatibility aliases with explicit units
            "unique_blocks": self.unique_blocks,
            "planned_capacity_blocks": self.planned_capacity_blocks,
            "tsdf_bytes": self.tsdf_bytes,
            "extraction_peak_bytes": self.extraction_peak_bytes,
            "extraction_peak_mb": self.extraction_peak_mb,
            "planner": "cpu_vbg_coordinates",
        }


class ExtractionMode(str, Enum):
    MESH_ONLY = "mesh_only"
    PCD_ONLY = "pcd_only"
    MESH_AND_PCD = "mesh_and_pcd"


def _bytes_per_block(store_color: bool = True, weight_dtype_bytes: int = 4) -> int:
    """Bytes per 16^3 block; color adds 3*f32 (12B) per voxel (§15)."""
    # tsdf f32=4 + weight (f32=4 or u16=2) + color 3*f32 if present
    per_voxel = 4 + weight_dtype_bytes + (12 if store_color else 0)
    return 16**3 * per_voxel


def _make_vbg(voxel_m: float, trunc_m: float, block_count: int, device,
              store_color: bool = False, weight_dtype: str = "float32"):
    """Create VoxelBlockGrid. Geometry fusion defaults to no-color (§15)."""
    import open3d as o3d
    import open3d.core as o3c

    if store_color:
        return o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=voxel_m,
            block_resolution=16,
            block_count=block_count,
            device=device,
        )
    # §15: 8 bytes/voxel vs 20 when color removed; §16 weight dtype experiment
    if weight_dtype == "uint16":
        return o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight"),
            attr_dtypes=(o3c.float32, o3c.uint16),
            attr_channels=((1,), (1,)),
            voxel_size=voxel_m,
            block_resolution=16,
            block_count=block_count,
            device=device,
        )
    return o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"),
        attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1,), (1,)),
        voxel_size=voxel_m,
        block_resolution=16,
        block_count=block_count,
        device=device,
    )


_BYTES_PER_BLOCK = 16**3 * (1 + 1 + 3) * 4  # 4096 voxels * (tsdf+w+rgb f32) legacy
_BYTES_PER_BLOCK_NO_COLOR = 16**3 * (1 + 1) * 4  # 8192*? actually 4096*8=32768
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


# ---- ActiveBlockPlanner (§13/§14) ----
def _planner_bytes_per_block(store_color: bool, weight_dtype: str = "float32") -> int:
    wt = 2 if weight_dtype == "uint16" else 4
    return _bytes_per_block(store_color=store_color, weight_dtype_bytes=wt)


def plan_active_blocks(
    frame_ids: list,
    pose_by_frame: dict,
    K: np.ndarray,
    width: int,
    height: int,
    voxel_m: float,
    trunc_mult: float = 4.0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 4.0,
    load_depth_mm=None,
    load_mask=None,
    store_color: bool = False,
    weight_dtype: str = "float32",
    sample_stride: int = 1,
) -> dict:
    """Estimate actually-visible TSDF block coordinates (§13).

    Uses a tiny CPU VBG (no GPU) to call compute_unique_block_coordinates for
    representative frames and unions block indices. Frustum/orientation-aware,
    unlike bbox-diagonal cube model.

    Returns dict with unique_block_count, estimated_hash_capacity,
    estimated_tsdf_bytes, estimated_extraction_peak, safe_block_count.
    Falls back to bbox estimate when planner cannot run.
    """
    if load_depth_mm is None or not frame_ids:
        return {"unique_block_count": _estimate_active_blocks(10.0, voxel_m),
                "estimated_hash_capacity": _estimate_active_blocks(10.0, voxel_m),
                "estimated_tsdf_bytes": 0, "estimated_extraction_peak": 0,
                "safe_block_count": _estimate_active_blocks(10.0, voxel_m),
                "planner": "fallback_no_loader"}
    try:
        import open3d as o3d
        import open3d.core as o3c
        dev = o3c.Device("CPU:0")
        # tiny VBG just for coordinate helper — block_count minimal
        vbg = _make_vbg(voxel_m, trunc_mult, block_count=4096, device=dev,
                        store_color=store_color, weight_dtype=weight_dtype)
        K_t = o3c.Tensor(np.ascontiguousarray(K, dtype=np.float64))
        depth_scale = 1000.0
        block_set = set()
        ids = frame_ids[::max(1, sample_stride)]
        for fid in ids:
            if fid not in pose_by_frame:
                continue
            depth_mm = load_depth_mm(fid)
            if depth_mm is None or not np.any(depth_mm):
                continue
            if load_mask is not None:
                m = load_mask(fid)
                if m is not None:
                    depth_mm = np.where(m, depth_mm, 0)
                    if not np.any(depth_mm):
                        continue
            # clip depth_max per §17 (planner must respect same depth_max as fusion)
            d = depth_mm.astype(np.uint16)
            # mask out beyond depth_max
            if depth_max_m < 6.0:
                d = np.where(d.astype(np.float32) / 1000.0 <= depth_max_m, d, 0).astype(np.uint16)
                if not np.any(d):
                    continue
            T_wc = pose_by_frame[fid]
            extrinsic = np.linalg.inv(T_wc)
            depth_t = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d), device=dev))
            extrinsic_t = o3c.Tensor(np.asarray(extrinsic, dtype=np.float64))
            try:
                coords = vbg.compute_unique_block_coordinates(
                    depth_t, K_t, extrinsic_t,
                    depth_scale=depth_scale, depth_max=depth_max_m,
                    trunc_voxel_multiplier=float(trunc_mult),
                )
                # coords is Tensor of shape [N, 3] int — convert to python set
                arr = coords.cpu().numpy() if hasattr(coords, "cpu") else np.asarray(coords)
                for c in arr:
                    block_set.add((int(c[0]), int(c[1]), int(c[2])))
            except Exception:
                continue
        del vbg
        uniq = len(block_set) if block_set else _estimate_active_blocks(10.0, voxel_m)
        bpb = _planner_bytes_per_block(store_color, weight_dtype)
        # hash capacity is next pow2 > uniq * safety factor (§13 2.0)
        cap = 1
        need = int(uniq * _OCCUPANCY_SAFETY_FACTOR)
        while cap < need:
            cap <<= 1
        cap = max(4096, cap)
        tsdf_bytes_alloc = int(cap * bpb)
        tsdf_bytes_unique = int(uniq * bpb)
        extraction_peak_bytes = tsdf_bytes_alloc + int(tsdf_bytes_unique * (_MC_OVERHEAD_FACTOR - 1.0) * 1.05)
        # keep at least unique*MC to avoid under-reporting on tiny scenes
        extraction_peak_bytes = max(extraction_peak_bytes,
                                    int(tsdf_bytes_unique * _MC_OVERHEAD_FACTOR))
        return {
            "unique_block_count": int(uniq),
            "estimated_hash_capacity": int(cap),
            "estimated_tsdf_bytes": int(tsdf_bytes_alloc),
            "estimated_tsdf_bytes_unique": int(tsdf_bytes_unique),
            "estimated_extraction_peak": int(extraction_peak_bytes),
            "estimated_extraction_peak_mb": float(extraction_peak_bytes / 1e6),
            "safe_block_count": int(cap),
            "planner": "cpu_vbg_coordinates",
            "sampled_frames": len(ids),
        }
    except Exception as exc:
        return {
            "unique_block_count": _estimate_active_blocks(10.0, voxel_m),
            "estimated_hash_capacity": _estimate_active_blocks(10.0, voxel_m),
            "estimated_tsdf_bytes": 0, "estimated_extraction_peak": 0,
            "safe_block_count": _estimate_active_blocks(10.0, voxel_m),
            "planner": f"fallback_error:{exc}",
        }


def required_vram_mb(bbox_diag_m: float, voxel_m: float,
                     vram_budget_mb: float | None = None,
                     store_color: bool = False,
                     weight_dtype: str = "float32") -> int:
    """Total VRAM (TSDF buffer + MC assistance) the fusion would want."""
    bpb = _bytes_per_block(store_color=store_color,
                           weight_dtype_bytes=(2 if weight_dtype == "uint16" else 4))
    return int(_estimate_active_blocks(bbox_diag_m, voxel_m)
               * bpb * _MC_OVERHEAD_FACTOR / 1e6)


def required_vram_mb_planned(planner_out: dict) -> int:
    """VRAM from ActiveBlockPlanner output (planner includes MC overhead).

    Contract: estimated_extraction_peak is BYTES, returned MB = bytes/1e6.
    Also accepts estimated_extraction_peak_mb alias.
    """
    if "estimated_extraction_peak_mb" in planner_out and planner_out["estimated_extraction_peak_mb"]:
        return int(float(planner_out["estimated_extraction_peak_mb"]))
    return int(planner_out.get("estimated_extraction_peak", 0) / 1e6)


def required_vram_mb_planned_bytes(planner_out: dict) -> int:
    """Return bytes contract explicitly."""
    return int(planner_out.get("estimated_extraction_peak", 0) or
               int(float(planner_out.get("estimated_extraction_peak_mb", 0)) * 1e6))


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
                         cap_blocks: int = 100000,
                         store_color: bool = False,
                         weight_dtype: str = "float32") -> int:
    """Block-count estimate, hard-capped by the usable VRAM budget.

    Open3D VBG allocates its full block buffer up front, so an inflated
    estimate is a guaranteed OOM on small GPUs: the VRAM cap is mandatory
    whenever a budget is supplied.  Includes §15 color-removal byte correction.
    """
    voxels_along = bbox_diag_m / max(voxel_m, 1e-6)
    est = _estimate_active_blocks(bbox_diag_m, voxel_m)
    bpb = _bytes_per_block(store_color=store_color,
                           weight_dtype_bytes=(2 if weight_dtype == "uint16" else 4))
    limit = cap_blocks
    if vram_budget_mb is not None and vram_budget_mb > 0:
        usable_mb = vram_budget_mb / _MC_OVERHEAD_FACTOR
        limit = min(limit, int(usable_mb * 1e6 / bpb))
    return max(4096, min(est, limit))


def integrate_frames(
    fusion_input: FusionInput,
    intrinsics_matrix: np.ndarray,
    width: int,
    height: int,
    voxel_m: float,
    trunc_mult: float = 4.0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 4.0,
    bbox_diag_m: float = 10.0,
    use_cuda: bool = True,
    prefetch: int = 4,
    depth_scale: float = 1000.0,
    vram_budget_mb: float | None = None,
    frames_per_chunk: int = 400,
    chunk_pause_s: float = 8.0,
    store_color: bool = False,
    weight_dtype: str = "float32",
    extraction_mode: str = "mesh_only",
    allow_cpu_migration: bool = False,
    planned_block_count: int | None = None,
) -> FusionOutput:
    """Integrate FUSE frames; returns extracted mesh stats without disk IO.

    Duty-cycle (L3 barrier): after every `frames_per_chunk` integrated frames
    the GPU idles `chunk_pause_s` seconds, capping sustained power draw.

    §15: store_color=False (default) reduces 20->8 bytes/voxel. Geometry mesh
    does not need TSDF color; texture is baked from RGB frames.
    §17: depth_max 4.0m default (indoor D435i, p98+0.3), not 6m unconditional.
    §18: extraction_mode MESH_ONLY/PCD_ONLY/MESH_AND_PCD controls peak.
    §19: CPU migration only for tiny canary when allow_cpu_migration=True.
    """
    import open3d as o3d
    import open3d.core as o3c

    if isinstance(extraction_mode, ExtractionMode):
        extraction_mode = extraction_mode.value

    # P0 #7: no-color CPU test must run without CUDA; allow explicit CPU path
    if not use_cuda:
        return _run(o3d, o3c, "CPU:0", fusion_input, intrinsics_matrix, width, height,
                    voxel_m, trunc_mult, depth_min_m, depth_max_m, bbox_diag_m, prefetch,
                    depth_scale, vram_budget_mb, frames_per_chunk, chunk_pause_s,
                    store_color, weight_dtype, extraction_mode, allow_cpu_migration,
                    planned_block_count)
    try:
        if not o3c.cuda.is_available():
            raise RuntimeError("cuda unavailable")
        return _run(o3d, o3c, "CUDA:0", fusion_input, intrinsics_matrix, width, height,
                    voxel_m, trunc_mult, depth_min_m, depth_max_m, bbox_diag_m, prefetch,
                    depth_scale, vram_budget_mb, frames_per_chunk, chunk_pause_s,
                    store_color, weight_dtype, extraction_mode, allow_cpu_migration,
                    planned_block_count)
    except Exception as cuda_err:
        n_frames = len(fusion_input.frame_ids)
        # L4 barrier: never auto-retry a huge integration on CPU — the
        # resulting minutes-long full-CPU load is its own hazard. The caller
        # degrades (coarser voxel / fewer frames) instead.
        if n_frames > 600:
            raise
        # §19: CPU fallback only for small jobs or explicit allow
        if not allow_cpu_migration and n_frames > 200:
            raise
        print(f"[open3d_vbg] CUDA path failed ({cuda_err}); retrying on CPU")
        return _run(o3d, o3c, "CPU:0", fusion_input, intrinsics_matrix, width, height,
                    voxel_m, trunc_mult, depth_min_m, depth_max_m, bbox_diag_m, prefetch,
                    depth_scale, vram_budget_mb, frames_per_chunk, chunk_pause_s,
                    store_color, weight_dtype, extraction_mode, False,
                    planned_block_count)


_CHUNKED_FUSION_THRESHOLD = 900  # frames; beyond this a single VBG cannot be
# trusted to stay within an 8GB-class VRAM budget on large scenes.


def _run(
    o3d, o3c, device_str, fi: FusionInput, K, width, height,
    voxel_m, trunc_mult, dmin, dmax, bbox_diag, prefetch, depth_scale,
    vram_budget_mb=None, frames_per_chunk=400, chunk_pause_s=8.0,
    store_color=False, weight_dtype="float32", extraction_mode="mesh_only",
    allow_cpu_migration=False,
    planned_block_count: int | None = None,
) -> FusionOutput:
    import time as _time

    # §18/§15: when PCD is requested, keep color channel for point-cloud extraction
    # (legacy Open3D pcd extraction expects color tensor; no-color pcd path segfaults)
    if extraction_mode in ("pcd_only", "mesh_and_pcd") and not store_color:
        store_color = True

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
                if depth_mm is None or not np.any(depth_mm):
                    continue
                T_wc = fi.pose_by_frame[fid]
                extrinsic = np.linalg.inv(T_wc)
                if fi.load_mask is not None:
                    m = fi.load_mask(fid)
                    if m is not None:
                        depth_mm = np.where(m, depth_mm, 0)
                        if not np.any(depth_mm):
                            continue
                # P0 #6: no-color geometry fusion must NOT decode RGB and must use depth-only overload
                extrinsic_t = o3c.Tensor(np.asarray(extrinsic, dtype=np.float64))
                depth_t = o3d.t.geometry.Image(
                    o3c.Tensor(np.ascontiguousarray(depth_mm.astype(np.uint16)), device=dev))
                coords = vbg.compute_unique_block_coordinates(
                    depth_t, K_t, extrinsic_t,
                    depth_scale=depth_scale, depth_max=dmax,
                    trunc_voxel_multiplier=float(trunc_mult),
                )
                if store_color:
                    bgr = fi.load_rgb(fid)
                    if bgr is None:
                        continue
                    color_rgb = bgr[:, :, ::-1] if bgr.ndim == 3 else bgr
                    color_t = o3d.t.geometry.Image(
                        o3c.Tensor(np.ascontiguousarray(color_rgb[:, :, :3].astype(np.uint8)), device=dev))
                    vbg.integrate(
                        coords, depth_t, color_t,
                        K_t, K_t,
                        extrinsic_t,
                        depth_scale=depth_scale, depth_max=dmax,
                        trunc_voxel_multiplier=float(trunc_mult),
                    )
                else:
                    # depth-only overload for VBG with attrs (tsdf, weight)
                    try:
                        vbg.integrate(
                            coords, depth_t,
                            K_t,
                            extrinsic_t,
                            depth_scale=depth_scale, depth_max=dmax,
                            trunc_voxel_multiplier=float(trunc_mult),
                        )
                    except TypeError:
                        # fallback for Open3D versions where depth-only still expects 6 args but color empty
                        vbg.integrate(
                            coords, depth_t, K_t, extrinsic_t,
                            depth_scale=depth_scale, depth_max=dmax,
                            trunc_voxel_multiplier=float(trunc_mult),
                        )
                n += 1
        return n

    def extract(vbg):
        # §18/§20: sequential extract + early release reduces peak (mesh then pcd)
        mesh, pcd = _extract(vbg, o3c, dev, device_str,
                             mode=extraction_mode,
                             allow_cpu_migration=allow_cpu_migration)
        return mesh, pcd

    # §21: adaptive chunk sizing via ActiveBlockPlanner + VRAM budget, not fixed 800.
    # If planner estimates >budget for CHUNK=800, shrink chunk.
    def _adaptive_chunk_size() -> int:
        if vram_budget_mb is None or vram_budget_mb <= 0:
            return 800
        # target active blocks per chunk = budget/(bytes_per_block*MC* safety)
        bpb = _bytes_per_block(store_color=store_color,
                               weight_dtype_bytes=(2 if weight_dtype == "uint16" else 4))
        safe_blocks = int((vram_budget_mb * 1e6 / bpb / _MC_OVERHEAD_FACTOR) / _OCCUPANCY_SAFETY_FACTOR)
        # §21 corrected: hallway loop at 10mm measured 13.8 blocks/frame avg but
        # HashMap doubling + MC overhead pushes real by 2-3x. Use 150 divisor and
        # tighter clamp [200, 400] for sustained >800 to keep sustained power low.
        # For 8GB budget, safe_blocks ~33k => 33k/150=220 => 220 frames per chunk.
        divisor = 150 if vram_budget_mb < 5000 else 120
        est = max(200, min(400 if len(ids) > 1200 else 800, int(safe_blocks / divisor))) if safe_blocks > 0 else 400
        return est

    if len(ids) > _CHUNKED_FUSION_THRESHOLD:
        # ---- Chunked fusion: bounded VRAM regardless of scene size (#30) ----
        # Each frame chunk gets its own VBG sized by the chunk-local camera
        # bbox; meshes are extracted per chunk and merged. Peak GPU memory is
        # therefore governed by the chunk, never by the whole capture.
        import gc

        CHUNK = _adaptive_chunk_size()
        # also consider planner if we have loader (optional) — use conservative of two
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
                                                 vram_budget_mb,
                                                 store_color=store_color,
                                                 weight_dtype=weight_dtype), dev,
                            store_color=store_color, weight_dtype=weight_dtype)
            integrate_into(vbg, chunk_ids)
            cmesh, cpcd = extract(vbg)
            # §20 worker lifetime: release VBG before accumulating to keep peak low
            del vbg
            gc.collect()
            total_stats["mesh_vertices"] += len(cmesh.vertices)
            total_stats["mesh_triangles"] += len(cmesh.triangles)
            total_stats["pcd_points"] += len(cpcd.points) if cpcd is not None else 0
            if mesh_acc is None:
                mesh_acc, pcd_acc = cmesh, cpcd
            else:
                # §22 temporal chunk seam mitigation: vertex weld + cleanup after merge
                mesh_acc += cmesh
                if cpcd is not None and pcd_acc is not None:
                    pcd_acc += cpcd
                elif cpcd is not None:
                    pcd_acc = cpcd
            # minimal cleanup per chunk to avoid seam explosion
            if len(mesh_acc.vertices) > 200000:
                mesh_acc.remove_duplicated_vertices()
                mesh_acc.remove_duplicated_triangles()
                mesh_acc.remove_degenerate_triangles()
            print(f"[open3d_vbg] chunk {ci+1}/{n_chunks} done "
                  f"(CHUNK {CHUNK} diag {chunk_diag:.1f}m, tris so far "
                  f"{total_stats['mesh_triangles']})", flush=True)
            # §26 duty-cycle accounting: accumulate across chunks; pause once
            # frames_per_chunk is reached, then reset the counter.  The old
            # code passed len(chunk_ids) (per-chunk count) so the pause fired
            # on every chunk >= threshold and the accumulator was meaningless.
            n_integrated += len(chunk_ids)
            if frames_per_chunk > 0 and n_integrated >= frames_per_chunk \
                    and chunk_pause_s > 0:
                print(f"[open3d_vbg] duty-cycle pause {chunk_pause_s:.0f}s after "
                      f"{n_integrated} frames", flush=True)
                _time.sleep(chunk_pause_s)
                n_integrated = 0

        out = FusionOutput(device=device_str.lower())
        out.mesh_vertices = total_stats["mesh_vertices"]
        out.mesh_triangles = total_stats["mesh_triangles"]
        out.pcd_points = total_stats["pcd_points"]
        out.mesh_obj = mesh_acc
        out.pcd_obj = pcd_acc
        # final weld
        if out.mesh_obj is not None and len(out.mesh_obj.vertices) > 0:
            out.mesh_obj.remove_duplicated_vertices()
            out.mesh_obj.remove_duplicated_triangles()
            out.mesh_obj.remove_degenerate_triangles()
        return out

    # ---- Single-VBG path (small captures) ----
    # P0 #2: use planner capacity when available, fallback to bbox estimate
    est_blocks = estimate_block_count(bbox_diag, voxel_m, vram_budget_mb,
                                      store_color=store_color,
                                      weight_dtype=weight_dtype)
    alloc_blocks = int(planned_block_count) if planned_block_count and planned_block_count > 0 else est_blocks
    # planner capacity includes safety factor; do not exceed VRAM-capped limit
    if vram_budget_mb and alloc_blocks > est_blocks:
        # if planner wants more than VRAM cap, clamp and log (planner prediction already capped by cap logic)
        # but keep planner when within limit — it is more accurate than bbox
        pass
    block_count = max(4096, alloc_blocks)
    vbg = _make_vbg(voxel_m, trunc_mult, block_count, dev,
                    store_color=store_color, weight_dtype=weight_dtype)
    n_integrated = 0
    for start in range(0, len(ids), prefetch):
        batch = ids[start : start + prefetch]
        n_integrated = duty_pause(n_integrated)
        integrate_into(vbg, batch)
        n_integrated += len(batch)

    # §20 sequential extraction to reduce peak: mesh first, then optionally pcd
    mesh, pcd = extract(vbg)
    out = FusionOutput(device=device_str.lower())
    out.mesh_vertices = len(mesh.vertices) if mesh is not None else 0
    out.mesh_triangles = len(mesh.triangles) if mesh is not None else 0
    out.pcd_points = len(pcd.points) if pcd is not None else 0
    out.mesh_obj = mesh
    out.pcd_obj = pcd

    import gc

    del vbg
    gc.collect()
    return out


def _extract(vbg, o3c, dev, device_str, mode="mesh_only", allow_cpu_migration=False):
    """Extract mesh/point cloud; §19 CPU migration is guarded."""
    import open3d as o3d

    def _empty_mesh():
        return o3d.geometry.TriangleMesh()
    def _empty_pcd():
        return o3d.geometry.PointCloud()

    try:
        if mode == "mesh_only":
            mesh = vbg.extract_triangle_mesh().to_legacy()
            return mesh, None
        elif mode == "pcd_only":
            pcd = vbg.extract_point_cloud().to_legacy()
            return _empty_mesh(), pcd
        else:  # mesh_and_pcd — sequential to keep peak low (§20)
            mesh = vbg.extract_triangle_mesh().to_legacy()
            # explicitly free intermediate before second extract? vbg stays but temp freed
            pcd = vbg.extract_point_cloud().to_legacy()
            return mesh, pcd
    except Exception as exc:
        if dev == o3c.Device("CPU:0"):
            raise
        # §19: GPU extraction OOM → do NOT blindly copy entire VBG to host when
        # pressure is high. Only allow for tiny canary (allow_cpu_migration) else
        # fail and let caller replan with smaller tile/coarser voxel.
        if not allow_cpu_migration:
            print(f"[open3d_vbg] GPU extraction OOM (mode={mode}) — no CPU migration "
                  f"(allow_cpu_migration=False); raising to trigger replan: {exc}")
            raise RuntimeError(f"GPU extraction failed without migration: {exc}") from exc
        print(f"[open3d_vbg] GPU extraction failed ({exc}); migrating VBG to CPU (allowed)")
        vbg_cpu = vbg.to(o3c.Device("CPU:0"))
        if mode == "mesh_only":
            mesh = vbg_cpu.extract_triangle_mesh().to_legacy()
            return mesh, None
        elif mode == "pcd_only":
            pcd = vbg_cpu.extract_point_cloud().to_legacy()
            return _empty_mesh(), pcd
        else:
            mesh = vbg_cpu.extract_triangle_mesh().to_legacy()
            pcd = vbg_cpu.extract_point_cloud().to_legacy()
            return mesh, pcd
