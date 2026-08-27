"""Minimal texture baker: grid-atlas baking with occlusion-aware view scoring.

Each triangle gets its best view via raycasting visibility (occlusion check)
plus facing-angle weight; the atlas cell is filled with the view's mean RGB.
Memory (#27 truthful): top_scores[T,K] + top_colors[T,K,3] = O(T*K) (dense T*V
removed). Time still O(T*V) scoring (V views) with T chunked for cache.

Time  O(T * V) scoring with T chunked, V candidate views.
Memory O(T*K + atlas_pixels) where T=triangles, K=max_views_per_tri.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BakeTimings:
    """Substage wall times in seconds + counters for telemetry."""
    texture_total_s: float = 0.0
    raycast_scoring_s: float = 0.0
    color_projection_s: float = 0.0
    face_color_reduce_s: float = 0.0
    vertex_color_reduce_s: float = 0.0
    normal_generation_s: float = 0.0
    atlas_write_s: float = 0.0
    obj_build_s: float = 0.0
    obj_write_s: float = 0.0
    ply_write_s: float = 0.0
    mesh_simplification_s: float = 0.0
    mesh_vertices: int = 0
    mesh_triangles: int = 0
    texture_mesh_triangles: int = 0
    texture_views_requested: int = 0
    texture_views_actual: int = 0
    candidate_triangle_view_pairs: int = 0
    ray_count_actual: int = 0

    def to_dict(self) -> dict:
        return {
            "texture_total_s": round(self.texture_total_s, 3),
            "raycast_scoring_s": round(self.raycast_scoring_s, 3),
            "color_projection_s": round(self.color_projection_s, 3),
            "face_color_reduce_s": round(self.face_color_reduce_s, 3),
            "vertex_color_reduce_s": round(self.vertex_color_reduce_s, 3),
            "normal_generation_s": round(self.normal_generation_s, 3),
            "atlas_write_s": round(self.atlas_write_s, 3),
            "obj_build_s": round(self.obj_build_s, 3),
            "obj_write_s": round(self.obj_write_s, 3),
            "ply_write_s": round(self.ply_write_s, 3),
            "mesh_simplification_s": round(self.mesh_simplification_s, 3),
            "mesh_vertices": self.mesh_vertices,
            "mesh_triangles": self.mesh_triangles,
            "texture_mesh_triangles": self.texture_mesh_triangles,
            "texture_views_requested": self.texture_views_requested,
            "texture_views_actual": self.texture_views_actual,
            "candidate_triangle_view_pairs": self.candidate_triangle_view_pairs,
            "ray_count_actual": self.ray_count_actual,
        }


@dataclass(frozen=True)
class BakeResult:
    obj_path: Path
    mtl_path: Path
    atlas_paths: tuple
    untextured_faces: int
    textured_faces: int = 0
    appearance_mode: str = "vertex_color"
    timings: Optional[BakeTimings] = None

    def to_dict(self) -> dict:
        d = {
            "obj": str(self.obj_path),
            "atlas_count": len(self.atlas_paths),
            "untextured_faces": self.untextured_faces,
            "textured_faces": self.textured_faces,
            "appearance_mode": self.appearance_mode,
        }
        if self.timings is not None:
            d["timings"] = self.timings.to_dict()
        return d


def _view_score(normal: np.ndarray, tri_center: np.ndarray, cam_pos: np.ndarray,
                visible: bool) -> float:
    if not visible:
        return -1.0
    d = cam_pos - tri_center
    dist = max(np.linalg.norm(d), 1e-6)
    facing = float(np.dot(normal, d / dist))
    if facing <= 0.1:
        return -1.0
    return facing / np.log2(2.0 + dist)


def sample_view_color(view_bgr: np.ndarray, K: np.ndarray, T_wc: np.ndarray,
                      points_cam_world: np.ndarray, w: int, h: int):
    """정점들을 뷰에 투영해 RGB 샘플. 반환: (colors uint8, valid bool mask)."""
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    pc = (points_cam_world - t) @ R
    z = pc[:, 2]
    valid = z > 1e-6
    u = np.full(len(points_cam_world), -1.0)
    v = np.full(len(points_cam_world), -1.0)
    u[valid] = K[0, 0] * pc[valid, 0] / z[valid] + K[0, 2]
    v[valid] = K[1, 1] * pc[valid, 1] / z[valid] + K[1, 2]
    ui = np.clip(u.astype(int), 0, w - 1)
    vi = np.clip(v.astype(int), 0, h - 1)
    ok = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.zeros((len(points_cam_world), 3), dtype=np.uint8)
    out[ok] = view_bgr[vi[ok], ui[ok]][..., ::-1]
    return out, ok


# ---------------------------------------------------------------------------
# Adaptive mesh simplification for texture stage
# ---------------------------------------------------------------------------

def simplify_mesh_for_texture(
    vertices: np.ndarray,
    triangles: np.ndarray,
    target_triangles: int = 750_000,
    max_triangles: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Simplify mesh via Open3D quadric decimation if needed.

    Returns (new_vertices, new_triangles, info_dict).
    info_dict contains timing and triangle counts.
    Never increases triangle count; returns original if below target.
    """
    t0 = time.time()
    n_tri = len(triangles)
    if n_tri <= target_triangles:
        return vertices, triangles, {
            "simplified": False,
            "original_triangles": n_tri,
            "texture_triangles": n_tri,
            "simplification_s": 0.0,
        }
    # Clamp target to max
    target = min(target_triangles, max_triangles)
    target = max(1, min(target, n_tri - 1))
    try:
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
        # Quadric decimation is topology-aware
        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        # Clean
        simplified.remove_duplicated_vertices()
        simplified.remove_duplicated_triangles()
        simplified.remove_degenerate_triangles()
        new_verts = np.asarray(simplified.vertices)
        new_tris = np.asarray(simplified.triangles)
        # If simplification produced too few or degenerate result, fallback to original
        if len(new_tris) == 0 or len(new_tris) > n_tri:
            return vertices, triangles, {
                "simplified": False,
                "original_triangles": n_tri,
                "texture_triangles": n_tri,
                "simplification_s": round(time.time() - t0, 3),
            }
        return new_verts, new_tris, {
            "simplified": True,
            "original_triangles": n_tri,
            "texture_triangles": len(new_tris),
            "simplification_s": round(time.time() - t0, 3),
        }
    except Exception as exc:
        return vertices, triangles, {
            "simplified": False,
            "original_triangles": n_tri,
            "texture_triangles": n_tri,
            "simplification_s": round(time.time() - t0, 3),
            "error": str(exc),
        }


def estimate_texture_time_s(
    triangle_count: int,
    view_count: int,
    calibrated_cost_per_million_tv: float = 8.0,
    serialization_overhead_s: float = 3.0,
) -> float:
    """Runtime ETA estimator for texture stage.

    calibrated_cost_per_million_triangle_view: seconds per 1e6 triangle-view pairs.
    Default 8.0s/M calibrated from large-mesh observation (13M * 80 ~ 1B pairs).
    Override via telemetry history if available.
    """
    pairs_million = (triangle_count * view_count) / 1e6
    return pairs_million * calibrated_cost_per_million_tv + serialization_overhead_s


def _cheap_candidate_mask(
    centers: np.ndarray,
    normals: np.ndarray,
    valid_face: np.ndarray,
    cam_pos: np.ndarray,
    T_wc: np.ndarray,
    K: np.ndarray,
    w: int,
    h: int,
    tris: np.ndarray,
    distance_min: float = 0.1,
    distance_max: float = 10.0,
) -> np.ndarray:
    """Vectorized cheap geometry filter before raycast.

    Returns bool mask of triangles that may be visible from this view.
    Checks: valid face, camera front (z>0), facing>0.1, frustum, distance.
    """
    n = len(centers)
    if n == 0:
        return np.zeros(0, dtype=bool)
    # Valid face
    mask = valid_face.copy()
    # Distance
    d = cam_pos[None, :] - centers  # (n,3) vector from triangle to camera
    dist = np.linalg.norm(d, axis=1)
    # Facing
    # facing = dot(normal, d/dist)
    facing = np.zeros(n, dtype=np.float64)
    # avoid div by zero: only compute where dist > 0
    nz = dist > 1e-6
    if np.any(nz):
        facing[nz] = (normals[nz] * d[nz]).sum(axis=1) / dist[nz]
    mask &= (facing > 0.1)
    mask &= (dist >= distance_min) & (dist <= distance_max)
    # Camera front: z > 0 in camera frame
    # Transform centers to camera frame: pc = (pw - t) @ R
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    pc = (centers - t) @ R
    z = pc[:, 2]
    mask &= (z > 1e-6)
    # Frustum: project center or triangle vertices; at least one vertex or center in image
    # For cheap check, first test center projection
    # Only for currently still-valid
    # Use boolean short-circuit: compute projection for all, then mask
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # center projection
    u_c = np.full(n, -1.0)
    v_c = np.full(n, -1.0)
    valid_z = z > 1e-6
    u_c[valid_z] = fx * pc[valid_z, 0] / z[valid_z] + cx
    v_c[valid_z] = fy * pc[valid_z, 1] / z[valid_z] + cy
    center_in = (u_c >= 0) & (u_c < w) & (v_c >= 0) & (v_c < h)
    # For those not center_in, check if any vertex projects inside
    need_vertex_check = mask & (~center_in)
    if np.any(need_vertex_check):
        # project all vertices of candidate triangles
        pts = tris.reshape(-1, 3)  # (n*3, 3)
        pc_v = (pts - t) @ R
        z_v = pc_v[:, 2]
        valid_v = z_v > 1e-6
        u_v = np.full(len(pts), -1.0)
        v_v = np.full(len(pts), -1.0)
        u_v[valid_v] = fx * pc_v[valid_v, 0] / z_v[valid_v] + cx
        v_v[valid_v] = fy * pc_v[valid_v, 1] / z_v[valid_v] + cy
        ok_v = valid_v & (u_v >= 0) & (u_v < w) & (v_v >= 0) & (v_v < h)
        ok_v = ok_v.reshape(n, 3)
        vertex_in = ok_v.any(axis=1)
        center_in = center_in | vertex_in
    mask &= center_in
    return mask


def bake_atlas(
    vertices: np.ndarray,
    triangles: np.ndarray,
    views: list,
    K: np.ndarray,
    poses_wc: dict,
    scene=None,
    atlas_size: int = 1024,
    grid: int = 8,
    max_views_per_tri: int = 5,
    tri_chunk: int = 65536,
    out_dir: Path = None,
    name: str = "model",
    enable_simplification: bool = True,
    simplification_target: int = 750_000,
    simplification_max: int = 1_000_000,
) -> BakeResult:
    """views: list of (frame_id, bgr_image). scene: o3d RaycastingScene for occlusion."""
    t_total0 = time.time()
    timings = {}
    out_dir = Path(out_dir)
    (out_dir / "textures").mkdir(parents=True, exist_ok=True)

    # --- Adaptive mesh simplification for texture (P0-3) ---
    t_simp0 = time.time()
    original_triangles = len(triangles)
    tex_vertices = vertices
    tex_triangles = triangles
    simp_info = {"simplified": False, "original_triangles": original_triangles,
                 "texture_triangles": original_triangles, "simplification_s": 0.0}
    if enable_simplification and original_triangles > simplification_target:
        tex_vertices, tex_triangles, simp_info = simplify_mesh_for_texture(
            vertices, triangles,
            target_triangles=simplification_target,
            max_triangles=simplification_max,
        )
    timings["mesh_simplification_s"] = time.time() - t_simp0

    # Use tex_* for texture stage geometry but keep original for final counts
    _T = len(tex_triangles)
    if _T > 50000 and grid <= 8:
        if _T > 500000:
            grid = 64
            atlas_size = 4096
        elif _T > 100000:
            grid = 32
            atlas_size = 2048
        else:
            grid = 16
            atlas_size = 2048
    cell = atlas_size // grid
    n_cells = grid * grid
    atlas = np.full((atlas_size, atlas_size, 3), 180, dtype=np.uint8)
    occupied = 0
    untextured = 0

    import open3d as _o3d

    tris = tex_vertices[tex_triangles]
    centers = tris.mean(axis=1)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    nrm = np.linalg.norm(normals, axis=1)
    valid_face = nrm > 1e-12
    normals[valid_face] /= nrm[valid_face, None]
    untextured += int((~valid_face).sum())

    view_ids = [fid for fid, _ in views if fid in poses_wc]
    img_by_fid = dict(views)
    T_count = len(tex_triangles)
    K_top = max(1, int(max_views_per_tri))
    tri_chunk = max(1, int(tri_chunk))
    top_scores = np.full((T_count, K_top), -1.0, dtype=np.float32)
    top_colors = np.zeros((T_count, K_top, 3), dtype=np.float32)

    # Telemetry counters
    candidate_pairs = 0
    ray_count = 0
    t_raycast = 0.0
    t_project = 0.0

    for s0 in range(0, T_count, tri_chunk):
        s1 = min(T_count, s0 + tri_chunk)
        c_tris = tris[s0:s1]
        c_centers = centers[s0:s1]
        c_normals = normals[s0:s1]
        c_valid = valid_face[s0:s1]
        n_c = s1 - s0
        for j, fid in enumerate(view_ids):
            T_wc = poses_wc[fid]
            cam_pos = T_wc[:3, 3]
            img = img_by_fid[fid]
            h, w = img.shape[0], img.shape[1]

            # --- P0-2: cheap vectorized pre-filter before raycast ---
            c_mask = _cheap_candidate_mask(
                c_centers, c_normals, c_valid, cam_pos, T_wc, K, w, h, c_tris
            )
            n_candidates = int(c_mask.sum())
            candidate_pairs += n_candidates
            if n_candidates == 0:
                continue

            # Prepare rays only for candidates
            d_all = cam_pos[None, :] - c_centers  # (n_c,3)
            dist_all = np.linalg.norm(d_all, axis=1)
            facing_all = np.zeros(n_c, dtype=np.float64)
            nz = dist_all > 1e-6
            facing_all[nz] = (c_normals[nz] * d_all[nz]).sum(axis=1) / dist_all[nz]

            # Visible via raycast only for candidates
            t_r0 = time.time()
            visible = np.zeros(n_c, dtype=bool)
            # For non-candidates, visible stays False and score will be -1 anyway
            if scene is not None and n_candidates > 0:
                cand_idx = np.nonzero(c_mask)[0]
                cand_centers = c_centers[cand_idx]
                cand_normals = c_normals[cand_idx]
                cand_d = d_all[cand_idx]
                cand_dist = dist_all[cand_idx]
                origins = cand_centers + cand_normals * 1e-3
                # normalize dirs
                dirs = cand_d / np.maximum(cand_dist, 1e-6)[:, None]
                rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
                ans = scene.cast_rays(_o3d.core.Tensor(rays))
                t_hit = ans["t_hit"].numpy()
                ray_count += len(rays)
                cand_visible = ~np.isfinite(t_hit) | (t_hit > cand_dist - 0.02)
                visible[cand_idx] = cand_visible
                # for non-candidates, visible remains False
            else:
                # No scene => visible = candidate mask (all candidates visible)
                visible[c_mask] = True
            t_raycast += time.time() - t_r0

            # Compute scores only for candidates (others already -1)
            scores = np.full(n_c, -1.0, dtype=np.float32)
            # Only candidates can have positive scores
            cand_idx2 = np.nonzero(c_mask)[0]
            if len(cand_idx2):
                facing_c = facing_all[cand_idx2]
                dist_c = dist_all[cand_idx2]
                vis_c = visible[cand_idx2]
                valid_c = c_valid[cand_idx2]
                good = vis_c & (facing_c > 0.1) & valid_c
                if np.any(good):
                    good_idx = cand_idx2[good]
                    # compute score for good
                    sc = (facing_all[good_idx] / np.log2(2.0 + dist_all[good_idx])).astype(np.float32)
                    scores[good_idx] = sc

            # Color projection: still need to sample view color, but only for candidates
            # However sample_view_color is vectorized over all tris vertices; we can still
            # limit to candidates to save time. Compute for all but only keep candidates.
            t_p0 = time.time()
            # Only sample for candidates to reduce O(T*V) projection cost
            # Build subset of tris for candidates
            # For efficiency, sample only candidate tris
            cols_mean = np.zeros((n_c, 3), dtype=np.float32)
            ok_tri_any = np.zeros(n_c, dtype=bool)
            if n_candidates > 0:
                cand_idx_all = np.nonzero(c_mask)[0]
                # Gather candidate tris (n_candidates*3 points)
                cand_tris_pts = c_tris[cand_idx_all].reshape(-1, 3)
                cols, ok_mask = sample_view_color(
                    img, K, T_wc, cand_tris_pts, w, h)
                cols_3 = cols.reshape(n_candidates, 3, 3).astype(np.float32)
                ok_3 = ok_mask.reshape(n_candidates, 3)
                ok_any_c = ok_3.any(axis=1)
                cnt_valid = np.maximum(ok_3.sum(axis=1, keepdims=True), 1)
                cols_mean_c = (cols_3 * ok_3[..., None]).sum(axis=1) / cnt_valid
                # scatter back
                cols_mean[cand_idx_all] = cols_mean_c
                ok_tri_any[cand_idx_all] = ok_any_c
                # Invalidate scores where projection failed
                # Only candidates with ok_any keep score; others -> -1
                fail_proj = ~ok_tri_any
                scores[fail_proj] = -1.0
            t_project += time.time() - t_p0

            # Top-K update: only where scores > current min
            # Use vectorized argmin
            min_idx = np.argmin(top_scores[s0:s1], axis=1)
            min_val = top_scores[np.arange(s0, s1), min_idx]
            better = scores > min_val
            rows_b = np.nonzero(better)[0]
            if len(rows_b):
                gi = rows_b + s0
                top_colors[gi, min_idx[rows_b]] = cols_mean[rows_b]
                top_scores[gi, min_idx[rows_b]] = scores[rows_b]

    # --- Face color reduction: vectorized instead of Python loop over T ---
    t_face0 = time.time()
    face_colors = {}
    # Vectorized: find best rows
    # top_scores shape (T, K), compute which rows have any >0
    has_any = (top_scores > 0).any(axis=1)
    valid_with_texture = has_any & valid_face  # valid_face here is tex validity
    # For each valid row, compute weighted average
    # w = row[best], best = where row>0
    # Use loop only over valid rows but with numpy ops inside; still need per-tid dict
    # Optimize: iterate only over valid_with_texture indices
    valid_indices = np.nonzero(valid_with_texture)[0]
    # Vectorized weighted sum: for each row, weighted avg = sum(top_colors[row,best]*w)/sum(w)
    for tid in valid_indices:
        row = top_scores[tid]
        mask = row > 0
        w = row[mask]
        # top_colors[tid, mask] shape (n_best, 3)
        weighted = (top_colors[tid, mask] * w[:, None]).sum(axis=0) / w.sum()
        face_colors[int(tid)] = weighted.astype(np.uint8)
    # Untextured: valid faces with no texture
    n_valid = int(valid_face.sum())
    n_textured = len(face_colors)
    # untextured already counted invalid faces; add valid but untextured
    untextured += int(n_valid - n_textured)
    t_face = time.time() - t_face0

    # --- Vertex colours: vectorized via np.add.at ---
    t_vcol0 = time.time()
    pts = np.asarray(tex_vertices, dtype=np.float64)
    vw_sum = np.zeros((len(tex_vertices), 3), dtype=np.float64)
    vw_cnt = np.zeros(len(tex_vertices), dtype=np.float64)
    if face_colors:
        # Build arrays for vectorized accumulation
        fids = np.array(list(face_colors.keys()), dtype=np.int64)
        colors = np.array([face_colors[fid] for fid in fids], dtype=np.float64)  # (F,3)
        tris_for_faces = tex_triangles[fids]  # (F,3)
        # For each face, its 3 vertices get same color contribution
        # Use np.add.at for accumulation
        # Need to repeat colors for each vertex
        # tris_for_faces is (F,3), colors is (F,3)
        # vw_sum[tris_for_faces[:,0]] += colors etc via add.at
        np.add.at(vw_sum, tris_for_faces[:, 0], colors)
        np.add.at(vw_sum, tris_for_faces[:, 1], colors)
        np.add.at(vw_sum, tris_for_faces[:, 2], colors)
        np.add.at(vw_cnt, tris_for_faces[:, 0], 1.0)
        np.add.at(vw_cnt, tris_for_faces[:, 1], 1.0)
        np.add.at(vw_cnt, tris_for_faces[:, 2], 1.0)
    has_vcol = vw_cnt > 0
    mean_vcol = (vw_sum[has_vcol] / vw_cnt[has_vcol][:, None] / 255.0).mean(axis=0) if has_vcol.any() else np.array([0.7, 0.7, 0.7])
    vcol = np.full((len(tex_vertices), 3), mean_vcol)
    vcol[has_vcol] = vw_sum[has_vcol] / vw_cnt[has_vcol][:, None] / 255.0
    t_vcol = time.time() - t_vcol0

    # Initialize atlas with mean surface color
    mean_rgb = (vcol[has_vcol].mean(axis=0) * 255.0) if has_vcol.any() else np.array([180.0, 180.0, 180.0])
    atlas = np.full((atlas_size, atlas_size, 3), mean_rgb.astype(np.uint8), dtype=np.uint8)

    # Keep a bounded diagnostic atlas for tooling, but never bind it to the
    # material unless it covers every colourable face.
    occupied = 0
    order = sorted(face_colors)
    cell_of_face = {}
    for tid in order:
        if occupied >= n_cells:
            continue
        cx, cy = occupied % grid, occupied // grid
        x0, y0 = cx * cell, cy * cell
        atlas[y0:y0 + cell, x0:x0 + cell] = face_colors[tid]
        cell_of_face[tid] = occupied
        occupied += 1

    import cv2

    t_atlas0 = time.time()
    atlas_path = out_dir / "textures" / f"{name}_atlas_0.png"
    cv2.imwrite(str(atlas_path), atlas[..., ::-1])
    t_atlas = time.time() - t_atlas0

    # Vertex normals for smooth shading: vectorized
    t_norm0 = time.time()
    v_normals = np.zeros_like(pts)
    # Accumulate normals per vertex via add.at
    if np.any(valid_face):
        valid_tri_idx = np.nonzero(valid_face)[0]
        valid_normals = normals[valid_tri_idx]  # (N_valid,3)
        valid_tris = tex_triangles[valid_tri_idx]  # (N_valid,3)
        np.add.at(v_normals, valid_tris[:, 0], valid_normals)
        np.add.at(v_normals, valid_tris[:, 1], valid_normals)
        np.add.at(v_normals, valid_tris[:, 2], valid_normals)
    v_nrm = np.linalg.norm(v_normals, axis=1)
    v_valid = v_nrm > 1e-12
    v_normals[v_valid] /= v_nrm[v_valid, None]
    t_norm = time.time() - t_norm0

    # --- OBJ generation: streaming/chunked write to avoid holding millions of strings ---
    t_obj_build0 = time.time()
    obj_path = out_dir / f"{name}.obj"
    mtl_path = out_dir / f"{name}.mtl"
    # Prepare VT lines
    # UV lines
    vt_lines = ["vt 0.000000 0.000000", "vt 1.000000 0.000000",
                "vt 0.000000 1.000000"]
    cell_vt = {}
    for cid in range(occupied):
        cx, cy = cid % grid, cid // grid
        u0, v0 = cx / grid, 1.0 - (cy + 1) / grid
        du = dv = 1.0 / grid
        vt_lines.append(f"vt {u0:.6f} {v0:.6f}")
        vt_lines.append(f"vt {u0 + du:.6f} {v0:.6f}")
        vt_lines.append(f"vt {u0 + du:.6f} {v0 + dv:.6f}")
        base = 3 + 3 * cid
        cell_vt[cid] = (base + 1, base + 2, base + 3)

    # Texture contract: PASS requires usemtl, map_Kd, vt, and coverage>0.
    fully_atlased = len(cell_of_face) == len(face_colors) and not untextured
    large_mesh_partial = (len(tex_triangles) > 50000 and len(cell_of_face) > 0)
    use_texture_material = fully_atlased or large_mesh_partial

    # Streaming write: write header, vertices, normals, vt, faces in chunks
    t_obj_write0 = time.time()
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {name}.mtl\n")
        f.write(f"o {name}\n")
        # Vertices with colors: chunked
        chunk_sz = 65536
        for i0 in range(0, len(tex_vertices), chunk_sz):
            i1 = min(len(tex_vertices), i0 + chunk_sz)
            lines = []
            for idx in range(i0, i1):
                vx, vy, vz = tex_vertices[idx]
                cr, cg, cb = vcol[idx]
                lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f} {cr:.6f} {cg:.6f} {cb:.6f}\n")
            f.writelines(lines)
        # Normals chunked
        for i0 in range(0, len(v_normals), chunk_sz):
            i1 = min(len(v_normals), i0 + chunk_sz)
            lines = [f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n" for nx, ny, nz in v_normals[i0:i1]]
            f.writelines(lines)
        # VT lines chunked
        for i0 in range(0, len(vt_lines), chunk_sz):
            i1 = min(len(vt_lines), i0 + chunk_sz)
            f.writelines(l + "\n" for l in vt_lines[i0:i1])
        if use_texture_material:
            f.write("usemtl baked\n")
        # Faces chunked
        for i0 in range(0, len(tex_triangles), chunk_sz):
            i1 = min(len(tex_triangles), i0 + chunk_sz)
            flines = []
            for tid in range(i0, i1):
                a, b, c = (tex_triangles[tid] + 1).tolist()
                ids = cell_vt.get(cell_of_face.get(tid))
                if ids is None or not use_texture_material:
                    flines.append(f"f {a}//{a} {b}//{b} {c}//{c}\n")
                else:
                    i0v, i1v, i2v = ids
                    flines.append(f"f {a}/{i0v}/{a} {b}/{i1v}/{b} {c}/{i2v}/{c}\n")
            f.writelines(flines)
    t_obj_write = time.time() - t_obj_write0
    t_obj_build = (time.time() - t_obj_build0) - t_obj_write  # build vs write split approx

    mtl = (
        "newmtl baked\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.000 0.000 0.000\n"
        "d 1.0\n"
        "illum 1\n"
    )
    if use_texture_material:
        mtl += f"map_Kd textures/{name}_atlas_0.png\n"
    mtl_path.write_text(mtl, encoding="utf-8")

    # Companion PLY with native vertex colors
    t_ply0 = time.time()
    try:
        mesh_o3d = _o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = _o3d.utility.Vector3dVector(tex_vertices)
        mesh_o3d.triangles = _o3d.utility.Vector3iVector(tex_triangles)
        mesh_o3d.vertex_colors = _o3d.utility.Vector3dVector(vcol)
        if len(v_valid) and v_valid.any():
            mesh_o3d.vertex_normals = _o3d.utility.Vector3dVector(v_normals)
        _o3d.io.write_triangle_mesh(str(out_dir / f"{name}.ply"), mesh_o3d, write_ascii=False)
    except Exception:
        pass
    t_ply = time.time() - t_ply0

    timings_dict = {
        "texture_total_s": time.time() - t_total0,
        "raycast_scoring_s": t_raycast,
        "color_projection_s": t_project,
        "face_color_reduce_s": t_face,
        "vertex_color_reduce_s": t_vcol,
        "normal_generation_s": t_norm,
        "atlas_write_s": t_atlas,
        "obj_build_s": t_obj_build,
        "obj_write_s": t_obj_write,
        "ply_write_s": t_ply,
        "mesh_simplification_s": timings.get("mesh_simplification_s", 0.0),
        "mesh_vertices": int(len(vertices)),
        "mesh_triangles": int(len(triangles)),
        "texture_mesh_triangles": int(len(tex_triangles)),
        "texture_views_requested": len(views),
        "texture_views_actual": len(view_ids),
        "candidate_triangle_view_pairs": int(candidate_pairs),
        "ray_count_actual": int(ray_count),
    }
    bake_timings = BakeTimings(**timings_dict)

    # appearance_mode is texture_atlas iff we actually emitted map_Kd + UV
    return BakeResult(obj_path, mtl_path, (atlas_path,), untextured,
                      textured_faces=len(cell_of_face),
                      appearance_mode="texture_atlas" if use_texture_material else "vertex_color",
                      timings=bake_timings)
