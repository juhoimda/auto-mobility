"""Minimal texture baker: grid-atlas baking with occlusion-aware view scoring.

Each triangle gets its best view via raycasting visibility (occlusion check)
plus facing-angle weight; the atlas cell is filled with the view's mean RGB.
Memory (#27 truthful): top_scores[T,K] + top_colors[T,K,3] = O(T*K) (dense T*V
removed). Time still O(T*V) scoring (V views) with T chunked for cache.

Time  O(T * V) scoring with T chunked, V candidate views.
Memory O(T*K + atlas_pixels) where T=triangles, K=max_views_per_tri.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BakeResult:
    obj_path: Path
    mtl_path: Path
    atlas_paths: tuple
    untextured_faces: int

    def to_dict(self) -> dict:
        return {
            "obj": str(self.obj_path),
            "atlas_count": len(self.atlas_paths),
            "untextured_faces": self.untextured_faces,
        }


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
) -> BakeResult:
    """views: list of (frame_id, bgr_image). scene: o3d RaycastingScene for occlusion."""
    out_dir = Path(out_dir)
    (out_dir / "textures").mkdir(parents=True, exist_ok=True)
    cell = atlas_size // grid
    n_cells = grid * grid
    # Initialize atlas with neutral warm grey instead of solid black (0,0,0)
    # so unmapped fallback faces render as natural surface instead of pitch black
    atlas = np.full((atlas_size, atlas_size, 3), 180, dtype=np.uint8)
    occupied = 0
    untextured = 0

    import open3d as _o3d

    tris = vertices[triangles]
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
    T_count = len(triangles)
    K_top = max(1, int(max_views_per_tri))
    tri_chunk = max(1, int(tri_chunk))
    top_scores = np.full((T_count, K_top), -1.0, dtype=np.float32)
    top_colors = np.zeros((T_count, K_top, 3), dtype=np.float32)

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
            d = cam_pos[None, :] - c_centers
            dist = np.linalg.norm(d, axis=1)
            facing = (c_normals * d).sum(axis=1) / np.maximum(dist, 1e-6)
            visible = np.ones(n_c, dtype=bool)
            if scene is not None:
                origins = c_centers + c_normals * 1e-3
                dirs = d / np.maximum(dist, 1e-6)[:, None]
                rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
                ans = scene.cast_rays(_o3d.core.Tensor(rays))
                t_hit = ans["t_hit"].numpy()
                visible = ~np.isfinite(t_hit) | (t_hit > dist - 0.02)
            scores = np.where(
                visible & (facing > 0.1) & c_valid,
                (facing / np.log2(2.0 + dist)).astype(np.float32),
                np.float32(-1.0),
            )
            img = img_by_fid[fid]
            cols, _ = sample_view_color(
                img, K, T_wc, c_tris.reshape(-1, 3), img.shape[1], img.shape[0])
            cols = cols.reshape(n_c, 3, 3).mean(axis=1).astype(np.float32)
            min_idx = np.argmin(top_scores[s0:s1], axis=1)
            min_val = top_scores[np.arange(s0, s1), min_idx]
            better = scores > min_val
            rows_b = np.nonzero(better)[0]
            if len(rows_b):
                gi = rows_b + s0
                top_colors[gi, min_idx[rows_b]] = cols[rows_b]
                top_scores[gi, min_idx[rows_b]] = scores[rows_b]

    face_colors = {}
    for tid in range(T_count):
        row = top_scores[tid]
        best = [b for b in range(len(row)) if row[b] > 0]
        if not best:
            if valid_face[tid]:
                untextured += 1
            continue
        w = row[best]
        face_colors[tid] = ((top_colors[tid, best] * w[:, None]).sum(axis=0)
                            / w.sum()).astype(np.uint8)

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

    # Vertex colors blending (all views)
    pts = np.asarray(vertices, dtype=np.float64)
    vw_sum = np.zeros((len(vertices), 3), dtype=np.float64)
    vw_cnt = np.zeros(len(vertices), dtype=np.float64)
    for fid in view_ids:
        img = img_by_fid[fid]
        cols, ok = sample_view_color(
            img, K, poses_wc[fid], pts, img.shape[1], img.shape[0])
        vw_sum[ok] += cols[ok]
        vw_cnt[ok] += 1.0
    has_vcol = vw_cnt > 0
    vcol = np.full((len(vertices), 3), 0.7)
    vcol[has_vcol] = vw_sum[has_vcol] / vw_cnt[has_vcol][:, None] / 255.0

    # Initialize atlas with mean surface color so fallback UVs render naturally, not pitch black
    mean_rgb = (vcol[has_vcol].mean(axis=0) * 255.0) if has_vcol.any() else np.array([180.0, 180.0, 180.0])
    atlas = np.full((atlas_size, atlas_size, 3), mean_rgb.astype(np.uint8), dtype=np.uint8)

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

    atlas_path = out_dir / "textures" / f"{name}_atlas_0.png"
    cv2.imwrite(str(atlas_path), atlas[..., ::-1])

    # Vertex normals for smooth MeshLab / Blender shading
    v_normals = np.zeros_like(pts)
    for tid in range(T_count):
        if valid_face[tid]:
            v_normals[triangles[tid]] += normals[tid]
    v_nrm = np.linalg.norm(v_normals, axis=1)
    v_valid = v_nrm > 1e-12
    v_normals[v_valid] /= v_nrm[v_valid, None]

    obj_lines = [f"mtllib {name}.mtl", f"o {name}"]
    for (vx, vy, vz), (cr, cg, cb) in zip(vertices, vcol):
        obj_lines.append(
            f"v {vx:.6f} {vy:.6f} {vz:.6f} {cr:.6f} {cg:.6f} {cb:.6f}")

    for nx, ny, nz in v_normals:
        obj_lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")

    # UV lines
    vt_lines = ["vt 0.000000 0.000000", "vt 1.000000 0.000000",
                "vt 0.000000 1.000000"]
    cell_vt = {}
    du = dv = 1.0 / grid
    for cid in range(occupied):
        cx, cy = cid % grid, cid // grid
        u0, v0 = cx / grid, 1.0 - (cy + 1) / grid
        vt_lines.append(f"vt {u0:.6f} {v0:.6f}")
        vt_lines.append(f"vt {u0 + du:.6f} {v0:.6f}")
        vt_lines.append(f"vt {u0 + du:.6f} {v0 + dv:.6f}")
        base = 3 + 3 * cid
        cell_vt[cid] = (base + 1, base + 2, base + 3)

    obj_lines.extend(vt_lines)
    obj_lines.append("usemtl baked")
    for tid in range(T_count):
        a, b, c = (triangles[tid] + 1).tolist()
        ids = cell_vt.get(cell_of_face.get(tid))
        if ids is None:
            obj_lines.append(f"f {a}/1/{a} {b}/2/{b} {c}/3/{c}")
        else:
            i0, i1, i2 = ids
            obj_lines.append(f"f {a}/{i0}/{a} {b}/{i1}/{b} {c}/{i2}/{c}")

    obj_path = out_dir / f"{name}.obj"
    obj_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")
    (out_dir / f"{name}.mtl").write_text(
        "newmtl baked\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.000 0.000 0.000\n"
        "d 1.0\n"
        "illum 1\n"
        f"map_Kd textures/{name}_atlas_0.png\n",
        encoding="utf-8",
    )
    return BakeResult(obj_path, out_dir / f"{name}.mtl", (atlas_path,), untextured)



