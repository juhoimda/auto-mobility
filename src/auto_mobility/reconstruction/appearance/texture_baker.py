"""Minimal texture baker: grid-atlas baking with occlusion-aware view scoring.

Each triangle gets its best view via raycasting visibility (occlusion check)
plus facing-angle weight; the atlas cell is filled with the view's mean RGB.
Bounded memory (#41): triangles are processed in chunks; per triangle only a
running top-K (K = max_views_per_tri) score/color buffer is kept.

Time  O(T * V) scoring with T chunked, V candidate views.
Memory O(atlas_pixels + tri_chunk * K + per-chunk transients).
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
    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)
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
    # #41/#42: bounded top-K buffers instead of full dense S[T,V]/C[T,V,3].
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
            # running top-K merge: candidate enters if it beats the current min
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
            continue  # 아틀라스 포화: 해당 면은 텍스처/정점색 없음(폴백 UV)
        cx, cy = occupied % grid, occupied // grid
        x0, y0 = cx * cell, cy * cell
        atlas[y0:y0 + cell, x0:x0 + cell] = face_colors[tid]
        cell_of_face[tid] = occupied
        occupied += 1

    import cv2

    atlas_path = out_dir / "textures" / f"{name}_atlas_0.png"
    cv2.imwrite(str(atlas_path), atlas[..., ::-1])

    # 정점 색상: 아틀라스 셀 예산과 무관하게 모든 뷰의 유효 샘플을 평균 블렌딩
    # (텍스처 미지원 뷰어(Open3D 등)에서 전체 컬러 표현용)
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
    vcol = np.full((len(vertices), 3), 0.5)
    vcol[has_vcol] = vw_sum[has_vcol] / vw_cnt[has_vcol][:, None] / 255.0

    obj_lines = [f"mtllib {name}.mtl", f"o {name}"]
    for (vx, vy, vz), (cr, cg, cb) in zip(vertices, vcol):
        obj_lines.append(
            f"v {vx:.6f} {vy:.6f} {vz:.6f} {cr:.6f} {cg:.6f} {cb:.6f}")

    # 더미 UV 3개(포화/무색 면 폴백) + 셀별 코너 UV 3개씩 (면들이 공유)
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
            obj_lines.append(f"f {a}/1 {b}/2 {c}/3")
        else:
            i0, i1, i2 = ids
            obj_lines.append(f"f {a}/{i0} {b}/{i1} {c}/{i2}")

    obj_path = out_dir / f"{name}.obj"
    obj_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")
    (out_dir / f"{name}.mtl").write_text(
        f"newmtl baked\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\n"
        f"map_Kd textures/{name}_atlas_0.png\n",
        encoding="utf-8",
    )
    return BakeResult(obj_path, out_dir / f"{name}.mtl", (atlas_path,), untextured)
