"""Texture baker invariants: OBJ/MTL/atlas produced, occlusion-aware coloring."""

import numpy as np

from auto_mobility.reconstruction.appearance import bake_atlas, sample_view_color


def _cube_mesh(size=1.0):
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float64)
    triangles = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ])
    return vertices, triangles


def test_sample_view_color_projects_into_image():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    T_wc = np.diag([1.0, 1.0, -1.0, 1.0])
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    pts = np.array([[0, 0, -3.0]])
    cols, ok = sample_view_color(img, K, T_wc, pts, 640, 480)
    assert ok[0]
    assert cols[0].tolist() == [200, 200, 200]


def test_bake_produces_obj_mtl_atlas(tmp_path):
    import cv2

    vertices, triangles = _cube_mesh()
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])

    view_img = np.zeros((480, 640, 3), dtype=np.uint8)
    view_img[:, :, 0] = 255
    views = [(0, view_img)]
    poses = {0: np.eye(4) @ np.diag([1, 1, -1, 1])}
    poses[0][:3, 3] = [0, 0, -3.0]

    result = bake_atlas(vertices, triangles, views, K, poses,
                        scene=None, out_dir=tmp_path, name="model")

    assert result.obj_path.is_file()
    assert result.mtl_path.is_file()
    assert result.atlas_paths[0].is_file()
    assert result.to_dict()["atlas_count"] == 1

    obj_text = result.obj_path.read_text()
    assert obj_text.count("v ") >= 8
    assert "mtllib model.mtl" in obj_text
    # Partial fixed-grid atlases must not be bound as map_Kd: that used to
    # paint every unmapped face the same fallback colour.  Vertex colours are
    # the authoritative appearance mode until a complete unwrap is available.
    assert result.appearance_mode == "vertex_color"
    assert "map_Kd" not in result.mtl_path.read_text()


def test_bake_writes_per_face_uvs_and_vertex_colors(tmp_path):
    """회귀: 면별 UV가 실제로 기록되어야 하고(더미 3개 공유 금지),
    텍스처 미지원 뷰어용 정점 색이 전체 정점에 존재해야 한다."""
    import cv2

    vertices, triangles = _cube_mesh()
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    view_img = np.zeros((480, 640, 3), dtype=np.uint8)
    view_img[:, :, 2] = 255  # BGR red
    views = [(0, view_img)]
    poses = {0: np.eye(4) @ np.diag([1, 1, -1, 1])}
    poses[0][:3, 3] = [0, 0, -3.0]

    result = bake_atlas(vertices, triangles, views, K, poses,
                        scene=None, out_dir=tmp_path, name="model")

    lines = result.obj_path.read_text().splitlines()
    assert all("//" in l for l in lines if l.startswith("f "))
    assert result.textured_faces > 0

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(result.obj_path))
    assert mesh.has_vertex_colors(), "정점 색이 기록되어야 함 (Open3D 뷰어 폴백)"
