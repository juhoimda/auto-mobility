"""
auto_mobility.evaluation.render_depth

Open3D RaycastingScene을 활용한 고속 Depth 렌더링 및 가상 관측 모듈.

좌표계 규약:
  - T_world_camera: (4x4) 월드 좌표계 기준 카메라 포즈 (위치: T[:3, 3], 회전: T[:3, :3])
  - 카메라 공간 Ray 방향: d_cam = ((u - cx)/fx, (v - cy)/fy, 1.0)
  - 월드 공간 Ray 방향: d_world = R_world_camera * (d_cam / ||d_cam||)
  - Ray 원점: o_world = T_world_camera[:3, 3]
  - Z-Depth (카메라 광축 수직 거리): z = t_hit * (1 / ||d_cam||)
"""

import numpy as np
import open3d as o3d
import open3d.core as o3c
from typing import Tuple, Union, Optional
from auto_mobility.dataset.frame_dataset import CameraIntrinsics


def create_raycasting_scene(mesh: Union[o3d.geometry.TriangleMesh, o3d.t.geometry.TriangleMesh]) -> o3d.t.geometry.RaycastingScene:
    """Open3D TriangleMesh로부터 RaycastingScene 생성."""
    scene = o3d.t.geometry.RaycastingScene()
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    else:
        mesh_t = mesh
    scene.add_triangles(mesh_t)
    return scene


def generate_camera_rays(
    T_world_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> Tuple[o3d.core.Tensor, np.ndarray]:
    """카메라 포즈와 Intrinsics로부터 Open3D RaycastingScene용 Ray 텐서 (H, W, 6) 생성.

    Returns:
        rays_tensor: o3d.core.Tensor of shape (H, W, 6) [ox, oy, oz, dx, dy, dz]
        norm_d_cam: np.ndarray of shape (H, W) representing ||d_cam|| (z-depth 변환용 계수)
    """
    w = width or intrinsics.width
    h = height or intrinsics.height
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = np.ones_like(x_cam)

    d_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    norm_d = np.linalg.norm(d_cam, axis=-1, keepdims=True)  # (H, W, 1)
    d_cam_unit = d_cam / np.maximum(norm_d, 1e-8)  # (H, W, 3)

    R_world_cam = T_world_camera[:3, :3]
    pos_world = T_world_camera[:3, 3]

    # Transform ray directions to world frame: d_world = d_cam_unit @ R^T
    d_world = np.dot(d_cam_unit, R_world_cam.T)  # (H, W, 3)

    # Origin tensor broadcasted across all pixels
    origins = np.broadcast_to(pos_world, (h, w, 3))

    rays = np.concatenate([origins, d_world], axis=-1).astype(np.float32)  # (H, W, 6)
    rays_tensor = o3c.Tensor(rays)
    return rays_tensor, norm_d.squeeze(-1)


def render_depth_map(
    scene: o3d.t.geometry.RaycastingScene,
    T_world_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_min_m: float = 0.2,
    depth_max_m: float = 6.0,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> np.ndarray:
    """RaycastingScene에서 주어진 카메라 포즈로 Predicted Z-Depth Map (단위: mm, uint16 또는 float32 mm) 렌더링.

    Returns:
        rendered_depth_mm: np.ndarray of shape (H, W), dtype float32 (0.0은 no-hit/유효범위 밖)
    """
    rays_tensor, norm_d = generate_camera_rays(T_world_camera, intrinsics, width, height)
    ans = scene.cast_rays(rays_tensor)
    t_hit = ans["t_hit"].numpy()  # (H, W) float32 distance along ray

    # Perpendicular Z-depth in meters: z = t_hit / norm_d
    # Note: t_hit is distance along normalized unit vector d_cam_unit.
    # Since d_cam_z = 1.0, and unit vector d_z = 1.0 / norm_d, the z-depth is t_hit * (1.0 / norm_d)
    z_depth_m = t_hit / np.maximum(norm_d, 1e-8)

    valid_mask = (t_hit < np.inf) & (z_depth_m >= depth_min_m) & (z_depth_m <= depth_max_m)
    rendered_depth_mm = np.zeros_like(z_depth_m, dtype=np.float32)
    rendered_depth_mm[valid_mask] = z_depth_m[valid_mask] * 1000.0
    return rendered_depth_mm
