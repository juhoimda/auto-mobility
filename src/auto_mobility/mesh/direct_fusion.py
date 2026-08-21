#!/usr/bin/env python3
"""
direct_fusion.py — Direct Point Cloud Fusion Baseline (Bypassing TSDF).

Backprojects Canonical RGB-D train frames into 3D world coordinates using trajectory poses:
  Canonical RGB-D Frames + Trajectory Poses
                     ↓
  Backproject Depth Pixels to World 3D (R * p_cam + t)
                     ↓
  Accumulate Point Batches + Colors
                     ↓
  Voxel Downsample + Statistical Outlier Removal
                     ↓
  Estimate & Orient Normals
                     ↓
  Direct Point Cloud (.ply)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import open3d as o3d
import open3d.core as o3c

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import POINTCLOUD_DIR, FRAME_DIR, PROJECT_DIR
from auto_mobility.dataset.frame_dataset import FrameDataset, CameraIntrinsics
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.trajectory.association import associate_trajectory_to_frames
from auto_mobility.evaluation.split import load_split_json


def backproject_frame_to_world(
    depth_raw: np.ndarray,
    color_rgb: Optional[np.ndarray],
    T_world_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
    stride: int = 1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Backproject a single depth image into 3D world coordinates with optional color."""
    h, w = depth_raw.shape[:2]
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    v_grid, u_grid = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing='ij')
    depth_sub = depth_raw[v_grid, u_grid]

    min_mm = int(depth_min_m * 1000.0)
    max_mm = int(depth_max_m * 1000.0)
    valid = (depth_sub >= min_mm) & (depth_sub <= max_mm)

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), (np.empty((0, 3), dtype=np.float32) if color_rgb is not None else None)

    u_valid = u_grid[valid].astype(np.float32)
    v_valid = v_grid[valid].astype(np.float32)
    z_m = depth_sub[valid].astype(np.float32) / 1000.0

    x_m = (u_valid - cx) * z_m / fx
    y_m = (v_valid - cy) * z_m / fy
    pts_cam = np.stack([x_m, y_m, z_m], axis=-1)

    R = T_world_camera[:3, :3].astype(np.float32)
    t = T_world_camera[:3, 3].astype(np.float32)
    pts_world = np.dot(pts_cam, R.T) + t

    colors_world = None
    if color_rgb is not None:
        color_sub = color_rgb[v_grid, u_grid]
        colors_valid = color_sub[valid].astype(np.float32) / 255.0
        colors_world = colors_valid

    return pts_world.astype(np.float32), colors_world


def direct_pointcloud_fusion(
    dataset: Union[FrameDataset, str, Path],
    trajectory: Union[Trajectory, str, Path, Dict[int, np.ndarray]],
    output_ply: Optional[str] = None,
    train_indices: Optional[List[int]] = None,
    voxel_size: float = 0.010,
    depth_min: float = 0.3,
    depth_max: float = 3.0,
    stride: int = 1,
    pixel_stride: int = 2,
    max_pose_gap_ms: float = 50.0,
    no_color: bool = False,
    orient: str = "centroid",
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    """Fuse RGB-D frames directly into a clean, normal-oriented point cloud without TSDF grid."""
    t0 = time.time()
    if isinstance(dataset, (str, Path)):
        dataset = FrameDataset(dataset)

    # Pose association
    if isinstance(trajectory, dict):
        poses = trajectory
    else:
        if isinstance(trajectory, (str, Path)):
            traj_obj = Trajectory.from_tum_file(str(trajectory))
        else:
            traj_obj = trajectory
        timestamps = dataset.get_timestamps(use_rgb=True)
        poses, _, assoc_summary = associate_trajectory_to_frames(
            timestamps, traj_obj, max_pose_gap_ms=max_pose_gap_ms, enable_interpolation=True
        )
        if assoc_summary.warning:
            print(f"⚠️ [Direct Fusion Warning] {assoc_summary.warning}")

    intr = dataset.intrinsics
    indices_to_integrate = set(train_indices) if train_indices is not None else set(range(len(dataset)))
    target_indices = [idx for idx in range(0, len(dataset), max(1, int(stride))) if idx in indices_to_integrate]

    print(f"🚀 [Direct Point Cloud Fusion] Processing {len(target_indices)} frames (Voxel: {voxel_size*1000:.1f}mm, Stride: {stride})")

    accumulated_pts: List[np.ndarray] = []
    accumulated_colors: List[np.ndarray] = []
    chunk_pts_count = 0
    intermediate_clouds: List[o3d.geometry.PointCloud] = []

    integrated = 0
    skipped = 0

    for f_cnt, idx in enumerate(target_indices):
        if idx not in poses or poses[idx] is None:
            skipped += 1
            continue

        depth = dataset.get_depth(idx)
        if depth is None:
            skipped += 1
            continue

        color = None if no_color else dataset.get_rgb_tensor(idx)
        T_world_cam = poses[idx]

        pts, cols = backproject_frame_to_world(
            depth_raw=depth,
            color_rgb=color,
            T_world_camera=T_world_cam,
            intrinsics=intr,
            depth_min_m=depth_min,
            depth_max_m=depth_max,
            stride=pixel_stride
        )

        if len(pts) > 0:
            accumulated_pts.append(pts)
            if cols is not None:
                accumulated_colors.append(cols)
            chunk_pts_count += len(pts)
            integrated += 1

        # Periodic intermediate voxel downsampling every ~25 frames to bound memory footprint
        if chunk_pts_count >= 1_500_000:
            chunk_xyz = np.concatenate(accumulated_pts, axis=0)
            chunk_pcd = o3d.geometry.PointCloud()
            chunk_pcd.points = o3d.utility.Vector3dVector(chunk_xyz.astype(np.float64))
            if accumulated_colors:
                chunk_rgb = np.concatenate(accumulated_colors, axis=0)
                chunk_pcd.colors = o3d.utility.Vector3dVector(chunk_rgb.astype(np.float64))

            chunk_pcd = chunk_pcd.voxel_down_sample(voxel_size)
            intermediate_clouds.append(chunk_pcd)
            accumulated_pts.clear()
            accumulated_colors.clear()
            chunk_pts_count = 0

        if integrated % 30 == 0:
            print(f"  integrated {integrated}/{len(target_indices)} frames...")

    # Flush remaining points
    if accumulated_pts:
        chunk_xyz = np.concatenate(accumulated_pts, axis=0)
        chunk_pcd = o3d.geometry.PointCloud()
        chunk_pcd.points = o3d.utility.Vector3dVector(chunk_xyz.astype(np.float64))
        if accumulated_colors:
            chunk_rgb = np.concatenate(accumulated_colors, axis=0)
            chunk_pcd.colors = o3d.utility.Vector3dVector(chunk_rgb.astype(np.float64))
        chunk_pcd = chunk_pcd.voxel_down_sample(voxel_size)
        intermediate_clouds.append(chunk_pcd)

    if not intermediate_clouds:
        print("⚠️ Direct fusion integrated 0 valid points! Returning empty PointCloud.")
        pcd = o3d.geometry.PointCloud()
        if output_ply:
            os.makedirs(os.path.dirname(os.path.abspath(output_ply)), exist_ok=True)
            o3d.io.write_point_cloud(output_ply, pcd)
        return pcd

    # Merge intermediate clouds and perform global voxel downsampling
    print(f"⚙️ Merging {len(intermediate_clouds)} chunks and performing global voxel filtering...")
    merged_pts = []
    merged_cols = []
    has_cols = intermediate_clouds[0].has_colors()

    for c in intermediate_clouds:
        merged_pts.append(np.asarray(c.points))
        if has_cols and c.has_colors():
            merged_cols.append(np.asarray(c.colors))

    full_pts = np.concatenate(merged_pts, axis=0)
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(full_pts)
    if merged_cols:
        final_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(merged_cols, axis=0))

    # Global voxel downsampling
    final_pcd = final_pcd.voxel_down_sample(voxel_size)

    # Statistical outlier removal
    if len(final_pcd.points) >= 10:
        print(f"🧹 Statistical outlier removal (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
        cl, ind = final_pcd.remove_statistical_outlier(
            nb_neighbors=min(nb_neighbors, len(final_pcd.points) - 1),
            std_ratio=std_ratio
        )
        final_pcd = final_pcd.select_by_index(ind)

    # Estimate and orient normals
    if len(final_pcd.points) >= 4:
        print("🧭 Estimating & orienting point cloud normals...")
        final_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 4.0, 0.03), max_nn=30),
            fast_normal_computation=True
        )
        if orient.lower() == "tangent":
            final_pcd.orient_normals_consistent_tangent_plane(k=15)
        else:
            final_pcd.orient_normals_towards_camera_location(final_pcd.get_center())

    elapsed = time.time() - t0
    print(f"✅ Direct Point Cloud Fusion Complete in {elapsed:.2f}s! ({len(final_pcd.points):,} points)")

    if output_ply:
        os.makedirs(os.path.dirname(os.path.abspath(output_ply)), exist_ok=True)
        o3d.io.write_point_cloud(output_ply, final_pcd)
        print(f"💾 Direct PointCloud saved to: {output_ply}")

    return final_pcd


def main():
    parser = argparse.ArgumentParser(description="Direct Point Cloud Fusion from Canonical RGB-D Frames & Trajectory")
    parser.add_argument("dataset", help="Canonical Frame Dataset path or name")
    parser.add_argument("trajectory", help="TUM Trajectory file path")
    parser.add_argument("--output", "-o", default=None, help="Output PLY path")
    parser.add_argument("--voxel", type=float, default=0.010, help="Voxel size in meters (default: 0.010)")
    parser.add_argument("--depth-min", type=float, default=0.3, help="Min depth in meters (default: 0.3)")
    parser.add_argument("--depth-max", type=float, default=3.0, help="Max depth in meters (default: 3.0)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1)")
    parser.add_argument("--split", default=None, help="Holdout split.json path")
    parser.add_argument("--no-color", action="store_true", help="Exclude color")
    parser.add_argument("--orient", choices=["centroid", "tangent"], default="centroid")
    args = parser.parse_args()

    train_indices = None
    if args.split and os.path.exists(args.split):
        s_data = load_split_json(args.split)
        train_indices = s_data.get("train_indices")

    out_ply = args.output
    if not out_ply:
        d_name = Path(args.dataset).name
        out_ply = str(POINTCLOUD_DIR / f"{d_name}_direct_cloud.ply")

    direct_pointcloud_fusion(
        dataset=args.dataset,
        trajectory=args.trajectory,
        output_ply=out_ply,
        train_indices=train_indices,
        voxel_size=args.voxel,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        stride=args.stride,
        no_color=args.no_color,
        orient=args.orient
    )


if __name__ == "__main__":
    main()
