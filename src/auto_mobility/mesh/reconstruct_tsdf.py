#!/usr/bin/env python3
"""
reconstruct_tsdf.py — Canonical Frame Dataset + Trajectory → Open3D Tensor TSDF → 3D Mesh (.obj) & Point Cloud (.ply)

표준 아키텍처:
  FrameDataset (Canonical RGB-D)
        +
  TUM Trajectory (SLAM 결과)
        ↓
  Open3D Tensor VoxelBlockGrid TSDF
        ↓
  3D Mesh (.obj) & Point Cloud (.ply)
"""

import os
import sys

# Limit multi-threading CPU peak overload to prevent hardware thermal/power throttling shutdowns
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "8"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "8"

import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import numpy as np
import cv2
import open3d as o3d
import open3d.core as o3c
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import DB_DIR, MESH_DIR, POINTCLOUD_DIR, FRAME_DIR, PROJECT_DIR
from auto_mobility.dataset.frame_dataset import FrameDataset, CameraIntrinsics
from auto_mobility.trajectory.io import Trajectory
from auto_mobility.trajectory.association import associate_trajectory_to_frames
from auto_mobility.evaluation.split import load_split_json


def _cuda_available() -> bool:
    try:
        return o3c.cuda.is_available() and o3c.cuda.device_count() > 0
    except Exception:
        return False


def _fast_inv_se3(T: np.ndarray) -> np.ndarray:
    """Fast inversion for SE(3) rigid transform T = [R t; 0 1].
    inv(T) = [R^T, -R^T * t; 0, 1] avoiding expensive general matrix inversion.
    """
    inv_T = np.eye(4, dtype=T.dtype)
    R_T = T[:3, :3].T
    inv_T[:3, :3] = R_T
    inv_T[:3, 3] = -R_T @ T[:3, 3]
    return inv_T


def _bytes_per_block(no_color: bool, block_resolution: int = 16) -> int:
    """VoxelBlockGrid가 블록당 선할당하는 바이트 수.
    - tsdf(float32) + weight(float32) = 8B/voxel
    - + color(float32 x3) = 20B/voxel
    """
    voxel_bytes = 8 if no_color else 20
    return block_resolution ** 3 * voxel_bytes


def _estimate_block_count(
    poses: Dict[int, np.ndarray],
    voxel_size: float,
    depth_max: float,
    no_color: bool = False,
    block_resolution: int = 16,
    safety_factor: float = 1.5,
    min_blocks: int = 2048,
    memory_budget_gb: float = 2.0,
) -> int:
    """궤적 범위(장면 규모)를 기반으로 VoxelBlockGrid의 block_count를 추정한다.

    block_count는 TSDF 품질 파라미터가 아니라 해시맵/속성 버퍼의 **용량 상한**이다.
    Open3D는 block_count만큼 메모리를 선할당하므로, 장면 규모에 비해 과도하게 크면
    OOM이 발생하고, 실제 활성 블록 수보다 작으면 블록이 방출되어 품질이 떨어진다.

    추정 방식:
      1) 궤적 bounding box를 depth_max(레이 최대 도달 거리)만큼 확장한 부피를
         블록 부피(voxel_size * block_resolution)^3 로 나눠 필요한 블록 수 계산
      2) safety_factor 배 여유 적용
      3) memory_budget_gb 한도를 넘지 않도록 상한(모자이크 방지)
    """
    positions = []
    for T in poses.values():
        if T is not None and getattr(T, "shape", None) == (4, 4):
            positions.append(T[:3, 3])
    if not positions:
        return min_blocks

    pos = np.asarray(positions, dtype=np.float64)
    # Filter extreme outlier jumping poses (1st to 99th percentile) to avoid gigantic bounding boxes
    if len(pos) >= 20:
        lo = np.percentile(pos, 1, axis=0) - depth_max
        hi = np.percentile(pos, 99, axis=0) + depth_max
    else:
        lo = pos.min(axis=0) - depth_max
        hi = pos.max(axis=0) + depth_max
    ext = np.maximum(hi - lo, voxel_size * block_resolution)

    block_size = voxel_size * block_resolution
    n_blocks = int(np.ceil(ext[0] / block_size) * np.ceil(ext[1] / block_size) * np.ceil(ext[2] / block_size))
    n_blocks = int(n_blocks * safety_factor)

    max_by_mem = int(memory_budget_gb * 1e9 / _bytes_per_block(no_color, block_resolution))
    n_blocks = min(n_blocks, max_by_mem)
    return max(min_blocks, n_blocks)


import queue
import concurrent.futures


class AsyncFramePrefetcher:
    """ThreadPool background loader to prefetch depth/color images to eliminate disk I/O stall."""
    def __init__(self, dataset, indices, no_color=False, max_workers=4, prefetch_size=12):
        self.dataset = dataset
        self.indices = indices
        self.no_color = no_color
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.queue = queue.Queue(maxsize=prefetch_size)
        self._stop_event = False
        self._future = self.executor.submit(self._worker_loop)

    def _fetch_one(self, idx):
        depth = self.dataset.get_depth(idx)
        color = None if self.no_color else self.dataset.get_rgb_tensor(idx)
        return idx, depth, color

    def _worker_loop(self):
        for idx in self.indices:
            if self._stop_event:
                break
            res = self._fetch_one(idx)
            self.queue.put(res)
        self.queue.put(None)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item

    def shutdown(self):
        self._stop_event = True
        self.executor.shutdown(wait=False)


def reconstruct(
    dataset: Union[FrameDataset, str, Path],
    trajectory: Union[Trajectory, str, Path, Dict[int, np.ndarray]],
    voxel_size: float = 0.02,
    output_mesh: Optional[str] = None,
    output_pcd: Optional[str] = None,
    train_indices: Optional[List[int]] = None,
    trunc_mult: float = 5.0,
    depth_max: float = 4.0,
    depth_min: float = 0.3,
    weight_thr: float = 1.5,
    block_count: int = 0,
    no_color: bool = False,
    no_gpu: bool = False,
    max_pose_gap_ms: float = 50.0,
    stride: int = 1
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """Canonical Dataset과 Trajectory를 받아 Open3D TSDF로 3D Mesh 및 Point Cloud를 재구성한다.

    좌표계 규약:
      - Trajectory poses: T_world_camera (카메라 좌표계 -> 월드 좌표계)
      - TSDF extrinsic: T_camera_world = inv(T_world_camera) (월드 좌표계 -> 카메라 좌표계)
    """
    if isinstance(dataset, (str, Path)):
        dataset = FrameDataset(dataset)

    # Load / associate poses
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
            print(f"⚠️ [TSDF Warning] {assoc_summary.warning}")

    # NOTE: no_gpu parameter accepted for API compatibility but CUDA is intentionally disabled.
    # In the benchmark environment, mixing ROS and Open3D CUDA paths causes SIGSEGV
    # (LD_LIBRARY_PATH conflict between ROS OpenCV/PCL and Open3D bundled libraries).
    # CPU:0 is safe and reproducible. To re-enable GPU, sanitize LD_LIBRARY_PATH first
    # (compare.sh already strips /opt/ros before calling the benchmark).
    device = o3c.Device("CPU:0")
    print(f"🖥️ TSDF Device: {device}, Voxel: {voxel_size*1000:.1f}mm, Total Frames: {len(dataset)}")

    intr = dataset.intrinsics
    intrinsic_t = o3c.Tensor(np.array([
        [intr.fx, 0.0, intr.cx],
        [0.0, intr.fy, intr.cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64))

    bc = int(block_count) if block_count and block_count > 0 else _estimate_block_count(
        poses, voxel_size, depth_max, no_color=no_color
    )
    if block_count and block_count > 0:
        print(f"🧮 VoxelBlockGrid block_count (명시) = {bc:,} (~{bc * _bytes_per_block(no_color) / 1e9:.1f} GB)")
    else:
        print(f"🧮 VoxelBlockGrid block_count (자동 추정) = {bc:,} (~{bc * _bytes_per_block(no_color) / 1e9:.1f} GB)")
    if no_color:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight'),
            attr_dtypes=(o3c.float32, o3c.float32),
            attr_channels=((1,), (1,)),
            voxel_size=voxel_size, block_resolution=16,
            block_count=bc, device=device
        )
    else:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=voxel_size, block_resolution=16,
            block_count=bc, device=device
        )

    # Frame filtering (holdout frames exclusion if train_indices specified)
    indices_to_integrate = set(train_indices) if train_indices is not None else set(range(len(dataset)))

    integrated = 0
    skipped = 0

    step_stride = max(1, int(stride))
    target_indices = [idx for idx in range(0, len(dataset), step_stride) if idx in indices_to_integrate]

    prefetcher = AsyncFramePrefetcher(dataset, target_indices, no_color=no_color, max_workers=4, prefetch_size=12)

    t0 = time.time()
    try:
        for idx, depth, color_rgb in prefetcher:
            if idx not in poses or poses[idx] is None or depth is None:
                skipped += 1
                continue

            T_world_cam = poses[idx]
            # Open3D extrinsic: world to camera (using fast SE(3) inversion)
            extrinsic = _fast_inv_se3(T_world_cam)

            if depth_min > 0:
                min_mm = int(depth_min * 1000.0)
                depth[depth < min_mm] = 0

            depth_t = o3d.t.geometry.Image(o3c.Tensor(np.asarray(depth, dtype=np.uint16), device=device))
            extrinsic_t = o3c.Tensor(extrinsic.astype(np.float64))

            try:
                coords = vbg.compute_unique_block_coordinates(
                    depth_t, intrinsic_t, extrinsic_t, depth_scale=1000.0,
                    depth_max=depth_max, trunc_voxel_multiplier=trunc_mult
                )
            except Exception:
                skipped += 1
                continue

            if no_color or color_rgb is None:
                vbg.integrate(coords, depth_t, intrinsic_t, extrinsic_t,
                              depth_scale=1000.0, depth_max=depth_max,
                              trunc_voxel_multiplier=trunc_mult)
            else:
                color_t = o3d.t.geometry.Image(o3c.Tensor(np.asarray(color_rgb, dtype=np.uint8), device=device))
                vbg.integrate(coords, depth_t, color_t, intrinsic_t, intrinsic_t,
                              extrinsic_t, depth_scale=1000.0, depth_max=depth_max,
                              trunc_voxel_multiplier=trunc_mult)
            integrated += 1
            if integrated % 30 == 0:
                print(f"  integrate {integrated}/{len(target_indices)} frames")
    finally:
        prefetcher.shutdown()

    print(f"✅ TSDF 적분 완료 ({time.time()-t0:.2f}s): {integrated} frames integrated (skipped {skipped})")

    if integrated == 0:
        print("⚠️ TSDF integrated 0 frames! Returning empty triangle mesh and point cloud.")
        mesh = o3d.geometry.TriangleMesh()
        pcd = o3d.geometry.PointCloud()
    else:
        try:
            vbg_cpu = vbg.to(o3c.Device("CPU:0"))
        except Exception as e:
            print(f"  ⚠️ Device CPU copy notice: {e}")
            vbg_cpu = vbg

        # ── Point Cloud 추출 + 즉시 저장 ──
        # Open3D extract_triangle_mesh()가 대용량 VoxelBlockGrid에서
        # SIGSEGV될 수 있어, pcd를 먼저 추출·저장해 crash 시에도 보존한다.
        print("  • PointCloud 추출 중...")
        try:
            pcd_t = vbg_cpu.extract_point_cloud(weight_threshold=weight_thr)
            pcd = pcd_t.to_legacy()
        except Exception as e:
            print(f"  ⚠️ extract_point_cloud 실패: {e}. 빈 PointCloud 사용.")
            pcd = o3d.geometry.PointCloud()

        if output_pcd:
            os.makedirs(os.path.dirname(os.path.abspath(output_pcd)), exist_ok=True)
            o3d.io.write_point_cloud(output_pcd, pcd)
            print(f"☁️ PointCloud 저장: {output_pcd}")

        # ── Triangle Mesh 추출 ──
        # ``tsdf_direct`` must be the TSDF zero-crossing surface.  The previous
        # implementation silently converted a 3 cm downsampled point cloud with
        # Poisson(depth=8), which discarded the requested voxel resolution and
        # produced a closed, inflated surface.  Poisson is available explicitly
        # through mesh_open3d.py; it is not a valid fallback for direct TSDF.
        print("  • Triangle Mesh 추출 중...")
        mesh = o3d.geometry.TriangleMesh()
        if output_mesh:
            try:
                mesh_t = vbg_cpu.extract_triangle_mesh(weight_threshold=weight_thr)
                mesh = mesh_t.to_legacy()
                if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                    mesh.remove_degenerate_triangles()
                    mesh.remove_duplicated_triangles()
                    mesh.remove_duplicated_vertices()
                    mesh.remove_non_manifold_edges()
            except Exception as e:
                raise RuntimeError(f"TSDF direct mesh extraction failed: {e}") from e

            print(f"🔺 Mesh 결과: {len(mesh.vertices):,} vertices / {len(mesh.triangles):,} triangles")
            print(f"☁️ PointCloud 결과: {len(pcd.points):,} points")

            os.makedirs(os.path.dirname(os.path.abspath(output_mesh)), exist_ok=True)
            o3d.io.write_triangle_mesh(output_mesh, mesh)
            print(f"💾 Mesh 저장: {output_mesh}")

    return mesh, pcd


# ────────────────────────── Legacy DB Adapter ──────────────────────────
def _extract_frames_legacy_db(db_path: str, workdir: str) -> Tuple[List, Dict, Tuple[float, float, float, float]]:
    """Legacy RTAB-Map DB에서 frames & poses 추출."""
    build_script = os.path.join(PROJECT_DIR, "scripts", "utils", "build_extractor.sh")
    build = subprocess.run([build_script], capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError("extract_db_rgbd build failed:\n" + build.stderr)
    exe = build.stdout.strip().splitlines()[-1]
    r = subprocess.run([exe, db_path, workdir], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("extract_db_rgbd run failed:\n" + r.stdout + r.stderr)

    frames_txt = os.path.join(workdir, "frames.txt")
    from auto_mobility.mesh.reconstruct_tsdf import load_frames, export_poses, load_poses
    frames = []
    with open(frames_txt) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                frames.append((int(p[0]), int(p[1]), float(p[2]), p[3], p[4]))
            elif len(p) == 4:
                frames.append((int(p[0]), int(p[1]), 0.0, p[2], p[3]))

    poses_file = export_poses(db_path, workdir, 0)
    poses = load_poses(poses_file)

    with open(os.path.join(workdir, "intrinsics.txt")) as f:
        vals = f.read().split()
    intrinsics = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
    return frames, poses, intrinsics


def export_poses(db_path: str, workdir: str, opt: int) -> str:
    """rtabmap-export --poses_camera 로 최적화 pose 취득 (Legacy)."""
    import re
    base = os.path.splitext(os.path.basename(db_path))[0]
    cmd = ["rtabmap-export", "--poses_camera", f"--opt={opt}",
           f"--output={base}_tsdf_poses", db_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = re.search(r'"([^"]*_camera_poses\.txt)"', out)
    if m:
        poses_file = m.group(1)
    else:
        poses_file = os.path.join(os.path.dirname(db_path), f"{base}_tsdf_poses_camera_poses.txt")
    return poses_file


def load_poses(poses_file: str) -> dict:
    """'#timestamp x y z qx qy qz qw id' → {node_id: T_map_cam(4x4)}"""
    poses = {}
    with open(poses_file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            q = np.array([float(p[4]), float(p[5]), float(p[6]), float(p[7])])
            t = np.array([float(p[1]), float(p[2]), float(p[3])])
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(q).as_matrix()
            T[:3, 3] = t
            poses[int(p[8])] = T
    return poses


def load_frames(frames_txt: str) -> list:
    out = []
    with open(frames_txt) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                out.append((int(p[0]), int(p[1]), float(p[2]), p[3], p[4]))
            elif len(p) == 4:
                out.append((int(p[0]), int(p[1]), 0.0, p[2], p[3]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Canonical Dataset / DB + Trajectory → TSDF Reconstruction")
    parser.add_argument("input", nargs="?", default=None, help="Canonical Frame Dataset 디렉터리, 데이터셋 이름 또는 RTAB DB 경로")
    parser.add_argument("output_pos", nargs="?", default=None, help="출력 mesh (.obj) 경로 (위치 인자)")
    parser.add_argument("--output", "-o", default=None, help="출력 mesh (.obj) 경로")
    parser.add_argument("--dataset", default=None, help="Canonical Frame Dataset 경로")
    parser.add_argument("--trajectory", default=None, help="TUM Trajectory 파일 (.txt)")
    parser.add_argument("--pcd-output", default=None, help="출력 pointcloud (.ply) 경로")
    parser.add_argument("--holdout-split", default=None, help="split.json 경로 (지정 시 train 프레임만 적분)")
    parser.add_argument("--voxel", type=float, default=0.02, help="TSDF Voxel 크기 (m, 기본 0.02)")
    parser.add_argument("--trunc-mult", type=float, default=5.0, help="truncation multiplier (기본 5.0)")
    parser.add_argument("--depth-max", type=float, default=4.0, help="최대 depth (m, 기본 4.0)")
    parser.add_argument("--depth-min", type=float, default=0.3, help="최소 depth (m, 기본 0.3)")
    parser.add_argument("--poses-opt", type=int, default=0, choices=[0, 2], help="Legacy DB export opt: 0=전역 최적화, 2=DB 원본")
    parser.add_argument("--weight-thr", type=float, default=1.5, help="표면 추출 weight 임계값 (기본 1.5)")
    parser.add_argument("--block-count", type=int, default=0, help="Voxel block 해시맵 크기 (0=자동 추정, 양수=명시 지정)")
    parser.add_argument("--stride", type=int, default=1, help="프레임 샘플링 간격 (기본 1: 모든 프레임, 2: 1/2 속도/메모리 최적화)")
    parser.add_argument("--no-color", action="store_true", help="Geometry 전용 (컬러 적분 생략)")
    parser.add_argument("--no-gpu", action="store_true", help="CUDA 비활성 (CPU 강제)")
    parser.add_argument("--view", action="store_true", help="완료 후 Open3D 뷰어 실행")
    args = parser.parse_args()

    # Determine input dataset vs legacy DB
    input_str = args.dataset or args.input
    if not input_str:
        parser.print_help()
        sys.exit(1)

    dataset_path = None
    input_path = Path(input_str)

    # 1. Check if input is a Canonical Dataset (directory or name in FRAME_DIR)
    if input_path.is_dir() and (input_path / "frames.csv").exists():
        dataset_path = input_path
    elif (FRAME_DIR / input_str).is_dir() and ((FRAME_DIR / input_str) / "frames.csv").exists():
        dataset_path = FRAME_DIR / input_str
    elif (FRAME_DIR / input_path.stem).is_dir() and ((FRAME_DIR / input_path.stem) / "frames.csv").exists():
        dataset_path = FRAME_DIR / input_path.stem

    # 2. If trajectory not specified, try to find default trajectory for dataset
    traj_path = args.trajectory
    base_name = dataset_path.name if dataset_path else input_path.stem

    if traj_path is None and dataset_path:
        candidates = [
            PROJECT_DIR / "ros2_data" / "trajectories" / f"rtab_{base_name}_trajectory.txt",
            PROJECT_DIR / "ros2_data" / "trajectories" / f"orbslam3_{base_name}_trajectory.txt",
        ]
        for c in candidates:
            if c.exists():
                traj_path = str(c)
                print(f"📍 기본 Trajectory 감지: {traj_path}")
                break

    # Determine output paths
    out_mesh = args.output or args.output_pos or os.path.join(MESH_DIR, f"{base_name}_rtab_tsdf.obj")
    out_pcd = args.pcd_output or os.path.join(POINTCLOUD_DIR, f"{base_name}_rtab_tsdf_cloud.ply")

    # Load holdout split if specified
    train_indices = None
    if args.holdout_split and os.path.exists(args.holdout_split):
        split_info = load_split_json(args.holdout_split)
        train_indices = split_info.get("train_indices")
        print(f"✂️ Hold-out split 적용: {len(train_indices)} train frames 사용 ({len(split_info.get('holdout_indices', []))} holdout frames 제외)")

    if dataset_path and traj_path and os.path.exists(traj_path):
        print(f"🚀 표준 Canonical FrameDataset 경로로 TSDF 실행: {dataset_path}")
        dataset = FrameDataset(dataset_path)
        mesh, pcd = reconstruct(
            dataset=dataset,
            trajectory=traj_path,
            voxel_size=args.voxel,
            output_mesh=out_mesh,
            output_pcd=out_pcd,
            train_indices=train_indices,
            trunc_mult=args.trunc_mult,
            depth_max=args.depth_max,
            depth_min=args.depth_min,
            weight_thr=args.weight_thr,
            block_count=args.block_count,
            no_color=args.no_color,
            no_gpu=args.no_gpu,
            stride=args.stride
        )
    elif str(input_str).endswith(".db") and os.path.exists(input_str):
        print(f"⚙️ Legacy RTAB-Map DB 추출 경로로 TSDF 실행: {input_str}")
        workdir = f"/tmp/tsdf_{base_name}"
        os.makedirs(workdir, exist_ok=True)
        try:
            frames, poses, intrinsics = _extract_frames_legacy_db(input_str, workdir)
            if traj_path and os.path.exists(traj_path):
                print(f"📍 외부 Trajectory 파일 적용: {traj_path}")
                traj = Trajectory.from_tum_file(traj_path)
                traj_poses = traj.to_pose_matrix_dict()
                # associate with sequence
                for seq, node_id, stamp, _, _ in frames:
                    if node_id in traj_poses:
                        poses[node_id] = traj_poses[node_id]

            # Run TSDF with legacy frames/poses adapter
            from auto_mobility.mesh.reconstruct_tsdf import run_tsdf
            mesh, pcd = run_tsdf(frames, poses, workdir, intrinsics, args.voxel, args)
            os.makedirs(os.path.dirname(os.path.abspath(out_mesh)), exist_ok=True)
            o3d.io.write_triangle_mesh(out_mesh, mesh)
            os.makedirs(os.path.dirname(os.path.abspath(out_pcd)), exist_ok=True)
            o3d.io.write_point_cloud(out_pcd, pcd)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    else:
        raise ValueError(f"Cannot find valid dataset or DB for {input_str}. Please prepare canonical dataset with prepare_dataset.sh")

    if args.view:
        o3d.visualization.draw_geometries([mesh], window_name=f"TSDF Mesh - {base_name}", width=1280, height=720)


if __name__ == "__main__":
    main()
