#!/usr/bin/env python3
"""
reconstruct_tsdf.py — RTAB-Map DB / ORB-SLAM3 Trajectory → Open3D Tensor TSDF → 3D Mesh (.obj) & Point Cloud (.ply)
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import numpy as np
import cv2
import open3d as o3d
import open3d.core as o3c
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import DB_DIR, MESH_DIR, POINTCLOUD_DIR, PROJECT_DIR


def _cuda_available() -> bool:
    try:
        return o3c.cuda.is_available() and o3c.cuda.device_count() > 0
    except Exception:
        return False


def extract_frames(db_path: str, workdir: str, build_script: str) -> str:
    """RTAB-Map DB에서 extract_db_rgbd C++ 도구를 빌드/실행하여 RGB-D 프레임 추출."""
    build = subprocess.run([build_script], capture_output=True, text=True)
    if build.returncode != 0:
        print("❌ extract_db_rgbd 빌드 실패:\n" + build.stderr)
        sys.exit(1)
    exe = build.stdout.strip().splitlines()[-1]
    r = subprocess.run([exe, db_path, workdir], capture_output=True, text=True)
    if r.returncode != 0:
        print("❌ extract_db_rgbd 실행 실패:\n" + r.stdout + r.stderr)
        sys.exit(1)
    print(r.stdout.strip())
    frames_txt = os.path.join(workdir, "frames.txt")
    if not os.path.exists(frames_txt):
        print("❌ frames.txt 없음 — DB에 RGB-D 프레임이 없는 것 같습니다.")
        sys.exit(1)
    return frames_txt


def export_poses(db_path: str, workdir: str, opt: int) -> str:
    """rtabmap-export --poses_camera 로 최적화 pose 취득."""
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
    if not os.path.exists(poses_file):
        print("❌ pose 추출 실패:\n" + out)
        sys.exit(1)
    n = sum(1 for _ in open(poses_file) if not _.startswith("#") and _.strip())
    print(f"✅ poses ({opt=}): {n} 프레임")
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
    """frames.txt → [(seq, node_id, stamp, color_rel, depth_rel)]"""
    out = []
    with open(frames_txt) as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                out.append((int(p[0]), int(p[1]), float(p[2]), p[3], p[4]))
            elif len(p) == 4:
                out.append((int(p[0]), int(p[1]), 0.0, p[2], p[3]))
    return out


def run_tsdf(frames, poses, workdir, intrinsics, voxel_size, args):
    device = o3c.Device("CUDA:0") if (_cuda_available() and not args.no_gpu) else o3c.Device("CPU:0")
    print(f"🖥️  device: {device}, Voxel: {voxel_size*1000:.1f}mm")

    fx, fy, cx, cy = intrinsics
    intrinsic_t = o3c.Tensor(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64))

    block_count = max(args.block_count, 100000)
    if args.no_color:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight'),
            attr_dtypes=(o3c.float32, o3c.float32),
            attr_channels=((1,), (1,)),
            voxel_size=voxel_size, block_resolution=16,
            block_count=block_count, device=device)
    else:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=voxel_size, block_resolution=16,
            block_count=block_count, device=device)

    integrated = 0
    skipped = 0
    for frame_info in frames:
        seq, node_id = frame_info[0], frame_info[1]
        color_rel, depth_rel = frame_info[-2], frame_info[-1]
        if node_id not in poses:
            skipped += 1
            continue
        T_map_cam = poses[node_id]
        extrinsic = np.linalg.inv(T_map_cam)

        depth_path = os.path.join(workdir, depth_rel)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            skipped += 1
            continue
        if args.depth_min > 0:
            depth = depth.copy()
            depth[depth < args.depth_min * 1000.0] = 0

        depth_t = o3d.t.geometry.Image(o3c.Tensor(np.asarray(depth, dtype=np.uint16), device=device))
        extrinsic_t = o3c.Tensor(extrinsic.astype(np.float64))

        try:
            coords = vbg.compute_unique_block_coordinates(
                depth_t, intrinsic_t, extrinsic_t, depth_scale=1000.0,
                depth_max=args.depth_max, trunc_voxel_multiplier=args.trunc_mult)
        except Exception:
            skipped += 1
            continue

        if args.no_color or color_rel == "none":
            vbg.integrate(coords, depth_t, intrinsic_t, extrinsic_t,
                          depth_scale=1000.0, depth_max=args.depth_max,
                          trunc_voxel_multiplier=args.trunc_mult)
        else:
            color_path = os.path.join(workdir, color_rel)
            color = cv2.imread(color_path, cv2.IMREAD_COLOR)
            if color is None:
                vbg.integrate(coords, depth_t, intrinsic_t, extrinsic_t,
                              depth_scale=1000.0, depth_max=args.depth_max,
                              trunc_voxel_multiplier=args.trunc_mult)
            else:
                color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                color_t = o3d.t.geometry.Image(o3c.Tensor(np.asarray(color_rgb, dtype=np.uint8), device=device))
                vbg.integrate(coords, depth_t, color_t, intrinsic_t, intrinsic_t,
                              extrinsic_t, depth_scale=1000.0, depth_max=args.depth_max,
                              trunc_voxel_multiplier=args.trunc_mult)
        integrated += 1
        if integrated % 20 == 0:
            print(f"  integrate {integrated}/{len(frames)} frames")

    print(f"✅ 통합 완료: {integrated} 프레임 (skip {skipped})")

    # Mesh 추출 시도 (실패 시 적응형 Voxel 확장 복구)
    try:
        mesh_t = vbg.extract_triangle_mesh(weight_threshold=args.weight_thr)
        mesh = mesh_t.to_legacy()
    except Exception as e:
        if voxel_size < 0.02:
            print(f"⚠️ Voxel {voxel_size*1000:.0f}mm에서 GPU 메모리 한계 발생. 안정적인 20mm Voxel로 자동 재구성합니다...")
            return run_tsdf(frames, poses, workdir, intrinsics, 0.02, args)
        else:
            raise e

    # Point Cloud 추출
    try:
        pcd_t = vbg.extract_point_cloud(weight_threshold=args.weight_thr)
        pcd = pcd_t.to_legacy()
    except Exception:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors

    print(f"🔺 mesh: {len(mesh.vertices):,} vertices / {len(mesh.triangles):,} triangles")
    print(f"☁️  pointcloud: {len(pcd.points):,} points")
    return mesh, pcd


def main():
    parser = argparse.ArgumentParser(description="RTAB-Map DB / ORB-SLAM3 → Open3D TSDF → Mesh (.obj) & PCD (.ply)")
    parser.add_argument("input", help="RTAB-Map DB (.db) 경로")
    parser.add_argument("output", nargs="?", default=None, help="출력 mesh (.obj). 기본: meshes/<session>_tsdf.obj")
    parser.add_argument("--pcd-output", default=None, help="출력 pointcloud (.ply). 기본: pointclouds/<session>_tsdf_cloud.ply")
    parser.add_argument("--voxel", type=float, default=0.02, help="voxel 크기 (m, 기본 0.02)")
    parser.add_argument("--trunc-mult", type=float, default=5.0, help="truncation = voxel×N (기본 5.0)")
    parser.add_argument("--depth-max", type=float, default=4.0, help="최대 depth (m, 기본 4.0)")
    parser.add_argument("--depth-min", type=float, default=0.3, help="최소 depth (m, 기본 0.3)")
    parser.add_argument("--poses-opt", type=int, default=0, choices=[0, 2],
                        help="rtabmap-export --opt: 0=전역 최적화, 2=DB 저장 pose")
    parser.add_argument("--trajectory", default=None,
                        help="외부 TUM Trajectory 파일 경로 (.txt). 지정 시 DB 내부 pose 대신 이 궤적을 사용하여 TSDF 적분.")
    parser.add_argument("--weight-thr", type=float, default=1.5, help="표면 추출 weight 임계값 (기본 1.5)")
    parser.add_argument("--block-count", type=int, default=100000, help="voxel block hash map 용량")
    parser.add_argument("--no-color", action="store_true", help="geometry 전용 (컬러 통합 생략)")
    parser.add_argument("--no-gpu", action="store_true", help="CUDA 비활성 (CPU)")
    parser.add_argument("--view", action="store_true", help="완료 후 Open3D 뷰어")
    parser.add_argument("--keep", action="store_true", help="추출 작업 디렉터리 유지")
    parser.add_argument("--workdir", default=None, help="작업 디렉터리 (기본: /tmp/tsdf_<session>)")
    args = parser.parse_args()

    db_path = os.path.abspath(args.input)
    if not os.path.exists(db_path):
        print(f"❌ DB 없음: {db_path}")
        sys.exit(1)

    base = os.path.splitext(os.path.basename(db_path))[0]
    if args.output is None:
        out_path = os.path.join(MESH_DIR, f"{base}_tsdf.obj")
    else:
        out_path = args.output

    workdir = args.workdir or f"/tmp/tsdf_{base}"
    os.makedirs(workdir, exist_ok=True)
    build_script = os.path.join(PROJECT_DIR, "scripts", "utils", "build_extractor.sh")

    t0 = time.time()
    frames_txt = extract_frames(db_path, workdir, build_script)
    frames = load_frames(frames_txt)

    if args.trajectory and os.path.exists(args.trajectory):
        print(f"📍 외부 Trajectory 파일 사용: {args.trajectory}")
        from auto_mobility.trajectory.io import Trajectory
        traj = Trajectory.from_tum_file(args.trajectory)
        traj_matrices = []
        for pos, quat in zip(traj.positions, traj.orientations):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            traj_matrices.append(T)
        traj_stamps = traj.timestamps

        poses = {}
        for frame_info in frames:
            seq, node_id, stamp = frame_info[0], frame_info[1], frame_info[2]
            if stamp > 0 and len(traj_stamps) > 0:
                idx = int(np.argmin(np.abs(traj_stamps - stamp)))
                dt = abs(traj_stamps[idx] - stamp)
                if dt < 0.25:  # 250ms 이내 매칭
                    poses[node_id] = traj_matrices[idx]
                elif (seq - 1) < len(traj_matrices):
                    poses[node_id] = traj_matrices[seq - 1]
            elif (seq - 1) < len(traj_matrices):
                poses[node_id] = traj_matrices[seq - 1]
    else:
        poses_file = export_poses(db_path, workdir, args.poses_opt)
        poses = load_poses(poses_file)
        if poses_file.startswith(os.path.dirname(db_path)):
            try:
                os.unlink(poses_file)
            except OSError:
                pass

    print(f"🎞️  프레임 {len(frames)} / pose {len(poses)}")

    with open(os.path.join(workdir, "intrinsics.txt")) as f:
        vals = f.read().split()
    intrinsics = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))

    mesh, pcd = run_tsdf(frames, poses, workdir, intrinsics, args.voxel, args)

    # Topology 정리
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"💾 Mesh 저장: {out_path} ({time.time()-t0:.1f}s)")

    if args.pcd_output:
        pcd_path = args.pcd_output
    else:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        base_name = os.path.splitext(os.path.basename(out_path))[0]
        if "benchmarks" in out_dir:
            pcd_path = os.path.join(out_dir, f"{base_name.replace('_mesh', '')}_cloud.ply")
        else:
            pcd_path = os.path.join(POINTCLOUD_DIR, f"{base}_tsdf_cloud.ply")

    os.makedirs(os.path.dirname(os.path.abspath(pcd_path)), exist_ok=True)
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"☁️  PointCloud 저장: {pcd_path} ({len(pcd.points):,} points)")

    if not args.keep and workdir.startswith("/tmp/"):
        shutil.rmtree(workdir, ignore_errors=True)
    if args.view:
        o3d.visualization.draw_geometries([mesh], window_name="TSDF Mesh", width=1280, height=720)


if __name__ == "__main__":
    main()
