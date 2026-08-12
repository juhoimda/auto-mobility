#!/usr/bin/env python3
"""
reconstruct_tsdf.py — RTAB-Map DB → Open3D Tensor TSDF → Mesh

현재 파이프라인의 "누적 Point Cloud → Poisson" 대신, DB에 저장된
원본 RGB-D 프레임 + 최적화 pose를 그대로 TSDF에 적분해 mesh를 만든다.
여러 관측을 voxel에서 평균화하므로 단일 뷰 depth 노이즈(1~2cm)가
표면으로 남지 않는다.

동작 흐름:
  1) extract_db_rgbd (C++ 헬퍼) → RGB-D 프레임 + intrinsics
  2) rtabmap-export --poses_camera --opt {0|2} → 최적화 pose (map→depth_optical)
  3) pose와 프레임을 node id로 매칭
  4) Open3D Tensor VoxelBlockGrid (CUDA) 적분 → Marching Cubes → OBJ

사용법:
  python3 reconstruct_tsdf.py <session.db> <output.obj>
    [--voxel 0.01] [--trunc-mult 8] [--depth-max 4.0] [--depth-min 0.3]
    [--poses-opt 0] [--weight-thr 3.0] [--block-count 50000]
    [--no-color] [--view] [--keep] [--workdir PATH]

실측 검증 (2026-08-12):
  - pose 규약: rtabmap-export --poses_camera 출력 = map→depth_optical (PLY 정합 p50 0.4cm 확인)
  - 저장된 color는 depth 평면에 정렬됨 (depth 유효 마스크 전폭 커버 확인)
  - intrinsics: calibration blob의 depth 모델 (fx≈606, 640x480)
"""

import sys
import os
import time
import shutil
import argparse
import subprocess

# repo 소스 실행 / 설치 실행 양쪽에서 auto_mobility 패키지 임포트 보장
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    import numpy as np
    import cv2
    import open3d as o3d
    import open3d.core as o3c
    from scipy.spatial.transform import Rotation
except ImportError:
    print("Error: open3d, numpy, opencv-python, scipy 필요 (`pip install open3d numpy opencv-python scipy`)")
    sys.exit(1)

from auto_mobility.config import MESH_DIR  # noqa: E402


# ────────────────────────────── 유틸 ──────────────────────────────

def _cuda_available() -> bool:
    try:
        return o3c.cuda.device_count() > 0
    except Exception:
        return False


def _find_file(root: str, suffix: str) -> str:
    for f in os.listdir(root):
        if f.endswith(suffix):
            return os.path.join(root, f)
    return ""


# ────────────────────────────── 추출 ──────────────────────────────

def extract_frames(db_path: str, workdir: str, build_script: str) -> str:
    """DB → RGB-D 프레임 + intrinsics. 성공 시 frames.txt 경로 반환."""
    build = subprocess.run([build_script], capture_output=True, text=True)
    if build.returncode != 0 or not build.stdout.strip():
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
    """rtabmap-export --poses_camera 로 최적화 pose 취득. pose 파일 경로 반환."""
    import re
    base = os.path.splitext(os.path.basename(db_path))[0]
    cmd = ["rtabmap-export", "--poses_camera", f"--opt={opt}",
           f"--output={base}_tsdf_poses", db_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    # "NN camera poses exported to "PATH"." 파싱 (경로 규칙이 DB dir 기준이라 메시지에서 취득)
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
    """frames.txt → [(seq, node_id, color_rel, depth_rel)]"""
    out = []
    with open(frames_txt) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                out.append((int(p[0]), int(p[1]), p[2], p[3]))
    return out


# ────────────────────────────── TSDF ──────────────────────────────

def run_tsdf(frames, poses, workdir, intrinsics, args):
    device = o3c.Device("CUDA:0") if (_cuda_available() and not args.no_gpu) else o3c.Device("CPU:0")
    print(f"🖥️  device: {device}")

    fx, fy, cx, cy = intrinsics
    # 주의: 이 Open3D 0.19 빌드에서 compute_unique_block_coordinates는
    # intrinsic/extrinsic을 CPU 텐서로 요구함 (CUDA 텐서 전달 시 역변환 오류 실측).
    intrinsic_t = o3c.Tensor(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64))

    if args.no_color:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight'),
            attr_dtypes=(o3c.float32, o3c.float32),
            attr_channels=((1,), (1,)),
            voxel_size=args.voxel, block_resolution=16,
            block_count=args.block_count, device=device)
    else:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=args.voxel, block_resolution=16,
            block_count=args.block_count, device=device)

    integrated = 0
    skipped = 0
    for seq, node_id, color_rel, depth_rel in frames:
        if node_id not in poses:
            skipped += 1
            continue
        T_map_cam = poses[node_id]
        extrinsic = np.linalg.inv(T_map_cam)  # world→camera (Open3D 규약)

        depth_path = os.path.join(workdir, depth_rel)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            skipped += 1
            continue
        if args.depth_min > 0:
            depth = depth.copy()
            depth[depth < args.depth_min * 1000.0] = 0

        depth_t = o3d.t.geometry.Image(o3c.Tensor(np.asarray(depth, dtype=np.uint16), device=device))
        extrinsic_t = o3c.Tensor(extrinsic.astype(np.float64))  # CPU

        try:
            coords = vbg.compute_unique_block_coordinates(
                depth_t, intrinsic_t, extrinsic_t, depth_scale=1000.0,
                depth_max=args.depth_max, trunc_voxel_multiplier=args.trunc_mult)
        except Exception as e:
            print(f"  ⚠ frame {seq}: compute_unique_block_coordinates 실패 ({e})")
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
        if integrated % 10 == 0:
            print(f"  integrate {integrated}/{len(frames)} frames")

    print(f"✅ 통합 완료: {integrated} 프레임 (skip {skipped})")

    mesh_t = vbg.extract_triangle_mesh(weight_threshold=args.weight_thr)
    mesh = mesh_t.to_legacy()
    print(f"🔺 mesh: {len(mesh.vertices):,} vertices / {len(mesh.triangles):,} triangles")
    return mesh, vbg


# ────────────────────────────── main ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RTAB-Map DB → Open3D Tensor TSDF → Mesh")
    parser.add_argument("input", help="RTAB-Map DB (.db) 경로")
    parser.add_argument("output", nargs="?", default=None, help="출력 mesh (.obj/.ply). 기본: meshes/<session>_tsdf.obj")
    parser.add_argument("--voxel", type=float, default=0.01, help="voxel 크기 (m, 기본 0.01)")
    parser.add_argument("--trunc-mult", type=float, default=8.0, help="truncation = voxel×N (기본 8)")
    parser.add_argument("--depth-max", type=float, default=4.0, help="최대 depth (m, 기본 4.0)")
    parser.add_argument("--depth-min", type=float, default=0.3, help="최소 depth (m, 기본 0.3)")
    parser.add_argument("--poses-opt", type=int, default=0, choices=[0, 2],
                        help="rtabmap-export --opt: 0=전역 최적화(파이프라인과 동일), 2=DB 저장 pose")
    parser.add_argument("--weight-thr", type=float, default=3.0, help="표면 추출 weight 임계값 (기본 3.0)")
    parser.add_argument("--block-count", type=int, default=50000, help="voxel block hash map 용량")
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
    poses_file = export_poses(db_path, workdir, args.poses_opt)
    poses = load_poses(poses_file)
    if poses_file.startswith(os.path.dirname(db_path)):
        try:
            os.unlink(poses_file)  # rtabmap-export가 DB 디렉터리에 남긴 임시 pose 파일 정리
        except OSError:
            pass
    frames = load_frames(frames_txt)
    print(f"🎞️  프레임 {len(frames)} / pose {len(poses)}")

    with open(os.path.join(workdir, "intrinsics.txt")) as f:
        vals = f.read().split()
    intrinsics = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))

    mesh, _ = run_tsdf(frames, poses, workdir, intrinsics, args)

    # topology 정리
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f"💾 저장: {out_path} ({time.time()-t0:.1f}s)")
    if not args.keep and workdir.startswith("/tmp/"):
        shutil.rmtree(workdir, ignore_errors=True)
    if args.view:
        o3d.visualization.draw_geometries([mesh], window_name="TSDF Mesh", width=1280, height=720)


if __name__ == "__main__":
    main()
