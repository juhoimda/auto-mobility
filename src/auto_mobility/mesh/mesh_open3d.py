#!/usr/bin/env python3
import sys
import os
import argparse
import time

# repo 소스 실행 / 설치 실행 양쪽에서 auto_mobility 패키지 임포트 보장
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import open3d as o3d
    import numpy as np
    from scipy.spatial import cKDTree
except ImportError:
    print("Error: Open3D, NumPy, or SciPy is not installed. Install via `pip install open3d numpy scipy`")
    sys.exit(1)

from auto_mobility.config import MESH_DEFAULTS

# CUDA voxel downsample이 CPU 대비 실질 이득을 내는 최소 포인트 수.
# 실측(2026-08-12, WSL2 paravirtualized GPU): GPU copy-in 오버헤드가 커서
# 2M 포인트에서도 CPU(tensor)와 비슷(0.94s vs 0.92s), 작은 클라우드는 CPU가 더 빠름.
# 실제 이산형 NVIDIA GPU를 사용한다면 이 임계값을 낮추거나 CPU 경로를 유지해도 무방.
GPU_VOXEL_MIN_POINTS = 5_000_000


def _cuda_available() -> bool:
    """CUDA 사용 가능 여부 (Open3D Tensor API)"""
    try:
        import open3d.core as o3c
        return o3c.cuda.device_count() > 0
    except Exception:
        return False


def _voxel_down(pcd, voxel_size, use_cuda=True):
    """Tensor API 기반 voxel downsampling.

    실측(2026-08-12): legacy voxel_down_sample 대비 더 빠르고 (2.2M pts: legacy
    1.67s vs tensor 0.92s), CUDA는 paravirtualized GPU에서 copy-in 오버헤드로
    이득이 없어 기본 CPU 사용. 대규모 클라우드에 한해 자동 CUDA 선택.
    """
    import open3d.core as o3c
    pts = np.asarray(pcd.points).astype(np.float32)
    use_gpu = bool(use_cuda and _cuda_available() and len(pts) >= GPU_VOXEL_MIN_POINTS)
    device = o3c.Device("CUDA:0") if use_gpu else o3c.Device("CPU:0")
    if use_gpu:
        print(f"  [GPU] voxel_down_sample (CUDA Tensor, {len(pts):,} pts)")
    else:
        print(f"  [CPU] voxel_down_sample (Tensor, {len(pts):,} pts)")
    tpc = o3d.t.geometry.PointCloud(o3c.Tensor(pts, device=device))
    if pcd.has_colors():
        tpc.point.colors = o3c.Tensor(np.asarray(pcd.colors).astype(np.float32), device=device)
    tpc = tpc.voxel_down_sample(voxel_size)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(tpc.point.positions.cpu().numpy().astype(np.float64))
    if tpc.point.colors is not None:
        out.colors = o3d.utility.Vector3dVector(tpc.point.colors.cpu().numpy().astype(np.float64))
    return out


def generate_mesh(input_ply, output_mesh, depth=MESH_DEFAULTS["depth"], voxel_size=MESH_DEFAULTS["voxel_size"],
                  method=MESH_DEFAULTS["method"], view_result=False, clean_density=True,
                  simplify_target=MESH_DEFAULTS["simplify_target"], use_cuda=True,
                  orient="centroid"):
    t0 = time.time()
    print(f"Loading point cloud: {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)

    if pcd.is_empty():
        print("Error: Point cloud is empty!")
        sys.exit(1)

    num_orig_points = len(pcd.points)
    print(f"Original point count: {num_orig_points:,}")

    print(f"Downsampling with fine resolution (voxel_size={voxel_size}m)...")
    raw_points = np.asarray(pcd.points)
    raw_normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    pcd = _voxel_down(pcd, voxel_size, use_cuda=use_cuda)

    # Tensor voxel downsampling은 법선을 보존하지 않으므로,
    # RTAB-Map이 제공한 법선이 있으면 최근접점으로 복사 (재계산보다 빠르고 SLAM 법선 유지).
    # 없을 경우에만 다중스레드 법선 추정 수행.
    if raw_normals is not None and len(pcd.points) > 0:
        print("Reusing normals from input point cloud (RTAB-Map)...")
        tree = cKDTree(raw_points)
        _, idx = tree.query(np.asarray(pcd.points), k=1, workers=-1)
        pcd.normals = o3d.utility.Vector3dVector(raw_normals[idx])
    else:
        print("Estimating normals using fast multi-threaded computation...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 5.0, 0.03), max_nn=30),
            fast_normal_computation=True
        )

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    print(f"Cleaned point count: {len(pcd.points):,}")

    # 법선 방향 정렬: 기본은 전역 MST(tangent plane) 대신 장면 중심(centroid) 방향 정렬.
    # 실측(2026-08-12): 실내 캡처에서 centroid 방향이 더 빠르고(~0s vs 16-49s)
    # 최종 mesh의 포인트클라우드 충실도도 더 우수함 (p99 오차 6.7cm vs 14.1cm).
    # --orient=tangent 로 기존 동작 복원 가능.
    if orient.lower() == "tangent":
        print("Orienting normals using tangent-plane consistent orientation (global MST)...")
        pcd.orient_normals_consistent_tangent_plane(k=15)
    else:
        print("Orienting normals towards scene centroid...")
        pcd.orient_normals_towards_camera_location(pcd.get_center())

    if method.lower() == "bpa":
        print("Reconstructing surface using Ball Pivoting Algorithm (BPA)...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances) if len(distances) > 0 else voxel_size
        radii = [avg_dist * 0.5, avg_dist, avg_dist * 2.0, avg_dist * 4.0, avg_dist * 8.0]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    else:
        # 기본: Poisson (watertight, 구멍 없는 폐곡면) - BPA 대비 품질 우수
        print(f"Reconstructing surface using Poisson Surface Reconstruction (depth={depth}, linear_fit=True, n_threads=-1)...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=1.1, linear_fit=True, n_threads=-1
        )

        if clean_density:
            print("Cleaning low-density Poisson reconstruction artifacts...")
            densities_arr = np.asarray(densities)
            density_threshold = np.quantile(densities_arr, 0.03)
            vertices_to_remove = densities_arr < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

    # Crop to bounding box of point cloud to prevent floating shell artifacts
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Simplify: Isaac Sim / viewer 로딩 성능을 위해 target 비율로 경량화 (기본 50%)
    if simplify_target and 0.0 < simplify_target < 1.0:
        n_before = len(mesh.triangles)
        target = max(int(n_before * simplify_target), 1000)
        print(f"Simplifying mesh: {n_before:,} → {target:,} triangles (target {simplify_target:.0%})...")
        try:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        except TypeError:
            # Open3D 구버전 호환 (positional)
            mesh = mesh.simplify_quadric_decimation(target)
        except Exception as e:
            print(f"  ⚠ simplify 실패 (무시): {e}")

    # Topology cleanup to eliminate degenerate and floating artifacts
    print("Performing mesh topology cleanup (removing degenerate/duplicated/non-manifold elements)...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Transfer RGB colors from point cloud to mesh vertices using nearest-neighbor search
    if pcd.has_colors() and len(pcd.points) > 0 and len(mesh.vertices) > 0:
        print("Transferring RGB colors from point cloud using nearest-neighbor search...")
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        mesh_vertices = np.asarray(mesh.vertices)

        tree = cKDTree(pcd_points)
        _, indices = tree.query(mesh_vertices, k=1, workers=-1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_colors[indices])

    mesh.compute_vertex_normals()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_mesh)), exist_ok=True)

    print(f"Saving generated mesh to: {output_mesh}")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    o3d.io.write_triangle_mesh(output_mesh, mesh)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    elapsed = time.time() - t0
    print(f"🎉 High-detail Mesh generation complete in {elapsed:.2f}s! Vertices: {len(mesh.vertices):,}, Triangles: {len(mesh.triangles):,}")

    if view_result:
        print("Opening interactive 3D Mesh Viewer (Close window to exit)...")
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f"3D Mesh Viewer - {os.path.basename(output_mesh)}",
            width=1280,
            height=720,
            mesh_show_back_face=True
        )

def main():
    parser = argparse.ArgumentParser(description="Generate high-quality 3D Mesh using Open3D from Point Cloud")
    parser.add_argument("input", help="Input .ply or .pcd point cloud file")
    parser.add_argument("output", help="Output mesh file (.obj or .ply)")
    parser.add_argument("--depth", type=int, default=MESH_DEFAULTS["depth"], help=f"Poisson reconstruction depth (default: {MESH_DEFAULTS['depth']}, 8=고품질, 7=약 2배 빠름)")
    parser.add_argument("--voxel", type=float, default=MESH_DEFAULTS["voxel_size"], help=f"Voxel size for downsampling (default: {MESH_DEFAULTS['voxel_size']})")
    parser.add_argument("--method", choices=["poisson", "bpa"], default=MESH_DEFAULTS["method"], help="Reconstruction method: poisson or bpa (default: poisson)")
    parser.add_argument("--view", action="store_true", help="Visualize generated mesh in interactive 3D window")
    parser.add_argument("--no-clean", action="store_true", help="Disable density cleaning filter")
    parser.add_argument("--no-simplify", action="store_true", help="Disable mesh simplification")
    parser.add_argument("--simplify", type=float, default=MESH_DEFAULTS["simplify_target"], help="Triangle simplification target ratio (default: 0.5)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable CUDA voxel downsampling (use CPU)")
    parser.add_argument("--orient", choices=["centroid", "tangent"], default="centroid",
                        help="Normal orientation method: centroid(빠름, 기본) or tangent(전역 MST, 느림)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)

    generate_mesh(
        args.input,
        args.output,
        depth=args.depth,
        voxel_size=args.voxel,
        method=args.method,
        view_result=args.view,
        clean_density=not args.no_clean,
        simplify_target=0.0 if args.no_simplify else args.simplify,
        use_cuda=not args.no_gpu,
        orient=args.orient
    )

if __name__ == "__main__":
    main()
