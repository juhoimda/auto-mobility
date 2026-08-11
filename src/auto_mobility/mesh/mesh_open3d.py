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


def generate_mesh(input_ply, output_mesh, depth=MESH_DEFAULTS["depth"], voxel_size=MESH_DEFAULTS["voxel_size"],
                  method=MESH_DEFAULTS["method"], view_result=False, clean_density=True,
                  simplify_target=MESH_DEFAULTS["simplify_target"]):
    t0 = time.time()
    print(f"Loading point cloud: {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)
    
    if pcd.is_empty():
        print("Error: Point cloud is empty!")
        sys.exit(1)
        
    num_orig_points = len(pcd.points)
    print(f"Original point count: {num_orig_points:,}")
    
    print(f"Downsampling with fine resolution (voxel_size={voxel_size}m) and statistical outlier removal...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    print(f"Cleaned point count: {len(pcd.points):,}")
    
    # 640x480 depth 기반 point cloud 는 노이즈/홀이 많다.
    # 충분한 밀도 확보를 위해 가벼운 hole filling 은 BPA 복원 시 자연 처리.
    print("Estimating normals using fast multi-threaded computation...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 5.0, 0.03), max_nn=30),
        fast_normal_computation=True
    )
    # Z축 단일 강제 정렬 대신 Tangent Plane 기반 3D 일관성 정렬 적용 (벽/천장 mesh 뒤집힘 방지)
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
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

    # Transfer RGB colors from point cloud to mesh vertices using k=3 IDW (Inverse Distance Weighting)
    if pcd.has_colors() and len(pcd.points) > 0 and len(mesh.vertices) > 0:
        print("Transferring RGB colors from point cloud using k=3 Inverse Distance Weighting...")
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        mesh_vertices = np.asarray(mesh.vertices)
        
        tree = cKDTree(pcd_points)
        k_neighbors = min(3, len(pcd_points))
        dists, indices = tree.query(mesh_vertices, k=k_neighbors, workers=-1)
        
        if k_neighbors == 1:
            mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_colors[indices])
        else:
            weights = 1.0 / (dists + 1e-8)
            weights /= np.sum(weights, axis=1, keepdims=True)
            interpolated_colors = np.sum(pcd_colors[indices] * weights[:, :, np.newaxis], axis=1)
            mesh.vertex_colors = o3d.utility.Vector3dVector(interpolated_colors)

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
    parser.add_argument("--depth", type=int, default=MESH_DEFAULTS["depth"], help=f"Poisson reconstruction depth (default: {MESH_DEFAULTS['depth']})")
    parser.add_argument("--voxel", type=float, default=MESH_DEFAULTS["voxel_size"], help=f"Voxel size for downsampling (default: {MESH_DEFAULTS['voxel_size']})")
    parser.add_argument("--method", choices=["poisson", "bpa"], default=MESH_DEFAULTS["method"], help="Reconstruction method: poisson or bpa (default: poisson)")
    parser.add_argument("--view", action="store_true", help="Visualize generated mesh in interactive 3D window")
    parser.add_argument("--no-clean", action="store_true", help="Disable density cleaning filter")
    parser.add_argument("--no-simplify", action="store_true", help="Disable mesh simplification")
    parser.add_argument("--simplify", type=float, default=MESH_DEFAULTS["simplify_target"], help="Triangle simplification target ratio (default: 0.5)")

    
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
        simplify_target=0.0 if args.no_simplify else args.simplify
    )

if __name__ == "__main__":
    main()


