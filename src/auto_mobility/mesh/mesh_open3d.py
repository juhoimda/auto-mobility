#!/usr/bin/env python3
import sys
import os
import argparse
import time

try:
    import open3d as o3d
    import numpy as np
    from scipy.spatial import cKDTree
except ImportError:
    print("Error: Open3D, NumPy, or SciPy is not installed. Install via `pip install open3d numpy scipy`")
    sys.exit(1)

def generate_mesh(input_ply, output_mesh, depth=9, voxel_size=0.003, method="bpa", view_result=False, clean_density=True):
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
    
    print("Estimating normals using fast multi-threaded computation...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(voxel_size * 5.0, 0.03), max_nn=30),
        fast_normal_computation=True
    )
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))
    
    if method.lower() == "bpa":
        print("Reconstructing surface using Ball Pivoting Algorithm (BPA)...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances) if len(distances) > 0 else voxel_size
        radii = [avg_dist * 0.5, avg_dist, avg_dist * 2.0, avg_dist * 4.0, avg_dist * 8.0]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    else:
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

    # Topology cleanup to eliminate degenerate and floating artifacts
    print("Performing mesh topology cleanup (removing degenerate/duplicated/non-manifold elements)...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Transfer RGB colors from point cloud to mesh vertices using vectorized multi-threaded cKDTree
    if pcd.has_colors() and len(pcd.points) > 0 and len(mesh.vertices) > 0:
        print("Transferring RGB colors from point cloud using vectorized multi-threaded cKDTree...")
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
    o3d.io.write_triangle_mesh(output_mesh, mesh)
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
    parser.add_argument("--depth", type=int, default=9, help="Poisson reconstruction depth (default: 9)")
    parser.add_argument("--voxel", type=float, default=0.003, help="Voxel size for downsampling (default: 0.003)")
    parser.add_argument("--method", choices=["poisson", "bpa"], default="bpa", help="Reconstruction method: poisson or bpa (default: bpa)")
    parser.add_argument("--view", action="store_true", help="Visualize generated mesh in interactive 3D window")
    parser.add_argument("--no-clean", action="store_true", help="Disable density cleaning filter")

    
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
        clean_density=not args.no_clean
    )

if __name__ == "__main__":
    main()


