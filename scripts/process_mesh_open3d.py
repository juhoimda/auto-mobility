#!/usr/bin/env python3
import sys
import os
import argparse

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D is not installed. Install via `pip install open3d`")
    sys.exit(1)

def generate_mesh(input_ply, output_mesh, depth=8, voxel_size=0.01):
    print(f"Loading point cloud: {input_ply}")
    pcd = o3d.io.read_point_cloud(input_ply)
    
    if pcd.is_empty():
        print("Error: Point cloud is empty!")
        sys.exit(1)
        
    print("Downsampling and statistical outlier removal...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(10)
    
    print(f"Reconstructing surface using Poisson Surface Reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    print(f"Saving generated mesh to: {output_mesh}")
    o3d.io.write_triangle_mesh(output_mesh, mesh)
    print("Mesh generation complete!")

def main():
    parser = argparse.ArgumentParser(description="Generate 3D Mesh using Open3D from Point Cloud")
    parser.add_argument("input", help="Input .ply or .pcd point cloud file")
    parser.add_argument("output", help="Output mesh file (.obj or .ply)")
    parser.add_argument("--depth", type=int, default=8, help="Poisson reconstruction depth (default: 8)")
    parser.add_argument("--voxel", type=float, default=0.01, help="Voxel size for downsampling (default: 0.01)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)
        
    generate_mesh(args.input, args.output, args.depth, args.voxel)

if __name__ == "__main__":
    main()
