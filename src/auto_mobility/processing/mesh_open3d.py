#!/usr/bin/env python3
import sys
import os
import argparse

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("Error: Open3D is not installed. Install via `pip install open3d numpy`")
    sys.exit(1)

def generate_mesh(input_ply, output_mesh, depth=8, voxel_size=0.01, view_result=False, clean_density=True):
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
    
    if clean_density:
        print("Cleaning low-density Poisson reconstruction artifacts...")
        densities_arr = np.asarray(densities)
        density_threshold = np.quantile(densities_arr, 0.05)
        vertices_to_remove = densities_arr < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Crop to bounding box of point cloud to prevent floating shell artifacts
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Transfer RGB colors from point cloud to mesh vertices if available
    if pcd.has_colors() and len(pcd.points) > 0:
        print("Transferring RGB colors from point cloud to 3D mesh...")
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_vertices = np.asarray(mesh.vertices)
        pcd_colors = np.asarray(pcd.colors)
        colors = []
        for v in mesh_vertices:
            [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)
            colors.append(pcd_colors[idx[0]])
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))

    mesh.compute_vertex_normals()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_mesh)), exist_ok=True)
    
    print(f"Saving generated mesh to: {output_mesh}")
    o3d.io.write_triangle_mesh(output_mesh, mesh)
    print("Mesh generation complete!")

    # Isaac Sim 바로 호환용 .usda Scene 파일 자동 생성
    usd_output_path = os.path.splitext(output_mesh)[0] + ".usd"
    obj_filename = os.path.basename(output_mesh)
    
    usda_content = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World"
{{
    def "DigitalTwinMesh" (
        references = @./{obj_filename}@
        apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform token physics:approximation = "triangleMesh"
        bool physics:collisionEnabled = 1
    }}
}}
"""
    try:
        with open(usd_output_path, "w") as f:
            f.write(usda_content)
        print(f"✨ Isaac Sim Direct-Ready USD Generated: {usd_output_path}")
    except Exception as e:
        print(f"⚠️ USD Scene 파일 생성 실패: {e}")

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
    parser = argparse.ArgumentParser(description="Generate 3D Mesh using Open3D from Point Cloud")
    parser.add_argument("input", help="Input .ply or .pcd point cloud file")
    parser.add_argument("output", help="Output mesh file (.obj or .ply)")
    parser.add_argument("--depth", type=int, default=8, help="Poisson reconstruction depth (default: 8)")
    parser.add_argument("--voxel", type=float, default=0.01, help="Voxel size for downsampling (default: 0.01)")
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
        view_result=args.view, 
        clean_density=not args.no_clean
    )

if __name__ == "__main__":
    main()
