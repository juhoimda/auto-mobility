#!/usr/bin/env python3

import sys
import os
import argparse

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("Error: Open3D is not installed")
    sys.exit(1)


def generate_mesh(
    input_ply,
    output_mesh,
    depth=8,
    voxel_size=0.01,
    view_result=False,
    clean_density=True
):

    print(f"Loading point cloud: {input_ply}")

    pcd = o3d.io.read_point_cloud(input_ply)

    if pcd.is_empty():
        print("❌ Error: Point cloud is empty")
        sys.exit(1)


    # Down sampling
    print("Downsampling point cloud...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)


    # Outlier 제거
    print("Removing outliers...")
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    pcd = pcd.select_by_index(ind)


    # Normal 계산
    print("Estimating normals...")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,
            max_nn=30
        )
    )

    pcd.orient_normals_consistent_tangent_plane(10)


    # Poisson reconstruction
    print(
        f"Poisson reconstruction "
        f"(depth={depth})"
    )

    mesh, densities = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_poisson(
            pcd,
            depth=depth
        )
    )


    # Density 낮은 noise 제거
    if clean_density:

        print("Removing low density artifacts...")

        densities = np.asarray(densities)

        threshold = np.quantile(
            densities,
            0.05
        )

        remove_mask = densities < threshold

        mesh.remove_vertices_by_mask(
            remove_mask
        )


    # Point cloud 영역 밖 제거
    bbox = pcd.get_axis_aligned_bounding_box()

    mesh = mesh.crop(bbox)


    # Color transfer
    if pcd.has_colors():

        print("Transferring colors...")

        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        vertices = np.asarray(mesh.vertices)

        colors = []

        pcd_colors = np.asarray(
            pcd.colors
        )

        for v in vertices:

            _, idx, _ = (
                pcd_tree.search_knn_vector_3d(
                    v,
                    1
                )
            )

            colors.append(
                pcd_colors[idx[0]]
            )


        mesh.vertex_colors = (
            o3d.utility.Vector3dVector(
                np.array(colors)
            )
        )


    mesh.compute_vertex_normals()


    # 저장 폴더 생성
    os.makedirs(
        os.path.dirname(
            os.path.abspath(output_mesh)
        ),
        exist_ok=True
    )


    print(f"Saving mesh: {output_mesh}")

    o3d.io.write_triangle_mesh(
        output_mesh,
        mesh,
        write_vertex_colors=True
    )


    print("✅ OBJ mesh generation complete")


    if view_result:

        o3d.visualization.draw_geometries(
            [mesh],
            window_name="Generated Mesh",
            width=1280,
            height=720,
            mesh_show_back_face=True
        )



def main():

    parser = argparse.ArgumentParser(
        description="Generate OBJ mesh from point cloud"
    )


    parser.add_argument(
        "input",
        help="Input point cloud (.ply)"
    )


    parser.add_argument(
        "output",
        help="Output mesh (.obj)"
    )


    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Poisson reconstruction depth"
    )


    parser.add_argument(
        "--voxel",
        type=float,
        default=0.01,
        help="Voxel downsampling size"
    )


    parser.add_argument(
        "--view",
        action="store_true",
        help="Show mesh viewer"
    )


    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable density cleaning"
    )


    args = parser.parse_args()


    if not os.path.exists(args.input):

        print(
            f"❌ Input file not found: {args.input}"
        )

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