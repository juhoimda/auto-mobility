#!/usr/bin/env python3
"""
view_mesh.py — 3D Mesh (.obj, .stl) 및 Point Cloud (.ply, .pcd) 통합 인터랙티브 뷰어
"""

import sys
import os
import argparse

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("Error: Open3D is not installed. Install via `pip install open3d numpy`!")
    sys.exit(1)


def view_geometry(file_path, wireframe=False, back_face=True, simplify_triangles=None, full_res=False):
    # 환경변수에서 Wayland 제거 보장
    os.environ.pop("WAYLAND_DISPLAY", None)
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    print(f"📂 3D 파일 로드 중: {file_path}")
    
    # 1. First try loading as TriangleMesh
    mesh = o3d.io.read_triangle_mesh(file_path)
    is_mesh = not mesh.is_empty() and len(mesh.triangles) > 0

    geom = None
    if is_mesh:
        num_vertices = len(mesh.vertices)
        num_triangles = len(mesh.triangles)
        has_normals = mesh.has_vertex_normals()
        has_colors = mesh.has_vertex_colors()

        print("==========================================================")
        print(f" 📊 3D Mesh 구조 정보")
        print(f"  - 타입                : TriangleMesh (3D 면 메쉬)")
        print(f"  - Vertices  (정점 수) : {num_vertices:,}")
        print(f"  - Triangles (삼각형 수): {num_triangles:,}")
        print(f"  - Vertex Normals      : {'예' if has_normals else '아니오 (자동 계산 적용)'}")
        print(f"  - Vertex Colors       : {'예' if has_colors else '아니오'}")
        print("==========================================================")

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        geom = mesh
    else:
        # 2. Fallback: Load as PointCloud
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty() or len(pcd.points) == 0:
            print(f"❌ 오류: 파일이 비어있거나 지원되지 않는 형식입니다: {file_path}")
            sys.exit(1)

        num_points = len(pcd.points)
        has_normals = pcd.has_normals()
        has_colors = pcd.has_colors()

        print("==========================================================")
        print(f" 📊 3D Point Cloud 구조 정보")
        print(f"  - 타입                : PointCloud (3D 점군)")
        print(f"  - Points (점의 수)    : {num_points:,}")
        print(f"  - Normals             : {'예' if has_normals else '아니오'}")
        print(f"  - Colors              : {'예' if has_colors else '아니오'}")
        print("==========================================================")
        geom = pcd

    print("🎨 3D Interactive Viewer 창을 엽니다... (마우스 회전/줌, 창을 닫거나 Q/Ctrl+C로 종료)")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"3D Viewer - {os.path.basename(file_path)}",
        width=1280,
        height=720,
        visible=True
    )
    
    vis.add_geometry(geom)
    
    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.array([0.15, 0.15, 0.15])
        if is_mesh:
            opt.mesh_show_back_face = back_face
            opt.mesh_show_wireframe = wireframe
        else:
            opt.point_size = 3.0
            opt.point_show_normal = False
    
    import time
    import signal

    running = True

    def sigint_handler(signum, frame):
        nonlocal running
        running = False
        print("\n👋 뷰어를 종료합니다 (Ctrl+C)...")

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        while running:
            if not vis.poll_events():
                break
            vis.update_renderer()
            time.sleep(0.01)
    finally:
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="View 3D Mesh or Point Cloud (.obj, .ply, .pcd, .stl) using Open3D")
    parser.add_argument("input", help="Path to 3D file (.obj, .ply, .pcd, .stl)")
    parser.add_argument("--wireframe", action="store_true", help="Display mesh in wireframe mode")
    parser.add_argument("--no-backface", action="store_true", help="Disable back-face rendering")
    parser.add_argument("--simplify", type=int, default=None, help="Target number of triangles for simplification (default: 300,000 for large meshes)")
    parser.add_argument("--full-res", action="store_true", help="Disable auto-simplification and render full resolution mesh")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 오류: 파일이 존재하지 않습니다: {args.input}")
        sys.exit(1)
        
    view_geometry(
        args.input,
        wireframe=args.wireframe,
        back_face=not args.no_backface,
        simplify_triangles=args.simplify,
        full_res=args.full_res
    )


if __name__ == "__main__":
    main()

