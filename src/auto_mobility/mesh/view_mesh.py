#!/usr/bin/env python3
import sys
import os
import argparse

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("Error: Open3D is not installed. Install via `pip install open3d numpy`!")
    sys.exit(1)

def view_mesh(mesh_path, wireframe=False, back_face=True):
    print(f"📂 Mesh 파일 로드 중: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    if mesh.is_empty():
        print(f"❌ 오류: Mesh 파일이 비어있거나 읽을 수 없습니다: {mesh_path}")
        sys.exit(1)
        
    num_vertices = len(mesh.vertices)
    num_triangles = len(mesh.triangles)
    has_normals = mesh.has_vertex_normals()
    has_colors = mesh.has_vertex_colors()
    
    print("==========================================================")
    print(f" 📊 Mesh 구조 정보")
    print(f"  - Vertices  (정점 수) : {num_vertices:,}")
    print(f"  - Triangles (삼각형 수): {num_triangles:,}")
    print(f"  - Vertex Normals      : {'예' if has_normals else '아니오 (자동 계산 적용)'}")
    print(f"  - Vertex Colors       : {'예' if has_colors else '아니오'}")
    print("==========================================================")

    if not has_normals:
        mesh.compute_vertex_normals()

    print("🎨 Open3D 3D Mesh Interactive Viewer 창을 엽니다... (창을 닫으면 종료됩니다)")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"3D Mesh Viewer - {os.path.basename(mesh_path)}",
        width=1280,
        height=720
    )
    
    vis.add_geometry(mesh)
    
    opt = vis.get_render_option()
    if opt is not None:
        opt.mesh_show_back_face = back_face
        opt.mesh_show_wireframe = wireframe
        opt.background_color = np.array([0.1, 0.1, 0.1])
    
    try:
        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        print("\nViewer closed by Ctrl+C")

    finally:
        vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="View 3D Mesh (.obj, .ply, .stl, etc.) using Open3D")
    parser.add_argument("input", help="Path to 3D mesh file (.obj, .ply, .stl)")
    parser.add_argument("--wireframe", action="store_true", help="Display mesh in wireframe mode")
    parser.add_argument("--no-backface", action="store_true", help="Disable back-face rendering")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 오류: 파일이 존재하지 않습니다: {args.input}")
        sys.exit(1)
        
    view_mesh(args.input, wireframe=args.wireframe, back_face=not args.no_backface)

if __name__ == "__main__":
    main()
