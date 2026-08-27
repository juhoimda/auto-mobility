#!/usr/bin/env python3
"""
view_pointcloud.py — 3D Point Cloud (.ply, .pcd, .pts) 전용 인터랙티브 뷰어
"""

import sys
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

# WSLg GPU 활성화 및 Wayland 비활성화 (Open3D GLEW/GLFW는 Wayland 네이티브에서 실패하므로 X11 강제)
for _k in ["MESA_LOADER_DRIVER_OVERRIDE", "LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER"]:
    os.environ.pop(_k, None)
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("Error: Open3D is not installed. Install via `pip install open3d numpy`!")
    sys.exit(1)


def view_pointcloud(pcd_path, point_size=3.0, show_normals=False, voxel_size=0.0):
    print(f"📂 Point Cloud 파일 로드 중: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    if pcd.is_empty() or len(pcd.points) == 0:
        print(f"❌ 오류: Point Cloud 파일이 비어있거나 읽을 수 없습니다: {pcd_path}")
        sys.exit(1)

    raw_points = len(pcd.points)
    if voxel_size > 0.0:
        print(f"🧹 Voxel 다운샘플링 적용 중 (voxel_size={voxel_size}m)...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    num_points = len(pcd.points)
    has_normals = pcd.has_normals()
    has_colors = pcd.has_colors()
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent().tolist()

    print("==========================================================")
    print(f" 📊 Point Cloud 구조 정보")
    print(f"  - 원본 점의 수 (Raw Points)   : {raw_points:,}")
    if voxel_size > 0.0:
        print(f"  - 필터 점의 수 (Voxel Points) : {num_points:,} (축소율: {num_points/max(raw_points, 1)*100:.1f}%)")
    print(f"  - Point Normals (법선 벡터)   : {'예' if has_normals else '아니오'}")
    print(f"  - Point Colors (RGB 색상)     : {'예' if has_colors else '아니오'}")
    print(f"  - Bounding Box 크기 (X/Y/Z)   : {extent[0]:.2f}m x {extent[1]:.2f}m x {extent[2]:.2f}m")
    print("==========================================================")
    print(f"🎨 Open3D Point Cloud Viewer 창을 엽니다... (Point Size: {point_size})")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Point Cloud Viewer - {os.path.basename(pcd_path)}",
        width=1280,
        height=720
    )
    
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = float(point_size)
        opt.point_show_normal = show_normals
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
    parser = argparse.ArgumentParser(description="View 3D Point Cloud (.ply, .pcd) using Open3D")
    parser.add_argument("input", help="Path to point cloud file (.ply, .pcd)")
    parser.add_argument("--point-size", type=float, default=3.0, help="Rendering point size (default: 3.0)")
    parser.add_argument("--show-normals", action="store_true", help="Display normal vectors")
    parser.add_argument("--voxel", type=float, default=0.0, help="Downsample voxel size in meters (e.g. 0.02)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 오류: 파일이 존재하지 않습니다: {args.input}")
        sys.exit(1)
        
    view_pointcloud(
        args.input,
        point_size=args.point_size,
        show_normals=args.show_normals,
        voxel_size=args.voxel
    )


if __name__ == "__main__":
    main()
