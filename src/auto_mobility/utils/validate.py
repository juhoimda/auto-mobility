#!/usr/bin/env python3
"""
Database, PointCloud, and Mesh Integrity Validator.
Checks file sizes, SQLite table integrity, vertex density, bounding box scale, and color channels.
"""

import sys
import os
import argparse
import sqlite3

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("❌ Error: Open3D or NumPy is not installed. Install via `pip install open3d numpy`")
    sys.exit(1)

def validate_db(db_path):
    print(f"\n--- [1/3] Database (.db) File Integrity Check ---")
    if not os.path.exists(db_path):
        print(f"❌ Error: Database file does not exist: {db_path}")
        return False
    
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"📁 DB Path   : {db_path}")
    print(f"📊 File Size : {file_size_mb:.2f} MB")
    
    if file_size_mb < 0.1:
        print("❌ [오류] DB 파일 용량이 100KB 미만입니다. 데이터가 비어있거나 생성이 중단되었습니다.")
        return False
    elif file_size_mb < 5.0:
        print("⚠️  [경고] DB 파일 크기가 5MB 미만으로 작습니다. 매핑 데이터가 부족할 수 있습니다.")
    else:
        print("✅ DB 파일 용량이 정상 범위입니다 (> 5MB).")

    # SQLite3 Integrity Check
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        conn.close()
        if result and result[0] == "ok":
            print("✅ SQLite3 데이터베이스 구조 무결성 검사: 정상 (ok)")
        else:
            print(f"❌ [오류] SQLite3 데이터베이스 손상 감지: {result}")
            return False
    except Exception as e:
        print(f"❌ [오류] DB 읽기 중 예외 발생: {e}")
        return False

    return True

def validate_pointcloud(ply_path):
    print(f"\n--- [2/3] Point Cloud (.ply) Quality & Scale Check ---")
    if not os.path.exists(ply_path):
        print(f"❌ Error: PLY file does not exist: {ply_path}")
        return False
    
    file_size_kb = os.path.getsize(ply_path) / 1024
    if file_size_kb < 1.0:
        print(f"❌ [오류] Point Cloud 파일 용량이 1KB 미만입니다. 빈 파일일 수 있습니다 ({file_size_kb:.2f} KB).")
        return False

    pcd = o3d.io.read_point_cloud(ply_path)
    num_points = len(pcd.points)
    print(f"📌 Total Points : {num_points:,} 개")
    
    if num_points < 100:
        print("❌ [오류] 포인트 수가 100개 미만으로 극도로 적습니다. Mesh 생성이 불가능합니다.")
        return False
    elif num_points < 10000:
        print("⚠️  [주의] 포인트 수가 10,000개 미만입니다. Mesh 품질이 저하될 수 있습니다.")
    else:
        print("✅ 포인트 밀도가 충분합니다 (> 10,000개).")
        
    # Check RGB Colors
    has_colors = pcd.has_colors()
    if has_colors:
        print("✅ RGB 컬러 데이터: 정상 포함됨")
    else:
        print("⚠️  RGB 컬러 데이터: 없음 (흑백 또는 단색 Mesh로 변환됨)")
        
    # Check Bounding Box / Dimensions
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent() # [dx, dy, dz]
    max_dim = max(extent)
    print(f"📐 공간 규격 (Bounding Box) : {extent[0]:.2f}m x {extent[1]:.2f}m x {extent[2]:.2f}m")
    
    if max_dim > 50.0:
        print("⚠️  [경고] 최대 공간 스팬이 50m를 초과합니다. Visual Odometry 드리프트 현상을 확인하세요.")
    elif max_dim < 0.1:
        print("⚠️  [경고] 공간 크기가 0.1m 미만입니다. 매핑 스케일 오류일 가능성이 있습니다.")
    else:
        print("✅ 공간 규격(크기) 범위 정상.")
        
    return True

def validate_mesh(mesh_path):
    print(f"\n--- [3/3] 3D Mesh (.obj/.ply) Integrity Check ---")
    if not os.path.exists(mesh_path):
        print(f"❌ Error: Mesh file does not exist: {mesh_path}")
        return False
        
    file_size_kb = os.path.getsize(mesh_path) / 1024
    if file_size_kb < 1.0:
        print(f"❌ [오류] Mesh 파일 용량이 1KB 미만입니다 ({file_size_kb:.2f} KB).")
        return False

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    num_vertices = len(mesh.vertices)
    num_triangles = len(mesh.triangles)
    
    print(f"🔺 Vertices  : {num_vertices:,} 개")
    print(f"📐 Triangles : {num_triangles:,} 개")
    
    if num_vertices == 0 or num_triangles == 0:
        print("❌ [오류] Mesh의 정점(Vertex) 또는 삼각형(Triangle) 수가 0개입니다.")
        return False

    # ===== Mesh 품질 메트릭 (2026-08-10 추가) =====
    # 1. 공간 규모 확인
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_dim = max(extent)
    print(f"📐 Mesh 공간 규격 : {extent[0]:.2f}m x {extent[1]:.2f}m x {extent[2]:.2f}m")
    if max_dim < 0.3:
        print("❌ [오류] Mesh 공간 크기가 0.3m 미만으로 너무 작습니다 (데이터 수집 실패 가능성).")
        return False
    elif max_dim < 1.0:
        print("⚠️  [경고] Mesh 공간 크기가 1m 미만으로 작습니다. 더 넓은 공간을 촬영하세요.")
    else:
        print("✅ Mesh 공간 규모 정상.")

    # 2. 삼각형 밀도 (표면 품질 지표)
    if max_dim > 0:
        area_m2 = 0.0
        try:
            area_m2 = mesh.get_surface_area()
        except Exception:
            pass
        density = num_triangles / max(area_m2, 1e-6) if area_m2 > 0 else 0
        print(f"📊 표면적 : {area_m2:.2f} m²  |  삼각형 밀도 : {density:.0f} tri/m²")
        if area_m2 < 0.01:
            print("⚠️  [경고] 표면적이 0.01m² 미만입니다. 빈 껍질(empty shell)일 가능성이 있습니다.")

    # 3. Watertight (폐곡면) 여부 — Poisson 복원 핵심 지표
    try:
        watertight = mesh.is_watertight()
        print(f"🔒 Watertight(폐곡면) : {'✅ 예' if watertight else '⚠️ 아니오 (구멍 존재)'}")
    except Exception:
        watertight = None
        print("🔒 Watertight 검사 불가 (오래된 Open3D 버전).")

    # 4. 컬러 여부
    if mesh.has_vertex_colors():
        print("✅ Mesh Vertex Colors: 정상 포함됨")
    else:
        print("⚠️  Mesh Vertex Colors: 없음 (단색 Mesh)")

    print("✅ 3D Mesh 무결성 및 구조 검사 통과!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate Database, PointCloud, and Mesh integrity")
    parser.add_argument("--db", help="Path to rtabmap database (.db)")
    parser.add_argument("--ply", help="Path to point cloud (.ply)")
    parser.add_argument("--mesh", help="Path to 3D mesh (.obj/.ply)")
    
    args = parser.parse_args()
    
    if not args.db and not args.ply and not args.mesh:
        print("Usage: python3 validate.py [--db DB_PATH] [--ply PLY_PATH] [--mesh MESH_PATH]")
        sys.exit(1)
        
    success = True
    if args.db and not validate_db(args.db):
        success = False
            
    if args.ply and not validate_pointcloud(args.ply):
        success = False

    if args.mesh and not validate_mesh(args.mesh):
        success = False

    print("\n==================================================")
    if success:
        print("🎉 [무결성 검증 성공] 모든 데이터 검증 장벽(Barrier)을 통과했습니다.")
    else:
        print("❌ [무결성 검증 실패] 파이프라인 전이 조건 미달입니다.")
    print("==================================================")
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
