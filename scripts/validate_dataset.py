#!/usr/bin/env python3
import sys
import os
import argparse

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("Error: Open3D or NumPy is not installed. Install via `pip install open3d numpy`")
    sys.exit(1)

def validate_db(db_path):
    print(f"\n--- [1/2] Database (.db) File Status ---")
    if not os.path.exists(db_path):
        print(f"❌ Error: Database file does not exist: {db_path}")
        return False
    
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"📁 DB Path : {db_path}")
    print(f"📊 File Size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 5.0:
        print("⚠️  [경고] DB 파일 크기가 5MB 미만으로 매우 작습니다. SLAM 매핑이 정상적으로 수행되지 않았을 수 있습니다.")
    else:
        print("✅ DB 파일 용량이 양호합니다 (> 5MB).")
    return True

def validate_pointcloud(ply_path):
    print(f"\n--- [2/2] Point Cloud (.ply) Quality & Scale Check ---")
    if not os.path.exists(ply_path):
        print(f"❌ Error: PLY file does not exist: {ply_path}")
        return False
    
    pcd = o3d.io.read_point_cloud(ply_path)
    num_points = len(pcd.points)
    print(f"📌 Total Points : {num_points:,} 개")
    
    if num_points < 10000:
        print("❌ [부적합] 포인트 수가 10,000개 미만입니다. Mesh 생성 시 껍데기만 남거나 형상을 얻기 어렵습니다.")
        return False
    elif num_points < 50000:
        print("⚠️  [주의] 포인트 수가 50,000개 미만으로 적습니다. Mesh 세부 표현력이 떨어질 수 있습니다.")
    else:
        print("✅ 포인트 밀도가 충분합니다 (> 50,000개).")
        
    # Check RGB Colors
    has_colors = pcd.has_colors()
    if has_colors:
        print("✅ RGB 컬러 데이터: 포함됨")
    else:
        print("⚠️  RGB 컬러 데이터: 없음 (흑백 Mesh로 생성됨)")
        
    # Check Bounding Box / Dimensions
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent() # [dx, dy, dz]
    max_dim = max(extent)
    print(f"📐 공간 규격 (Bounding Box) : {extent[0]:.2f}m (가로) x {extent[1]:.2f}m (세로) x {extent[2]:.2f}m (높이)")
    print(f"📏 최대 공간 스팬 : {max_dim:.2f} m")
    
    if max_dim > 30.0:
        print("⚠️  [경고] 공간 크기가 30m를 초과합니다! Visual Odometry가 이탈(Drift)하여 지도 궤적이 공중에 튄 자국이 있는지 확인하세요.")
    elif max_dim < 0.3:
        print("⚠️  [경고] 공간 크기가 0.3m 미만입니다. 매핑 스케일 오류를 점검하세요.")
    else:
        print("✅ 실내 공간 규격(크기)이 수긍 가능한 범위 내에 있습니다.")
        
    # Check Outlier Ratio
    print("🔍 노이즈 비율 분석 (Statistical Outlier Detection)...")
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_count = len(ind)
    noise_ratio = (1.0 - (inlier_count / num_points)) * 100.0
    print(f"🧹 이상치(노이즈) 비율: {noise_ratio:.1f}% (약 {num_points - inlier_count:,} 개 포인트가 노이즈로 감지됨)")
    
    if noise_ratio > 30.0:
        print("⚠️  [주의] 노이즈 비율이 30%를 넘습니다. Mesh 표면에 허공 아티팩트가 생성될 수 있습니다.")
    else:
        print("✅ 노이즈 비율이 정상 범위(30% 이하)입니다.")
        
    print("\n==================================================")
    if num_points >= 10000 and max_dim <= 40.0:
        print("🎉 [데이터 검증 성공] Mesh 생성을 진행해도 좋은 고품질 데이터입니다!")
        return True
    else:
        print("⚠️  [데이터 검증 주의] 위 경고 항목을 확인 후 Mesh 생성을 진행하세요.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate Database and PointCloud quality before Mesh generation")
    parser.add_argument("--db", help="Path to rtabmap database (.db)")
    parser.add_argument("--ply", help="Path to point cloud (.ply)")
    
    args = parser.parse_args()
    
    if not args.db and not args.ply:
        print("Usage: python3 validate_dataset.py --db DB_PATH or --ply PLY_PATH")
        sys.exit(1)
        
    success = True
    if args.db:
        if not validate_db(args.db):
            success = False
            
    if args.ply:
        if not validate_pointcloud(args.ply):
            success = False
            
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
