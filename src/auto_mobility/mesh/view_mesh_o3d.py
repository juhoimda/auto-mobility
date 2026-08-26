"""
Thin wrapper - view_mesh.py 와 동일 로직 공유 (중복 방지)
Usage:
  python3 src/auto_mobility/mesh/view_mesh_o3d.py ros2_data/meshes/base3_rtab_reconstructed.obj
"""
import os
import sys

# 패키지/스크립트 두 경우 모두 동작하도록 경로 보정
sys.path.insert(0, os.path.dirname(__file__))
try:
    from view_mesh import main
except ImportError:
    from auto_mobility.mesh.view_mesh import main  # type: ignore

if __name__ == "__main__":
    main()
