import unittest
import os
import tempfile
import numpy as np
import open3d as o3d
from auto_mobility.mesh import mesh_open3d, view_mesh

class TestMeshOpen3DUnit(unittest.TestCase):
    def test_mesh_open3d_exports(self):
        self.assertTrue(hasattr(mesh_open3d, "main"))
        self.assertTrue(hasattr(mesh_open3d, "generate_mesh"))

    def test_view_mesh_exports(self):
        self.assertTrue(hasattr(view_mesh, "main"))

    def test_generate_mesh_execution(self):
        # Create dummy point cloud with colors
        pcd = o3d.geometry.PointCloud()
        pts = np.random.rand(200, 3)
        colors = np.random.rand(200, 3)
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = os.path.join(tmpdir, "test.ply")
            obj_path = os.path.join(tmpdir, "test.obj")
            o3d.io.write_point_cloud(ply_path, pcd)
            
            # Execute generate_mesh
            mesh_open3d.generate_mesh(ply_path, obj_path, depth=6, voxel_size=0.05, method="poisson", clean_density=False)
            self.assertTrue(os.path.exists(obj_path))
            self.assertGreater(os.path.getsize(obj_path), 0)

if __name__ == "__main__":
    unittest.main()

