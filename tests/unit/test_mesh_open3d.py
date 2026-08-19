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

    def test_alpha_shape_reconstruction(self):
        # Create synthetic sphere point cloud
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        pcd = mesh_sphere.sample_points_poisson_disk(number_of_points=300)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = os.path.join(tmpdir, "sphere.ply")
            obj_path = os.path.join(tmpdir, "sphere_alpha.obj")
            o3d.io.write_point_cloud(ply_path, pcd)
            
            # Execute generate_mesh with alpha_shape
            mesh_open3d.generate_mesh(
                ply_path, obj_path, method="alpha_shape", voxel_size=0.02, alpha_factor=3.0
            )
            self.assertTrue(os.path.exists(obj_path))
            self.assertGreater(os.path.getsize(obj_path), 0)

            # Test explicit alpha parameter
            obj_path_alpha = os.path.join(tmpdir, "sphere_explicit_alpha.obj")
            mesh_open3d.generate_mesh(
                ply_path, obj_path_alpha, method="alpha", alpha=0.3
            )
            self.assertTrue(os.path.exists(obj_path_alpha))

    def test_alpha_shape_empty_handling(self):
        # Sparse point cloud where very tiny alpha would produce empty mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float64))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = os.path.join(tmpdir, "sparse.ply")
            obj_path = os.path.join(tmpdir, "sparse_alpha.obj")
            o3d.io.write_point_cloud(ply_path, pcd)
            
            # Should handle empty gracefully without crash
            mesh_open3d.generate_mesh(
                ply_path, obj_path, method="alpha_shape", alpha=0.0001
            )
            self.assertTrue(os.path.exists(obj_path))

    def test_cgal_polygonal_surface_reconstruction(self):
        from auto_mobility.mesh.cgal_surface import reconstruct_cgal_polygonal, is_cgal_available
        
        # Synthetic box/cube room point cloud
        box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
        pcd = box.sample_points_poisson_disk(number_of_points=500)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = os.path.join(tmpdir, "cgal_room.obj")
            mesh = reconstruct_cgal_polygonal(pcd, obj_path)
            self.assertTrue(os.path.exists(obj_path))
            self.assertGreater(len(mesh.triangles), 0)
            self.assertGreater(len(mesh.vertices), 0)

            # Empty cloud handling
            empty_pcd = o3d.geometry.PointCloud()
            with self.assertRaises(ValueError):
                reconstruct_cgal_polygonal(empty_pcd, os.path.join(tmpdir, "empty.obj"))


if __name__ == "__main__":
    unittest.main()

