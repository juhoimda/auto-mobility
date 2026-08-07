import unittest
from auto_mobility.mesh import mesh_open3d, view_mesh

class TestMeshOpen3DUnit(unittest.TestCase):
    def test_mesh_open3d_exports(self):
        self.assertTrue(hasattr(mesh_open3d, "main"))

    def test_view_mesh_exports(self):
        self.assertTrue(hasattr(view_mesh, "main"))

if __name__ == "__main__":
    unittest.main()
