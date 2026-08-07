import unittest

class TestDomainImportsIntegration(unittest.TestCase):
    def test_import_all_domain_modules(self):
        # 1. Config
        import auto_mobility.config as config
        self.assertTrue(hasattr(config, "TOPICS_CONFIG"))

        # 2. Launch
        import auto_mobility.launch.launch_common as lc
        self.assertTrue(hasattr(lc, "RTABMAP_ARGS"))
        self.assertTrue(hasattr(lc, "create_republish_node"))
        self.assertTrue(hasattr(lc, "create_imu_filter_node"))

        # 3. Nodes
        import auto_mobility.nodes.republish as rep
        self.assertTrue(hasattr(rep, "CompressedRepublisher"))

        # 4. Mesh Domain
        import auto_mobility.mesh.mesh_open3d
        import auto_mobility.mesh.view_mesh

        # 5. Isaac Domain
        import auto_mobility.isaac.load_isaac_mesh

        # 6. SLAM Domain
        import auto_mobility.slam.benchmark_slam

        # 7. Utils Domain
        import auto_mobility.utils.validate
        import auto_mobility.utils.benchmark_hw
        import auto_mobility.utils.inspect_system

if __name__ == "__main__":
    unittest.main()
