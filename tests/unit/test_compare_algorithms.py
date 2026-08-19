import unittest
import os
import json
import tempfile
from pathlib import Path

from auto_mobility.slam.compare_algorithms import get_git_commit, get_system_hardware_info, get_software_info, _generate_modular_markdown_report


class TestCompareAlgorithmsUnit(unittest.TestCase):
    def test_system_info_queries(self):
        commit = get_git_commit()
        self.assertIsInstance(commit, str)
        
        hw = get_system_hardware_info()
        self.assertIn("cpu_count", hw)
        self.assertIn("ram_total_mb", hw)
        self.assertIn("gpu_name", hw)
        
        sw = get_software_info()
        self.assertIn("open3d", sw)
        self.assertIn("python", sw)
        self.assertIn("ros_distro", sw)

    def test_generate_modular_markdown_report(self):
        dummy_manifest = {
            "benchmark_id": "test_bench_001",
            "bag_name": "test_bag",
            "evaluated_at": "2026-08-19_120000",
            "hardware": {"cpu_count": 8, "ram_total_mb": 16000, "gpu_name": "RTX GPU", "vram_total_mb": 8000},
            "software": {"ros_distro": "humble", "open3d": "0.19.0", "python": "3.10.12", "git_commit": "abcdef1"},
            "phase_a_slam_results": [
                {
                    "candidate_name": "rtab_rgbd",
                    "trajectory_metrics": {"num_frames": 100},
                    "geometry": {"depth_mae_mm": 12.5, "depth_p95_mm": 24.1, "depth_coverage_ratio": 0.95, "within_20mm_ratio": 0.90},
                    "runtime_sec": 1.5
                }
            ],
            "phase_b_tsdf_results": [
                {
                    "candidate_name": "tsdf_10mm",
                    "voxel_size_m": 0.010,
                    "geometry": {"depth_mae_mm": 11.2, "depth_p95_mm": 22.0, "depth_coverage_ratio": 0.96},
                    "mesh": {"num_triangles": 54000},
                    "runtime_sec": 2.1
                }
            ],
            "phase_c_surface_results": [
                {
                    "candidate_name": "tsdf_direct",
                    "geometry": {"depth_mae_mm": 11.2, "point_to_mesh_p95_mm": 15.0, "depth_coverage_ratio": 0.96},
                    "mesh": {"non_manifold_edges": 0, "num_triangles": 54000},
                    "runtime_sec": 0.5
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "report.md"
            _generate_modular_markdown_report(dummy_manifest, md_path)
            self.assertTrue(md_path.exists())
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("[SLAM Ranking]", content)
            self.assertIn("[TSDF Ranking]", content)
            self.assertIn("[Surface Ranking]", content)
            self.assertIn("rtab_rgbd", content)
            self.assertIn("tsdf_10mm", content)


if __name__ == "__main__":
    unittest.main()
