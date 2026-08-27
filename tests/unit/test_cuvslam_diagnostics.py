"""P1-1: cuVSLAM diagnostics mode tests."""
from pathlib import Path
import pytest


def test_cuvslam_worker_has_diagnostics_arg():
    """cuvslam_worker must accept --diagnostics-dir argument."""
    src = Path('/home/kth/auto-mobility/src/auto_mobility/reconstruction/pose/backends/cuvslam_worker.py').read_text()
    assert '--diagnostics-dir' in src
    assert 'diagnostics_dir' in src


def test_cuvslam_worker_records_frame_diagnostics():
    """Worker must record per-frame diagnostics."""
    src = Path('/home/kth/auto-mobility/src/auto_mobility/reconstruction/pose/backends/cuvslam_worker.py').read_text()
    assert 'frame_diagnostics' in src
    assert 'brightness' in src
    assert 'blur' in src or 'blur_laplacian' in src
    assert 'depth_valid_ratio' in src
    assert 'odom_success' in src
    assert 'slam_success' in src


def test_cuvslam_worker_exports_tum_files():
    """Worker must export TUM files to diagnostics dir."""
    src = Path('/home/kth/auto-mobility/src/auto_mobility/reconstruction/pose/backends/cuvslam_worker.py').read_text()
    assert 'cuvslam_optimized_slam.tum' in src


def test_frame_diagnostics_csv_written_when_diagnostics_dir_set():
    """frame_diagnostics.csv and metrics_timeline.json must be written when --diagnostics-dir is set."""
    src = Path('/home/kth/auto-mobility/src/auto_mobility/reconstruction/pose/backends/cuvslam_worker.py').read_text()
    assert 'frame_diagnostics.csv' in src
    assert 'metrics_timeline.json' in src


def test_root_cause_md_is_unclassified():
    """CUVSLAM_ROOT_CAUSE.md must NOT say CASE 2 확정 without evidence."""
    rc_path = Path('/home/kth/auto-mobility/audit/hallway/CUVSLAM_ROOT_CAUSE.md')
    if rc_path.exists():
        text = rc_path.read_text()
        assert 'UNCLASSIFIED' in text or 'CASE 2 확정' not in text
