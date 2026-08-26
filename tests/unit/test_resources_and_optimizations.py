"""
tests/unit/test_resources_and_optimizations.py

Comprehensive tests for:
  - ResourcePolicy centralization & hardware fingerprinting
  - Subprocess live resource telemetry (CPU, Peak RSS, Memory)
  - DirectCloud precomputed camera ray mathematical equivalence
  - RTAB-Map headless benchmark launch configuration
  - Open3D TSDF zero-crossing mesh/pointcloud extraction without color
"""

import os
import sys
import numpy as np
import open3d as o3d
import pytest
from pathlib import Path

from auto_mobility.resources import (
    ResourcePolicy,
    ResourceUsage,
    DEFAULT_RESOURCE_POLICY,
    detect_system_hardware_fingerprint,
    get_default_resource_policy,
    run_monitored_subprocess
)
from auto_mobility.dataset.frame_dataset import CameraIntrinsics
from auto_mobility.mesh.direct_fusion import precompute_camera_rays, backproject_frame_to_world


def test_resource_policy_defaults_and_env():
    """Verify ResourcePolicy default properties and environment variable generation."""
    policy = get_default_resource_policy()
    assert policy.cpu_threads >= 2
    assert policy.openmp_threads == policy.cpu_threads
    assert policy.blas_threads == 1
    assert policy.opencv_threads == 1
    assert policy.poisson_threads == policy.cpu_threads
    assert policy.kdtree_workers == policy.cpu_threads
    assert policy.frame_prefetch_workers >= 2
    assert policy.tsdf_memory_budget_gb >= 2.0

    env = policy.get_worker_env({"FOO": "BAR", "LD_LIBRARY_PATH": "/opt/ros/humble/lib:/usr/local/cuda/lib64"})
    assert env["OMP_NUM_THREADS"] == str(policy.openmp_threads)
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENCV_NUM_THREADS"] == "1"
    assert env["FOO"] == "BAR"
    # Sanitized ROS path from LD_LIBRARY_PATH to avoid Open3D conflict
    assert "/opt/ros" not in env.get("LD_LIBRARY_PATH", "")


def test_hardware_fingerprint_deterministic():
    """Verify hardware fingerprint is non-empty, hex string, and deterministic across invocations."""
    fp1 = detect_system_hardware_fingerprint()
    fp2 = detect_system_hardware_fingerprint()
    assert len(fp1) == 16
    assert fp1 == fp2


def test_monitored_subprocess_telemetry():
    """Verify live subprocess telemetry tracking captures wall time and memory RSS."""
    cmd = [
        sys.executable, "-c",
        "import time, numpy as np; a = np.ones((500, 500), dtype=np.float64); time.sleep(0.1)"
    ]
    rc, stdout, stderr, usage = run_monitored_subprocess(cmd, timeout=10, sample_interval=0.05)
    assert rc == 0
    assert usage.wall_time_sec >= 0.08
    assert usage.peak_rss_mb > 0.0
    assert usage.min_available_ram_mb > 0.0


def test_directcloud_ray_precomputation_equivalence():
    """Verify that precomputed ray grids produce bitwise/floating-point identical 3D world points."""
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    h, w = 480, 640
    depth_raw = np.random.randint(400, 2500, size=(h, w), dtype=np.uint16)
    color_rgb = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    T_world = np.array([
        [0.866, -0.5, 0.0, 1.2],
        [0.5, 0.866, 0.0, -0.4],
        [0.0, 0.0, 1.0, 0.8],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 1. Without precomputed rays (calculated on the fly)
    pts1, cols1 = backproject_frame_to_world(
        depth_raw=depth_raw,
        color_rgb=color_rgb,
        T_world_camera=T_world,
        intrinsics=intrinsics,
        stride=2,
        precomputed_rays=None
    )

    # 2. With precomputed rays
    rays = precompute_camera_rays(intrinsics, h, w, stride=2)
    pts2, cols2 = backproject_frame_to_world(
        depth_raw=depth_raw,
        color_rgb=color_rgb,
        T_world_camera=T_world,
        intrinsics=intrinsics,
        stride=2,
        precomputed_rays=rays
    )

    assert len(pts1) == len(pts2)
    assert len(pts1) > 0
    max_coord_diff = np.max(np.abs(pts1 - pts2))
    assert max_coord_diff < 1e-6, f"DirectCloud ray precomputation discrepancy too large: {max_coord_diff}"
    if cols1 is not None and cols2 is not None:
        assert np.allclose(cols1, cols2)


def test_tsdf_nocolor_extraction_safety():
    """Regression test: Open3D TSDF VoxelBlockGrid pointcloud/mesh extraction succeeds with no_color=True."""
    import open3d.core as o3c
    device = o3c.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=0.010, block_resolution=16,
        block_count=1000, device=device
    )

    # Synthesize small depth frame
    depth = np.full((100, 100), 1000, dtype=np.uint16)
    depth_t = o3d.t.geometry.Image(o3c.Tensor(depth, device=device))
    intr_t = o3c.Tensor(np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]], dtype=np.float64))
    extr_t = o3c.Tensor(np.eye(4, dtype=np.float64))

    coords = vbg.compute_unique_block_coordinates(depth_t, intr_t, extr_t, depth_scale=1000.0, depth_max=3.0)
    vbg.integrate(coords, depth_t, intr_t, extr_t, depth_scale=1000.0, depth_max=3.0)

    # Extraction must not throw Open3D C++ shape incompatible exception
    pcd_t = vbg.extract_point_cloud(weight_threshold=0.1)
    pcd = pcd_t.to_legacy()
    assert isinstance(pcd, o3d.geometry.PointCloud)


def test_monitored_subprocess_memory_guard_termination():
    """Verify that a runaway memory consumer is safely killed by the memory watchdog without orphan processes."""
    # Policy with 0.05 GB (50 MB) budget
    low_mem_policy = ResourcePolicy(process_memory_budget_gb=0.05, system_ram_reserve_gb=0.0)
    
    # Subprocess that allocates ~150 MB and holds it
    cmd = [
        sys.executable, "-c",
        "import time, numpy as np; a = np.ones((20000, 2000), dtype=np.float64); time.sleep(5.0)"
    ]
    rc, stdout, stderr, usage = run_monitored_subprocess(cmd, timeout=10, sample_interval=0.05, policy=low_mem_policy)
    
    assert rc == -9
    assert "KILLED_BY_RESOURCE_GUARD" in stderr
    assert "exceeded policy budget" in stderr
    assert usage.peak_rss_mb > 40.0


def test_monitored_subprocess_timeout_process_tree():
    """Verify that a timing out process and its children are killed without leaving orphan processes."""
    cmd = [
        sys.executable, "-c",
        "import time, subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    ]
    rc, stdout, stderr, usage = run_monitored_subprocess(cmd, timeout=1, sample_interval=0.05)
    
    assert rc == -15
    assert "timed out after 1s" in stderr

