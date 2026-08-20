"""
tests/unit/test_benchmark_workers.py

Unit tests for candidate subprocess crash isolation:
  - SIGSEGV (-11 / 139) isolation
  - OOM (-9 / 137) isolation
  - Timeout isolation
  - SKIPPED_UNAVAILABLE handling for missing dependencies (CGAL)
"""

import subprocess
import pytest
from unittest.mock import patch, MagicMock

from auto_mobility.benchmark.workers import (
    WorkerStatus,
    WorkerResult,
    _classify_returncode,
    run_tsdf_worker,
    run_surface_worker
)


def test_classify_returncode():
    assert _classify_returncode(0) == WorkerStatus.SUCCESS
    assert _classify_returncode(-11) == WorkerStatus.FAIL_SEGFAULT
    assert _classify_returncode(139) == WorkerStatus.FAIL_SEGFAULT
    assert _classify_returncode(-9) == WorkerStatus.FAIL_OOM
    assert _classify_returncode(137) == WorkerStatus.FAIL_OOM
    assert _classify_returncode(2) == WorkerStatus.SKIPPED_UNAVAILABLE
    assert _classify_returncode(1, stderr="Segmentation fault (core dumped)") == WorkerStatus.FAIL_SEGFAULT
    assert _classify_returncode(1, stderr="std::bad_alloc (Out of memory)") == WorkerStatus.FAIL_OOM
    assert _classify_returncode(1, stderr="Some general python error") == WorkerStatus.FAIL_EXCEPTION


@patch("subprocess.run")
def test_segfault_isolation_in_tsdf_worker(mock_subproc):
    # Mock subprocess crashing with SIGSEGV (returncode -11)
    mock_subproc.return_value = MagicMock(returncode=-11, stdout="", stderr="Fatal: Segmentation fault (SIGSEGV)")
    
    result = run_tsdf_worker(
        dataset_dir="/fake/dataset",
        traj_file="/fake/traj.txt",
        mesh_path="/fake/mesh.obj",
        pcd_path="/fake/cloud.ply",
        voxel=0.010
    )
    
    assert result.status == WorkerStatus.FAIL_SEGFAULT
    assert not result.is_success
    assert result.returncode == -11
    assert "Segmentation fault" in (result.error_message or "")


@patch("subprocess.run")
def test_oom_isolation_in_tsdf_worker(mock_subproc):
    # Mock subprocess killed by OOM killer (returncode -9)
    mock_subproc.return_value = MagicMock(returncode=-9, stdout="", stderr="Killed")
    
    result = run_tsdf_worker(
        dataset_dir="/fake/dataset",
        traj_file="/fake/traj.txt",
        mesh_path="/fake/mesh.obj",
        pcd_path="/fake/cloud.ply",
        voxel=0.005
    )
    
    assert result.status == WorkerStatus.FAIL_OOM
    assert not result.is_success
    assert result.returncode == -9


@patch("subprocess.run")
def test_timeout_isolation_in_tsdf_worker(mock_subproc):
    # Mock subprocess timing out
    mock_subproc.side_effect = subprocess.TimeoutExpired(cmd=["python3", "worker.py"], timeout=10)
    
    result = run_tsdf_worker(
        dataset_dir="/fake/dataset",
        traj_file="/fake/traj.txt",
        mesh_path="/fake/mesh.obj",
        voxel=0.010,
        timeout=10
    )
    
    assert result.status == WorkerStatus.FAIL_TIMEOUT
    assert not result.is_success
    assert "Timed out" in (result.error_message or "")


@patch("auto_mobility.benchmark.workers.is_cgal_available", return_value=(False, "CGAL binary not found"))
def test_cgal_skipped_unavailable_when_missing(mock_cgal):
    result = run_surface_worker(
        input_ply="/fake/cloud.ply",
        output_mesh="/fake/mesh.obj",
        method="cgal_polygonal"
    )
    assert result.status == WorkerStatus.SKIPPED_UNAVAILABLE
    assert result.returncode == 2
    assert not result.is_success
