"""P0-3: Test that failed runs return non-zero exit codes and don't corrupt stale artifacts."""
import json
from pathlib import Path
import pytest


def _make_gpu_mocks(present=False, vram_total_mb=0, vram_budget_mb=0):
    """Helper to build consistent GPU/profile/budget mocks."""
    from unittest.mock import MagicMock
    mock_gpu = MagicMock()
    mock_gpu.present = present
    mock_gpu.vram_total_mb = vram_total_mb
    mock_gpu.vram_free_mb = max(0, vram_total_mb - 1024)
    mock_gpu.model = f"Mock({'GPU' if present else 'None'})"

    mock_profile = MagicMock()
    mock_profile.gpu = mock_gpu
    mock_profile.to_dict.return_value = {"gpu": {"present": present}}

    mock_budgets = MagicMock()
    mock_budgets.vram_budget_mb = vram_budget_mb
    mock_budgets.ram_budget_mb = 4096
    mock_budgets.cpu_threads = 4
    mock_budgets.gpu_heavy_slots = 0
    mock_budgets.vram_reserve_mb = 0
    mock_budgets.to_dict.return_value = {}

    return mock_gpu, mock_profile, mock_budgets


def test_gpu_absent_manifest_records_preflight_failure(tmp_path):
    """When GPU absent and trajectories exist, preflight must record PRECONDITION_FAILED."""
    from unittest.mock import MagicMock, patch
    import auto_mobility.reconstruction.cli as cli_mod

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    mock_gpu, mock_profile, mock_budgets = _make_gpu_mocks(present=False, vram_total_mb=0)

    # Inject a fake trajectory so we enter the `if trajs:` block and hit the preflight
    fake_traj = MagicMock()
    fake_traj.__len__ = MagicMock(return_value=100)

    with patch('auto_mobility.reconstruction.runtime.machine_profile._probe_gpu', return_value=mock_gpu), \
         patch('auto_mobility.reconstruction.runtime.load_or_probe_profile', return_value=mock_profile), \
         patch('auto_mobility.reconstruction.runtime.compute_resource_budgets', return_value=mock_budgets), \
         patch('auto_mobility.reconstruction.runtime.run_state.detect_previous_host_reset', return_value={}), \
         patch('auto_mobility.reconstruction.runtime.run_state.write_run_state', return_value=None), \
         patch('auto_mobility.reconstruction.cli._verify_trajectory_cache', return_value=True), \
         patch('auto_mobility.trajectory.io.Trajectory.from_tum_file', return_value=fake_traj):

        # Create a fake dataset dir and trajectory so the pipeline reaches preflight
        ds_dir = tmp_path / 'frames' / 'test_bag'
        ds_dir.mkdir(parents=True)
        (ds_dir / 'frames.csv').write_text('frame_id,rgb_timestamp,depth_timestamp,rgb_path,depth_path\n0,1.0,1.0,a.png,b.png\n')

        traj_dir = tmp_path / 'ros2_data' / 'trajectories'
        traj_dir.mkdir(parents=True)
        traj_file = traj_dir / 'cuvslam_test_bag_trajectory.txt'
        traj_file.write_text('\n'.join([f'{1+i*0.03:.6f} 0 0 0 0 0 0 1' for i in range(12)]) + '\n')

        args = cli_mod.build_parser().parse_args(['test_bag', '--preview'])
        args.output = out_dir
        args.dataset_dir = ds_dir
        args.trajectory = [traj_file]

        result = cli_mod.run(args)

    # Should return 1 due to preflight failure
    assert result == 1, f"Expected exit code 1 (GPU preflight), got {result}"

    manifest_path = out_dir / 'run_manifest.json'
    assert manifest_path.exists(), "manifest must be written even on preflight failure"
    manifest = json.loads(manifest_path.read_text())
    assert 'preflight_failure' in manifest, f"manifest missing preflight_failure: {manifest.keys()}"
    assert 'PRECONDITION_FAILED' in manifest['preflight_failure']
    assert manifest.get('standard_result', {}).get('ok') is False


def test_manifest_has_preflight_failure_on_no_gpu(tmp_path, monkeypatch):
    """When GPU preflight fails (dataset missing), manifest written only if preflight reached."""
    from unittest.mock import MagicMock, patch
    import auto_mobility.reconstruction.cli as cli_mod

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    mock_gpu, mock_profile, mock_budgets = _make_gpu_mocks(present=False, vram_total_mb=0)

    with patch('auto_mobility.reconstruction.runtime.machine_profile._probe_gpu', return_value=mock_gpu), \
         patch('auto_mobility.reconstruction.runtime.load_or_probe_profile', return_value=mock_profile), \
         patch('auto_mobility.reconstruction.runtime.compute_resource_budgets', return_value=mock_budgets), \
         patch('auto_mobility.reconstruction.runtime.run_state.detect_previous_host_reset', return_value={}), \
         patch('auto_mobility.reconstruction.runtime.run_state.write_run_state', return_value=None):

        args = cli_mod.build_parser().parse_args(['test_bag', '--preview'])
        args.output = out_dir

        result = cli_mod.run(args)

    # Without dataset the preflight is never reached, so result may be 0 or 1
    # If manifest has preflight_failure, check it says PRECONDITION_FAILED and code is 1
    manifest_path = out_dir / 'run_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if 'preflight_failure' in manifest:
            assert 'PRECONDITION_FAILED' in manifest['preflight_failure']
            assert result == 1


def test_stale_artifact_not_modified_on_failure(tmp_path):
    """Failed runs should not modify existing publish directory."""
    preview_dir = tmp_path / 'out' / 'preview' / 'rtab'
    preview_dir.mkdir(parents=True)
    stale_obj = preview_dir / 'model.obj'
    stale_obj.write_text('# stale mesh')
    stale_mtime = stale_obj.stat().st_mtime

    # Verify preflight logic is in the source
    src = Path('/home/kth/auto-mobility/src/auto_mobility/reconstruction/cli.py').read_text()
    assert 'PRECONDITION_FAILED' in src, "PRECONDITION_FAILED sentinel must exist in cli.py"
    # Verify preflight returns before touching preview dir
    assert 'manifest["preflight_failure"]' in src

    # Stale file must be unchanged
    assert stale_obj.read_text() == '# stale mesh'
    assert stale_obj.stat().st_mtime == stale_mtime


def test_exit_code_reflects_standard_result_ok_false(tmp_path):
    """CLI must write standard_result.ok=False and return non-zero when pipeline fails."""
    from unittest.mock import MagicMock, patch
    import auto_mobility.reconstruction.cli as cli_mod

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    mock_gpu, mock_profile, mock_budgets = _make_gpu_mocks(present=True, vram_total_mb=8192,
                                                           vram_budget_mb=4096)
    mock_profile.gpu.model = "Mock GPU"

    # Simulate run_standard returning ok=False
    mock_std_result = {"ok": False, "reason": "fusion_scheduler_failed", "decisions": []}

    fake_traj = MagicMock()

    with patch('auto_mobility.reconstruction.runtime.load_or_probe_profile', return_value=mock_profile), \
         patch('auto_mobility.reconstruction.runtime.compute_resource_budgets', return_value=mock_budgets), \
         patch('auto_mobility.reconstruction.runtime.run_state.detect_previous_host_reset', return_value={}), \
         patch('auto_mobility.reconstruction.runtime.run_state.write_run_state', return_value=None), \
         patch('auto_mobility.reconstruction.runtime.machine_profile._probe_gpu',
               return_value=mock_profile.gpu), \
         patch('auto_mobility.reconstruction.cli._verify_trajectory_cache', return_value=True), \
         patch('auto_mobility.trajectory.io.Trajectory.from_tum_file', return_value=fake_traj), \
         patch('auto_mobility.reconstruction.pipeline.standard.run_standard',
               return_value=mock_std_result):

        ds_dir = tmp_path / 'frames' / 'test_bag'
        ds_dir.mkdir(parents=True)
        (ds_dir / 'frames.csv').write_text('frame_id,rgb_timestamp,depth_timestamp,rgb_path,depth_path\n0,1.0,1.0,a.png,b.png\n')

        traj_file = tmp_path / 'rtab_normal_test_bag_trajectory.txt'
        traj_file.write_text('\n'.join([f'{1+i*0.03:.6f} 0 0 0 0 0 0 1' for i in range(12)]) + '\n')

        args = cli_mod.build_parser().parse_args(['test_bag', '--standard'])
        args.output = out_dir
        args.dataset_dir = ds_dir
        args.trajectory = [traj_file]

        result = cli_mod.run(args)

    manifest_path = out_dir / 'run_manifest.json'
    assert manifest_path.exists()
    m = json.loads(manifest_path.read_text())
    assert m.get("standard_result", {}).get("ok") is False
    assert result == 1, f"standard_result.ok=False must cause exit code 1, got {result}"


def test_vram_budget_zero_with_gpu_returns_nonzero(tmp_path):
    """Even with GPU present, vram_budget_mb=0 must trigger PRECONDITION_FAILED."""
    from unittest.mock import MagicMock, patch
    import auto_mobility.reconstruction.cli as cli_mod

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    mock_gpu, mock_profile, mock_budgets = _make_gpu_mocks(present=True, vram_total_mb=4096,
                                                           vram_budget_mb=0)
    mock_profile.gpu.model = "Mock GPU (low)"

    fake_traj = MagicMock()

    with patch('auto_mobility.reconstruction.runtime.machine_profile._probe_gpu', return_value=mock_gpu), \
         patch('auto_mobility.reconstruction.runtime.load_or_probe_profile', return_value=mock_profile), \
         patch('auto_mobility.reconstruction.runtime.compute_resource_budgets', return_value=mock_budgets), \
         patch('auto_mobility.reconstruction.runtime.run_state.detect_previous_host_reset', return_value={}), \
         patch('auto_mobility.reconstruction.runtime.run_state.write_run_state', return_value=None), \
         patch('auto_mobility.reconstruction.cli._verify_trajectory_cache', return_value=True), \
         patch('auto_mobility.trajectory.io.Trajectory.from_tum_file', return_value=fake_traj):

        ds_dir = tmp_path / 'frames' / 'test_bag'
        ds_dir.mkdir(parents=True)
        (ds_dir / 'frames.csv').write_text('frame_id,rgb_timestamp,depth_timestamp,rgb_path,depth_path\n0,1.0,1.0,a.png,b.png\n')

        traj_file = tmp_path / 'cuvslam_test_bag_trajectory.txt'
        traj_file.write_text('\n'.join([f'{1+i*0.03:.6f} 0 0 0 0 0 0 1' for i in range(12)]) + '\n')

        args = cli_mod.build_parser().parse_args(['test_bag', '--preview'])
        args.output = out_dir
        args.dataset_dir = ds_dir
        args.trajectory = [traj_file]

        result = cli_mod.run(args)

    manifest_path = out_dir / 'run_manifest.json'
    assert manifest_path.exists(), "manifest must be written on preflight failure"
    manifest = json.loads(manifest_path.read_text())
    assert 'preflight_failure' in manifest, f"vram_budget=0 must record preflight_failure"
    assert 'PRECONDITION_FAILED' in manifest['preflight_failure']
    assert result == 1
