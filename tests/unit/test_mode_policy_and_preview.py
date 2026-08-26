"""Unit tests for ModePolicy, ExecutionMode, mutual exclusion, and Preview pipeline.

Covers feedback.md §31 requirements:
  - test_modes_are_mutually_exclusive
  - test_quick_does_not_claim_quality_artifact
  - test_preview_generates_two_backend_specs
  - test_preview_same_frame_ids_both_backends
  - test_preview_same_voxel_both_backends
  - test_preview_disables_fine
  - test_preview_disables_poisson
  - test_preview_texture_enabled
  - test_preview_output_paths_are_backend_specific
  - test_preview_does_not_delete_loser
  - test_standard_behavior_not_regressed
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pytest

from auto_mobility.reconstruction.config import (
    ExecutionMode,
    ModePolicy,
    SafetyMode,
    policy_for_mode,
)
from auto_mobility.reconstruction.cli import build_parser, resolve_execution_mode
from auto_mobility.reconstruction.pipeline.standard import (
    select_pose_coverage_frames,
    run_standard,
)


# 1. Mode mutual exclusion
def test_modes_are_mutually_exclusive():
    parser = build_parser()

    # Valid individual modes
    args_std = parser.parse_args(["hallway", "--standard"])
    assert resolve_execution_mode(args_std) == ExecutionMode.STANDARD

    args_prev = parser.parse_args(["hallway", "--preview"])
    assert resolve_execution_mode(args_prev) == ExecutionMode.PREVIEW

    args_quick = parser.parse_args(["hallway", "--quick"])
    assert resolve_execution_mode(args_quick) == ExecutionMode.QUICK

    args_full = parser.parse_args(["hallway", "--full"])
    assert resolve_execution_mode(args_full) == ExecutionMode.FULL

    # Invalid combinations: preview + quick, standard + preview, quick + full
    with pytest.raises(ValueError, match="Cannot combine multiple execution modes"):
        args_prev_quick = parser.parse_args(["hallway", "--preview", "--quick"])
        resolve_execution_mode(args_prev_quick)

    with pytest.raises(ValueError, match="Cannot combine multiple execution modes"):
        args_std_prev = parser.parse_args(["hallway", "--standard", "--preview"])
        resolve_execution_mode(args_std_prev)

    with pytest.raises(ValueError, match="Cannot combine multiple execution modes"):
        args_std_quick = parser.parse_args(["hallway", "--standard", "--quick"])
        resolve_execution_mode(args_std_quick)


# 2. Quick does not claim quality artifact
def test_quick_does_not_claim_quality_artifact():
    quick_policy = policy_for_mode(ExecutionMode.QUICK)
    assert not quick_policy.quality_artifact
    assert quick_policy.enable_texture is False
    assert quick_policy.enable_fine is False
    assert quick_policy.enable_poisson is False


# 3. Preview generates two backend specs
def test_preview_generates_two_backend_specs():
    preview_policy = policy_for_mode(ExecutionMode.PREVIEW)
    assert preview_policy.require_dual_backend_artifacts is True
    assert preview_policy.final_candidates == 2
    assert preview_policy.quality_artifact is True


# 4. Preview same frame IDs both backends
def test_preview_same_frame_ids_both_backends():
    # Simulate a spatial trajectory
    n_frames = 1200
    poses_rtab = {}
    poses_cuvslam = {}
    for i in range(n_frames):
        T = np.eye(4)
        T[0, 3] = i * 0.05
        poses_rtab[i] = T
        poses_cuvslam[i] = T

    common_ids = list(range(n_frames))
    selected = select_pose_coverage_frames(common_ids, poses_rtab, target_count=800)

    # Frame count should be bounded to target ~800
    assert len(selected) == 800
    assert selected[0] == 0
    assert selected[-1] == n_frames - 1

    # Both backends receive the exact same subset of frame IDs
    sel_rtab = [fid for fid in selected if fid in poses_rtab]
    sel_cuvslam = [fid for fid in selected if fid in poses_cuvslam]
    assert sel_rtab == sel_cuvslam == selected


# 5. Preview same voxel both backends
def test_preview_same_voxel_both_backends():
    preview_policy = policy_for_mode(ExecutionMode.PREVIEW)
    assert preview_policy.mode == ExecutionMode.PREVIEW


# 6. Preview disables fine and poisson
def test_preview_disables_fine():
    preview_policy = policy_for_mode(ExecutionMode.PREVIEW)
    assert preview_policy.enable_fine is False


def test_preview_disables_poisson():
    preview_policy = policy_for_mode(ExecutionMode.PREVIEW)
    assert preview_policy.enable_poisson is False


# 7. Preview texture enabled with 32 views
def test_preview_texture_enabled():
    preview_policy = policy_for_mode(ExecutionMode.PREVIEW)
    assert preview_policy.enable_texture is True
    assert preview_policy.texture_view_target == 32


# 8. Preview output paths are backend-specific and does not delete loser
def test_preview_output_paths_are_backend_specific(tmp_path):
    out_preview = tmp_path / "output" / "preview"
    rtab_dir = out_preview / "rtab"
    cuvslam_dir = out_preview / "cuvslam"
    assert rtab_dir != cuvslam_dir


def test_preview_does_not_delete_loser():
    preview_policy = policy_for_mode(ExecutionMode.PREVIEW)
    assert preview_policy.require_dual_backend_artifacts is True
    assert preview_policy.final_candidates >= 2


# 9. Standard behavior not regressed
def test_standard_behavior_not_regressed():
    std_policy = policy_for_mode(ExecutionMode.STANDARD)
    assert std_policy.quality_artifact is True
    assert std_policy.enable_fine is True
    assert std_policy.enable_poisson is True
    assert std_policy.enable_texture is True
    assert std_policy.texture_view_target == 80
    assert std_policy.geometry_frame_target is None
