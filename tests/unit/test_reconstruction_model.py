"""V2 model + config invariants: coordinate convention, typed config."""

import numpy as np
import pytest

from auto_mobility.reconstruction.config import (
    BudgetConfig,
    ReconstructionConfig,
    ResourcePolicyConfig,
    load_config,
)
from auto_mobility.reconstruction.model import (
    CameraIntrinsics,
    invert_pose,
    validate_pose,
)


def test_trajectory_coordinate_convention():
    t_wc = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    validate_pose(t_wc)

    p_world = np.array([5.0, 6.0, 7.0, 1.0])
    p_cam = np.linalg.inv(t_wc) @ p_world
    manual = invert_pose(t_wc) @ p_world
    assert np.allclose(p_cam, manual, atol=1e-12)
    assert np.allclose(invert_pose(invert_pose(t_wc)), t_wc, atol=1e-12)

    pure_t = np.eye(4)
    pure_t[:3, 3] = [10, 20, 30]
    assert np.allclose(invert_pose(pure_t)[:3, 3], [-10, -20, -30])


def test_validate_pose_rejects_bad_transforms():
    with pytest.raises(ValueError):
        validate_pose(np.eye(3))

    bad_rot = np.eye(4)
    bad_rot[:3, :3] = np.diag([2.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        validate_pose(bad_rot)

    nan = np.eye(4)
    nan[0, 3] = np.nan
    with pytest.raises(ValueError):
        validate_pose(nan)


def test_intrinsics_roundtrip_strict():
    intr = CameraIntrinsics(width=640, height=480, fx=385.0, fy=385.0, cx=320.0, cy=240.0)
    K = intr.matrix()
    assert K[0, 2] == 320.0 and K[1, 2] == 240.0

    parsed = CameraIntrinsics.from_dict(intr.to_dict())
    assert parsed == intr

    with pytest.raises(ValueError):
        CameraIntrinsics.from_dict({"width": 640})
    with pytest.raises(ValueError):
        CameraIntrinsics.from_dict({**intr.to_dict(), "bogus": 1})


def test_config_defaults_and_rejects_unknown_keys(tmp_path):
    cfg = load_config(None)
    assert isinstance(cfg, ReconstructionConfig)
    assert cfg.resources.ram_budget_fraction > 0

    good = tmp_path / "recon.yaml"
    good.write_text(
        "resources:\n"
        "  ram_budget_fraction: 0.60\n"
        "budget:\n"
        "  total_minutes: 45\n"
    )
    loaded = load_config(good)
    assert loaded.resources.ram_budget_fraction == 0.60
    assert loaded.budget.total_minutes == 45
    assert isinstance(loaded.resources, ResourcePolicyConfig)
    assert isinstance(loaded.budget, BudgetConfig)
    assert loaded.schema_version == cfg.schema_version

    bad = tmp_path / "bad.yaml"
    bad.write_text("resources:\n  not_a_real_key: 1\n")
    with pytest.raises(ValueError):
        load_config(bad)

    worse = tmp_path / "worse.yaml"
    worse.write_text("unknown_section:\n  x: 1\n")
    with pytest.raises(ValueError):
        load_config(worse)


def test_stage_decision_trace_shape():
    from auto_mobility.reconstruction.model import StageDecision

    d = StageDecision(
        stage="fusion_refinement",
        decision="SKIP_FINE_VOXEL",
        reason="trajectory residual dominates",
        evidence={"residual_mm": 4.2},
    ).to_dict()
    assert d["decision"] == "SKIP_FINE_VOXEL"
    assert set(d) == {"stage", "decision", "reason", "evidence"}
