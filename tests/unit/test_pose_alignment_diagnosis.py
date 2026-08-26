import numpy as np

from auto_mobility.diagnostics.pose_alignment import diagnose_pose_alignment
from auto_mobility.diagnostics.pipeline_diagnosis import diagnose_pipeline
from auto_mobility.diagnostics.frame_quality import analyze_frame_quality
from auto_mobility.trajectory.io import Trajectory


def _traj(times):
    times = np.asarray(times, dtype=float)
    return Trajectory(
        times,
        np.column_stack([times, np.zeros_like(times), np.zeros_like(times)]),
        np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1)),
    )


def test_pose_alignment_detects_timestamp_offset():
    # Frames are one second ahead of the exported trajectory.
    result = diagnose_pose_alignment(
        np.arange(1.0, 5.0, 0.1),
        _traj(np.arange(0.0, 4.0, 0.1)),
        max_pose_gap_ms=50.0,
        offset_search_s=2.0,
        offset_step_s=0.1,
    )
    assert result["status"] == "FAIL"
    assert result["cause"] == "TIME_ALIGNMENT"
    assert result["best_offset_coverage_ratio"] > result["pose_coverage_ratio"]


def test_pipeline_diagnosis_blocks_downstream_cause_attribution():
    result = diagnose_pipeline(
        {"rtab_rgbd": {"status": "FAIL", "cause": "TIME_ALIGNMENT"}},
        [{"candidate_name": "rtab_rgbd_voxel10mm", "status": "FAIL"}],
        [{"candidate_name": "rtab_tsdf", "status": "BLOCKED"}],
        [{"candidate_name": "rtab_surface", "status": "BLOCKED"}],
    )
    assert result["primary_cause"] == "TIME_OR_POSE_ALIGNMENT"
    assert result["stages"]["tsdf_fusion"]["status"] == "BLOCKED"
    assert result["stages"]["surface_reconstruction"]["status"] == "BLOCKED"


def test_frame_quality_flags_missing_canonical_images(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "frames.csv").write_text(
        "frame_id,rgb_timestamp,depth_timestamp,rgb_path,depth_path,rgb_depth_dt_ms,bag_timestamp,camera_frame_id,width,height\n"
        "0,0.0,0.0,rgb/missing.png,depth/missing.png,0,0,camera,640,480\n",
        encoding="utf-8",
    )
    (frame_dir / "camera_info.json").write_text(
        '{"fx":400,"fy":400,"cx":320,"cy":240,"width":640,"height":480}',
        encoding="utf-8",
    )
    from auto_mobility.dataset.frame_dataset import FrameDataset

    result = analyze_frame_quality(FrameDataset(frame_dir))
    assert result["overall_status"] == "FAIL"
    assert result["missing_rgb"] == 1
    assert result["missing_depth"] == 1
