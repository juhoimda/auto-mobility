"""
tests/unit/test_frame_dataset.py

Unit tests for FrameDataset loader, camera intrinsics parsing, and dataset validation.
"""

import os
import json
import csv
import numpy as np
import cv2
import pytest
from pathlib import Path

from auto_mobility.dataset.frame_dataset import FrameDataset, CameraIntrinsics
from auto_mobility.dataset.validate_dataset import validate_dataset
from auto_mobility.evaluation.split import create_holdout_split


def test_frame_dataset_load_and_access(tmp_path):
    dataset_dir = tmp_path / "test_dataset"
    rgb_dir = dataset_dir / "rgb"
    depth_dir = dataset_dir / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    dummy_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    dummy_depth = np.full((240, 320), 1200, dtype=np.uint16)

    cv2.imwrite(str(rgb_dir / "000000.png"), dummy_rgb)
    cv2.imwrite(str(depth_dir / "000000.png"), dummy_depth)

    # frames.csv
    with open(dataset_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "rgb_timestamp", "depth_timestamp", "rgb_path", "depth_path",
            "rgb_depth_dt_ms", "bag_timestamp", "camera_frame_id", "width", "height"
        ])
        writer.writeheader()
        writer.writerow({
            "frame_id": 0,
            "rgb_timestamp": "100.000000",
            "depth_timestamp": "100.005000",
            "rgb_path": "rgb/000000.png",
            "depth_path": "depth/000000.png",
            "rgb_depth_dt_ms": "5.0",
            "bag_timestamp": "100.000000",
            "camera_frame_id": "camera_color_optical_frame",
            "width": 320,
            "height": 240
        })

    # camera_info.json
    cam_info = {
        "fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0,
        "width": 320, "height": 240, "distortion_model": "plumb_bob"
    }
    with open(dataset_dir / "camera_info.json", "w", encoding="utf-8") as f:
        json.dump(cam_info, f)

    dataset = FrameDataset(dataset_dir)
    assert len(dataset) == 1
    assert dataset.intrinsics.fx == 300.0
    assert dataset.intrinsics.cx == 160.0

    rgb = dataset.get_rgb(0)
    assert rgb is not None
    assert rgb.shape == (240, 320, 3)

    depth_m = dataset.get_depth_meters(0)
    assert depth_m is not None
    assert abs(depth_m[0, 0] - 1.2) < 1e-4

    val_res = validate_dataset(str(dataset_dir))
    assert val_res["pass"] is True
    assert val_res["quality_metrics"]["num_frames"] == 1


def test_holdout_split_deterministic():
    split = create_holdout_split(total_frames=20, policy="every_nth", nth=5)
    assert split["total_frames"] == 20
    assert split["holdout_indices"] == [4, 9, 14, 19]
    assert split["train_count"] == 16
    assert split["holdout_count"] == 4
    assert abs(split["holdout_ratio"] - 0.20) < 1e-4
