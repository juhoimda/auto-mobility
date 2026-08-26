"""
tests/integration/test_synthetic_evaluation.py

Synthetic integration test to rigorously verify evaluator geometry accuracy:
1) Known synthetic flat plane at Z=1.5m, known intrinsics, synthetic depth frames & TUM trajectory.
2) Evaluates ground-truth matching mesh -> asserts Depth MAE ~ 0mm and Point-to-Mesh ~ 0mm.
3) Translates mesh by +5cm (+50mm) -> asserts Depth MAE and P95 strictly increase to ~50mm.
"""

import os
import json
import csv
import shutil
import tempfile
import numpy as np
import cv2
import open3d as o3d
import pytest
from pathlib import Path

from auto_mobility.evaluation.evaluator import evaluate_reconstruction
from auto_mobility.trajectory.io import Trajectory


@pytest.fixture
def synthetic_plane_dataset(tmp_path):
    dataset_dir = tmp_path / "synthetic_room"
    rgb_dir = dataset_dir / "rgb"
    depth_dir = dataset_dir / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    w, h = 640, 480
    fx, fy = 400.0, 400.0
    cx, cy = 320.0, 240.0

    # Plane at Z=1.5m (1500 mm) in front of camera
    depth_val_mm = 1500
    depth_img = np.full((h, w), depth_val_mm, dtype=np.uint16)
    rgb_img = np.full((h, w, 3), 128, dtype=np.uint8)

    # 10 frames at 10Hz (t=0.0 to 0.9) with static camera at (0, 0, 0)
    frames_csv_path = dataset_dir / "frames.csv"
    with open(frames_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_id", "rgb_timestamp", "depth_timestamp", "rgb_path", "depth_path",
            "rgb_depth_dt_ms", "bag_timestamp", "camera_frame_id", "width", "height"
        ])
        writer.writeheader()

        for idx in range(10):
            t = idx * 0.1
            f_str = f"{idx:06d}"
            r_rel = f"rgb/{f_str}.png"
            d_rel = f"depth/{f_str}.png"

            cv2.imwrite(str(dataset_dir / r_rel), rgb_img)
            cv2.imwrite(str(dataset_dir / d_rel), depth_img)

            writer.writerow({
                "frame_id": idx,
                "rgb_timestamp": f"{t:.3f}",
                "depth_timestamp": f"{t:.3f}",
                "rgb_path": r_rel,
                "depth_path": d_rel,
                "rgb_depth_dt_ms": "0.0",
                "bag_timestamp": f"{t:.3f}",
                "camera_frame_id": "camera_color_optical_frame",
                "width": w,
                "height": h
            })

    # camera_info.json
    cam_info = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": w, "height": h, "distortion_model": "plumb_bob",
        "distortion_coefficients": [0.0]*5,
        "K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    }
    with open(dataset_dir / "camera_info.json", "w", encoding="utf-8") as f:
        json.dump(cam_info, f)

    # trajectory.txt
    traj_path = tmp_path / "trajectory.txt"
    stamps = np.array([idx * 0.1 for idx in range(10)])
    positions = np.zeros((10, 3))
    orientations = np.zeros((10, 4))
    orientations[:, 3] = 1.0 # identity quaternion
    traj = Trajectory(stamps, positions, orientations)
    traj.to_tum_file(str(traj_path))

    # Perfect Ground Truth Mesh at Z=1.5m (spanning X: -2 to 2, Y: -2 to 2)
    mesh_gt_path = tmp_path / "mesh_gt.obj"
    plane_v = np.array([
        [-2.0, -2.0, 1.5],
        [ 2.0, -2.0, 1.5],
        [ 2.0,  2.0, 1.5],
        [-2.0,  2.0, 1.5]
    ], dtype=np.float64)
    plane_tri = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)
    mesh_gt = o3d.geometry.TriangleMesh()
    mesh_gt.vertices = o3d.utility.Vector3dVector(plane_v)
    mesh_gt.triangles = o3d.utility.Vector3iVector(plane_tri)
    mesh_gt.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(mesh_gt_path), mesh_gt)

    # Perturbed Mesh translated by +5cm in Z (Z = 1.55m)
    mesh_shifted_path = tmp_path / "mesh_shifted_5cm.obj"
    mesh_shifted = o3d.geometry.TriangleMesh(mesh_gt)
    mesh_shifted.translate(np.array([0.0, 0.0, 0.05])) # +5cm in Z
    o3d.io.write_triangle_mesh(str(mesh_shifted_path), mesh_shifted)

    return {
        "dataset_dir": dataset_dir,
        "traj_path": traj_path,
        "mesh_gt_path": mesh_gt_path,
        "mesh_shifted_path": mesh_shifted_path,
        "out_dir": tmp_path / "eval_out"
    }


def test_synthetic_evaluator_exact_and_shifted(synthetic_plane_dataset):
    ds_dir = synthetic_plane_dataset["dataset_dir"]
    traj_p = synthetic_plane_dataset["traj_path"]
    gt_mesh = synthetic_plane_dataset["mesh_gt_path"]
    shifted_mesh = synthetic_plane_dataset["mesh_shifted_path"]
    out_dir = synthetic_plane_dataset["out_dir"]

    # 1. Evaluate Ground Truth Matching Mesh
    summary_gt = evaluate_reconstruction(
        dataset_input=ds_dir,
        trajectory_input=traj_p,
        mesh_input=gt_mesh,
        output_dir=out_dir / "gt",
        candidate_name="gt_plane"
    )

    geom_gt = summary_gt["geometry"]
    # Depth MAE should be virtually 0 mm (< 1.0mm numerical precision)
    assert geom_gt["depth_mae_mm"] is not None
    assert geom_gt["depth_mae_mm"] < 1.0
    assert geom_gt["depth_p95_mm"] < 1.0
    assert geom_gt["depth_coverage_ratio"] >= 0.99
    assert geom_gt["within_10mm_ratio"] >= 0.99
    assert geom_gt["point_to_mesh_mean_mm"] < 1.0
    assert summary_gt["overall_status"] == "PASS"

    # 2. Evaluate Shifted Mesh (+50mm offset)
    summary_shifted = evaluate_reconstruction(
        dataset_input=ds_dir,
        trajectory_input=traj_p,
        mesh_input=shifted_mesh,
        output_dir=out_dir / "shifted",
        candidate_name="shifted_5cm"
    )

    geom_shifted = summary_shifted["geometry"]
    # Depth MAE should accurately measure the 50mm shift (49.0 ~ 51.0 mm)
    assert abs(geom_shifted["depth_mae_mm"] - 50.0) < 1.5
    assert abs(geom_shifted["depth_p95_mm"] - 50.0) < 1.5
    assert geom_shifted["within_10mm_ratio"] == 0.0
    assert abs(geom_shifted["point_to_mesh_mean_mm"] - 50.0) < 1.5

    # Proves that error metric strictly increases with geometry deviation
    assert geom_shifted["depth_mae_mm"] > geom_gt["depth_mae_mm"] + 45.0
