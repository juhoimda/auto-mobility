#!/usr/bin/env python3
import os, glob, pathlib, sys
for p in pathlib.Path.home().glob('.local/lib/python3.10/site-packages/nvidia/*/lib'):
    os.environ['LD_LIBRARY_PATH'] = f"{p}:{os.environ.get('LD_LIBRARY_PATH','')}"
for p in pathlib.Path('/usr/local/lib/python3.10/dist-packages').glob('nvidia/*/lib'):
    os.environ['LD_LIBRARY_PATH'] = f"{p}:{os.environ.get('LD_LIBRARY_PATH','')}"

import json, csv, time
from pathlib import Path
import numpy as np
import cv2
import cuvslam

DATASET = Path("ros2_data/frames/base3")
OUT_TUM = Path("ros2_data/trajectories/cuvslam_base3_trajectory.txt")
MAX_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cam_info = json.load(open(DATASET / "camera_info.json"))
K = np.array(cam_info["K"]).reshape(3,3) if "K" in cam_info else None
fx, fy = float(cam_info["fx"]), float(cam_info["fy"])
cx, cy = float(cam_info["cx"]), float(cam_info["cy"])
W, H = int(cam_info["width"]), int(cam_info["height"])

rig = cuvslam.Rig()
cam = cuvslam.Camera()
cam.focal = [fx, fy]
cam.principal = [cx, cy]
cam.size = [W, H]
cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
cam.rig_from_camera = cuvslam.Pose(rotation=[0,0,0,1], translation=[0,0,0])
rig.cameras = [cam]

odom_cfg = cuvslam.Tracker.OdometryConfig(
    async_sba=False,
    enable_observations_export=False,
    enable_final_landmarks_export=False,
    rectified_stereo_camera=False,
    odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
    use_gpu=True,
)
odom_cfg.rgbd_settings = cuvslam.Tracker.OdometryRGBDSettings(
    depth_scale_factor=1000.0,  # uint16 mm -> meters
    depth_camera_id=0,
)

tracker = cuvslam.Tracker(rig, odom_cfg)
print(f"[cuvslam] RGBD tracker {W}x{H} fx={fx:.1f} depth_scale=0.001")

rows = list(csv.DictReader(open(DATASET / "frames.csv")))
rows.sort(key=lambda r: float(r["rgb_timestamp"]))
if MAX_FRAMES and len(rows) > MAX_FRAMES:
    rows = rows[:MAX_FRAMES]

traj = []
t0 = time.time()
for i, r in enumerate(rows):
    ts = float(r["rgb_timestamp"])
    ts_ns = int(ts * 1e9)
    rgb_path = DATASET / r["rgb_path"] if not Path(r["rgb_path"]).is_absolute() else Path(r["rgb_path"])
    depth_path = DATASET / r["depth_path"] if not Path(r["depth_path"]).is_absolute() else Path(r["depth_path"])
    bgr = cv2.imread(str(rgb_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        continue
    odom_est, slam_est = tracker.track(ts_ns, [bgr], depths=[depth])
    pose = odom_est.world_from_rig if odom_est and odom_est.world_from_rig is not None else None
    # fallback to SLAM pose if odometry lost but SLAM has LC pose
    if pose is None and slam_est is not None and slam_est.world_from_rig is not None:
        pose = slam_est.world_from_rig
    if pose is None:
        if i % 200 == 0:
            print(f"  [{i}/{len(rows)}] lost")
        continue
    traj.append((ts, list(pose.pose.translation), list(pose.pose.rotation)))
    if i % 200 == 0:
        print(f"  [{i}/{len(rows)}] ok t={traj[-1][1]}")

OUT_TUM.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_TUM, "w") as fh:
    for ts, t, q in traj:
        fh.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
print(f"[cuvslam] wrote {len(traj)}/{len(rows)} -> {OUT_TUM} {time.time()-t0:.1f}s")
