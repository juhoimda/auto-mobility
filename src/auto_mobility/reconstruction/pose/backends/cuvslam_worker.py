"""cuVSLAM worker subprocess: isolated CUDA context (§4).

Runs in a short-lived child process so parent VRAM baseline recovers after exit.
Consumes canonical RGB-D dataset (frames.csv + camera_info.json) and emits TUM trajectory.
Keep imports lazy so parent never imports cuvslam.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _dataset_fingerprint(dataset_dir: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in ("frames.csv", "camera_info.json"):
        fp = dataset_dir / name
        if fp.is_file():
            h.update(fp.read_bytes())
    return h.hexdigest()[:16]


def _load_dataset(dataset_dir: Path):
    cam_info = json.loads((dataset_dir / "camera_info.json").read_text())
    K = np.array(cam_info["K"]).reshape(3, 3) if "K" in cam_info else None
    fx, fy = float(cam_info.get("fx", K[0, 0])), float(cam_info.get("fy", K[1, 1]))
    cx, cy = float(cam_info.get("cx", K[0, 2])), float(cam_info.get("cy", K[1, 2]))
    W, H = int(cam_info["width"]), int(cam_info["height"])
    rows = list(csv.DictReader(open(dataset_dir / "frames.csv")))
    rows.sort(key=lambda r: float(r["rgb_timestamp"]))
    return W, H, fx, fy, cx, cy, K, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out_traj = Path(args.out)
    # preload nvidia libraries if available in site-packages
    try:
        import ctypes
        import glob
        for p in sys.path:
            for lib_dir in glob.glob(str(Path(p) / "nvidia" / "*" / "lib")):
                for so in sorted(glob.glob(str(Path(lib_dir) / "*.so*"))):
                    try:
                        ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass
    except Exception:
        pass

    # lazy import cuvslam only in child
    try:
        import cuvslam
    except ImportError as e:
        print(f"[cuvslam_worker] cuvslam not available: {e}", file=sys.stderr)
        return 2

    W, H, fx, fy, cx, cy, K, rows = _load_dataset(dataset)
    if args.max_frames and len(rows) > args.max_frames:
        rows = rows[: args.max_frames]

    rig = cuvslam.Rig()
    cam = cuvslam.Camera()
    cam.focal = [fx, fy]
    cam.principal = [cx, cy]
    cam.size = [W, H]
    cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
    cam.rig_from_camera = cuvslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
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
        depth_scale_factor=1000.0, depth_camera_id=0,
    )
    # SLAM must be explicitly enabled.  Without this config the second return
    # value of track() is always None, so a corridor is exported as raw VO.
    slam_cfg = cuvslam.Tracker.SlamConfig(
        use_gpu=True, sync_mode=True, retention_time_ms=0, max_map_size=0
    )
    tracker = cuvslam.Tracker(rig, odom_cfg, slam_cfg)
    print(f"[cuvslam_worker] RGBD SLAM tracker {W}x{H} fx={fx:.1f} (max_map_size=0 unlimited)")

    # Tracking samples are used only for progress logging.  The delivered
    # trajectory must be the retrospectively optimized SLAM export.
    t0 = time.time()
    for i, r in enumerate(rows):
        ts = float(r["rgb_timestamp"])
        ts_ns = int(ts * 1e9)
        rgb_path = Path(r["rgb_path"])
        depth_path = Path(r["depth_path"])
        if not rgb_path.is_absolute():
            rgb_path = dataset / rgb_path
        if not depth_path.is_absolute():
            depth_path = dataset / depth_path
        bgr = cv2.imread(str(rgb_path))
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None:
            continue
        odom_est, slam_est = tracker.track(ts_ns, [bgr], depths=[depth])
        pose = None
        if slam_est is not None:
            pose_estimate = getattr(slam_est, "world_from_rig", None)
            pose = getattr(pose_estimate, "pose", pose_estimate)
        if pose is None and odom_est is not None:
            pose_estimate = getattr(odom_est, "world_from_rig", None)
            pose = getattr(pose_estimate, "pose", pose_estimate)
        if pose is None:
            if i % 200 == 0:
                print(f"  [{i}/{len(rows)}] lost")
            continue
        if i % 200 == 0:
            print(f"  [{i}/{len(rows)}] ok t={list(pose.translation)}")

    optimized = tracker.get_all_slam_poses()
    traj = [(p.timestamp_ns / 1e9, list(p.pose.translation), list(p.pose.rotation))
            for p in optimized if getattr(p, "pose", None) is not None]
    if len(traj) < 10:
        print("[cuvslam_worker] optimized SLAM export unavailable; refusing unsafe VO fallback",
              file=sys.stderr)
        return 3

    out_traj.parent.mkdir(parents=True, exist_ok=True)
    with open(out_traj, "w") as fh:
        for ts, t, q in traj:
            fh.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    # sidecar (#50): content hash + dataset fingerprint for verified reuse
    import hashlib

    meta = {
        "schema_version": "recon-v3/sidecar-3",
        "backend": "cuvslam",
        "pose_convention": "T_world_camera",
        "pose_frame": "camera_color_optical_frame",
        "pose_export": "retrospective_slam",
        "pose_export_semantics": "optical_frame_retrospective_slam",
        "profile": "standard",
        "n_frames": len(rows),
        "n_poses": len(traj),
        "trajectory_sha256": hashlib.sha256(out_traj.read_bytes()).hexdigest(),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
    }
    meta_json = json.dumps(meta, indent=2)
    Path(str(out_traj) + ".meta.json").write_text(meta_json)
    print(f"[cuvslam_worker] wrote {len(traj)}/{len(rows)} -> {out_traj} {time.time()-t0:.1f}s")
    return 0 if len(traj) >= 10 else 3


if __name__ == "__main__":
    sys.exit(main())
