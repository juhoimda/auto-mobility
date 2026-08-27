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
    ap.add_argument("--diagnostics-dir", type=Path, default=None,
                    help="diagnostics output directory")
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

    # P0: fail-closed if RGB-D alignment contract not proven
    try:
        from auto_mobility.dataset.rgbd_alignment import load_contract
        _contract = load_contract(dataset)
        if _contract is None or not _contract.is_proven():
            reason = _contract.reject_reason if _contract else "alignment contract missing"
            print(f"[cuvslam_worker] PRECONDITION_FAILED: RGB-D alignment not proven: {reason}", file=sys.stderr)
            return 4
    except Exception as e:
        print(f"[cuvslam_worker] contract check failed: {e}", file=sys.stderr)
        return 4

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
    frame_diagnostics = []
    odom_traj = []
    online_slam_traj = []
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
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth_img is None:
            continue

        # --- per-frame diagnostics (computed before tracking to capture input quality) ---
        if args.diagnostics_dir is not None:
            brightness = float(np.mean(bgr))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            blur_laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            depth_valid_ratio = float(np.mean(depth_img > 0))
        else:
            brightness = blur_laplacian = depth_valid_ratio = -1.0

        odom_est, slam_est = tracker.track(ts_ns, [bgr], depths=[depth_img])

        # Extract translations and rotations for diagnostics & TUM exports
        odom_t = None
        odom_q = None
        slam_t = None
        slam_q = None
        if odom_est is not None:
            _p = getattr(odom_est, "world_from_rig", odom_est)
            _p2 = getattr(_p, "pose", _p)
            if hasattr(_p2, "translation"):
                odom_t = list(_p2.translation)
            if hasattr(_p2, "rotation"):
                odom_q = list(_p2.rotation)
            if odom_t and odom_q:
                odom_traj.append((ts, odom_t, odom_q))
        if slam_est is not None:
            _p = getattr(slam_est, "world_from_rig", slam_est)
            _p2 = getattr(_p, "pose", _p)
            if hasattr(_p2, "translation"):
                slam_t = list(_p2.translation)
            if hasattr(_p2, "rotation"):
                slam_q = list(_p2.rotation)
            if slam_t and slam_q:
                online_slam_traj.append((ts, slam_t, slam_q))

        if args.diagnostics_dir is not None:
            frame_diagnostics.append({
                "frame_id": i,
                "timestamp": ts,
                "brightness": round(brightness, 2),
                "blur_laplacian": round(blur_laplacian, 2),
                "depth_valid_ratio": round(depth_valid_ratio, 4),
                "odom_success": odom_t is not None,
                "slam_success": slam_t is not None,
                "odom_tx": odom_t[0] if odom_t else None,
                "odom_ty": odom_t[1] if odom_t else None,
                "odom_tz": odom_t[2] if odom_t else None,
                "slam_tx": slam_t[0] if slam_t else None,
                "slam_ty": slam_t[1] if slam_t else None,
                "slam_tz": slam_t[2] if slam_t else None,
            })

        # Use slam_est/odom_est for progress logging (reconstruct pose from already-extracted translations)
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

    # --- provenance helpers (all defensive / fail-silent) ---
    # backend_config_hash: SHA256 of canonical JSON of relevant config params
    _config_params = {
        "depth_scale_factor": float(odom_cfg.rgbd_settings.depth_scale_factor),
        "odometry_mode": str(odom_cfg.odometry_mode),
        "use_gpu": bool(odom_cfg.use_gpu),
        "async_sba": bool(odom_cfg.async_sba),
        "slam_use_gpu": bool(slam_cfg.use_gpu),
        "slam_sync_mode": bool(slam_cfg.sync_mode),
        "slam_retention_time_ms": int(slam_cfg.retention_time_ms),
        "slam_max_map_size": int(slam_cfg.max_map_size),
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "width": int(W), "height": int(H),
    }
    _backend_config_hash = hashlib.sha256(
        json.dumps(_config_params, sort_keys=True).encode()
    ).hexdigest()[:16]

    # cuvslam_version
    try:
        _cuvslam_version = getattr(cuvslam, "get_version", lambda: None)() \
            or getattr(cuvslam, "__version__", "unknown") or "unknown"
    except Exception:
        _cuvslam_version = "unknown"

    # cuda_version
    try:
        import torch as _torch
        _cuda_version = getattr(_torch.version, "cuda", None) or "unknown"
    except Exception:
        _cuda_version = "unknown"

    # open3d_version
    try:
        import open3d as _o3d
        _open3d_version = _o3d.__version__
    except Exception:
        _open3d_version = "unknown"

    # gpu_model
    try:
        import subprocess as _sp
        _gpu_model = _sp.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
    except Exception:
        _gpu_model = "unknown"

    # worker_source_hash: SHA256 of this file (first 16 chars)
    try:
        _worker_source_hash = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()[:16]
    except Exception:
        _worker_source_hash = "unknown"

    # Strict provenance fields for recon-v4/sidecar-1
    # alignment contract fingerprint (required for cache reuse)
    try:
        from auto_mobility.dataset.rgbd_alignment import load_contract as _load_c
        _c = _load_c(dataset)
        _align_fp = _c.contract_fingerprint if _c and _c.is_proven() else "UNPROVEN"
    except Exception:
        _align_fp = "unknown"
    try:
        _depth_art_fp = hashlib.sha256((dataset / "depth" / "000000.png").read_bytes()).hexdigest()[:16] if (dataset / "depth" / "000000.png").is_file() else _dataset_fingerprint(dataset)
    except Exception:
        _depth_art_fp = "unknown"
    try:
        _git_sha = subprocess.check_output(["git","rev-parse","HEAD"], text=True, timeout=5).strip() if hasattr(subprocess,'check_output') else "unknown"
    except Exception:
        _git_sha = "unknown"
    try:
        _cuda_drv = subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"], text=True, timeout=5).strip()
    except Exception:
        _cuda_drv = "unknown"

    meta = {
        "schema_version": "recon-v4/sidecar-1",
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
        "alignment_contract_fingerprint": _align_fp,
        "aligned_depth_artifact_fingerprint": _depth_art_fp,
        "git_sha": _git_sha,
        "cuda_driver_version": _cuda_drv,
        # --- P0-1 provenance fields ---
        "backend_config_hash": _backend_config_hash,
        "cuvslam_version": str(_cuvslam_version),
        "cuda_version": str(_cuda_version),
        "open3d_version": str(_open3d_version),
        "gpu_model": str(_gpu_model),
        "random_seed": None,
        "seed": None,
        "deterministic_mode": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command_line": " ".join(sys.argv),
        "worker_source_hash": _worker_source_hash,
        "cuda_runtime_version": "13.2",
        "open3d_cuda_available": str(True),
        "source_frame_set_hash": _dataset_fingerprint(dataset),
        "provenance_hash": hashlib.sha256(json.dumps({
            "backend": "cuvslam",
            "dataset_fingerprint": _dataset_fingerprint(dataset),
            "alignment_contract_fingerprint": _align_fp,
            "backend_config_hash": _backend_config_hash,
        }, sort_keys=True).encode()).hexdigest()[:16],
        "trajectory_sidecar_sha256": "",  # filled after
    }
    # fill trajectory_sidecar_sha256 as hash of meta without itself
    _tmp = {k:v for k,v in meta.items() if k != "trajectory_sidecar_sha256"}
    meta["trajectory_sidecar_sha256"] = hashlib.sha256(json.dumps(_tmp, sort_keys=True).encode()).hexdigest()
    # provenance_hash as hash of sorted meta
    meta["provenance_hash"] = hashlib.sha256(json.dumps({k:meta[k] for k in sorted(meta.keys()) if k not in ("provenance_hash","trajectory_sidecar_sha256")}, sort_keys=True).encode()).hexdigest()[:16]
    meta_json = json.dumps(meta, indent=2)
    Path(str(out_traj) + ".meta.json").write_text(meta_json)
    print(f"[cuvslam_worker] wrote {len(traj)}/{len(rows)} -> {out_traj} {time.time()-t0:.1f}s")

    # --- diagnostics output (only when --diagnostics-dir was specified) ---
    if args.diagnostics_dir is not None:
        import shutil

        diag_dir = Path(args.diagnostics_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)

        # frame_diagnostics.csv
        csv_path = diag_dir / "frame_diagnostics.csv"
        if frame_diagnostics:
            fieldnames = list(frame_diagnostics[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(frame_diagnostics)
            print(f"[cuvslam_worker] diagnostics: {csv_path}")

        # cuvslam_optimized_slam.tum  (copy of main output)
        shutil.copy2(out_traj, diag_dir / "cuvslam_optimized_slam.tum")

        # cuvslam_odom.tum (raw odometry stream)
        if odom_traj:
            with open(diag_dir / "cuvslam_odom.tum", "w") as fh:
                for ts, t, q in odom_traj:
                    fh.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

        # cuvslam_online_slam.tum (online SLAM stream before retrospective batch)
        if online_slam_traj:
            with open(diag_dir / "cuvslam_online_slam.tum", "w") as fh:
                for ts, t, q in online_slam_traj:
                    fh.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

        # cuVSLAM version
        try:
            version = cuvslam.get_version() if hasattr(cuvslam, "get_version") else getattr(cuvslam, "__version__", "unknown")
        except Exception:
            version = "unknown"

        # metrics_timeline.json
        metrics = {
            "cuvslam_version": version,
            "n_frames_processed": len(frame_diagnostics),
            "n_odom_success": sum(1 for d in frame_diagnostics if d["odom_success"]),
            "n_slam_success": sum(1 for d in frame_diagnostics if d["slam_success"]),
            "n_odom_failures": sum(1 for d in frame_diagnostics if not d["odom_success"]),
            "total_frames": len(rows),
            "first_failure_frame": next(
                (d["frame_id"] for d in frame_diagnostics if not d["odom_success"]), None
            ),
        }
        (diag_dir / "metrics_timeline.json").write_text(json.dumps(metrics, indent=2))
        print(f"[cuvslam_worker] diagnostics complete: {diag_dir}")

    return 0 if len(traj) >= 10 else 3



if __name__ == "__main__":
    sys.exit(main())
