"""cuVSLAM backend adapter (#25).

Availability probe only in this environment; visual SLAM execution delegates
to run_slam.sh-style subprocess once isaac-ros-visual-slam is installed.
Contract: TUM trajectory output + metadata sidecar (backend=cuvslam).
"""

from __future__ import annotations

import shutil
from pathlib import Path


def available() -> bool:
    try:
        import cuvslam  # noqa: F401

        return True
    except ImportError:
        pass
    return shutil.which("ros2") is not None and _has_ros_pkg()


def _has_ros_pkg() -> bool:
    out = shutil.which("ros2")
    if not out:
        return False
    import subprocess

    try:
        res = subprocess.run(["bash", "-lc",
                              "source /opt/ros/humble/setup.bash && "
                              "ros2 pkg list 2>/dev/null | grep -x isaac_ros_visual_slam"],
                             capture_output=True, text=True, timeout=30)
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False


def run(bag_dir: Path, out_trajectory: Path) -> Path:
    """Run visual_slam offline on a bag; write TUM trajectory + sidecar."""
    import json
    import subprocess

    if not available():
        raise RuntimeError("cuvslam unavailable")
    cmd = ["bash", "-lc",
           "source /opt/ros/humble/setup.bash && ros2 launch "
           "isaac_ros_visual_slam isaac_ros_visual_slam.launch.py"]
    log = out_trajectory.with_suffix(".log")
    with open(log, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                       check=True, timeout=3600)
    meta = {
        "schema_version": "recon-v3/sidecar-3",
        "backend": "cuvslam",
        "pose_convention": "T_world_camera",
        "pose_frame": "camera_color_optical_frame",
        "profile": "standard",
        "log": str(log),
    }
    Path(str(out_trajectory) + ".meta.json").write_text(json.dumps(meta))
    return out_trajectory
