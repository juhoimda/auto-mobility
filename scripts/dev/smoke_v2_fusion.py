"""GPU smoke: integrate a base2 frame subset via the V2 fusion path."""

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from auto_mobility.reconstruction.fusion import (
    FusionInput,
    choose_fusion_backend,
    integrate_frames,
)
from auto_mobility.reconstruction.runtime import probe_machine

DATA = Path("ros2_data/frames/base2")
N_FRAMES = 200
STRIDE = 1


def main():
    rows = list(csv.DictReader(open(DATA / "frames.csv")))
    cam = json.load(open(DATA / "camera_info.json"))
    K = np.array(cam["K"] if "K" in cam else [
        [cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=np.float64)
    if K.ndim == 1:
        K = K.reshape(3, 3)
    w, h = int(cam.get("width", 640)), int(cam.get("height", 480))

    ids = [r["frame_id"] for r in rows[: N_FRAMES * STRIDE : STRIDE]]

    def rgb_path(i):
        return DATA / f"rgb/{int(i):06d}.png"

    def depth_path(i):
        return DATA / f"depth/{int(i):06d}.png"

    load_rgb = lambda i: cv2.imread(str(rgb_path(i)))
    load_depth = lambda i: cv2.imread(str(depth_path(i)), cv2.IMREAD_UNCHANGED)

    t0 = time.time()
    straight = {}
    for n, fid in enumerate(ids):
        T = np.eye(4)
        T[0, 3] = 0.01 * n
        straight[fid] = T
    fi = FusionInput(frame_ids=ids, load_depth_mm=load_depth, load_rgb=load_rgb,
                     pose_by_frame=straight)

    profile = probe_machine(probe_open3d=True)
    backend = choose_fusion_backend(profile)
    print("backend:", backend.to_dict())

    out = integrate_frames(
        fi, K, w, h, voxel_m=0.010, trunc_mult=4.0,
        bbox_diag_m=3.0, use_cuda=True,
    )
    print(json.dumps({**out.to_dict(), "wall_s": round(time.time() - t0, 1)}, indent=1))
    assert out.mesh_triangles > 10000, "degenerate reconstruction"


if __name__ == "__main__":
    main()
