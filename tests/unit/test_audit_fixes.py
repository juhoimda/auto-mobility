"""Regression tests for the review.md audit fixes.

Covers:
  - §9/§11 truthful resource accounting (no min() RAM clamp)
  - §10/§11 watchdog VRAM baseline/delta semantics + lazy baseline
  - §50 trajectory cache content verification (sha256 + dataset fingerprint)
  - pose association gap rejection (no stale-pose bridging)
  - §15 bytes-per-block layout math
  - #41 texture baker bounded memory (chunk-size independence)
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from auto_mobility.reconstruction.appearance.texture_baker import bake_atlas


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_src(rel: str) -> str:
    return (_REPO_ROOT / "src" / rel).read_text()


# ------------------------------------------------- §9/§11 accounting -------

def test_fusion_ram_estimate_is_truthful_not_clamped():
    """RAM estimate must grow with block count and never be capped at budget."""
    from auto_mobility.reconstruction.pipeline.standard import (
        estimate_fusion_ram_mb)

    small = estimate_fusion_ram_mb(10_000)
    large = estimate_fusion_ram_mb(1_000_000)
    assert large > small
    # 1M blocks * 32768 B * 2.0 / 1e6 = 65_536 MB — the old min() clamp
    # would have reported ~4096 MB here (a lie to the scheduler).
    assert large == pytest.approx(65536, rel=0.01)


def test_no_min_clamp_left_in_standard_pipeline_source():
    src = _read_src("auto_mobility/reconstruction/pipeline/standard.py")
    assert "min(max(int(estimate_block_count" not in src
    # admission must consider RAM overruns too, not only VRAM
    assert "ram_want > avail_ram" in src


def test_scheduler_rejects_job_exceeding_vram_budget():
    """required_vram=6000 vs allowed 4000 -> reject; job must never run."""
    from auto_mobility.reconstruction.runtime.scheduler import (
        CapacityError, JobSpec, Scheduler)

    s = Scheduler(cpu_threads=4, ram_mb=8000, gpu_slots=1,
                  vram_mb=4000).start()
    try:
        ran = {"v": False}

        def job():
            ran["v"] = True

        with pytest.raises(CapacityError):
            s.submit(job, JobSpec(name="hog", gpu_slots=1, vram_mb=6000))
        assert not ran["v"], "over-budget VRAM job must not execute"
    finally:
        s.shutdown()


# ---------------------------------------------- §10/§11 watchdog semantics -

def test_vram_barrier_delta_semantics_pure():
    from auto_mobility.reconstruction.runtime.process import vram_barrier_breach

    # delta within budget: no kill even when absolute usage >> budget
    reason, base = vram_barrier_breach(7000, 6800, 500, None)
    assert reason is None and base == 6800
    # delta breach kills on used-baseline, not absolute
    reason, _ = vram_barrier_breach(7400, 6800, 500, None)
    assert reason and "job_delta" in reason
    # hard ceiling IS an absolute comparison
    reason, _ = vram_barrier_breach(8100, 1000, None, 8000)
    assert reason and "hard_ceiling" in reason


def test_watchdog_no_false_kill_when_baseline_sample_fails(
        tmp_path, monkeypatch):
    """Old code compared ABSOLUTE memory.used against the INCREMENTAL budget
    whenever the start-of-job baseline sample failed -> immediate false kill
    on machines with a large desktop/compositor baseline."""
    import auto_mobility.reconstruction.runtime.process as proc_mod

    calls = {"n": 0}

    def fake_sample():
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # initial baseline probe fails
        return (1200.0, 50.0, 20.0)  # absolute usage far above the budget

    monkeypatch.setattr(proc_mod, "_gpu_sample", fake_sample)
    out = proc_mod.run_monitored_process(
        ["python3", "-c", "import time; time.sleep(2)"],
        log_path=tmp_path / "w.log",
        timeout_s=15,
        poll_interval_s=0.05,
        gpu_limits={"vram_mb": 500},
    )
    assert out.ok, f"false kill: {out.to_dict()}"


def test_watchdog_delta_breach_kills_and_writes_barrier(tmp_path, monkeypatch):
    import auto_mobility.reconstruction.runtime.process as proc_mod

    calls = {"n": 0}

    def fake_sample():
        calls["n"] += 1
        return (1000.0 + 600.0 * calls["n"], 50.0, 20.0)

    monkeypatch.setattr(proc_mod, "_gpu_sample", fake_sample)
    log = tmp_path / "b.log"
    out = proc_mod.run_monitored_process(
        ["python3", "-c", "import time; time.sleep(30)"],
        log_path=log,
        timeout_s=60,
        poll_interval_s=0.05,
        gpu_limits={"vram_mb": 300},
    )
    assert not out.ok
    barrier = tmp_path / "b.barrier"
    assert barrier.is_file() and "job_delta" in barrier.read_text()


# ------------------------------------------------------------ §50 cache ---

def _write_traj(path: Path):
    lines = [f"{1.0 + i * 0.03:.6f} {i} 0 0 0 0 0 1" for i in range(12)]
    path.write_text("\n".join(lines) + "\n")


def _sidecar(path: Path, dataset_dir, *, sha=None, fp=True):
    # Strict provenance sidecar-1 for Task2
    import time as _time
    backend = "cuvslam" if "cuvslam" in path.name else "rtab"
    meta = {
        "schema_version": "recon-v4/sidecar-1",
        "backend": backend,
        "pose_convention": "T_world_camera",
        "pose_frame": "camera_color_optical_frame",
        "profile": "standard",
        "command_line": "pytest",
        "created_at_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "seed": None,
        "deterministic_mode": False,
        "gpu_model": "test_gpu",
        "open3d_version": "0.19.0",
        "open3d_cuda_available": "True",
        "backend_config_hash": "abc123",
        "worker_source_hash": "abc123",
        "git_sha": "abc123",
        "cuda_driver_version": "596.58",
        "cuda_runtime_version": "13.2",
    }
    if backend == "cuvslam":
        meta["cuvslam_version"] = "17.0.0"
    else:
        meta["rtab_version"] = "0.21.5"
        meta["rtab_binary_hash"] = "abc123"
    meta["trajectory_sha256"] = sha or hashlib.sha256(
        path.read_bytes()).hexdigest()
    meta["dataset_fingerprint"] = (
        _fp_of(dataset_dir) if fp else "stale-fingerprint")
    # alignment fingerprint: use dummy proven value matching dataset's contract if exists
    try:
        from auto_mobility.dataset.rgbd_alignment import load_contract
        c = load_contract(dataset_dir)
        meta["alignment_contract_fingerprint"] = c.contract_fingerprint if c and c.is_proven() else "dummy_align_fp"
    except Exception:
        meta["alignment_contract_fingerprint"] = "dummy_align_fp"
    meta["aligned_depth_artifact_fingerprint"] = "dummy_depth_fp"
    meta["source_frame_set_hash"] = meta["dataset_fingerprint"]
    meta["provenance_hash"] = hashlib.sha256(json.dumps({k:meta[k] for k in sorted(meta.keys())}, sort_keys=True).encode()).hexdigest()[:16]
    meta["trajectory_sidecar_sha256"] = hashlib.sha256(json.dumps({k:meta[k] for k in sorted(meta.keys()) if k!="trajectory_sidecar_sha256"}, sort_keys=True).encode()).hexdigest()
    Path(str(path) + ".meta.json").write_text(json.dumps(meta))


def _fp_of(dataset_dir):
    h = hashlib.sha256()
    h.update((Path(dataset_dir) / "frames.csv").read_bytes())
    cam = Path(dataset_dir) / "camera_info.json"
    if cam.is_file():
        h.update(cam.read_bytes())
    return h.hexdigest()[:16]


def test_trajectory_cache_valid_roundtrip(tmp_path):
    from auto_mobility.reconstruction.cli import _verify_trajectory_cache

    ds = tmp_path / "frames"
    ds.mkdir()
    (ds / "frames.csv").write_text("a\n")
    tp = tmp_path / "cuvslam_x_trajectory.txt"
    _write_traj(tp)
    _sidecar(tp, ds)
    assert _verify_trajectory_cache(tp, ds)


def test_trajectory_cache_rejects_tampered_content(tmp_path):
    from auto_mobility.reconstruction.cli import _verify_trajectory_cache

    ds = tmp_path / "frames"
    ds.mkdir()
    (ds / "frames.csv").write_text("a\n")
    tp = tmp_path / "cuvslam_x_trajectory.txt"
    _write_traj(tp)
    _sidecar(tp, ds)
    tp.write_text(tp.read_text() + "99 9 9 9 0 0 0 1\n")  # silent edit
    assert not _verify_trajectory_cache(tp, ds)


def test_trajectory_cache_rejects_missing_sidecar(tmp_path):
    from auto_mobility.reconstruction.cli import _verify_trajectory_cache

    tp = tmp_path / "cuvslam_y_trajectory.txt"
    _write_traj(tp)
    assert not _verify_trajectory_cache(tp, None)


def test_trajectory_cache_invalidates_on_dataset_change(tmp_path):
    from auto_mobility.reconstruction.cli import _verify_trajectory_cache

    ds = tmp_path / "frames"
    ds.mkdir()
    (ds / "frames.csv").write_text("v1\n")
    tp = tmp_path / "rtab_x_trajectory.txt"
    _write_traj(tp)
    _sidecar(tp, ds)
    assert _verify_trajectory_cache(tp, ds)
    # re-extracted frames change fingerprint -> stale cache must regenerate
    (ds / "frames.csv").write_text("v2-different-sync\n")
    assert not _verify_trajectory_cache(tp, ds)


def test_trajectory_cache_fail_closed_on_legacy_sidecar(tmp_path):
    from auto_mobility.reconstruction.cli import _verify_trajectory_cache

    ds = tmp_path / "frames"
    ds.mkdir()
    (ds / "frames.csv").write_text("a\n")
    tp = tmp_path / "cuvslam_z_trajectory.txt"
    _write_traj(tp)
    _sidecar(tp, ds, fp=False)  # legacy sidecar without fingerprint match
    assert not _verify_trajectory_cache(tp, ds)


# ------------------------------------------- pose association gap guard ---


class _NS:
    pass


def _traj(ts):
    t = _NS()
    t.timestamps = list(ts)
    t.positions = [[float(i), 0, 0] for i in range(len(ts))]
    t.orientations = [[0, 0, 0, 1]] * len(ts)
    return t


def _frame(fid, ts):
    f = _NS()
    f.frame_id = fid
    f.rgb_timestamp = ts
    return f


def test_nearest_pose_map_drops_frames_inside_tracking_gap():
    from auto_mobility.reconstruction.pipeline.standard import _nearest_pose_map

    # tracking-loss hole between t=3 and t=8
    traj = _traj([1.0, 2.0, 3.0, 8.0, 9.0])
    frames = [_frame(f"f{i}", 1.0 + i) for i in range(9)]  # t=1..9
    dropped = []
    poses = _nearest_pose_map(traj, frames, max_pose_gap_ms=200.0,
                              dropped=dropped)
    # frames inside the hole are DROPPED, not bridged by stale poses;
    # f6 (t=7) nearest pose is 8.0 -> 1000 ms -> dropped.
    # f7 (t=8) matches pose t=8.0 exactly -> kept.
    assert set(dropped) == {"f3", "f4", "f5", "f6"}, dropped
    for fid in ("f3", "f4", "f5", "f6"):
        assert fid not in poses
    assert {"f0", "f1", "f2", "f7", "f8"} <= set(poses)


def test_nearest_pose_map_keeps_dense_normal_frames():
    from auto_mobility.reconstruction.pipeline.standard import _nearest_pose_map

    ts = [i / 30.0 for i in range(30)]
    traj = _traj(ts)
    frames = [_frame(f"f{i}", t + 0.004) for i, t in enumerate(ts)]
    poses = _nearest_pose_map(traj, frames, max_pose_gap_ms=200.0)
    assert len(poses) == len(frames)
    assert not any(f.frame_id not in poses for f in frames)


# ------------------------------------------------------- §15 bpb layout ---

def test_bytes_per_block_matches_attr_layout_math():
    from auto_mobility.reconstruction.fusion.open3d_vbg import (
        _BYTES_PER_BLOCK_NO_COLOR, _bytes_per_block)

    # tsdf f32(4B) + weight f32(4B), 16^3 voxels/block, no color
    assert _bytes_per_block(store_color=False) == 16 ** 3 * 8
    assert _BYTES_PER_BLOCK_NO_COLOR == _bytes_per_block(False)
    # color adds float32x3 (12 B/voxel)
    assert _bytes_per_block(store_color=True) == 16 ** 3 * 20
    # uint16 weight variant
    assert _bytes_per_block(store_color=False,
                            weight_dtype_bytes=2) == 16 ** 3 * 6


def test_production_vbg_defaults_to_nocolor_attrs():
    """The production VBG must allocate (tsdf, weight) only — no color TSDF
    (final texture is baked from raw RGB frames)."""
    src = _read_src("auto_mobility/reconstruction/fusion/open3d_vbg.py")
    assert 'attr_names=("tsdf", "weight")' in src
    # color may only re-enter when PCD extraction explicitly requested
    assert 'extraction_mode in ("pcd_only", "mesh_and_pcd")' in src


def test_planner_and_runtime_share_block_byte_model():
    from auto_mobility.reconstruction.fusion.open3d_vbg import (
        _bytes_per_block, _planner_bytes_per_block)

    for color in (False, True):
        for wd in ("float32", "uint16"):
            expected = _bytes_per_block(
                store_color=color,
                weight_dtype_bytes=(2 if wd == "uint16" else 4))
            assert _planner_bytes_per_block(color, wd) == expected


# -------------------------------------------------- #41 texture chunking --

def _cube_mesh(size=1.0):
    s = size / 2.0
    vertices = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float64)
    triangles = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ])
    return vertices, triangles


def _bake_scene(tmp_path, tri_chunk):
    """Camera looks along +z (identity rotation) so projections are valid."""
    import cv2

    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    img_a = np.zeros((480, 640, 3), dtype=np.uint8)
    img_a[:, :, 2] = 255  # BGR red
    img_b = np.zeros((480, 640, 3), dtype=np.uint8)
    img_b[:, :, 0] = 255  # BGR blue
    views = [(0, img_a), (1, img_b)]
    poses = {0: np.eye(4), 1: np.eye(4)}
    poses[0][:3, 3] = [0, 0, -3.0]   # near camera
    poses[1][:3, 3] = [0, 0, -10.0]  # far camera (lower score)
    vertices, triangles = _cube_mesh()
    out = tmp_path / f"bake_{tri_chunk}"
    return bake_atlas(vertices, triangles, views, K, poses, scene=None,
                      out_dir=out, name="model", max_views_per_tri=3,
                      tri_chunk=tri_chunk)


def test_texture_baker_chunk_size_independence(tmp_path):
    """O(chunk*K) streaming refactor must be output-identical to the dense
    behavior regardless of tri_chunk size."""
    import cv2

    r_small = _bake_scene(tmp_path, tri_chunk=1)
    r_big = _bake_scene(tmp_path, tri_chunk=100000)
    assert r_small.obj_path.read_text() == r_big.obj_path.read_text()
    assert r_small.untextured_faces == r_big.untextured_faces
    a1 = cv2.imread(str(r_small.atlas_paths[0]))
    a2 = cv2.imread(str(r_big.atlas_paths[0]))
    assert a1 is not None and a2 is not None
    assert np.array_equal(a1, a2)


def test_texture_baker_prefers_near_view_colors(tmp_path):
    """With two valid views, the higher-scoring (near) view must dominate
    blended face colors."""
    import cv2

    r = _bake_scene(tmp_path, tri_chunk=4)
    atlas = cv2.imread(str(r.atlas_paths[0]))[..., ::-1]  # back to RGB
    cells = atlas[atlas.sum(axis=2) > 0]
    assert len(cells) > 0, "faces must receive colors"
    # red (near view) dominates over blue (far view)
    assert int(cells[:, 0].astype(int).sum()) > \
        int(cells[:, 2].astype(int).sum())


def test_texture_baker_does_not_allocate_full_txv_matrices(tmp_path):
    """Regression (#41): dense S[T,V]/C[T,V,3] allocations must stay gone."""
    src = _read_src("auto_mobility/reconstruction/appearance/"
                    "texture_baker.py")
    assert "np.full((T_count, len(view_ids))" not in src
    assert "np.zeros((T_count, len(view_ids)" not in src
