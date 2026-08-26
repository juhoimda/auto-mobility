"""Conservative pose refinement via short-baseline RGB-D ICP pose graph (#34-36).

Keyframes by translation/rotation novelty. Camera-frame clouds per keyframe;
adjacent point-to-point ICP constraints -> o3d global optimization ->
pose-dependent guard validation -> ACCEPT / ROLLBACK.

Convention: poses are T_world_camera; pose-graph node pose == T_world_camera
(so a node pose directly maps its camera-frame cloud into the world/global
frame). Edge transformation maps camera_i into camera_{i-1}:
    T_edge = inv(T_wc_prev) @ T_wc_cur,  constraint: P_prev @ T_edge == P_cur.
Loop closures are left to RTAB's optimized graph; we only smooth drift.

Guard metrics are POSE-DEPENDENT: heldout/structural residual is the median
cross-cloud alignment distance of adjacent keyframe pairs under the candidate
poses. A collapsed or folded trajectory inflates it and is rolled back (#35).

Complexity: O(K * icp) for K<=300 keyframes + O(pairs * nn-query) guards.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def select_keyframes(frames, pose_by_frame, max_kf=300,
                     min_trans_m=0.03, min_rot_rad=0.02):
    kfs, last = [], None
    for f in frames:
        T = pose_by_frame.get(f.frame_id)
        if T is None:
            continue
        if last is None:
            kfs.append(f.frame_id)
            last = T
            continue
        dt = float(np.linalg.norm(T[:3, 3] - last[:3, 3]))
        cos = np.clip((np.trace(last[:3, :3].T @ T[:3, :3]) - 1.0) / 2.0, -1, 1)
        if dt >= min_trans_m or float(np.arccos(cos)) >= min_rot_rad:
            kfs.append(f.frame_id)
            last = T
    if len(kfs) > max_kf:
        stride = int(np.ceil(len(kfs) / max_kf))
        kfs = kfs[::stride]
    return kfs


def _depth_stats(poses, load_depth_mm):
    """Pose-INDEPENDENT sanity stat kept for reporting only (not a guard)."""
    meds = []
    for fid in list(poses)[::20]:
        d = load_depth_mm(fid)
        if d is None:
            continue
        z = d.astype(np.float32)[d > 0]
        if len(z) > 100:
            meds.append(float(np.median(z)))
    return float(np.std(meds)) if len(meds) > 2 else 0.0


def _load_cloud_cam(load_depth_mm, fid, intrinsic):
    import open3d as o3d

    d = load_depth_mm(fid)
    if d is None:
        return None
    pc = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(d.astype(np.uint16)), intrinsic,
        extrinsic=np.eye(4), depth_scale=1000.0, depth_trunc=5.0
    ).voxel_down_sample(0.03)
    return pc if len(pc.points) >= 500 else None


def _world_points(pc_cam, T_wc):
    pts = np.asarray(pc_cam.points)
    return pts @ np.asarray(T_wc)[:3, :3].T + np.asarray(T_wc)[:3, 3]


def alignment_residual_mm(clouds_cam, poses_wc, order, max_pairs: int = 24) -> float:
    """Median adjacent-pair alignment distance (mm) under the given poses.

    Pose-dependent: collapses/folds inflate this metric even when raw
    inter-pose distances shrink.
    """
    if not _HAS_SCIPY or len(order) < 2:
        return -1.0
    step = max(1, (len(order) - 1) // max_pairs)
    pairs = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, step)]
    meds = []
    for a, b in pairs:
        ca, cb = clouds_cam.get(a), clouds_cam.get(b)
        Ta, Tb = poses_wc.get(a), poses_wc.get(b)
        if ca is None or cb is None or Ta is None or Tb is None:
            continue
        tgt = _world_points(ca, Ta)
        src = _world_points(cb, Tb)
        tree = cKDTree(tgt)
        dists, _ = tree.query(src, k=1)
        meds.append(float(np.median(dists)))
    return float(np.mean(meds)) * 1000.0 if meds else -1.0


def refine_trajectory(frames, pose_by_frame, load_depth_mm, K,
                      width=None, height=None) -> dict:
    import open3d as o3d

    from auto_mobility.reconstruction.pose.refiner import (
        PoseQualitySnapshot, evaluate_refinement,
    )

    w = int(width) if width else int(K.shape[0] and 640)
    h = int(height) if height else 480
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    kfs = select_keyframes(frames, pose_by_frame)

    def snapshot(poses, clouds=None, order=None):
        ids = sorted(poses)
        steps = [float(np.linalg.norm(poses[a][:3, 3] - poses[b][:3, 3]))
                 for a, b in zip(ids[::10], ids[10::10])]
        disc = float(np.percentile(steps, 99)) * 1000.0 if steps else 1e-3
        if clouds and order:
            align = alignment_residual_mm(clouds, poses, order)
            align = max(align, 1e-2)
        else:
            align = 1e-2
        return PoseQualitySnapshot(heldout_residual=align,
                                   loop_consistency=max(disc, 1e-3),
                                   structural_residual=align,
                                   discontinuity=max(disc, 1e-3))

    # camera-frame clouds built once; reused by ICP and by both guard snapshots
    clouds_cam, order = {}, []
    for fid in kfs:
        if pose_by_frame.get(fid) is None:
            continue
        pc = _load_cloud_cam(load_depth_mm, fid, intrinsic)
        if pc is not None:
            clouds_cam[fid] = pc
            order.append(fid)

    before = snapshot(pose_by_frame, clouds_cam, order)

    pg = o3d.pipelines.registration.PoseGraph()
    icp = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    prev_fid = None
    for i, fid in enumerate(order):
        T_wc = pose_by_frame[fid]
        if i == 0:
            pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.asarray(T_wc)))
        else:
            init = np.linalg.inv(pose_by_frame[prev_fid]) @ T_wc
            reg = o3d.pipelines.registration.registration_icp(
                clouds_cam[fid], clouds_cam[prev_fid], 0.05, init, icp,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                clouds_cam[fid], clouds_cam[prev_fid], 0.05, reg.transformation)
            pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                i - 1, i, reg.transformation, info, uncertain=False))
            pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.asarray(T_wc)))
        prev_fid = fid

    if len(pg.nodes) < 6:
        return {"accepted": False, "reason": "insufficient graph",
                "pose_by_frame": pose_by_frame}

    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25, reference_node=0))

    refined = dict(pose_by_frame)
    for i, fid in enumerate(order):
        refined[fid] = np.asarray(pg.nodes[i].pose)

    after = snapshot(refined, clouds_cam, order)
    decision = evaluate_refinement(before, after)
    return {"accepted": decision.accepted, "reason": decision.reason,
            "decision": decision.to_dict(), "before": before.as_dict(),
            "after": after.as_dict(),
            "pose_by_frame": refined if decision.accepted else pose_by_frame}
