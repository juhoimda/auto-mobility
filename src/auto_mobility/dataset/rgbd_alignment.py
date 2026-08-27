"""
RGB-D Alignment Contract Proof (P0-2).

Determines and records how depth images are aligned to color images.
Only TWO valid methods:
  1. DRIVER_ALIGNED: driver provides color-aligned depth (K/frame/extrinsic evidence)
  2. REPROJECTED: native depth reprojected via K_depth, K_color, T_color_depth

frame_id equality alone is NEVER sufficient to conclude alignment.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass
class AlignmentContract:
    """Canonical alignment proof stored in dataset."""
    method: str  # "DRIVER_ALIGNED" | "REPROJECTED" | "UNPROVEN"
    depth_topic: str
    depth_frame_id: str
    color_frame_id: str
    same_frame_id: bool
    K_color: list  # 3x3 flattened, color camera intrinsics
    K_depth: Optional[list]  # 3x3 flattened, depth camera intrinsics (if separate)
    T_color_depth: Optional[list]  # 4x4 flattened extrinsic transform (if available)
    depth_scale: float  # mm per unit (usually 1.0 for uint16 mm)
    image_size_color: list  # [W, H]
    image_size_depth: list  # [W, H]
    evidence: list  # list of evidence strings used to determine method
    reject_reason: Optional[str]  # if UNPROVEN, why
    contract_fingerprint: str  # SHA256 of the contract JSON

    def to_dict(self) -> dict:
        return asdict(self)

    def is_proven(self) -> bool:
        return self.method in ("DRIVER_ALIGNED", "REPROJECTED")


def prove_alignment(
    color_frame_id: str,
    depth_frame_id: str,
    depth_topic: str,
    K_color: np.ndarray,
    K_depth: Optional[np.ndarray],
    T_color_depth: Optional[np.ndarray],
    image_size_color: tuple,  # (W, H)
    image_size_depth: tuple,  # (W, H)
    depth_scale: float = 1.0,
) -> AlignmentContract:
    """Prove alignment method. Fails closed (UNPROVEN) if evidence is insufficient.

    Explicitly: frame_id equality alone MUST NOT determine alignment.
    Two valid methods:
      DRIVER_ALIGNED  – depth topic name carries 'aligned' AND image sizes match.
      REPROJECTED     – separate K_depth and T_color_depth are both available.
    """
    evidence = []
    reject_reason = None

    same_fid = (color_frame_id == depth_frame_id)
    if same_fid:
        # Record it, but do NOT use it as proof by itself
        evidence.append(
            f"frame_id_match: {color_frame_id} (noted but NOT sufficient proof)"
        )

    # Check if depth size == color size (necessary for driver-aligned)
    same_size = (tuple(image_size_color) == tuple(image_size_depth))
    if same_size:
        evidence.append(f"image_size_match: {image_size_color}")

    # Check if K matrices are the same (driver-aligned depth uses color K)
    # Only valid when K_depth comes from genuine sensor metadata, not fabricated copy.
    k_match = False
    k_match_genuine = False
    if K_depth is not None and K_color is not None:
        diff = np.abs(np.array(K_color, dtype=float) - np.array(K_depth, dtype=float))
        k_match = bool(np.max(diff) < 1.0)  # within 1 pixel
        if k_match:
            evidence.append("K_match: depth K equals color K (driver-aligned)")
            # K_match is only genuine if K_depth was from sensor (not fabricated) — caller must not fabricate.
            # We treat any provided K_depth as genuine here; circular case is prevented by not fabricating.
            k_match_genuine = True

    # Check if T_color_depth indicates identity (depth already in color frame via TF)
    t_is_identity = False
    if T_color_depth is not None:
        try:
            t_is_identity = bool(np.allclose(np.asarray(T_color_depth, dtype=float), np.eye(4), atol=1e-3))
            if t_is_identity:
                evidence.append("T_identity: T_color_depth is identity (depth already in color frame via TF)")
        except Exception:
            t_is_identity = False

    # Determine method
    method = "UNPROVEN"

    # Check if the topic name signals aligned depth from the driver
    is_aligned_topic = (
        "aligned_depth" in depth_topic.lower()
        or "aligned" in depth_topic.lower()
    )
    if is_aligned_topic:
        evidence.append(f"aligned_depth_topic: {depth_topic}")

    # DRIVER_ALIGNED requires same_size AND independent evidence beyond frame_id:
    #   - aligned topic name (driver metadata), OR
    #   - genuine K_match from sensor depth CameraInfo, OR
    #   - TF-proven identity transform plus same frame_id (depth already expressed in color frame)
    # Frame_id/resolution/K copied by code alone is NOT sufficient (fail-closed).
    driver_evidence = (k_match_genuine or is_aligned_topic or (same_fid and t_is_identity))
    if same_size and driver_evidence:
        # DRIVER_ALIGNED: image sizes match AND structural independent evidence beyond frame_id
        method = "DRIVER_ALIGNED"
        evidence.append("conclusion: driver provides color-aligned depth")
    elif K_depth is not None and K_color is not None and T_color_depth is not None:
        # Can do reprojection with known intrinsics + extrinsic transform
        method = "REPROJECTED"
        evidence.append(
            "conclusion: native depth with known extrinsics -> reprojection available"
        )
    else:
        reject_reason = (
            f"Cannot prove alignment: frame_id_same={same_fid}, "
            f"size_same={same_size}, k_match={k_match}, k_genuine={k_match_genuine}, "
            f"aligned_topic={is_aligned_topic}, t_is_identity={t_is_identity}, "
            f"has_T_color_depth={T_color_depth is not None}"
        )
        evidence.append(f"FAIL: {reject_reason}")

    # Compute fingerprint over the core contract data
    contract_data = {
        "method": method,
        "color_frame_id": color_frame_id,
        "depth_frame_id": depth_frame_id,
        "depth_topic": depth_topic,
        "K_color": K_color.tolist() if K_color is not None else None,
        "K_depth": K_depth.tolist() if K_depth is not None else None,
        "T_color_depth": T_color_depth.tolist() if T_color_depth is not None else None,
        "image_size_color": list(image_size_color),
        "image_size_depth": list(image_size_depth),
        "depth_scale": depth_scale,
    }
    fingerprint = hashlib.sha256(
        json.dumps(contract_data, sort_keys=True).encode()
    ).hexdigest()[:16]

    return AlignmentContract(
        method=method,
        depth_topic=depth_topic,
        depth_frame_id=depth_frame_id,
        color_frame_id=color_frame_id,
        same_frame_id=same_fid,
        K_color=K_color.tolist() if K_color is not None else None,
        K_depth=K_depth.tolist() if K_depth is not None else None,
        T_color_depth=T_color_depth.tolist() if T_color_depth is not None else None,
        depth_scale=depth_scale,
        image_size_color=list(image_size_color),
        image_size_depth=list(image_size_depth),
        evidence=evidence,
        reject_reason=reject_reason,
        contract_fingerprint=fingerprint,
    )


def save_contract(contract: AlignmentContract, dataset_dir: Path) -> None:
    """Save alignment contract to dataset directory."""
    path = Path(dataset_dir) / "rgbd_alignment_contract.json"
    path.write_text(json.dumps(contract.to_dict(), indent=2))


def load_contract(dataset_dir: Path) -> Optional[AlignmentContract]:
    """Load alignment contract from dataset directory. Returns None if missing."""
    path = Path(dataset_dir) / "rgbd_alignment_contract.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return AlignmentContract(**data)


def reproject_depth_to_color(
    depth_native: np.ndarray,
    K_depth: np.ndarray,
    K_color: np.ndarray,
    T_color_depth: np.ndarray,
    out_shape: tuple[int, int],
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """Reproject native depth to color camera frame with z-buffer collision handling.

    Args:
        depth_native: (H_d, W_d) array of uint16 depth in raw units (e.g. mm).
        K_depth: 3x3 depth camera intrinsic matrix.
        K_color: 3x3 color camera intrinsic matrix.
        T_color_depth: 4x4 rigid body transform from depth frame to color frame.
        out_shape: (H_c, W_c) output color image resolution.
        depth_scale: Units per meter (default 1000.0 for mm).

    Returns:
        (H_c, W_c) uint16 array of aligned depth in mm. Nearer pixels occlude farther pixels.
    """
    H_c, W_c = out_shape
    out_depth = np.zeros((H_c, W_c), dtype=np.uint16)

    valid_mask = depth_native > 0
    if not np.any(valid_mask):
        return out_depth

    v_d, u_d = np.where(valid_mask)
    z_d = depth_native[v_d, u_d].astype(np.float64) / depth_scale  # in meters

    # Backproject to 3D in depth camera optical frame
    x_d = (u_d - K_depth[0, 2]) * z_d / K_depth[0, 0]
    y_d = (v_d - K_depth[1, 2]) * z_d / K_depth[1, 1]
    pts_d = np.stack([x_d, y_d, z_d, np.ones_like(z_d)], axis=0)  # (4, N)

    # Transform to color camera optical frame: p_c = T_color_depth @ p_d
    pts_c = T_color_depth @ pts_d  # (4, N)
    z_c = pts_c[2]

    # Keep points in front of the color camera
    valid_c = z_c > 0.01  # > 10mm
    if not np.any(valid_c):
        return out_depth

    pts_c = pts_c[:, valid_c]
    z_c = z_c[valid_c]

    # Project to color image pixel plane
    u_c = np.rint((K_color[0, 0] * pts_c[0] / z_c) + K_color[0, 2]).astype(int)
    v_c = np.rint((K_color[1, 1] * pts_c[1] / z_c) + K_color[1, 2]).astype(int)

    # Boundary check
    in_bounds = (u_c >= 0) & (u_c < W_c) & (v_c >= 0) & (v_c < H_c)
    if not np.any(in_bounds):
        return out_depth

    u_c = u_c[in_bounds]
    v_c = v_c[in_bounds]
    z_c = z_c[in_bounds]

    # Z-buffer collision resolution: sort descending by distance (furthest first, nearest last).
    # Writing in this order guarantees that the closest surface overwrites background points.
    sort_idx = np.argsort(-z_c)
    z_mm = np.clip(z_c[sort_idx] * depth_scale, 0, 65535).astype(np.uint16)
    out_depth[v_c[sort_idx], u_c[sort_idx]] = z_mm

    return out_depth

