"""Unit tests for P0-2: RGB-D Alignment Contract Proof (rgbd_alignment.py)."""
import json
import numpy as np
import pytest
from pathlib import Path

from auto_mobility.dataset.rgbd_alignment import (
    AlignmentContract,
    prove_alignment,
    save_contract,
    load_contract,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

K_COLOR = np.array([
    [615.0,   0.0, 320.0],
    [  0.0, 615.0, 240.0],
    [  0.0,   0.0,   1.0],
])

K_DEPTH_NATIVE = np.array([
    [580.0,   0.0, 316.0],
    [  0.0, 580.0, 236.0],
    [  0.0,   0.0,   1.0],
])

T_COLOR_DEPTH = np.eye(4)  # identity extrinsic (co-located)

SIZE_COLOR = (640, 480)
SIZE_DEPTH = (640, 480)
SIZE_DEPTH_NATIVE = (640, 480)  # same size but different K

ALIGNED_TOPIC = "/camera/aligned_depth_to_color/image_raw"
NATIVE_TOPIC  = "/camera/depth/image_rect_raw"


# ---------------------------------------------------------------------------
# 1. aligned_depth topic name -> DRIVER_ALIGNED
# ---------------------------------------------------------------------------

def test_aligned_depth_topic_proves_driver_aligned():
    contract = prove_alignment(
        color_frame_id="camera_color_optical_frame",
        depth_frame_id="camera_color_optical_frame",
        depth_topic=ALIGNED_TOPIC,
        K_color=K_COLOR,
        K_depth=None,
        T_color_depth=None,
        image_size_color=SIZE_COLOR,
        image_size_depth=SIZE_DEPTH,
    )
    assert contract.method == "DRIVER_ALIGNED", (
        f"Expected DRIVER_ALIGNED, got {contract.method}. Evidence: {contract.evidence}"
    )
    assert contract.is_proven()
    assert contract.reject_reason is None


# ---------------------------------------------------------------------------
# 2. Same frame_id but no size / topic evidence -> UNPROVEN
# ---------------------------------------------------------------------------

def test_frame_id_equality_alone_is_insufficient():
    """frame_id equality must NEVER be the sole proof of alignment."""
    contract = prove_alignment(
        color_frame_id="camera_optical_frame",
        depth_frame_id="camera_optical_frame",   # same frame_id
        depth_topic=NATIVE_TOPIC,                 # no 'aligned' in topic
        K_color=K_COLOR,
        K_depth=None,
        T_color_depth=None,
        image_size_color=SIZE_COLOR,
        image_size_depth=(848, 480),              # different size -> not driver-aligned
    )
    assert contract.method == "UNPROVEN", (
        f"frame_id equality alone must not prove alignment; got {contract.method}"
    )
    assert not contract.is_proven()
    assert contract.reject_reason is not None
    # frame_id match should be noted but marked as insufficient
    assert any("noted but NOT sufficient" in e for e in contract.evidence)


# ---------------------------------------------------------------------------
# 3. Native depth with K_depth + T_color_depth -> REPROJECTED
# ---------------------------------------------------------------------------

def test_reprojection_method_with_extrinsics():
    contract = prove_alignment(
        color_frame_id="camera_color_optical_frame",
        depth_frame_id="camera_depth_optical_frame",
        depth_topic=NATIVE_TOPIC,
        K_color=K_COLOR,
        K_depth=K_DEPTH_NATIVE,
        T_color_depth=T_COLOR_DEPTH,
        image_size_color=SIZE_COLOR,
        image_size_depth=SIZE_DEPTH_NATIVE,
    )
    assert contract.method == "REPROJECTED", (
        f"Expected REPROJECTED, got {contract.method}. Evidence: {contract.evidence}"
    )
    assert contract.is_proven()
    assert contract.reject_reason is None


# ---------------------------------------------------------------------------
# 4. Different K -> different fingerprint
# ---------------------------------------------------------------------------

def test_contract_fingerprint_changes_with_K():
    K_alt = K_COLOR.copy()
    K_alt[0, 0] = 700.0  # different fx

    c1 = prove_alignment(
        color_frame_id="f", depth_frame_id="f",
        depth_topic=ALIGNED_TOPIC,
        K_color=K_COLOR, K_depth=None, T_color_depth=None,
        image_size_color=SIZE_COLOR, image_size_depth=SIZE_DEPTH,
    )
    c2 = prove_alignment(
        color_frame_id="f", depth_frame_id="f",
        depth_topic=ALIGNED_TOPIC,
        K_color=K_alt, K_depth=None, T_color_depth=None,
        image_size_color=SIZE_COLOR, image_size_depth=SIZE_DEPTH,
    )
    assert c1.contract_fingerprint != c2.contract_fingerprint, (
        "Fingerprints must differ when K_color differs"
    )


# ---------------------------------------------------------------------------
# 5. Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    contract = prove_alignment(
        color_frame_id="camera_color_optical_frame",
        depth_frame_id="camera_color_optical_frame",
        depth_topic=ALIGNED_TOPIC,
        K_color=K_COLOR,
        K_depth=None,
        T_color_depth=None,
        image_size_color=SIZE_COLOR,
        image_size_depth=SIZE_DEPTH,
    )
    save_contract(contract, tmp_path)

    loaded = load_contract(tmp_path)
    assert loaded is not None
    assert loaded.method == contract.method
    assert loaded.contract_fingerprint == contract.contract_fingerprint
    assert loaded.depth_topic == contract.depth_topic
    assert loaded.evidence == contract.evidence
    assert loaded.is_proven() == contract.is_proven()


# ---------------------------------------------------------------------------
# 6. Native depth without extrinsics -> UNPROVEN
# ---------------------------------------------------------------------------

def test_native_depth_without_extrinsics_is_unproven():
    """Native depth topic, no T_color_depth -> cannot prove REPROJECTED."""
    contract = prove_alignment(
        color_frame_id="camera_color_optical_frame",
        depth_frame_id="camera_depth_optical_frame",
        depth_topic=NATIVE_TOPIC,
        K_color=K_COLOR,
        K_depth=K_DEPTH_NATIVE,
        T_color_depth=None,          # no extrinsic -> REPROJECTED not possible
        image_size_color=SIZE_COLOR,
        image_size_depth=SIZE_DEPTH_NATIVE,
    )
    assert contract.method == "UNPROVEN", (
        f"Without T_color_depth, method must be UNPROVEN; got {contract.method}"
    )
    assert not contract.is_proven()
    assert contract.reject_reason is not None


# ---------------------------------------------------------------------------
# 7. load_contract returns None when file is absent
# ---------------------------------------------------------------------------

def test_load_contract_returns_none_when_missing(tmp_path):
    result = load_contract(tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# 8. UNPROVEN is not is_proven
# ---------------------------------------------------------------------------

def test_unproven_is_not_proven():
    c = AlignmentContract(
        method="UNPROVEN",
        depth_topic="/camera/depth/image_raw",
        depth_frame_id="depth_frame",
        color_frame_id="color_frame",
        same_frame_id=False,
        K_color=K_COLOR.tolist(),
        K_depth=None,
        T_color_depth=None,
        depth_scale=1.0,
        image_size_color=[640, 480],
        image_size_depth=[640, 480],
        evidence=["FAIL: no evidence"],
        reject_reason="no valid evidence",
        contract_fingerprint="deadbeef12345678",
    )
    assert not c.is_proven()


# ---------------------------------------------------------------------------
# 9. Reprojection: Identity transform retains depth map
# ---------------------------------------------------------------------------

def test_reproject_depth_to_color_identity():
    from auto_mobility.dataset.rgbd_alignment import reproject_depth_to_color

    H, W = 480, 640
    depth_in = np.zeros((H, W), dtype=np.uint16)
    # Put a 2m depth object in the center
    depth_in[200:280, 300:340] = 2000

    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    T_identity = np.eye(4)

    reprojected = reproject_depth_to_color(
        depth_native=depth_in,
        K_depth=K,
        K_color=K,
        T_color_depth=T_identity,
        out_shape=(H, W),
        depth_scale=1000.0,
    )

    # In center region, depth must be preserved
    assert reprojected[240, 320] == 2000
    assert np.count_nonzero(reprojected) == np.count_nonzero(depth_in)


# ---------------------------------------------------------------------------
# 10. Reprojection: Z-buffer collision test (nearer pixel wins)
# ---------------------------------------------------------------------------

def test_reproject_depth_to_color_zbuffer_collision():
    """When two 3D points map to the same color pixel, the nearer point (smaller z) must win."""
    from auto_mobility.dataset.rgbd_alignment import reproject_depth_to_color

    H, W = 480, 640
    depth_in = np.zeros((H, W), dtype=np.uint16)

    # Place a background point at (u=320, v=240, z=5.0m)
    # and a foreground point at (u=320, v=240, z=1.0m)
    # Both along optical axis (u=320, v=240)
    # In depth map, we can place two points in adjacent pixels that both map to the same color pixel
    # With optical center at 320, 240:
    # Point A: u=320, v=240, z=3000 mm -> maps to (320, 240) in color
    # Point B: u=321, v=240, z=1000 mm -> with slight shift, or let's use an extrinsic translation:
    # Let K_depth = K_color = 500 focal length, cx=320, cy=240.
    # Point 1 at (320, 240) with depth 4000mm -> 3D = (0, 0, 4)
    # Point 2 at (330, 240) with depth 1000mm -> 3D = ((330-320)*1/500, 0, 1) = (0.02, 0, 1)
    # If color camera is shifted in X by +0.02m: T_color_depth has t_x = -0.02
    # Then Point 1: p_c = (-0.02, 0, 4) -> u_c = 500*(-0.02)/4 + 320 = -2.5 + 320 = 317.5 ~ 318
    # Point 2: p_c = (0.02 - 0.02, 0, 1) = (0, 0, 1) -> u_c = 500*(0)/1 + 320 = 320
    # Let's create an exact collision:
    # We want two depth pixels (u1, v1) at z1 and (u2, v2) at z2 that map to exact same (u_c, v_c).
    # Point 1: (u1=320, v1=240, z1=4.0m) -> 3D=(0,0,4). With T=eye, u_c=320, v_c=240, z_c=4m
    # Point 2: (u2=320, v2=240, z2=1.0m) -> In depth_in, we put depth_in[240, 320] = 1000 (near)
    # But to test multiple inputs mapping to one, let's have two depth points that project to same pixel.
    # If K_depth has smaller resolution than K_color, or slight angle:
    # Say K_depth fx=250, K_color fx=500.
    # Point 1: u_d=160, z=2m -> x_d=(160-160)*2/250=0 -> u_c = 500*0/2 + 320 = 320, z=2m
    # Point 2: u_d=161, z=1m -> x_d=(161-160)*1/250=0.004 -> u_c = 500*0.004/1 + 320 = 322
    # If Point 3 at u_d=160 with depth 5m, and Point 4 at u_d=160 with depth 1m:
    # We test with depth map containing foreground (1000mm) and background (5000mm) overlapping in projection.
    depth_in[240, 320] = 5000  # Far point
    depth_in[240, 321] = 1000  # Near point

    K_d = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    K_c = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])

    # Color camera shifted such that (321, 240, 1.0m) maps to (320, 240) in color:
    # x_d(321) = (321-320)*1/500 = 1/500 = 0.002m
    # x_d(320) = 0
    # Let T_c_d translate by tx = -0.002m, then Point 2 has x_c = 0.002 - 0.002 = 0 -> u_c = 320 (z=1m)
    # Point 1 has x_c = 0 - 0.002 = -0.002 -> u_c = 500*(-0.002)/5 + 320 = -0.2 + 320 = 319.8 ~ 320 (z=5m)
    # BOTH map to u_c = 320, v_c = 240!
    T_shift = np.eye(4)
    T_shift[0, 3] = -0.002

    reprojected = reproject_depth_to_color(
        depth_native=depth_in,
        K_depth=K_d,
        K_color=K_c,
        T_color_depth=T_shift,
        out_shape=(H, W),
        depth_scale=1000.0,
    )

    # Collision at (240, 320): Nearest point (1000mm) must win over far point (5000mm)
    assert reprojected[240, 320] == 1000, (
        f"Z-buffer collision failure: expected near depth 1000mm, got {reprojected[240, 320]}mm"
    )

