"""Reconstruction V2: ROSBAG RGB-D -> SLAM -> geometry -> textured OBJ.

Package boundaries (hard):
    algorithm != orchestration != resource scheduling != artifact/cache != evaluation != reporting

Conventions (single source of truth):
    pose          T_world_camera, 4x4 float64, meters
    depth         uint16 millimeters (scale 1000)
    timestamps    float64 seconds
    quaternions   scalar-last xyzw
"""

from auto_mobility.reconstruction.model import (
    CameraIntrinsics,
    FrameMeta,
    invert_pose,
    SCHEMA_VERSION,
)
from auto_mobility.reconstruction.config import (
    ReconstructionConfig,
    default_config,
    load_config,
)

__all__ = [
    "CameraIntrinsics",
    "FrameMeta",
    "SCHEMA_VERSION",
    "ReconstructionConfig",
    "default_config",
    "load_config",
    "invert_pose",
]
