"""
auto_mobility.dataset.frame_dataset

알고리즘 독립적인 Canonical Frame Dataset 로더 및 데이터 구조.
Rosbag에서 추출된 표준화된 RGB-D 프레임, CameraInfo, IMU, 메타데이터를 관리한다.
"""

import os
import json
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import cv2


@dataclass
class FrameMetadata:
    frame_id: int
    rgb_timestamp: float
    depth_timestamp: float
    rgb_path: str
    depth_path: str
    rgb_depth_dt_ms: float
    bag_timestamp: float = 0.0
    camera_frame_id: str = "camera_color_optical_frame"
    depth_camera_frame_id: str = ""
    width: int = 640
    height: int = 480


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
    distortion_model: str = "plumb_bob"
    distortion_coefficients: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    K: List[float] = field(default_factory=list)
    R: List[float] = field(default_factory=list)
    P: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.K:
            self.K = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        if not self.R:
            self.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        if not self.P:
            self.P = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CameraIntrinsics":
        if "fx" in data and "fy" in data and "cx" in data and "cy" in data:
            return cls(
                fx=float(data["fx"]),
                fy=float(data["fy"]),
                cx=float(data["cx"]),
                cy=float(data["cy"]),
                width=int(data.get("width", 640)),
                height=int(data.get("height", 480)),
                distortion_model=data.get("distortion_model", "plumb_bob"),
                distortion_coefficients=list(data.get("distortion_coefficients", [0.0]*5)),
                K=list(data.get("K", [])),
                R=list(data.get("R", [])),
                P=list(data.get("P", []))
            )
        elif "K" in data and len(data["K"]) >= 9:
            K = data["K"]
            return cls(
                fx=float(K[0]),
                fy=float(K[4]),
                cx=float(K[2]),
                cy=float(K[5]),
                width=int(data.get("width", 640)),
                height=int(data.get("height", 480)),
                distortion_model=data.get("distortion_model", "plumb_bob"),
                distortion_coefficients=list(data.get("distortion_coefficients", [0.0]*5)),
                K=list(K),
                R=list(data.get("R", [])),
                P=list(data.get("P", []))
            )
        else:
            raise ValueError(f"Invalid camera info dict: {data}")


class FrameDataset:
    """Canonical Frame Dataset 표현 클래스."""

    def __init__(self, dataset_dir: Union[str, Path]):
        self.dataset_dir = Path(dataset_dir).resolve()
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.frames_csv = self.dataset_dir / "frames.csv"
        self.camera_info_json = self.dataset_dir / "camera_info.json"
        self.imu_csv = self.dataset_dir / "imu.csv"
        self.dataset_info_json = self.dataset_dir / "dataset_info.json"

        if not self.frames_csv.exists():
            raise FileNotFoundError(f"frames.csv not found in {self.dataset_dir}")

        self.frames: List[FrameMetadata] = self._load_frames_csv()
        self.intrinsics: CameraIntrinsics = self._load_camera_info()
        self.dataset_info: dict = self._load_dataset_info()

    def _load_frames_csv(self) -> List[FrameMetadata]:
        frames = []
        with open(self.frames_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames.append(FrameMetadata(
                    frame_id=int(row["frame_id"]),
                    rgb_timestamp=float(row["rgb_timestamp"]),
                    depth_timestamp=float(row["depth_timestamp"]),
                    rgb_path=row["rgb_path"],
                    depth_path=row["depth_path"],
                    rgb_depth_dt_ms=float(row.get("rgb_depth_dt_ms", 0.0)),
                    bag_timestamp=float(row.get("bag_timestamp", row["rgb_timestamp"])),
                    camera_frame_id=row.get("camera_frame_id", "camera_color_optical_frame"),
                    depth_camera_frame_id=row.get("depth_camera_frame_id", ""),
                    width=int(row.get("width", 640)),
                    height=int(row.get("height", 480))
                ))
        return frames

    def _load_camera_info(self) -> CameraIntrinsics:
        if self.camera_info_json.exists():
            with open(self.camera_info_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                return CameraIntrinsics.from_dict(data)
        # Default D435i intrinsics if missing
        return CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)

    def _load_dataset_info(self) -> dict:
        if self.dataset_info_json.exists():
            with open(self.dataset_info_json, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> FrameMetadata:
        return self.frames[idx]

    def get_frame_by_id(self, frame_id: int) -> Optional[FrameMetadata]:
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame
        return None

    def get_rgb_path(self, idx: int) -> Path:
        rel = self.frames[idx].rgb_path
        p = Path(rel)
        return p if p.is_absolute() else self.dataset_dir / p

    def get_depth_path(self, idx: int) -> Path:
        rel = self.frames[idx].depth_path
        p = Path(rel)
        return p if p.is_absolute() else self.dataset_dir / p

    def get_rgb(self, idx: int) -> Optional[np.ndarray]:
        """BGR uint8 image (OpenCV format) 반환."""
        path = self.get_rgb_path(idx)
        if not path.exists():
            return None
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    def get_rgb_tensor(self, idx: int) -> Optional[np.ndarray]:
        """RGB uint8 image 반환."""
        bgr = self.get_rgb(idx)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_depth(self, idx: int) -> Optional[np.ndarray]:
        """16-bit uint16 depth image in millimeters (16UC1) 반환."""
        path = self.get_depth_path(idx)
        if not path.exists():
            return None
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    def get_depth_meters(self, idx: int, depth_scale: float = 1000.0) -> Optional[np.ndarray]:
        """float32 depth array in meters 반환."""
        depth_raw = self.get_depth(idx)
        if depth_raw is None:
            return None
        depth_m = depth_raw.astype(np.float32) / depth_scale
        return depth_m

    def get_timestamps(self, use_rgb: bool = True) -> np.ndarray:
        if use_rgb:
            return np.array([f.rgb_timestamp for f in self.frames], dtype=np.float64)
        return np.array([f.depth_timestamp for f in self.frames], dtype=np.float64)
