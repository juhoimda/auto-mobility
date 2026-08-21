"""
artifacts.py — Artifact Identity, Cache Key, Validation, and Reuse Management.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from auto_mobility.config import (
    MESH_DIR, POINTCLOUD_DIR, TRAJECTORY_DIR, EVALUATION_DIR, FRAME_DIR, PROJECT_DIR
)


def _file_hash_or_stat(file_path: Union[str, Path]) -> str:
    """Return a compact deterministic representation (hash or size+mtime) of a file."""
    p = Path(file_path)
    if not p.exists():
        return "missing"
    try:
        st = p.stat()
        # For fast cache key without reading multi-gigabyte files, use size + mtime_ns
        return f"{st.st_size}_{st.st_mtime_ns}"
    except Exception:
        return "error"


def compute_cache_key(
    stage: str,
    dataset_name: str,
    upstream_files: Optional[List[Union[str, Path]]] = None,
    params: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None
) -> str:
    """Compute a deterministic SHA-256 cache key for an artifact generation step."""
    payload: Dict[str, Any] = {
        "stage": stage,
        "dataset": dataset_name,
        "version": version or "v1",
        "params": params or {},
        "upstream": {}
    }
    if upstream_files:
        for f in upstream_files:
            fp = Path(f).resolve()
            # Use absolute path as key to distinguish same-named files in different directories
            payload["upstream"][str(fp)] = _file_hash_or_stat(fp)

    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def is_artifact_valid(file_path: Union[str, Path], min_bytes: int = 100) -> bool:
    """Check if an artifact file exists, is non-empty, and meets minimum size threshold."""
    if not file_path:
        return False
    p = Path(file_path)
    return p.exists() and p.is_file() and p.stat().st_size >= min_bytes


def atomic_write_json(file_path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Writes JSON data atomically using a temporary file and atomic rename."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(f".tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def atomic_write_text(file_path: Union[str, Path], text: str) -> None:
    """Writes text data atomically using a temporary file and atomic rename."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(f".tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


class ArtifactManager:
    """Manages artifact paths, validity checks, and caching for the benchmark."""

    def __init__(self, bag_name: str, base_eval_dir: Optional[Path] = None):
        self.bag_name = bag_name
        self.eval_dir = Path(base_eval_dir) if base_eval_dir else (EVALUATION_DIR / bag_name)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def get_mesh_path(self, slam: str, voxel_mm: int, method: Optional[str] = None) -> Path:
        base = f"{self.bag_name}_{slam}_voxel{voxel_mm}mm"
        if method:
            return MESH_DIR / f"{base}_{method}.obj"
        return MESH_DIR / f"{base}.obj"

    def get_pcd_path(self, slam: str, voxel_mm: int) -> Path:
        return POINTCLOUD_DIR / f"{self.bag_name}_{slam}_voxel{voxel_mm}mm_cloud.ply"

    def get_direct_pcd_path(self, slam: str, voxel_mm: int) -> Path:
        return POINTCLOUD_DIR / f"{self.bag_name}_{slam}_direct_voxel{voxel_mm}mm_cloud.ply"

    def get_candidate_eval_dir(self, candidate_name: str) -> Path:
        return self.eval_dir / candidate_name

    def get_cached_eval_summary(self, candidate_name: str, expected_spec_hash: Optional[str] = None) -> Optional[dict]:
        cand_dir = self.get_candidate_eval_dir(candidate_name)
        summary_file = cand_dir / "evaluation_summary.json"
        if summary_file.exists() and summary_file.stat().st_size > 50:
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("geometry"):
                    if expected_spec_hash:
                        cached_hash = data.get("spec_hash") or data.get("metadata", {}).get("spec_hash")
                        if cached_hash and cached_hash != expected_spec_hash:
                            return None
                    return data
            except Exception:
                return None
        return None

    def should_reuse_reconstruction(
        self,
        mesh_path: Optional[Path],
        pcd_path: Optional[Path],
        candidate_name: str,
        force: bool = False
    ) -> bool:
        """Determine whether existing mesh/pcd and evaluation can be reused."""
        if force:
            return False

        # If mesh requested, must be valid
        if mesh_path and not is_artifact_valid(mesh_path):
            return False

        # If pcd requested, must be valid
        if pcd_path and not is_artifact_valid(pcd_path):
            return False

        return True

    def should_reuse_evaluation(
        self,
        candidate_name: str,
        force: bool = False,
        expected_spec_hash: Optional[str] = None
    ) -> Optional[dict]:
        """Check if evaluation summary exists and can be reused."""
        if force:
            return None
        return self.get_cached_eval_summary(candidate_name, expected_spec_hash=expected_spec_hash)

