"""
artifacts.py — Artifact Identity, Cache Validation, Trajectory Provenance, and Directory Isolation.

Provides:
  - Strict content-aware cache validation (CandidateSpec hash, dataset fingerprint, trajectory SHA, split hash).
  - Trajectory metadata storage & provenance verification (no silent profile fallback).
  - Isolated candidate artifact directories (candidate-id based, preventing full rebuild collisions).
  - Atomic JSON and text writers.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from auto_mobility.config import (
    MESH_DIR, POINTCLOUD_DIR, TRAJECTORY_DIR, EVALUATION_DIR, FRAME_DIR, PROJECT_DIR
)
from auto_mobility.benchmark.candidate import CandidateSpec, SlamProfileSpec


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute deterministic SHA-256 hash of a file."""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return "missing"
    sha = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def compute_cache_key(
    stage: str = "default",
    dataset_name: str = "default",
    upstream_files: Optional[List[Union[str, Path]]] = None,
    params: Optional[Dict[str, Any]] = None,
    version: str = "v1"
) -> str:
    """Computes a deterministic 16-character SHA-256 cache key."""
    sha = hashlib.sha256()
    sha.update(stage.encode("utf-8"))
    sha.update(dataset_name.encode("utf-8"))
    sha.update(version.encode("utf-8"))
    if params:
        sha.update(json.dumps(params, sort_keys=True).encode("utf-8"))
    if upstream_files:
        for uf in sorted([str(p) for p in upstream_files]):
            p = Path(uf)
            if p.exists():
                sha.update(p.name.encode("utf-8"))
                sha.update(str(p.stat().st_size).encode("utf-8"))
                sha.update(str(p.stat().st_mtime_ns).encode("utf-8"))
    return sha.hexdigest()[:16]


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
    tmp_path = p.with_suffix(f".tmp.{os.getpid()}_{id(data)}")
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
    tmp_path = p.with_suffix(f".tmp.{os.getpid()}_{hash(text) & 0xffffffff}")
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


# ───────────────────────────────────────────────────────────
# Trajectory Provenance & Metadata Management
# ───────────────────────────────────────────────────────────

def get_trajectory_meta_path(trajectory_path: Union[str, Path]) -> Path:
    p = Path(trajectory_path)
    return p.parent / f"{p.stem}.meta.json"


def count_tum_poses(trajectory_path: Union[str, Path]) -> int:
    """Counts valid pose rows (non-comment, >= 8 columns) in a TUM trajectory file."""
    p = Path(trajectory_path)
    if not p.exists():
        return 0
    count = 0
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if len(line.split()) >= 8:
                    count += 1
    except OSError:
        return 0
    return count


def save_trajectory_metadata(
    trajectory_path: Union[str, Path],
    profile_spec: SlamProfileSpec,
    bag_fingerprint: str = "unknown",
    slam_config_hash: str = "unknown",
    git_commit: str = "unknown",
    pose_count: Optional[int] = None
) -> Path:
    """Saves trajectory provenance metadata alongside the TUM trajectory file.

    If pose_count is None (default), it is counted automatically from the
    trajectory file so the metadata always reflects actual content.
    """
    traj_path = Path(trajectory_path)
    meta_path = get_trajectory_meta_path(traj_path)
    traj_sha = compute_file_sha256(traj_path) if traj_path.exists() else "missing"
    if pose_count is None:
        pose_count = count_tum_poses(traj_path)

    meta = {
        "candidate_key": profile_spec.candidate_key,
        "backend": profile_spec.backend,
        "profile": profile_spec.profile,
        "replay_rate": profile_spec.replay_rate,
        "bag_fingerprint": bag_fingerprint,
        "slam_config_hash": slam_config_hash,
        "git_commit": git_commit,
        "trajectory_sha256": traj_sha,
        "pose_count": pose_count,
        "provenance_status": "VERIFIED"
    }
    atomic_write_json(meta_path, meta)
    return meta_path


def load_trajectory_metadata(trajectory_path: Union[str, Path]) -> Optional[dict]:
    meta_path = get_trajectory_meta_path(trajectory_path)
    if meta_path.exists() and meta_path.stat().st_size > 10:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def verify_trajectory_provenance(
    trajectory_path: Union[str, Path],
    requested_spec: SlamProfileSpec,
    expected_bag_fingerprint: Optional[str] = None,
    strict: bool = True
) -> Tuple[bool, str, dict]:
    """Verifies that the trajectory file matches the requested SLAM profile and rate.

    Returns:
      (is_valid, status_code, metadata_dict)
    """
    traj_path = Path(trajectory_path)
    if not traj_path.exists() or traj_path.stat().st_size == 0:
        return False, "MISSING_TRAJECTORY", {}

    meta = load_trajectory_metadata(traj_path)
    if meta is None:
        if strict:
            # Unverified legacy trajectory without metadata
            return False, "LEGACY_UNVERIFIED", {"provenance_status": "LEGACY_UNVERIFIED"}
        return True, "LEGACY_UNVERIFIED", {"provenance_status": "LEGACY_UNVERIFIED"}

    # Verify fields
    backend_match = (meta.get("backend") == requested_spec.backend)
    profile_match = (meta.get("profile") == requested_spec.profile)
    rate_match = (abs(float(meta.get("replay_rate", 1.0)) - requested_spec.replay_rate) < 1e-3)

    current_sha = compute_file_sha256(traj_path)
    sha_match = (meta.get("trajectory_sha256") == current_sha)
    
    fp_match = True
    if expected_bag_fingerprint and meta.get("bag_fingerprint"):
        fp_match = (meta.get("bag_fingerprint") == expected_bag_fingerprint)

    if backend_match and profile_match and rate_match and sha_match and fp_match:
        return True, "VERIFIED", meta

    diffs = []
    if not backend_match:
        diffs.append(f"backend mismatch ({meta.get('backend')} != {requested_spec.backend})")
    if not profile_match:
        diffs.append(f"profile mismatch ({meta.get('profile')} != {requested_spec.profile})")
    if not rate_match:
        diffs.append(f"rate mismatch ({meta.get('replay_rate')} != {requested_spec.replay_rate})")
    if not sha_match:
        diffs.append("file modified after metadata creation")
    if not fp_match:
        diffs.append(f"bag fingerprint mismatch ({meta.get('bag_fingerprint')} != {expected_bag_fingerprint})")

    return False, "PROVENANCE_MISMATCH", {"reasons": diffs, "stored_meta": meta}


# ───────────────────────────────────────────────────────────
# Artifact Manager & Content-Aware Cache
# ───────────────────────────────────────────────────────────

class ArtifactManager:
    """Manages isolated artifact directories, deterministic paths, and strict validation."""

    def __init__(self, bag_name: str, base_eval_dir: Optional[Path] = None):
        self.bag_name = bag_name
        self.eval_dir = Path(base_eval_dir) if base_eval_dir else (EVALUATION_DIR / bag_name)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_root = self.eval_dir / "artifacts"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def get_candidate_eval_dir(self, candidate_name: str) -> Path:
        cand_dir = self.eval_dir / candidate_name
        cand_dir.mkdir(parents=True, exist_ok=True)
        return cand_dir

    def get_candidate_artifact_dir(self, candidate_id: str) -> Path:
        """Returns the isolated directory for a candidate's artifacts."""
        cand_dir = self.artifacts_root / candidate_id
        cand_dir.mkdir(parents=True, exist_ok=True)
        return cand_dir

    # Legacy/compatibility paths
    def get_mesh_path(self, slam: str, voxel_mm: int, method: Optional[str] = None) -> Path:
        base = f"{self.bag_name}_{slam}_voxel{voxel_mm}mm"
        if method:
            return MESH_DIR / f"{base}_{method}.obj"
        return MESH_DIR / f"{base}.obj"

    def get_pcd_path(self, slam: str, voxel_mm: int) -> Path:
        return POINTCLOUD_DIR / f"{self.bag_name}_{slam}_voxel{voxel_mm}mm_cloud.ply"

    def get_direct_pcd_path(self, slam: str, voxel_mm: int) -> Path:
        return POINTCLOUD_DIR / f"{self.bag_name}_{slam}_direct_voxel{voxel_mm}mm_cloud.ply"

    def get_artifact_meta_path(self, artifact_path: Union[str, Path]) -> Path:
        """Returns standard metadata file path adjacent to an artifact."""
        p = Path(artifact_path)
        return p.parent / f"{p.stem}.meta.json"

    def save_artifact_metadata(
        self,
        meta_path: Union[str, Path],
        candidate_spec: CandidateSpec,
        dataset_fingerprint: str,
        trajectory_sha256: str,
        split_hash: str,
        effective_params: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Writes artifact.meta.json for content-aware caching."""
        p = Path(meta_path)
        meta = {
            "candidate_id": candidate_spec.compute_candidate_id(),
            "candidate_spec_hash": candidate_spec.compute_spec_hash(),
            "fusion_hash": candidate_spec.compute_fusion_hash(),
            "dataset_fingerprint": dataset_fingerprint,
            "trajectory_sha256": trajectory_sha256,
            "split_hash": split_hash,
            "code_version": candidate_spec.code_version,
            "cache_schema_version": candidate_spec.cache_schema_version,
            "requested_params": candidate_spec.to_metadata_dict()["requested_params"],
            "effective_params": effective_params or candidate_spec.to_metadata_dict()["effective_params"]
        }
        atomic_write_json(p, meta)
        return p

    def should_reuse_reconstruction(
        self,
        mesh_path: Optional[Path],
        pcd_path: Optional[Path],
        candidate_spec: Optional[Union[CandidateSpec, str]] = None,
        fusion_hash: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        trajectory_sha256: Optional[str] = None,
        split_hash: Optional[str] = None,
        meta_path: Optional[Path] = None,
        force: bool = False
    ) -> bool:
        """Strict content-aware cache validation for reconstruction artifacts.

        If validation criteria are supplied, metadata file is strictly required.
        Missing or mismatched metadata causes a CACHE MISS.

        fusion_hash: when supplied, reconstruction identity is validated against
        the stored fusion_hash instead of the full candidate_spec_hash. This lets
        candidates whose fusion kernels are byte-identical (e.g. identical TSDF
        params under different surface_method labels) share one reconstruction.
        """
        if force:
            return False

        if mesh_path and not is_artifact_valid(mesh_path):
            return False
        if pcd_path and not is_artifact_valid(pcd_path):
            return False

        # If strict content validation criteria are supplied:
        has_content_checks = (
            isinstance(candidate_spec, CandidateSpec) or
            fusion_hash is not None or
            dataset_fingerprint is not None or
            trajectory_sha256 is not None or
            split_hash is not None or
            meta_path is not None
        )
        if has_content_checks:
            if not meta_path or not Path(meta_path).exists():
                return False
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if fusion_hash is not None:
                    if meta.get("fusion_hash") != fusion_hash:
                        return False
                elif candidate_spec and isinstance(candidate_spec, CandidateSpec):
                    if meta.get("candidate_spec_hash") != candidate_spec.compute_spec_hash():
                        return False
                if dataset_fingerprint is not None and meta.get("dataset_fingerprint") != dataset_fingerprint:
                    return False
                if trajectory_sha256 is not None and meta.get("trajectory_sha256") != trajectory_sha256:
                    return False
                if split_hash is not None and meta.get("split_hash") != split_hash:
                    return False
            except Exception:
                return False

        return True

    def should_reuse_evaluation(
        self,
        candidate_name: str,
        force: bool = False,
        expected_spec_hash: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        split_hash: Optional[str] = None
    ) -> Optional[dict]:
        """Check if evaluation summary exists and matches all spec and provenance hashes."""
        if force:
            return None
        cand_dir = self.get_candidate_eval_dir(candidate_name)
        summary_file = cand_dir / "evaluation_summary.json"
        if summary_file.exists() and summary_file.stat().st_size > 50:
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("geometry"):
                    if expected_spec_hash is not None:
                        cached_hash = data.get("spec_hash") or data.get("spec", {}).get("spec_hash")
                        if cached_hash != expected_spec_hash:
                            return None
                    if dataset_fingerprint is not None:
                        cached_ds = data.get("dataset_fingerprint")
                        if cached_ds != dataset_fingerprint:
                            return None
                    if split_hash is not None:
                        cached_sp = data.get("split_hash")
                        if cached_sp != split_hash:
                            return None
                    return data
            except Exception:
                return None
        return None
