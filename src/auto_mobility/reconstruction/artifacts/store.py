"""Content-addressed artifact store with atomic writes and sidecar metadata.

Reuse contract: an artifact is reused only when its sidecar exists AND the
recorded spec hash matches the requested semantic identity AND the content
SHA256 verifies. Existence of a file at a path is never sufficient.

Complexity: put/get are O(file size) for hashing. Memory: streamed in chunks.
"""

from dataclasses import dataclass
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

from auto_mobility.reconstruction.artifacts.identity import ArtifactIdentity, safe_segment
from auto_mobility.reconstruction.model import SCHEMA_VERSION

_CHUNK = 1 << 20
_SIDECAR_SUFFIX = ".meta.json"


@dataclass(frozen=True)
class StoredArtifact:
    path: Path
    sidecar_path: Path
    content_sha256: str

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "sidecar_path": str(self.sidecar_path),
            "content_sha256": self.content_sha256,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _kind_dir(self, identity: ArtifactIdentity, kind: str) -> Path:
        return self.root / Path(identity.relpath(safe_segment(kind)))

    def put(
        self,
        identity: ArtifactIdentity,
        kind: str,
        filename: str,
        source_path: Path,
        extra_meta: Optional[dict] = None,
    ) -> StoredArtifact:
        src = Path(source_path)
        if not src.is_file():
            raise FileNotFoundError(f"artifact source missing: {src}")
        digest = sha256_file(src)
        kdir = self._kind_dir(identity, kind)
        final = kdir / safe_segment(filename)
        sidecar = kdir / (final.name + _SIDECAR_SUFFIX)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "identity": identity.to_dict(),
            "kind": kind,
            "filename": final.name,
            "content_sha256": digest,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra_meta:
            meta.update(extra_meta)
        kdir.mkdir(parents=True, exist_ok=True)
        tmp_final = None
        tmp_side = None
        try:
            fd, tmp_final = tempfile.mkstemp(dir=kdir, prefix=".tmp_art_")
            os.close(fd)
            shutil.copyfile(src, tmp_final)
            fd, tmp_side = tempfile.mkstemp(dir=kdir, prefix=".tmp_meta_")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2, sort_keys=True)
            os.replace(tmp_final, final)
            tmp_final = None
            os.replace(tmp_side, sidecar)
            tmp_side = None
        finally:
            for leftover in (tmp_final, tmp_side):
                if leftover is not None and os.path.exists(leftover):
                    os.unlink(leftover)
        return StoredArtifact(path=final, sidecar_path=sidecar, content_sha256=digest)

    def get(self, identity: ArtifactIdentity, kind: str, filename: str) -> Optional[Path]:
        """Return the artifact path only when sidecar identity matches; else None."""
        kdir = self._kind_dir(identity, kind)
        final = kdir / safe_segment(filename)
        sidecar = kdir / (final.name + _SIDECAR_SUFFIX)
        return self._validated(final, sidecar, identity)

    def verify(self, artifact: StoredArtifact) -> bool:
        if not artifact.path.is_file():
            return False
        return sha256_file(artifact.path) == artifact.content_sha256

    def _validated(
        self, final: Path, sidecar: Path, identity: ArtifactIdentity
    ) -> Optional[Path]:
        if not final.is_file() or not sidecar.is_file():
            return None
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        recorded = meta.get("identity", {})
        if recorded != identity.to_dict():
            return None
        expected = meta.get("content_sha256")
        if not expected or sha256_file(final) != expected:
            return None
        return final
