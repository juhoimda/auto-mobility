"""Content-addressed artifact identity.

Artifact identity is the sha256 of a canonical JSON semantic spec (including
schema version and producer namespace). Two semantically different objects can
never share identity; one changed effective parameter always changes identity.

Complexity: O(size of spec dict). Memory: O(spec).
"""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

from auto_mobility.reconstruction.model import SCHEMA_VERSION

_HASH_LEN = 16
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def spec_hash(spec: Mapping[str, Any]) -> str:
    payload = {"schema_version": SCHEMA_VERSION, "spec": dict(spec)}
    return hashlib.sha256(canonical_json(payload)).hexdigest()[:_HASH_LEN]


def safe_segment(name: str) -> str:
    cleaned = _SAFE_SEGMENT.sub("_", name).strip("._")
    return cleaned or "x"


@dataclass(frozen=True)
class ArtifactIdentity:
    """Hierarchical semantic identity:

    artifacts/<dataset>/<trajectory>/<fusion>/<surface>/<kind>_<hash>
    """

    dataset_hash: str
    trajectory_hash: str
    fusion_hash: str
    surface_hash: str

    def __post_init__(self):
        for name in (
            "dataset_hash",
            "trajectory_hash",
            "fusion_hash",
            "surface_hash",
        ):
            value = getattr(self, name)
            if not re.fullmatch(r"[0-9a-f]{16}", value):
                raise ValueError(f"{name} must be a 16-hex hash, got {value!r}")

    def relpath(self, kind: str) -> str:
        kind_safe = safe_segment(kind)
        return f"{self.dataset_hash}/{self.trajectory_hash}/{self.fusion_hash}/{self.surface_hash}/{kind_safe}"

    def to_dict(self) -> dict:
        return {
            "dataset_hash": self.dataset_hash,
            "trajectory_hash": self.trajectory_hash,
            "fusion_hash": self.fusion_hash,
            "surface_hash": self.surface_hash,
        }


def make_identity(
    dataset_spec: Mapping[str, Any],
    trajectory_spec: Mapping[str, Any],
    fusion_spec: Mapping[str, Any],
    surface_spec: Mapping[str, Any],
) -> ArtifactIdentity:
    return ArtifactIdentity(
        dataset_hash=spec_hash({"stage": "dataset", **dataset_spec}),
        trajectory_hash=spec_hash({"stage": "trajectory", **trajectory_spec}),
        fusion_hash=spec_hash({"stage": "fusion", **fusion_spec}),
        surface_hash=spec_hash({"stage": "surface", **surface_spec}),
    )
