"""V2 artifact identity/store invariants (next.md #12, #13, §93).

Covers:
  test_artifact_identity_no_collision
  test_directcloud_tsdf_artifacts_separate
  test_candidate_spec_hash_changes_on_effective_param
"""

from auto_mobility.reconstruction.artifacts import (
    ArtifactStore,
    make_identity,
    spec_hash,
)
from auto_mobility.reconstruction.runtime.machine_profile import GpuInfo, MachineProfile


def _identity(fusion_method="tsdf", voxel_mm=10):
    return make_identity(
        dataset_spec={"bag": "base3", "fingerprint": "fp-123"},
        trajectory_spec={"backend": "rtabmap", "profile": "standard"},
        fusion_spec={"method": fusion_method, "voxel_mm": voxel_mm, "trunc_mm": 40},
        surface_spec={"method": "tsdf_direct"},
    )


def test_same_spec_same_identity():
    assert _identity() == _identity()


def test_candidate_spec_hash_changes_on_effective_param():
    base = spec_hash({"voxel_mm": 10})
    assert spec_hash({"voxel_mm": 8}) != base
    assert spec_hash({"voxel_mm": "10"}) != base
    assert spec_hash({"a": 1, "b": 2}) == spec_hash({"b": 2, "a": 1})


def test_artifact_identity_no_collision():
    id_a = _identity(voxel_mm=10)
    id_b = _identity(voxel_mm=8)
    assert id_a.relpath("pointcloud") != id_b.relpath("pointcloud")
    assert id_a.fusion_hash != id_b.fusion_hash
    assert id_a.dataset_hash == id_b.dataset_hash
    assert id_a.trajectory_hash == id_b.trajectory_hash


def test_directcloud_tsdf_artifacts_separate():
    tsdf_id = _identity(fusion_method="tsdf")
    direct_id = _identity(fusion_method="direct_pointcloud")
    assert tsdf_id.fusion_hash != direct_id.fusion_hash
    assert tsdf_id.relpath("pointcloud") != direct_id.relpath("pointcloud")


def test_identity_rejects_malformed_hash():
    import pytest

    from auto_mobility.reconstruction.artifacts.identity import ArtifactIdentity

    with pytest.raises(ValueError):
        ArtifactIdentity("not-hash", "0" * 16, "0" * 16, "0" * 16)


def test_store_roundtrip_and_rejects_tamper(tmp_path):
    src = tmp_path / "payload.bin"
    src.write_bytes(b"point-cloud-bytes")

    ident = _identity()
    store = ArtifactStore(tmp_path / "artifacts")
    stored = store.put(ident, "pointcloud", "cloud.ply", src, extra_meta={"producer": "unit"})

    assert store.get(ident, "pointcloud", "cloud.ply") == stored.path
    assert store.verify(stored)

    with open(stored.path, "ab") as fh:
        fh.write(b"corruption")

    assert store.verify(stored) is False
    assert store.get(ident, "pointcloud", "cloud.ply") is None


def test_store_rejects_wrong_semantic_spec(tmp_path):
    src = tmp_path / "payload.bin"
    src.write_bytes(b"x")
    store = ArtifactStore(tmp_path / "artifacts")
    store.put(_identity(), "pointcloud", "cloud.ply", src)

    assert store.get(_identity(voxel_mm=8), "pointcloud", "cloud.ply") is None
    assert store.get(_identity(), "mesh", "cloud.ply") is None


def test_store_atomic_no_tmp_leftovers(tmp_path):
    src = tmp_path / "p.bin"
    src.write_bytes(b"data")
    store = ArtifactStore(tmp_path / "arts")
    store.put(_identity(), "pointcloud", "c.ply", src)

    kdir = next((tmp_path / "arts").iterdir())
    names = [p.name for p in kdir.rglob("*")]
    assert all(not n.startswith(".tmp_") for n in names)


def test_profile_fingerprint_stable_and_serialization():
    p = MachineProfile(
        cpu_physical=4,
        cpu_logical=8,
        ram_total_mb=32000,
        ram_available_mb=16000,
        gpu=GpuInfo(model="RTX", vram_total_mb=8192, vram_free_mb=6000),
    )
    q = MachineProfile.from_dict(p.to_dict())
    assert q.software_fingerprint == p.software_fingerprint

    p2 = MachineProfile(
        cpu_physical=4,
        cpu_logical=8,
        ram_total_mb=32000,
        ram_available_mb=1000,
        gpu=GpuInfo(model="OtherGPU", vram_total_mb=4096, vram_free_mb=3000),
    )
    assert p.software_fingerprint != p2.software_fingerprint
