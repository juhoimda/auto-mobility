"""P1-2: Poisson acceptance/rejection gate and texture contract tests."""
from pathlib import Path
import pytest
import numpy as np
import cv2


def test_poisson_gate_exists_in_standard():
    """Standard pipeline must have a Poisson acceptance/rejection gate."""
    src = Path('/home/kth/auto-mobility/src/auto_mobility/reconstruction/pipeline/standard.py').read_text()
    assert 'POISSON_APPLIED' in src or 'enable_poisson' in src


def test_texture_contract_checker_classifies_no_usemtl_as_fail(tmp_path):
    """OBJ without usemtl must get APPEARANCE_FAIL."""
    from auto_mobility.reconstruction.appearance.texture_contract import check_texture_contract
    obj = tmp_path / 'model.obj'
    obj.write_text('# mesh\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    result = check_texture_contract(tmp_path)
    assert result.gate_status == 'APPEARANCE_FAIL'
    assert not result.ok
    assert 'usemtl' in result.reject_reason.lower() or 'map_Kd' in result.reject_reason


def test_texture_contract_checker_pass_with_map_kd(tmp_path):
    """OBJ with usemtl, MTL with map_Kd pointing to existing file, and UVs must PASS."""
    from auto_mobility.reconstruction.appearance.texture_contract import check_texture_contract

    # Create texture
    tex_dir = tmp_path / 'textures'
    tex_dir.mkdir()
    tex = tex_dir / 'atlas.png'
    cv2.imwrite(str(tex), np.zeros((64, 64, 3), dtype=np.uint8))

    # Create MTL
    mtl = tmp_path / 'model.mtl'
    mtl.write_text('newmtl mat0\nmap_Kd textures/atlas.png\n')

    # Create OBJ with UV
    obj = tmp_path / 'model.obj'
    obj.write_text(
        'mtllib model.mtl\n'
        'v 0 0 0\nv 1 0 0\nv 0 1 0\n'
        'vt 0 0\nvt 1 0\nvt 0 1\n'
        'usemtl mat0\n'
        'f 1/1 2/2 3/3\n'
    )

    result = check_texture_contract(tmp_path)
    assert result.ok
    assert result.gate_status == 'PASS'
    assert result.has_usemtl
    assert result.has_map_kd
    assert result.has_uv_coords
    assert result.textured_face_coverage > 0


def test_texture_contract_coverage_zero_is_fail(tmp_path):
    """OBJ with usemtl+map_Kd but no UV coords must be APPEARANCE_FAIL."""
    from auto_mobility.reconstruction.appearance.texture_contract import check_texture_contract

    tex = tmp_path / 'atlas.png'
    cv2.imwrite(str(tex), np.zeros((64, 64, 3), dtype=np.uint8))

    mtl = tmp_path / 'model.mtl'
    mtl.write_text('newmtl mat0\nmap_Kd atlas.png\n')

    # OBJ without vt lines
    obj = tmp_path / 'model.obj'
    obj.write_text(
        'mtllib model.mtl\n'
        'v 0 0 0\nv 1 0 0\nv 0 1 0\n'
        'usemtl mat0\n'
        'f 1 2 3\n'  # no UV indices
    )

    result = check_texture_contract(tmp_path)
    assert result.gate_status == 'APPEARANCE_FAIL'
    assert not result.ok


def test_artifact_bundle_hash_changes_with_obj(tmp_path):
    """Bundle hash must change when OBJ content changes."""
    from auto_mobility.reconstruction.appearance.texture_contract import check_texture_contract

    obj = tmp_path / 'model.obj'
    obj.write_text('v 0 0 0\nf 1 2 3\n')
    r1 = check_texture_contract(tmp_path)

    obj.write_text('v 1 0 0\nf 1 2 3\n')
    r2 = check_texture_contract(tmp_path)

    assert r1.artifact_bundle_hash != r2.artifact_bundle_hash
