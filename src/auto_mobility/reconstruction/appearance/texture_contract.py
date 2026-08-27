"""
OBJ/MTL texture delivery contract verification (P1-2).

A fully textured delivery requires:
  - OBJ file with `usemtl` directive
  - MTL file with `map_Kd` directive pointing to a real image file
  - UV coordinates (vt lines in OBJ)
  - At least some faces with UV assignments (f lines with UV indices)
  - Textured face coverage > 0

Vertex-color PLY is NOT a texture delivery. Diagnostic color atlases are NOT texture delivery.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TextureContractResult:
    ok: bool
    gate_status: str  # "PASS" | "APPEARANCE_FAIL" | "NO_OBJ"
    has_usemtl: bool
    has_map_kd: bool
    has_uv_coords: bool
    textured_face_coverage: float  # 0.0-1.0
    texture_files: list  # list of texture file paths that exist
    missing_files: list
    obj_hash: Optional[str]
    mtl_hash: Optional[str]
    atlas_hashes: dict  # {filename: sha256}
    artifact_bundle_hash: Optional[str]  # hash over all artifact hashes
    reject_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "gate_status": self.gate_status,
            "has_usemtl": self.has_usemtl,
            "has_map_kd": self.has_map_kd,
            "has_uv_coords": self.has_uv_coords,
            "textured_face_coverage": self.textured_face_coverage,
            "texture_files": self.texture_files,
            "missing_files": self.missing_files,
            "obj_hash": self.obj_hash,
            "mtl_hash": self.mtl_hash,
            "atlas_hashes": self.atlas_hashes,
            "artifact_bundle_hash": self.artifact_bundle_hash,
            "reject_reason": self.reject_reason,
        }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def check_texture_contract(mesh_dir: Path) -> TextureContractResult:
    """Check OBJ/MTL/texture delivery contract for a mesh directory.

    Returns a TextureContractResult. gate_status will be:
      - "PASS"            : full texture delivery confirmed
      - "APPEARANCE_FAIL" : OBJ exists but texture contract is not satisfied
                            (missing usemtl, map_Kd, UV coords, or zero coverage)
      - "NO_OBJ"          : model.obj does not exist in mesh_dir

    Vertex-color PLY files and diagnostic color atlases are explicitly NOT
    counted as texture delivery; only a proper OBJ+MTL+map_Kd+UV pipeline counts.
    """
    obj_path = mesh_dir / "model.obj"
    mtl_path = mesh_dir / "model.mtl"

    if not obj_path.is_file():
        return TextureContractResult(
            ok=False, gate_status="NO_OBJ",
            has_usemtl=False, has_map_kd=False, has_uv_coords=False,
            textured_face_coverage=0.0, texture_files=[], missing_files=[str(obj_path)],
            obj_hash=None, mtl_hash=None, atlas_hashes={}, artifact_bundle_hash=None,
            reject_reason="model.obj does not exist"
        )

    obj_text = obj_path.read_text(errors="replace")
    obj_hash = _sha256_file(obj_path)

    # Check OBJ for usemtl
    has_usemtl = "usemtl" in obj_text
    has_uv_coords = any(line.startswith("vt ") for line in obj_text.splitlines())

    # Count faces with UV
    total_faces = 0
    textured_faces = 0
    for line in obj_text.splitlines():
        if line.startswith("f "):
            total_faces += 1
            # Face with UV: "f v/vt/vn" or "f v/vt"
            parts = line.split()[1:]
            if all("/" in p and len(p.split("/")) >= 2 and p.split("/")[1] for p in parts):
                textured_faces += 1

    textured_face_coverage = textured_faces / total_faces if total_faces > 0 else 0.0

    # Check MTL
    mtl_hash = None
    has_map_kd = False
    texture_files = []
    missing_files = []
    atlas_hashes = {}

    if mtl_path.is_file():
        mtl_text = mtl_path.read_text(errors="replace")
        mtl_hash = _sha256_file(mtl_path)
        has_map_kd = "map_Kd" in mtl_text

        # Find map_Kd file(s)
        for line in mtl_text.splitlines():
            if line.strip().lower().startswith("map_kd"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    map_kd_file = parts[-1]
                    # Check if texture file exists relative to mesh_dir
                    tex_path = mesh_dir / map_kd_file
                    if tex_path.is_file():
                        texture_files.append(str(tex_path))
                        atlas_hashes[map_kd_file] = _sha256_file(tex_path)
                    else:
                        # Try textures subdirectory
                        tex_path2 = mesh_dir / "textures" / map_kd_file
                        if tex_path2.is_file():
                            texture_files.append(str(tex_path2))
                            atlas_hashes[map_kd_file] = _sha256_file(tex_path2)
                        else:
                            missing_files.append(str(tex_path))
                            has_map_kd = False  # map_Kd references non-existent file

    # Also index any additional images in textures/
    tex_dir = mesh_dir / "textures"
    if tex_dir.is_dir():
        for img in sorted(tex_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in (".png", ".jpg", ".jpeg"):
                if str(img) not in texture_files:
                    texture_files.append(str(img))
                    atlas_hashes[img.name] = _sha256_file(img)

    # Compute artifact bundle hash (covers OBJ + MTL + all atlas files)
    all_hashes = [obj_hash]
    if mtl_hash:
        all_hashes.append(mtl_hash)
    all_hashes.extend(sorted(atlas_hashes.values()))
    bundle_hash = hashlib.sha256("|".join(all_hashes).encode()).hexdigest()[:16]

    # Determine gate status
    ok = (
        has_usemtl
        and has_map_kd
        and has_uv_coords
        and textured_face_coverage > 0.0
        and len(texture_files) > 0
    )

    if not ok:
        reasons = []
        if not has_usemtl:
            reasons.append("no usemtl in OBJ")
        if not has_map_kd:
            reasons.append("no map_Kd in MTL (or MTL missing)")
        if not has_uv_coords:
            reasons.append("no UV coordinates in OBJ")
        if textured_face_coverage == 0.0:
            reasons.append("texture_coverage=0")
        if not texture_files:
            reasons.append("no texture image files found")
        reject_reason = "; ".join(reasons)
        gate_status = "APPEARANCE_FAIL"
    else:
        reject_reason = None
        gate_status = "PASS"

    return TextureContractResult(
        ok=ok,
        gate_status=gate_status,
        has_usemtl=has_usemtl,
        has_map_kd=has_map_kd,
        has_uv_coords=has_uv_coords,
        textured_face_coverage=textured_face_coverage,
        texture_files=texture_files,
        missing_files=missing_files,
        obj_hash=obj_hash,
        mtl_hash=mtl_hash,
        atlas_hashes=atlas_hashes,
        artifact_bundle_hash=bundle_hash,
        reject_reason=reject_reason,
    )
