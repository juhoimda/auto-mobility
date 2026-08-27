"""Appearance layer: texture/material baking."""

from auto_mobility.reconstruction.appearance.texture_baker import (
    BakeResult,
    bake_atlas,
    sample_view_color,
)
from auto_mobility.reconstruction.appearance.views import (
    atlas_metrics,
    normalize_exposure,
)
from auto_mobility.reconstruction.appearance.texture_contract import (
    TextureContractResult,
    check_texture_contract,
)

__all__ = [
    "BakeResult",
    "bake_atlas",
    "sample_view_color",
    "atlas_metrics",
    "normalize_exposure",
    "TextureContractResult",
    "check_texture_contract",
]
