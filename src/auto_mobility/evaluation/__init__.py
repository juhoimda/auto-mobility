"""
auto_mobility.evaluation

3D Reconstruction 정량 형상 품질(Geometry Quality) 평가 및 벤치마크 랭킹 모듈.
"""

from .split import create_holdout_split, save_split_json, load_split_json
from .render_depth import create_raycasting_scene, render_depth_map
from .geometry_metrics import (
    compute_depth_metrics,
    backproject_depth_to_world_points,
    compute_point_to_mesh_metrics,
    generate_error_visualization
)
from .mesh_metrics import compute_mesh_quality_metrics, compute_plane_quality_metrics
from .trajectory_metrics import compute_trajectory_quality
from .report import generate_markdown_report
from .evaluator import evaluate_reconstruction
from .compare_results import rank_candidates, compare_dataset_evaluations

__all__ = [
    "create_holdout_split",
    "save_split_json",
    "load_split_json",
    "create_raycasting_scene",
    "render_depth_map",
    "compute_depth_metrics",
    "backproject_depth_to_world_points",
    "compute_point_to_mesh_metrics",
    "generate_error_visualization",
    "compute_mesh_quality_metrics",
    "compute_plane_quality_metrics",
    "compute_trajectory_quality",
    "generate_markdown_report",
    "evaluate_reconstruction",
    "rank_candidates",
    "compare_dataset_evaluations",
]
