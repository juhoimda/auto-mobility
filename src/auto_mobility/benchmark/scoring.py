"""
scoring.py — Candidate-Independent Absolute Scoring & Hard Gate Filtering.

Provides:
  - HardGateFilter: Rejects corrupt/failed candidate artifacts.
  - compute_absolute_quality_score: Computes deterministic [0, 100] scores independent of candidate set.
  - rank_candidate_summaries: Applies Hard Gate, computes Absolute Quality & Cost scores,
    and sorts candidates (Quality primary, Cost tie-breaker).
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np


def _score_lower_better(val: Optional[float], p100: float = 0.0, p80: float = 25.0, p50: float = 50.0, p0: float = 150.0) -> float:
    """Piecewise linear mapping where lower values produce higher scores [0, 100]."""
    if val is None or np.isnan(val):
        return 0.0
    v = float(val)
    if v <= p100:
        return 100.0
    if v <= p80:
        return 100.0 - 20.0 * (v - p100) / max(p80 - p100, 1e-6)
    if v <= p50:
        return 80.0 - 30.0 * (v - p80) / max(p50 - p80, 1e-6)
    if v <= p0:
        return 50.0 - 50.0 * (v - p50) / max(p0 - p50, 1e-6)
    # Smooth exponential tail for higher errors (e.g. 385mm vs 624mm) so lower error is always rewarded
    return float(max(0.0, 20.0 * np.exp(-(v - p0) / 400.0)))


def _score_higher_better(val: Optional[float], p100: float = 1.0, p80: float = 0.85, p50: float = 0.50, p0: float = 0.20) -> float:
    """Piecewise linear mapping where higher values produce higher scores [0, 100]."""
    if val is None or np.isnan(val):
        return 0.0
    v = float(val)
    if v >= p100:
        return 100.0
    if v >= p80:
        return 80.0 + 20.0 * (v - p80) / max(p100 - p80, 1e-6)
    if v >= p50:
        return 50.0 + 30.0 * (v - p50) / max(p80 - p50, 1e-6)
    if v >= p0:
        return 0.0 + 50.0 * (v - p0) / max(p50 - p0, 1e-6)
    return 0.0


class HardGateFilter:
    """Hard Gate Filter to eliminate corrupt or non-functional candidate artifacts before scoring."""

    @staticmethod
    def evaluate(summary: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Returns (is_valid, failure_reason)."""
        status = summary.get("status")
        if status in ("FAIL_CRASH", "FAIL_SEGFAULT", "FAIL_OOM", "FAIL_TIMEOUT", "FAIL_EXCEPTION", "SKIPPED_UNAVAILABLE", "BLOCKED", "NOT_EVALUATED"):
            return False, f"Candidate execution status was {status}: {summary.get('error', 'N/A')}"

        geom = summary.get("geometry", {})
        if not geom:
            return False, "Geometry evaluation missing"

        mae = geom.get("depth_mae_mm")
        if mae is None or np.isnan(mae):
            return False, "Depth MAE is null or NaN"

        cov = geom.get("depth_coverage_ratio", 0.0)
        if cov is None or cov < 0.05:
            return False, f"Depth coverage severely low ({cov*100:.1f}% < 5%)"

        mesh_meta = summary.get("mesh", {})
        if mesh_meta:
            triangles = mesh_meta.get("num_triangles", 0)
            if triangles <= 0:
                return False, "Mesh has 0 triangles"

        # Free-space catastrophic check: if free space correctness is extremely low (< 20%), candidate is corrupted
        fs_corr = geom.get("free_space_correctness_ratio")
        if fs_corr is not None and fs_corr < 0.20:
            return False, f"Severe free space phantom violation (correctness {fs_corr*100:.1f}% < 20%)"

        return True, None


def compute_absolute_scores(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Computes candidate-independent absolute quality and cost scores for a single evaluation summary."""
    geom = summary.get("geometry", {})
    mesh_m = summary.get("mesh", {})
    perf = summary.get("performance", {})
    planes = summary.get("plane_analysis", {})

    # 1. Geometry Accuracy (40% weight)
    s_mae = _score_lower_better(geom.get("depth_mae_mm"), p100=0.0, p80=20.0, p50=40.0, p0=120.0)
    s_p95 = _score_lower_better(geom.get("depth_p95_mm"), p100=0.0, p80=50.0, p50=100.0, p0=250.0)
    s_p2m = _score_lower_better(geom.get("point_to_mesh_p95_mm"), p100=0.0, p80=40.0, p50=80.0, p0=200.0)
    accuracy_score = (s_mae + s_p95 + s_p2m) / 3.0

    # 2. Geometry Coverage & Completeness (25% weight)
    s_cov = _score_higher_better(geom.get("depth_coverage_ratio"), p100=0.98, p80=0.85, p50=0.60, p0=0.20)
    s_w20 = _score_higher_better(geom.get("within_20mm_ratio"), p100=0.95, p80=0.80, p50=0.55, p0=0.20)
    s_compl = _score_higher_better(geom.get("observed_surface_completeness", geom.get("within_50mm_ratio")), p100=0.90, p80=0.75, p50=0.50, p0=0.15)
    coverage_score = (s_cov + s_w20 + s_compl) / 3.0

    # 3. Free-space & Artifact Penalties (20% weight)
    s_fs = _score_higher_better(geom.get("free_space_correctness_ratio", 1.0), p100=0.99, p80=0.95, p50=0.80, p0=0.50)
    s_art = _score_lower_better(mesh_m.get("small_component_area_ratio"), p100=0.0, p80=0.03, p50=0.10, p0=0.30)
    freespace_artifact_score = (s_fs * 2.0 + s_art) / 3.0

    # 4. Topology & Structural Quality (15% weight)
    s_deg = _score_lower_better(mesh_m.get("degenerate_triangle_ratio"), p100=0.0, p80=0.001, p50=0.01, p0=0.05)
    plane_res = planes.get("dominant_plane_residual_mean_mm") or planes.get("mean_residual_mm")
    s_plane = _score_lower_better(plane_res, p100=0.0, p80=15.0, p50=30.0, p0=80.0) if plane_res is not None else 85.0
    topology_score = (s_deg + s_plane) / 2.0

    # Quality Score (0 to 100)
    quality_score = (
        0.40 * accuracy_score +
        0.25 * coverage_score +
        0.20 * freespace_artifact_score +
        0.15 * topology_score
    )

    # 5. Cost Score (Lower resources -> Higher cost score)
    runtime = summary.get("runtime_sec") or perf.get("runtime_sec")
    triangles = mesh_m.get("num_triangles")
    s_runtime = _score_lower_better(runtime, p100=0.0, p80=10.0, p50=60.0, p0=300.0)
    s_triangles = _score_lower_better(triangles, p100=10000.0, p80=100000.0, p50=500000.0, p0=2000000.0)
    cost_score = (s_runtime + s_triangles) / 2.0

    # Composite score: Quality 90%, Cost 10%
    composite_score = quality_score * 0.90 + cost_score * 0.10

    return {
        "quality_score": round(quality_score, 2),
        "cost_score": round(cost_score, 2),
        "composite_score": round(composite_score, 2),
        "component_scores": {
            "geometry_accuracy": round(accuracy_score, 1),
            "geometry_coverage": round(coverage_score, 1),
            "artifact_penalty": round(freespace_artifact_score, 1),
            "topology": round(topology_score, 1),
            "performance": round(cost_score, 1)
        }
    }


def rank_candidate_summaries(
    summaries: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """Filter summaries via Hard Gate, compute absolute scores, and rank candidates."""
    if not summaries:
        return []

    valid_entries = []
    failed_entries = []

    for s in summaries:
        is_valid, reason = HardGateFilter.evaluate(s)
        cand_name = s.get("candidate_name", "unknown")
        status = s.get("status") or s.get("overall_status", "PASS" if is_valid else "FAIL")

        if is_valid:
            scores = compute_absolute_scores(s)
            valid_entries.append({
                "candidate_name": cand_name,
                "composite_score": scores["composite_score"],
                "quality_score": scores["quality_score"],
                "cost_score": scores["cost_score"],
                "status": status,
                "hard_gate_pass": True,
                "component_scores": scores["component_scores"],
                "raw_metrics": {
                    "depth_mae_mm": s.get("geometry", {}).get("depth_mae_mm"),
                    "depth_p95_mm": s.get("geometry", {}).get("depth_p95_mm"),
                    "point_to_mesh_p95_mm": s.get("geometry", {}).get("point_to_mesh_p95_mm"),
                    "depth_coverage_ratio": s.get("geometry", {}).get("depth_coverage_ratio"),
                    "within_20mm_ratio": s.get("geometry", {}).get("within_20mm_ratio"),
                    "observed_surface_completeness": s.get("geometry", {}).get("observed_surface_completeness"),
                    "free_space_correctness_ratio": s.get("geometry", {}).get("free_space_correctness_ratio", 1.0),
                    "small_component_ratio": s.get("mesh", {}).get("small_component_area_ratio"),
                    "runtime_sec": s.get("runtime_sec") or s.get("performance", {}).get("runtime_sec")
                },
                "summary_data": s
            })
        else:
            failed_entries.append({
                "candidate_name": cand_name,
                "composite_score": 0.0,
                "quality_score": 0.0,
                "cost_score": 0.0,
                "status": status,
                "hard_gate_pass": False,
                "failure_reason": reason,
                "component_scores": {
                    "geometry_accuracy": 0.0,
                    "geometry_coverage": 0.0,
                    "artifact_penalty": 0.0,
                    "topology": 0.0,
                    "performance": 0.0
                },
                "raw_metrics": {
                    "depth_mae_mm": s.get("geometry", {}).get("depth_mae_mm"),
                    "depth_p95_mm": s.get("geometry", {}).get("depth_p95_mm"),
                    "point_to_mesh_p95_mm": s.get("geometry", {}).get("point_to_mesh_p95_mm"),
                    "depth_coverage_ratio": s.get("geometry", {}).get("depth_coverage_ratio"),
                    "within_20mm_ratio": s.get("geometry", {}).get("within_20mm_ratio"),
                    "small_component_ratio": s.get("mesh", {}).get("small_component_area_ratio"),
                    "runtime_sec": s.get("runtime_sec") or s.get("performance", {}).get("runtime_sec")
                },
                "summary_data": s
            })

    # Sort valid entries:
    # Primary: Quality Score (if difference > 0.5)
    # Secondary / Tie-break: Cost Score (higher is better / lower resources)
    def sort_key(item):
        q = item.get("quality_score", 0.0)
        c = item.get("cost_score", 0.0)
        # Quantize quality into 0.5 point bins for tie breaking by cost
        q_bin = round(q * 2.0) / 2.0
        return (q_bin, c, item.get("composite_score", 0.0))

    valid_entries.sort(key=sort_key, reverse=True)

    # Combine valid ranked candidates with failed candidates
    all_ranked = list(valid_entries) + failed_entries
    for idx, item in enumerate(all_ranked, 1):
        item["rank"] = idx

    return all_ranked


def explain_winner_decision(winner: Optional[Dict[str, Any]], runner_up: Optional[Dict[str, Any]]) -> List[str]:
    """Generate human-readable explainable rationale why the winner was chosen over the runner-up."""
    if not winner:
        return ["No candidate passed Hard Gate quality requirements; no winner could be selected."]
    if not runner_up:
        return [
            f"Candidate `{winner.get('candidate_name')}` was the sole candidate to pass all quality criteria.",
            f"Quality Score: {winner.get('quality_score', 0):.1f}/100, Composite: {winner.get('composite_score', 0):.1f}/100."
        ]

    w_name = winner.get("candidate_name")
    r_name = runner_up.get("candidate_name")
    w_qual = winner.get("quality_score", 0.0)
    r_qual = runner_up.get("quality_score", 0.0)
    w_cost = winner.get("cost_score", 0.0)
    r_cost = runner_up.get("cost_score", 0.0)
    q_diff = w_qual - r_qual

    w_raw = winner.get("raw_metrics", {})
    r_raw = runner_up.get("raw_metrics", {})

    lines = [f"**Winner Decision Rationale** (`{w_name}` vs Rank 2 `{r_name}`):"]

    w_mae = w_raw.get("depth_mae_mm")
    r_mae = r_raw.get("depth_mae_mm")
    if w_mae is not None and r_mae is not None:
        mae_comp = "superior (+)" if w_mae < r_mae else ("equal" if w_mae == r_mae else "inferior (-)")
        lines.append(f"- **Depth Accuracy (MAE)**: `{w_mae:.1f} mm` vs `{r_mae:.1f} mm` ({mae_comp})")

    w_cov = w_raw.get("depth_coverage_ratio")
    r_cov = r_raw.get("depth_coverage_ratio")
    if w_cov is not None and r_cov is not None:
        lines.append(f"- **Coverage**: `{w_cov*100:.1f}%` vs `{r_cov*100:.1f}%`")

    w_fs = w_raw.get("free_space_correctness_ratio", 1.0)
    r_fs = r_raw.get("free_space_correctness_ratio", 1.0)
    if w_fs is not None and r_fs is not None:
        lines.append(f"- **Free-Space Correctness**: `{w_fs*100:.1f}%` vs `{r_fs*100:.1f}%`")

    w_rt = w_raw.get("runtime_sec")
    r_rt = r_raw.get("runtime_sec")

    if abs(q_diff) <= 0.5:
        lines.append(f"- **Quality Comparison**: Quality difference (`{q_diff:+.1f}` pts) is within noise tolerance (<= 0.5 pts).")
        lines.append(f"- **Selection Basis**: **Cost Tie-Break** applied (Cost Score `{w_cost:.1f}` vs `{r_cost:.1f}`, runtime `{w_rt or 0:.1f}s` vs `{r_rt or 0:.1f}s). Lower computational complexity selected.")
    else:
        lines.append(f"- **Quality Comparison**: Substantial quality score improvement (`{q_diff:+.1f}` pts, `{w_qual:.1f}` vs `{r_qual:.1f}`).")
        if w_rt and r_rt and w_rt > r_rt:
            lines.append(f"- **Selection Basis**: Significant geometric accuracy and completeness gain justified higher runtime (`{w_rt:.1f}s` vs `{r_rt:.1f}s).")
        else:
            lines.append(f"- **Selection Basis**: Highest overall geometric fidelity with competitive compute cost.")

    return lines
