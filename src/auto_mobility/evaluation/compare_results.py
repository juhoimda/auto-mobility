#!/usr/bin/env python3
"""
compare_results.py — 동일 Dataset 내 다중 Reconstruction 결과의 정량 Ranking 및 비교 분석 도구
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import EVALUATION_DIR, get_evaluation_config


# 각 지표별 평가 방향 (True: 낮을수록 우수, False: 높을수록 우수)
METRIC_DIRECTIONS = {
    "depth_mae_mm": True,                # Lower is better
    "depth_p95_mm": True,                # Lower is better
    "point_to_mesh_p95_mm": True,        # Lower is better
    "depth_coverage_ratio": False,       # Higher is better
    "within_20mm_ratio": False,          # Higher is better
    "small_component_area_ratio": True,  # Lower is better (fewer floating artifacts)
    "degenerate_triangle_ratio": True,   # Lower is better
    "runtime_sec": True                  # Lower is better
}


def normalize_metrics(values: List[float], lower_is_better: bool) -> List[float]:
    """후보군 간 robust min-max normalization (0.0 ~ 1.0, 1.0이 항상 최고)."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    if not valid or len(valid) == 1:
        return [1.0 if v is not None else 0.0 for v in values]

    min_v = float(min(valid))
    max_v = float(max(valid))
    diff = max_v - min_v

    if diff < 1e-9:
        return [1.0 if v is not None else 0.0 for v in values]

    normed = []
    for v in values:
        if v is None or np.isnan(v):
            normed.append(0.0)
        else:
            ratio = (v - min_v) / diff
            score = 1.0 - ratio if lower_is_better else ratio
            normed.append(round(float(score), 4))
    return normed


def rank_candidates(eval_summaries: List[Dict[str, Any]], weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """동일 Dataset 내 후보군들을 정량 메트릭 가중합으로 공정 랭킹 산출."""
    if not eval_summaries:
        return []

    cfg = get_evaluation_config()
    w = weights or cfg.get("quality_weights", {
        "geometry_accuracy": 0.40,
        "geometry_coverage": 0.25,
        "artifact_penalty": 0.15,
        "topology": 0.10,
        "performance": 0.10
    })

    n = len(eval_summaries)

    # Extract raw vectors
    mae_list = [s.get("geometry", {}).get("depth_mae_mm") for s in eval_summaries]
    p95_list = [s.get("geometry", {}).get("depth_p95_mm") for s in eval_summaries]
    p2m_list = [s.get("geometry", {}).get("point_to_mesh_p95_mm") for s in eval_summaries]
    cov_list = [s.get("geometry", {}).get("depth_coverage_ratio") for s in eval_summaries]
    w20_list = [s.get("geometry", {}).get("within_20mm_ratio") for s in eval_summaries]
    art_list = [s.get("mesh", {}).get("small_component_area_ratio") for s in eval_summaries]
    deg_list = [s.get("mesh", {}).get("degenerate_triangle_ratio") for s in eval_summaries]
    run_list = [s.get("performance", {}).get("runtime_sec") for s in eval_summaries]

    # Normalize each dimension (1.0 = best)
    s_mae = normalize_metrics(mae_list, True)
    s_p95 = normalize_metrics(p95_list, True)
    s_p2m = normalize_metrics(p2m_list, True)
    s_cov = normalize_metrics(cov_list, False)
    s_w20 = normalize_metrics(w20_list, False)
    s_art = normalize_metrics(art_list, True)
    s_deg = normalize_metrics(deg_list, True)
    s_run = normalize_metrics(run_list, True)

    ranked_results = []

    for i, s in enumerate(eval_summaries):
        geom_acc_score = (s_mae[i] + s_p95[i] + s_p2m[i]) / 3.0
        geom_cov_score = (s_cov[i] + s_w20[i]) / 2.0
        artifact_score = s_art[i]
        topology_score = s_deg[i]
        perf_score = s_run[i]

        composite_score = (
            w.get("geometry_accuracy", 0.4) * geom_acc_score +
            w.get("geometry_coverage", 0.25) * geom_cov_score +
            w.get("artifact_penalty", 0.15) * artifact_score +
            w.get("topology", 0.10) * topology_score +
            w.get("performance", 0.10) * perf_score
        )

        ranked_results.append({
            "candidate_name": s.get("candidate_name", f"candidate_{i}"),
            "composite_score": round(composite_score * 100.0, 2),
            "status": s.get("overall_status", "PASS"),
            "component_scores": {
                "geometry_accuracy": round(geom_acc_score * 100.0, 1),
                "geometry_coverage": round(geom_cov_score * 100.0, 1),
                "artifact_penalty": round(artifact_score * 100.0, 1),
                "topology": round(topology_score * 100.0, 1),
                "performance": round(perf_score * 100.0, 1)
            },
            "raw_metrics": {
                "depth_mae_mm": mae_list[i],
                "depth_p95_mm": p95_list[i],
                "point_to_mesh_p95_mm": p2m_list[i],
                "depth_coverage_ratio": cov_list[i],
                "within_20mm_ratio": w20_list[i],
                "small_component_ratio": art_list[i],
                "runtime_sec": run_list[i]
            },
            "summary_data": s
        })

    # Sort descending by composite score
    ranked_results.sort(key=lambda x: x["composite_score"], reverse=True)
    for rank, item in enumerate(ranked_results, 1):
        item["rank"] = rank

    return ranked_results


def compare_dataset_evaluations(eval_dir_or_dataset: str) -> List[Dict[str, Any]]:
    p = Path(eval_dir_or_dataset)
    if not p.is_absolute():
        if (EVALUATION_DIR / eval_dir_or_dataset).exists():
            p = EVALUATION_DIR / eval_dir_or_dataset
        elif not p.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {eval_dir_or_dataset}")

    summaries = []
    # Search for evaluation_summary.json in subdirectories
    for cand_dir in p.iterdir():
        if cand_dir.is_dir():
            sum_file = cand_dir / "evaluation_summary.json"
            if sum_file.exists():
                with open(sum_file, "r", encoding="utf-8") as f:
                    summaries.append(json.load(f))

    if not summaries:
        print(f"⚠️ {p} 하위에 evaluation_summary.json 파일이 없습니다.")
        return []

    ranked = rank_candidates(summaries)

    # Print Table
    print("\n==========================================================================================")
    print(f" 🏆 Reconstruction Candidates Ranking ({p.name})")
    print("==========================================================================================")
    print(f"{'Rank':<5} {'Candidate':<24} {'Score':<8} {'Status':<8} {'Depth MAE':<12} {'Depth P95':<12} {'Coverage':<10} {'Runtime':<8}")
    print("-" * 90)
    for r in ranked:
        raw = r["raw_metrics"]
        mae_str = f"{raw['depth_mae_mm']} mm" if raw['depth_mae_mm'] is not None else "N/A"
        p95_str = f"{raw['depth_p95_mm']} mm" if raw['depth_p95_mm'] is not None else "N/A"
        cov_str = f"{raw['depth_coverage_ratio']*100:.1f}%" if raw['depth_coverage_ratio'] is not None else "N/A"
        run_str = f"{raw['runtime_sec']}s" if raw['runtime_sec'] is not None else "N/A"
        print(f"{r['rank']:<5} {r['candidate_name']:<24} {r['composite_score']:<8.1f} {r['status']:<8} {mae_str:<12} {p95_str:<12} {cov_str:<10} {run_str:<8}")

    print("==========================================================================================\n")
    if ranked:
        winner = ranked[0]
        print(f"🥇 1위 추천: `{winner['candidate_name']}` (Score: {winner['composite_score']})")
        print("   선정 근거:")
        cs = winner["component_scores"]
        print(f"   - 형상 정밀도 (Depth MAE/P95): {cs['geometry_accuracy']:.1f}/100")
        print(f"   - 센서 관측 커버리지       : {cs['geometry_coverage']:.1f}/100")
        print(f"   - 아티팩트/부유 조각 억제   : {cs['artifact_penalty']:.1f}/100")
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Compare & Rank Reconstruction Candidates on same Dataset")
    parser.add_argument("dataset", help="Dataset name or evaluations directory (e.g. ros2_data/evaluations/room01)")
    args = parser.parse_args()

    compare_dataset_evaluations(args.dataset)


if __name__ == "__main__":
    main()
