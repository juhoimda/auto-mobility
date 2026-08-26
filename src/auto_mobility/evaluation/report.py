"""
auto_mobility.evaluation.report

Reconstruction 품질 평가 보고서 (Markdown) 생성기.
"""

from typing import Dict, List, Any


def generate_markdown_report(summary: Dict[str, Any]) -> str:
    """QualityProfile 요약 딕셔너리로부터 구조화된 Markdown 리포트 생성."""
    candidate = summary.get("candidate_name", "candidate")
    dataset = summary.get("dataset_name", "dataset")
    timestamp = summary.get("evaluated_at", "")
    overall_status = summary.get("overall_status", "PASS")
    warnings = summary.get("warnings", [])

    status_badge = "🟢 PASS" if overall_status == "PASS" else ("🟡 WARN" if overall_status == "WARN" else "🔴 FAIL")

    lines = []
    lines.append(f"# 📊 3D Reconstruction Quality Report: `{candidate}`\n")
    lines.append(f"- **Dataset**: `{dataset}`")
    lines.append(f"- **Candidate**: `{candidate}`")
    lines.append(f"- **Evaluation Time**: `{timestamp}`")
    lines.append(f"- **Overall Quality Verdict**: **{status_badge}**\n")

    if warnings:
        lines.append("## ⚠️ Quality Warnings & Alerts")
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    # 1. Executive Summary Table
    lines.append("## 🎯 Executive Quality Summary\n")
    lines.append("| Category | Key Metric | Measured Value | Threshold Status |")
    lines.append("| :--- | :--- | :---: | :---: |")

    geom = summary.get("geometry", {})
    mesh = summary.get("mesh", {})
    pose = summary.get("pose_association", {})
    rules = summary.get("rule_evaluations", {})

    # Geometry accuracy
    lines.append(f"| **Geometry Accuracy** | Depth MAE | **{geom.get('depth_mae_mm', 'N/A')} mm** | {rules.get('geometry_accuracy', {}).get('status', 'N/A')} |")
    lines.append(f"| **Geometry Accuracy** | Depth P95 Error | **{geom.get('depth_p95_mm', 'N/A')} mm** | {rules.get('geometry_accuracy', {}).get('status', 'N/A')} |")
    lines.append(f"| **Geometry Accuracy** | Point-to-Mesh P95 | **{geom.get('point_to_mesh_p95_mm', 'N/A')} mm** | - |")
    lines.append(f"| **Geometry Coverage** | Sensor Depth Coverage | **{geom.get('depth_coverage_ratio', 0.0)*100:.1f}%** | {rules.get('geometry_coverage', {}).get('status', 'N/A')} |")
    lines.append(f"| **Geometry Coverage** | Within 20mm Ratio | **{geom.get('within_20mm_ratio', 0.0)*100:.1f}%** | - |")
    lines.append(f"| **Mesh Topology** | Degenerate Triangles | **{mesh.get('degenerate_triangle_ratio', 0.0)*100:.3f}%** | {rules.get('mesh_topology', {}).get('status', 'N/A')} |")
    lines.append(f"| **Mesh Topology** | Non-manifold Ratio | **{mesh.get('non_manifold_edge_ratio', 0.0)*100:.3f}%** | - |")
    lines.append(f"| **Pose Tracking** | Pose Coverage | **{pose.get('pose_coverage_ratio', 0.0)*100:.1f}%** | {rules.get('pose_association', {}).get('status', 'N/A')} |\n")

    # 2. Detailed Held-out Depth Metrics
    lines.append("## 🔍 1. Held-out Sensor Consistency (Depth Reprojection)\n")
    lines.append("> [!NOTE]")
    lines.append("> Ground Truth 계측기가 없는 환경이므로, Reconstruction에 사용되지 않은 **Held-out D435i Depth 프레임**을 Reference Observation으로 사용하여 기하학적 정합 오차를 정량 측정합니다.\n")

    lines.append("| Depth Metric | Value | Meaning & Unit |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(f"| **Depth MAE** | `{geom.get('depth_mae_mm', 'N/A')} mm` | 실제 관측과 메쉬 렌더링 간 평균 절대 오차 |")
    lines.append(f"| **Depth RMSE** | `{geom.get('depth_rmse_mm', 'N/A')} mm` | 제곱근 평균 제곱 오차 (큰 오차에 민감) |")
    lines.append(f"| **Depth Median Error** | `{geom.get('depth_median_error_mm', 'N/A')} mm` | 오차 중앙값 (노이즈/아웃라이어에 강건) |")
    lines.append(f"| **Depth P90 Error** | `{geom.get('depth_p90_mm', 'N/A')} mm` | 90% 신뢰 구간 최대 오차 |")
    lines.append(f"| **Depth P95 Error** | `{geom.get('depth_p95_mm', 'N/A')} mm` | 95% 신뢰 구간 최대 오차 (벽/모서리 변형 감지) |")
    lines.append(f"| **Within 10mm Ratio** | `{geom.get('within_10mm_ratio', 0.0)*100:.1f}%` | 1cm 이내 완벽 정합된 관측 비율 |")
    lines.append(f"| **Within 20mm Ratio** | `{geom.get('within_20mm_ratio', 0.0)*100:.1f}%` | 2cm 이내 정합된 관측 비율 |")
    lines.append(f"| **Within 50mm Ratio** | `{geom.get('within_50mm_ratio', 0.0)*100:.1f}%` | 5cm 이내 유효 정합 비율 |")
    lines.append(f"| **Depth Coverage** | `{geom.get('depth_coverage_ratio', 0.0)*100:.1f}%` | 실제 센서 관측 영역 중 메쉬 표면이 존재하는 비율 |\n")

    # 3. Point-to-Mesh & Plane Quality
    lines.append("## 📐 2. 3D Point-to-Mesh & Planar Residuals\n")
    plane = summary.get("plane_analysis", {})
    lines.append("| Metric | Value | Detail |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(f"| **Point-to-Mesh Mean** | `{geom.get('point_to_mesh_mean_mm', 'N/A')} mm` | 센서 3D 포인트에서 메쉬 표면까지의 평균 최단거리 |")
    lines.append(f"| **Point-to-Mesh Median** | `{geom.get('point_to_mesh_median_mm', 'N/A')} mm` | 포인트-메쉬 최단거리 중앙값 |")
    lines.append(f"| **Point-to-Mesh P95** | `{geom.get('point_to_mesh_p95_mm', 'N/A')} mm` | 포인트-메쉬 95 백분위 오차 |")
    lines.append(f"| **Major Planes Found** | `{plane.get('num_major_planes', 0)}` | RANSAC 실내 주 평면 (벽/바닥) 개수 |")
    lines.append(f"| **Plane Inlier Ratio** | `{plane.get('plane_inlier_ratio', 0.0)*100:.1f}%` | 주 평면에 소속된 포인트 비율 |")
    lines.append(f"| **Plane Residual Mean** | `{plane.get('plane_residual_mean_mm', 'N/A')} mm` | 평면 굴곡/왜곡 평균 잔차 |")
    lines.append(f"| **Plane Residual P95** | `{plane.get('plane_residual_p95_mm', 'N/A')} mm` | 평면 벌어짐/이중 벽 왜곡 지표 |\n")

    # 4. Mesh Quality & Topology
    lines.append("## 🔺 3. Mesh Structural & Topology Quality\n")
    lines.append("| Mesh Property | Measured Value | Evaluation Notes |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(f"| **Vertices** | `{mesh.get('num_vertices', 0):,}` | 정점 수 |")
    lines.append(f"| **Triangles** | `{mesh.get('num_triangles', 0):,}` | 삼각면 수 |")
    lines.append(f"| **Surface Area** | `{mesh.get('surface_area_m2', 0.0)} m²` | 총 표면적 |")
    lines.append(f"| **Triangle Density** | `{mesh.get('density_tri_per_m2', 0.0):,} tri/m²` | 단위 면적당 삼각면 밀도 |")
    lines.append(f"| **Bounding Box Extent** | `{mesh.get('bbox_extent_m', [])} m` | X, Y, Z 외곽 크기 |")
    lines.append(f"| **Connected Components** | `{mesh.get('connected_component_count', 1)}` | 독립 분리된 메쉬 조각 수 |")
    lines.append(f"| **Largest Component** | `{mesh.get('largest_component_ratio', 1.0)*100:.1f}%` | 주 메쉬 연결 비율 |")
    lines.append(f"| **Small Floating Shells** | `{mesh.get('small_component_count', 0)} ({mesh.get('small_component_area_ratio', 0.0)*100:.2f}%)` | 부유성 아티팩트 조각 비율 |")
    lines.append(f"| **Degenerate Triangles** | `{mesh.get('degenerate_triangle_count', 0)} ({mesh.get('degenerate_triangle_ratio', 0.0)*100:.4f}%)` | 찌그러진 무효 삼각면 |")
    lines.append(f"| **Non-manifold Edges** | `{mesh.get('non_manifold_edge_count', 0)} ({mesh.get('non_manifold_edge_ratio', 0.0)*100:.4f}%)` | 비다양체 엣지 수 |")
    lines.append(f"| **Watertight** | `{'✅ True' if mesh.get('is_watertight') else '❌ False (Open Surface)'}` | 폐곡면 여부 (실내 환경은 Open Surface 허용) |\n")

    # 5. Performance & Computational Cost
    perf = summary.get("performance", {})
    lines.append("## ⚡ 4. Computational Resource Usage\n")
    lines.append(f"- **Reconstruction Runtime**: `{perf.get('runtime_sec', 'N/A')} s`")
    lines.append(f"- **Peak RAM (RSS)**: `{perf.get('peak_rss_mb', 'N/A')} MB`")
    lines.append(f"- **Peak GPU VRAM**: `{perf.get('peak_gpu_memory_mb', 'N/A')}`\n")

    # 6. Artifact & Visualizations Reference
    renders = summary.get("render_samples", [])
    if renders:
        lines.append("## 🖼️ 5. Representative Error Heatmaps\n")
        for r in renders[:3]:
            lines.append(f"### Frame `{r.get('frame_id')}` (Hold-out sample)")
            lines.append(f"- Real Depth: `{r.get('real_path')}`")
            lines.append(f"- Rendered Depth: `{r.get('rendered_path')}`")
            lines.append(f"- Error Heatmap: `{r.get('heatmap_path')}`\n")

    return "\n".join(lines)
