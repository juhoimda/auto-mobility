"""Convert stage results into one explainable pipeline diagnosis."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _phase_has_real_result(results: Iterable[dict]) -> bool:
    return any(r.get("status") not in ("BLOCKED", "NOT_EVALUATED") for r in results)


def diagnose_pipeline(
    pose_alignment: Dict[str, dict],
    phase_a: List[dict],
    phase_b: List[dict],
    phase_c: List[dict],
    sensor_input: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return stage statuses and a primary root-cause classification.

    This intentionally distinguishes a blocked downstream stage from a stage
    that actually ran and produced a poor metric.
    """
    stages: Dict[str, dict] = {
        "sensor_input": {"status": "NOT_EVALUATED", "cause": "NONE"},
        "pose_alignment": {"status": "NOT_EVALUATED", "cause": "NONE"},
        "slam_tracking": {"status": "NOT_EVALUATED", "cause": "NONE"},
        "tsdf_fusion": {"status": "NOT_EVALUATED", "cause": "NONE"},
        "surface_reconstruction": {"status": "NOT_EVALUATED", "cause": "NONE"},
    }

    if sensor_input:
        sensor_status = sensor_input.get("overall_status")
        if sensor_status in ("FAIL", "WARN", "PASS"):
            stages["sensor_input"] = {
                "status": sensor_status,
                "cause": "SENSOR_INPUT" if sensor_status != "PASS" else "NONE",
                "details": sensor_input.get("issues", sensor_input.get("warnings", [])),
            }

    if pose_alignment:
        statuses = [d.get("status", "FAIL") for d in pose_alignment.values()]
        if all(s == "FAIL" for s in statuses):
            stages["pose_alignment"] = {"status": "FAIL", "cause": "TIME_OR_POSE_ALIGNMENT", "details": pose_alignment}
        elif any(s == "WARN" for s in statuses):
            stages["pose_alignment"] = {"status": "WARN", "cause": "TIME_OR_POSE_ALIGNMENT", "details": pose_alignment}
        else:
            stages["pose_alignment"] = {"status": "PASS", "cause": "NONE", "details": pose_alignment}
    elif not phase_a:
        stages["pose_alignment"] = {
            "status": "FAIL",
            "cause": "TRAJECTORY_MISSING",
            "details": "No trajectory candidates were available",
        }

    if phase_a:
        failed = [r for r in phase_a if r.get("status") in ("FAIL", "FAIL_EXCEPTION", "FAIL_SEGFAULT", "FAIL_OOM", "FAIL_TIMEOUT") or r.get("overall_status") == "FAIL"]
        stages["slam_tracking"] = {
            "status": "FAIL" if failed and len(failed) == len(phase_a) else ("WARN" if failed else "PASS"),
            "cause": "SLAM_TRACKING_OR_RECONSTRUCTION" if failed else "NONE",
            "candidate_count": len(phase_a),
            "failed_count": len(failed),
        }

    if phase_b:
        if any(r.get("status") == "BLOCKED" for r in phase_b):
            stages["tsdf_fusion"] = {"status": "BLOCKED", "cause": "POSE_ALIGNMENT", "details": phase_b}
        else:
            failed = [r for r in phase_b if str(r.get("status", "")).startswith("FAIL") or r.get("overall_status") == "FAIL"]
            stages["tsdf_fusion"] = {"status": "FAIL" if failed and len(failed) == len(phase_b) else ("WARN" if failed else "PASS"), "cause": "TSDF_FUSION" if failed else "NONE", "candidate_count": len(phase_b), "failed_count": len(failed)}

    if phase_c:
        if any(r.get("status") == "BLOCKED" for r in phase_c):
            stages["surface_reconstruction"] = {"status": "BLOCKED", "cause": "POSE_ALIGNMENT", "details": phase_c}
        else:
            failed = [r for r in phase_c if str(r.get("status", "")).startswith("FAIL") or r.get("overall_status") == "FAIL"]
            stages["surface_reconstruction"] = {"status": "FAIL" if failed and len(failed) == len(phase_c) else ("WARN" if failed else "PASS"), "cause": "SURFACE_RECONSTRUCTION" if failed else "NONE", "candidate_count": len(phase_c), "failed_count": len(failed)}

    if stages["pose_alignment"]["status"] == "FAIL":
        primary = "TIME_OR_POSE_ALIGNMENT"
        confidence = "high"
        rationale = "All available trajectories failed frame↔pose alignment; downstream geometry is not attributable to mesh code."
    elif stages["slam_tracking"]["status"] == "FAIL":
        primary = "SLAM_TRACKING"
        confidence = "medium"
        rationale = "Every evaluated SLAM candidate failed after pose alignment passed."
    elif stages["tsdf_fusion"]["status"] == "FAIL":
        primary = "TSDF_FUSION"
        confidence = "medium"
        rationale = "Pose/SLAM stages were usable, but all TSDF candidates failed."
    elif stages["surface_reconstruction"]["status"] == "FAIL":
        primary = "SURFACE_RECONSTRUCTION"
        confidence = "medium"
        rationale = "Upstream pose and TSDF stages were usable, but all surface candidates failed."
    else:
        primary = "INCONCLUSIVE"
        confidence = "low"
        rationale = "No single failing stage has enough evidence for attribution."

    return {
        "primary_cause": primary,
        "confidence": confidence,
        "rationale": rationale,
        "stages": stages,
    }
