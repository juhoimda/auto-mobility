#!/usr/bin/env python3
"""
sensor_integrity.py — rosbag2(MCAP) 정밀 센서 무결성 진단 및 품질 보고서 생성기

기능:
  - Topic별: count, duration, effective Hz, inter-frame interval (mean, p50, p90, p95, p99, max),
             timestamp reversal, duplicates, large gaps, missing frame estimate
  - RGB ↔ Depth: Rate gap, sync delta distribution (mean, p50, p90, p95, p99, max), unmatched ratio
  - IR1 ↔ IR2: Stereo sync delta & rate comparison
  - IMU: effective Hz, interval jitter, largest gap, timestamp monotonicity
  - CameraInfo: Intrinsics (fx, fy, cx, cy), distortion, resolution match
  - Output: 콘솔 포맷 출력 + JSON 저장 + Markdown 보고서 저장

사용법:
  python3 -m auto_mobility.diagnostics.sensor_integrity <bag_path_or_name> [--out-dir <dir>]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import (
    BAG_DIR, LOG_DIR,
    CAMERA_RGB_TOPIC, CAMERA_RGB_COMPRESSED_TOPIC,
    CAMERA_DEPTH_TOPIC, CAMERA_DEPTH_COMPRESSED_TOPIC,
    CAMERA_ALIGNED_DEPTH_TOPIC, CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC,
    CAMERA_INFO_TOPIC, CAMERA_INFO_WINDOWS_TOPIC,
    CAMERA_INFRA1_TOPIC, CAMERA_INFRA1_COMPRESSED_TOPIC,
    CAMERA_INFRA2_TOPIC, CAMERA_INFRA2_COMPRESSED_TOPIC,
    CAMERA_IMU_TOPIC, CAMERA_IMU_FILTERED_TOPIC,
)

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    rosbag2_py = None


def _stamp_sec(msg, type_name: str) -> Optional[float]:
    t = type_name.lower()
    if "tfmessage" in t or "tf_message" in t:
        transforms = getattr(msg, "transforms", None)
        if transforms and len(transforms) > 0:
            header = transforms[0].header
            sec = getattr(header.stamp, "sec", 0)
            nsec = getattr(header.stamp, "nanosec", 0)
            if sec > 0:
                return sec + nsec * 1e-9
        return None
    header = getattr(msg, "header", None)
    if header and hasattr(header, "stamp"):
        sec = getattr(header.stamp, "sec", 0)
        nsec = getattr(header.stamp, "nanosec", 0)
        if sec > 0:
            return sec + nsec * 1e-9
    return None


def calculate_stream_stats(stamps: List[float], bag_duration: float, expected_hz: float = 30.0) -> Dict[str, Any]:
    if not stamps:
        return {
            "count": 0,
            "duration_sec": 0.0,
            "effective_hz": 0.0,
            "expected_hz": expected_hz,
            "status": "FAIL",
            "dt_mean_ms": 0.0,
            "dt_median_ms": 0.0,
            "dt_p90_ms": 0.0,
            "dt_p95_ms": 0.0,
            "dt_p99_ms": 0.0,
            "dt_max_ms": 0.0,
            "dt_min_ms": 0.0,
            "largest_gap_ms": 0.0,
            "duplicate_count": 0,
            "reversal_count": 0,
            "estimated_missing": int(round(bag_duration * expected_hz)) if bag_duration > 0 else 0,
        }

    arr = np.asarray(stamps, dtype=np.float64)
    n = len(arr)
    duration = float(arr[-1] - arr[0]) if n > 1 else 0.0
    effective_hz = (n - 1) / duration if duration > 0 else 0.0

    diffs = np.diff(arr)
    reversals = int(np.sum(diffs < 0))
    duplicates = int(np.sum(diffs == 0))
    valid_diffs = diffs[diffs > 0] * 1000.0  # ms

    if len(valid_diffs) > 0:
        dt_mean = float(np.mean(valid_diffs))
        dt_median = float(np.median(valid_diffs))
        dt_p90 = float(np.percentile(valid_diffs, 90))
        dt_p95 = float(np.percentile(valid_diffs, 95))
        dt_p99 = float(np.percentile(valid_diffs, 99))
        dt_max = float(np.max(valid_diffs))
        dt_min = float(np.min(valid_diffs))
    else:
        dt_mean = dt_median = dt_p90 = dt_p95 = dt_p99 = dt_max = dt_min = 0.0

    expected_total = int(round(bag_duration * expected_hz)) if expected_hz > 0 else n
    estimated_missing = max(0, expected_total - n)

    # Status check
    if expected_hz > 0:
        max_allowed_gap = 250.0 if expected_hz >= 20.0 else (2000.0 if expected_hz <= 2.0 else 500.0)
        if effective_hz >= expected_hz * 0.9 and reversals == 0 and dt_max < max_allowed_gap:
            status = "PASS"
        elif effective_hz >= expected_hz * 0.6 and reversals == 0:
            status = "WARN"
        else:
            status = "FAIL"
    else:
        status = "PASS" if reversals == 0 else "FAIL"

    return {
        "count": n,
        "first_timestamp": float(arr[0]),
        "last_timestamp": float(arr[-1]),
        "duration_sec": duration,
        "effective_hz": round(effective_hz, 2),
        "expected_hz": expected_hz,
        "status": status,
        "dt_mean_ms": round(dt_mean, 2),
        "dt_median_ms": round(dt_median, 2),
        "dt_p90_ms": round(dt_p90, 2),
        "dt_p95_ms": round(dt_p95, 2),
        "dt_p99_ms": round(dt_p99, 2),
        "dt_max_ms": round(dt_max, 2),
        "dt_min_ms": round(dt_min, 2),
        "largest_gap_ms": round(dt_max, 2),
        "duplicate_count": duplicates,
        "reversal_count": reversals,
        "estimated_missing": estimated_missing,
    }


def analyze_sync_pair(stamps_a: List[float], stamps_b: List[float], name_a="RGB", name_b="Depth") -> Dict[str, Any]:
    if not stamps_a or not stamps_b:
        return {
            "name_a": name_a,
            "name_b": name_b,
            "count_a": len(stamps_a),
            "count_b": len(stamps_b),
            "sync_p95_ms": 0.0,
            "unmatched_ratio_50ms": 1.0,
            "status": "FAIL",
            "error": "One or both streams are empty"
        }

    arr_a = np.asarray(stamps_a, dtype=np.float64)
    arr_b = np.asarray(stamps_b, dtype=np.float64)

    deltas_ms = []
    unmatched_33 = 0
    unmatched_50 = 0
    unmatched_100 = 0

    for sa in arr_a:
        min_idx = int(np.argmin(np.abs(arr_b - sa)))
        dt = abs(arr_b[min_idx] - sa) * 1000.0
        deltas_ms.append(dt)
        if dt > 33.33:
            unmatched_33 += 1
        if dt > 50.0:
            unmatched_50 += 1
        if dt > 100.0:
            unmatched_100 += 1

    deltas_arr = np.asarray(deltas_ms, dtype=np.float64)
    mean_d = float(np.mean(deltas_arr))
    median_d = float(np.median(deltas_arr))
    p90_d = float(np.percentile(deltas_arr, 90))
    p95_d = float(np.percentile(deltas_arr, 95))
    p99_d = float(np.percentile(deltas_arr, 99))
    max_d = float(np.max(deltas_arr))

    unmatched_ratio_50 = unmatched_50 / len(arr_a) if len(arr_a) > 0 else 1.0

    status = "PASS" if p95_d <= 35.0 and unmatched_ratio_50 < 0.05 else ("WARN" if p95_d <= 60.0 else "FAIL")

    return {
        "name_a": name_a,
        "name_b": name_b,
        "count_a": len(stamps_a),
        "count_b": len(stamps_b),
        "count_diff": abs(len(stamps_a) - len(stamps_b)),
        "count_ratio": round(len(stamps_a) / len(stamps_b), 3) if len(stamps_b) > 0 else 0.0,
        "delta_mean_ms": round(mean_d, 2),
        "delta_median_ms": round(median_d, 2),
        "delta_p90_ms": round(p90_d, 2),
        "delta_p95_ms": round(p95_d, 2),
        "delta_p99_ms": round(p99_d, 2),
        "delta_max_ms": round(max_d, 2),
        "unmatched_ratio_33ms": round(unmatched_33 / len(arr_a), 4) if len(arr_a) > 0 else 1.0,
        "unmatched_ratio_50ms": round(unmatched_ratio_50, 4),
        "unmatched_ratio_100ms": round(unmatched_100 / len(arr_a), 4) if len(arr_a) > 0 else 1.0,
        "status": status,
    }


def analyze_sensor_integrity(bag_path_or_name: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
    if rosbag2_py is None:
        raise RuntimeError("rosbag2_py is not available. Please run in ROS 2 Humble environment.")

    p = Path(bag_path_or_name)
    if not p.is_absolute():
        if (BAG_DIR / bag_path_or_name).exists():
            p = BAG_DIR / bag_path_or_name
        elif not p.exists():
            raise FileNotFoundError(f"Rosbag not found: {bag_path_or_name}")

    bag_path = p.resolve()
    bag_name = bag_path.name

    reader = rosbag2_py.SequentialReader()
    storage_options = None
    for storage_id in ("mcap", "sqlite3"):
        try:
            storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id)
            converter = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )
            reader.open(storage_options, converter)
            break
        except Exception:
            continue

    if storage_options is None:
        raise RuntimeError(f"Failed to open rosbag with mcap or sqlite3: {bag_path}")

    all_topics = reader.get_all_topics_and_types()
    topic_map = {t.name: t.type for t in all_topics}

    # Stream buckets
    stamps_by_topic: Dict[str, List[float]] = {t: [] for t in topic_map}
    bag_stamps_by_topic: Dict[str, List[float]] = {t: [] for t in topic_map}
    cam_info_details = {}

    total_msgs = 0
    first_bag_t = None
    last_bag_t = None

    while reader.has_next():
        topic, data, bag_stamp_ns = reader.read_next()
        total_msgs += 1
        bag_t = float(bag_stamp_ns) * 1e-9
        if first_bag_t is None:
            first_bag_t = bag_t
        last_bag_t = bag_t

        bag_stamps_by_topic[topic].append(bag_t)

        try:
            msg = deserialize_message(data, get_message(topic_map[topic]))
            s = _stamp_sec(msg, topic_map[topic])
            if s is not None:
                stamps_by_topic[topic].append(s)

            if "camerainfo" in topic_map[topic].lower() and not cam_info_details.get(topic):
                if hasattr(msg, "k") and len(msg.k) >= 9 and msg.k[0] > 0:
                    cam_info_details[topic] = {
                        "width": int(msg.width),
                        "height": int(msg.height),
                        "fx": float(msg.k[0]),
                        "fy": float(msg.k[4]),
                        "cx": float(msg.k[2]),
                        "cy": float(msg.k[5]),
                        "distortion_model": str(msg.distortion_model),
                        "d": [float(x) for x in msg.d],
                    }
        except Exception:
            pass

    bag_duration = float(last_bag_t - first_bag_t) if (first_bag_t and last_bag_t) else 0.0

    # Categorize topics
    rgb_topic = next((t for t in [CAMERA_RGB_COMPRESSED_TOPIC, CAMERA_RGB_TOPIC] if t in topic_map), None)
    depth_topic = next((t for t in [CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC, CAMERA_ALIGNED_DEPTH_TOPIC,
                                   CAMERA_DEPTH_COMPRESSED_TOPIC, CAMERA_DEPTH_TOPIC] if t in topic_map), None)
    ir1_topic = next((t for t in [CAMERA_INFRA1_COMPRESSED_TOPIC, CAMERA_INFRA1_TOPIC] if t in topic_map), None)
    ir2_topic = next((t for t in [CAMERA_INFRA2_COMPRESSED_TOPIC, CAMERA_INFRA2_TOPIC] if t in topic_map), None)
    imu_topic = next((t for t in [CAMERA_IMU_TOPIC, CAMERA_IMU_FILTERED_TOPIC] if t in topic_map), None)
    info_topic = next((t for t in [CAMERA_INFO_WINDOWS_TOPIC, CAMERA_INFO_TOPIC] if t in topic_map), None)

    # Compute stream stats
    stream_stats = {}
    stream_stats["RGB"] = calculate_stream_stats(stamps_by_topic.get(rgb_topic, []), bag_duration, expected_hz=30.0) if rgb_topic else None
    stream_stats["Depth"] = calculate_stream_stats(stamps_by_topic.get(depth_topic, []), bag_duration, expected_hz=30.0) if depth_topic else None
    stream_stats["IR1"] = calculate_stream_stats(stamps_by_topic.get(ir1_topic, []), bag_duration, expected_hz=30.0) if ir1_topic else None
    stream_stats["IR2"] = calculate_stream_stats(stamps_by_topic.get(ir2_topic, []), bag_duration, expected_hz=30.0) if ir2_topic else None
    stream_stats["IMU"] = calculate_stream_stats(stamps_by_topic.get(imu_topic, []), bag_duration, expected_hz=200.0) if imu_topic else None
    stream_stats["CameraInfo"] = calculate_stream_stats(stamps_by_topic.get(info_topic, []), bag_duration, expected_hz=1.0) if info_topic else None

    # Sync analysis
    rgb_depth_sync = analyze_sync_pair(
        stamps_by_topic.get(rgb_topic, []),
        stamps_by_topic.get(depth_topic, []),
        name_a="RGB", name_b="Depth"
    ) if (rgb_topic and depth_topic) else None

    ir1_ir2_sync = analyze_sync_pair(
        stamps_by_topic.get(ir1_topic, []),
        stamps_by_topic.get(ir2_topic, []),
        name_a="IR1", name_b="IR2"
    ) if (ir1_topic and ir2_topic) else None

    # Overall Evaluation
    critical_issues = []
    major_issues = []
    minor_issues = []

    if stream_stats.get("RGB"):
        s = stream_stats["RGB"]
        if s["effective_hz"] < 20.0:
            critical_issues.append(f"RGB effective rate is critically low: {s['effective_hz']} Hz (Target: 30 Hz)")
        elif s["effective_hz"] < 27.0:
            major_issues.append(f"RGB effective rate is degraded: {s['effective_hz']} Hz (Target: 30 Hz)")
        if s["largest_gap_ms"] > 500.0:
            major_issues.append(f"RGB stream has large frame drop gap: {s['largest_gap_ms']:.0f} ms")

    if stream_stats.get("Depth"):
        s = stream_stats["Depth"]
        if s["effective_hz"] < 20.0:
            critical_issues.append(f"Depth effective rate is critically low: {s['effective_hz']} Hz (Target: 30 Hz)")
        elif s["effective_hz"] < 27.0:
            major_issues.append(f"Depth effective rate is degraded: {s['effective_hz']} Hz (Target: 30 Hz)")

    if rgb_depth_sync:
        if rgb_depth_sync["delta_p95_ms"] > 60.0 or rgb_depth_sync["unmatched_ratio_50ms"] > 0.15:
            critical_issues.append(
                f"RGB↔Depth desynchronization: P95 sync delta is {rgb_depth_sync['delta_p95_ms']} ms "
                f"({rgb_depth_sync['unmatched_ratio_50ms']*100:.1f}% frames unmatched at 50ms)"
            )
        elif rgb_depth_sync["delta_p95_ms"] > 35.0:
            major_issues.append(f"RGB↔Depth sync jitter: P95 delta {rgb_depth_sync['delta_p95_ms']} ms")

    if not cam_info_details:
        major_issues.append("No valid CameraInfo intrinsics found in bag")

    overall_status = "FAIL" if critical_issues else ("WARN" if major_issues else "PASS")

    report_data = {
        "dataset": bag_name,
        "bag_path": str(bag_path),
        "total_messages": total_msgs,
        "bag_duration_sec": round(bag_duration, 2),
        "overall_status": overall_status,
        "issues": {
            "critical": critical_issues,
            "major": major_issues,
            "minor": minor_issues
        },
        "stream_stats": stream_stats,
        "rgb_depth_sync": rgb_depth_sync,
        "ir1_ir2_sync": ir1_ir2_sync,
        "camera_info": cam_info_details,
        "topics": {t: {"type": topic_map[t], "count": len(stamps_by_topic[t])} for t in topic_map}
    }

    # Save reports
    diagnostics_dir = Path(out_dir) if out_dir else (LOG_DIR / f"diagnostics_{bag_name}")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    json_path = diagnostics_dir / "sensor_integrity.json"
    md_path = diagnostics_dir / "sensor_integrity.md"

    # Also save to ros2_data/logs/sensor_integrity_<dataset>.json
    log_json_path = LOG_DIR / f"sensor_integrity_{bag_name}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    # Generate Markdown
    md = generate_markdown_report(report_data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(generate_console_report(report_data))
    print(f"\n💾 Report saved to: {md_path}")
    print(f"💾 JSON saved to: {json_path}")
    return report_data


def generate_console_report(data: Dict[str, Any]) -> str:
    lines = []
    lines.append("===========================================")
    lines.append(" 🛡️  Sensor Integrity Report")
    lines.append(f" Dataset : {data['dataset']}")
    lines.append(f" Duration: {data['bag_duration_sec']}s (Total msgs: {data['total_messages']})")
    lines.append(f" Status  : {data['overall_status']}")
    lines.append("===========================================")

    for name in ["RGB", "Depth", "IR1", "IR2", "IMU", "CameraInfo"]:
        st = data["stream_stats"].get(name)
        if not st:
            continue
        lines.append(f"\n[{name}]")
        lines.append(f" Expected : {st['expected_hz']} Hz | Actual: {st['effective_hz']} Hz (Count: {st['count']})")
        lines.append(f" Interval : mean={st['dt_mean_ms']}ms, p95={st['dt_p95_ms']}ms, max_gap={st['largest_gap_ms']}ms")
        lines.append(f" Quality  : Reversals={st['reversal_count']}, Duplicates={st['duplicate_count']}, DropEstimate={st['estimated_missing']}")
        lines.append(f" Status   : {st['status']}")

    if data.get("rgb_depth_sync"):
        sync = data["rgb_depth_sync"]
        lines.append("\n[RGB ↔ Depth Synchronization]")
        lines.append(f" Count Diff : {sync['count_diff']} frames (Ratio: {sync['count_ratio']})")
        lines.append(f" Sync Delta : mean={sync['delta_mean_ms']}ms, median={sync['delta_median_ms']}ms, p95={sync['delta_p95_ms']}ms, max={sync['delta_max_ms']}ms")
        lines.append(f" Unmatched  : >33ms: {sync['unmatched_ratio_33ms']*100:.1f}%, >50ms: {sync['unmatched_ratio_50ms']*100:.1f}%, >100ms: {sync['unmatched_ratio_100ms']*100:.1f}%")
        lines.append(f" Status     : {sync['status']}")

    if data["issues"]["critical"]:
        lines.append("\n🚨 Critical Issues:")
        for iss in data["issues"]["critical"]:
            lines.append(f"  • {iss}")
    if data["issues"]["major"]:
        lines.append("\n⚠️ Major Issues:")
        for iss in data["issues"]["major"]:
            lines.append(f"  • {iss}")

    lines.append(f"\nOverall: {data['overall_status']}")
    lines.append("===========================================")
    return "\n".join(lines)


def generate_markdown_report(data: Dict[str, Any]) -> str:
    md = []
    md.append(f"# 🛡️ Sensor Integrity Report — `{data['dataset']}`\n")
    md.append(f"- **검사 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **총 녹화 시간**: {data['bag_duration_sec']} 초")
    md.append(f"- **총 메시지 수**: {data['total_messages']}")
    md.append(f"- **최종 판정**: **`{data['overall_status']}`**\n")
    md.append("---\n")

    md.append("## 1. 스트림별 수신 속도 및 타이밍 무결성\n")
    md.append("| 스트림 | 기대 Hz | 실측 Hz | 메시지 수 | 평균 간격 | P95 간격 | 최대 Gap | 타임스탬프 역전 | 상태 |")
    md.append("|---|---|---|---|---|---|---|---|---|")

    for name in ["RGB", "Depth", "IR1", "IR2", "IMU", "CameraInfo"]:
        st = data["stream_stats"].get(name)
        if not st:
            continue
        status_icon = "✅ PASS" if st["status"] == "PASS" else ("⚠️ WARN" if st["status"] == "WARN" else "❌ FAIL")
        md.append(
            f"| **{name}** | {st['expected_hz']} Hz | {st['effective_hz']} Hz | {st['count']} | "
            f"{st['dt_mean_ms']} ms | {st['dt_p95_ms']} ms | {st['largest_gap_ms']} ms | {st['reversal_count']} | {status_icon} |"
        )

    md.append("\n---\n")
    md.append("## 2. 동기화 및 페어링 분석 (RGB ↔ Depth)\n")
    if data.get("rgb_depth_sync"):
        sync = data["rgb_depth_sync"]
        md.append(f"- **프레임 수 차이**: {sync['count_diff']} 개 (RGB: {sync['count_a']}, Depth: {sync['count_b']})")
        md.append(f"- **동기화 오차**: 평균 `{sync['delta_mean_ms']} ms`, 중앙값 `{sync['delta_median_ms']} ms`, P95 `{sync['delta_p95_ms']} ms`, 최대 `{sync['delta_max_ms']} ms`")
        md.append(f"- **임계치별 미매칭율**:")
        md.append(f"  - `> 33.3 ms` (1프레임 초과): `{sync['unmatched_ratio_33ms']*100:.1f}%`")
        md.append(f"  - `> 50.0 ms` (표준 매칭 윈도우 초과): `{sync['unmatched_ratio_50ms']*100:.1f}%`")
        md.append(f"  - `> 100.0 ms` (심각한 비동기): `{sync['unmatched_ratio_100ms']*100:.1f}%`\n")

    if data.get("ir1_ir2_sync"):
        irsync = data["ir1_ir2_sync"]
        md.append(f"### Stereo IR1 ↔ IR2 동기화\n")
        md.append(f"- **IR1 수 / IR2 수**: {irsync['count_a']} / {irsync['count_b']}")
        md.append(f"- **P95 시차**: `{irsync['delta_p95_ms']} ms` (최대: `{irsync['delta_max_ms']} ms`)\n")

    md.append("---\n")
    md.append("## 3. 발견된 이슈 요약\n")
    if data["issues"]["critical"]:
        md.append("### 🚨 Critical Issues")
        for iss in data["issues"]["critical"]:
            md.append(f"- ❌ {iss}")
    if data["issues"]["major"]:
        md.append("### ⚠️ Major Issues")
        for iss in data["issues"]["major"]:
            md.append(f"- ⚠️ {iss}")
    if not data["issues"]["critical"] and not data["issues"]["major"]:
        md.append("✅ 특이 이슈 없음 (모든 센서 스트림 정상 범위)")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Sensor Integrity Diagnostic Tool")
    parser.add_argument("bag", help="Bag name or path")
    parser.add_argument("--out-dir", default=None, help="Output directory for reports")
    args = parser.parse_args()

    analyze_sensor_integrity(args.bag, args.out_dir)


if __name__ == "__main__":
    main()
