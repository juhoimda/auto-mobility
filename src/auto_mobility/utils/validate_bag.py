#!/usr/bin/env python3
"""
validate_bag.py — 녹화된 rosbag2(MCAP) 데이터셋 자동 검증 + 매니페스트 생성

검증 항목:
  - bag 존재 / 정상 판독 / 필수 토픽 존재 / 메시지 수 / 실제 Hz / zero-frame 토픽
  - CameraInfo 존재 / TF_static 존재 / IMU 존재 / timestamp 단조성 / 최대 gap
  - RGB↔Depth 동기화 delta 통계

매니페스트 생성:
  - git commit, 환경, 카메라 프로파일, 토픽 맵, config 해시

사용법:
  python3 validate_bag.py <bag_dir> [--out manifest.json] [--min-hz 15]

종료 코드:
  0 = 검증 통과 (경고 포함)
  1 = 필수 항목 실패
  2 = 실행 불가 (의존성/인자 오류)
"""
import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    import numpy as np
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    print(f"❌ [validate_bag] 의존성 부족: {e} (ROS2 환경에서 실행하세요: source /opt/ros/humble/setup.bash)")
    sys.exit(2)

# ────────────────────────────── 유틸 ──────────────────────────────

def _stamp_sec(stamp):
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", 0)
    nsec = getattr(stamp, "nanosec", 0)
    if sec <= 0:
        return None
    return sec + nsec * 1e-9


def _msg_stamp(msg, type_name):
    """메시지에서 header.stamp 추출 (타입별 대응)."""
    t = type_name.lower()
    if "tf_message" in t:
        transforms = getattr(msg, "transforms", None)
        if transforms:
            return _stamp_sec(transforms[0].header.stamp)
        return None
    header = getattr(msg, "header", None)
    if header is None:
        return None
    return _stamp_sec(header.stamp)


def _md5(path):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return None


def _git_commit():
    try:
        r = subprocess.run(
            ["git", "-C", PROJECT_DIR, "log", "-1", "--format=%H"],  # noqa: S603
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            dirty = subprocess.run(
                ["git", "-C", PROJECT_DIR, "status", "--porcelain"],  # noqa: S603
                capture_output=True, text=True, timeout=5)
            return r.stdout.strip(), bool(dirty.stdout.strip())
    except Exception:
        pass
    return None, None


def _gap_stats(stamps):
    """stamps: epoch 초 리스트 → 통계 dict."""
    if len(stamps) < 2:
        return None
    arr = np.asarray(stamps, dtype=np.float64)
    diffs = np.diff(arr)
    valid = diffs[diffs >= 0]
    out = {
        "n": int(len(arr)),
        "duration_s": float(arr[-1] - arr[0]),
        "max_gap_s": float(diffs.max()) if len(diffs) else 0.0,
        "p95_gap_s": float(np.percentile(valid, 95)) if len(valid) else 0.0,
        "mean_gap_s": float(valid.mean()) if len(valid) else 0.0,
        "monotonic_violations": int((diffs < 0).sum()),
        "zero_stamps": int((arr <= 0).sum()),
    }
    if out["duration_s"] > 0:
        out["hz"] = round((len(arr) - 1) / out["duration_s"], 2)
    else:
        out["hz"] = None
    return out


# ────────────────────────────── 주 검증 ──────────────────────────────

def read_bag(bag_dir: str):
    """bag 전체를 순회하며 토픽별 stamp 수집."""
    reader = rosbag2_py.SequentialReader()
    storage_options = None
    # Humble: get_default_storage_id() 는 인자 없음 → mcap/sqlite3 순서로 시도
    for storage_id in ("mcap", "sqlite3"):
        try:
            storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id)
            converter = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr")
            reader.open(storage_options, converter)
            break
        except Exception:
            continue
    if storage_options is None:
        print(f"❌ [validate_bag] bag 판독 실패 (mcap/sqlite3 모두 안 됨): {bag_dir}")
        sys.exit(1)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topic_stamps = {t: [] for t in type_map}
    topic_counts = {t: 0 for t in type_map}

    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic not in topic_stamps:
            continue
        topic_counts[topic] += 1
        try:
            msg = deserialize_message(data, get_message(type_map[topic]))
            s = _msg_stamp(msg, type_map[topic])
            if s is not None:
                topic_stamps[topic].append(s)
        except Exception:
            pass

    return type_map, topic_counts, topic_stamps


def classify_topic(topic: str):
    """토픽 → 역할 카테고리."""
    t = topic.lower()
    if "aligned_depth" in t or (("depth" in t) and ("image_rect_raw" in t)):
        return "depth"
    if ("color/image_raw" in t) and ("aligned_depth" not in t):
        return "rgb"
    if "camera_info" in t:
        return "camera_info"
    if t.endswith("/imu"):
        return "imu"
    if t == "/tf_static":
        return "tf_static"
    return "other"


def main():
    parser = argparse.ArgumentParser(description="rosbag2 데이터셋 검증 + 매니페스트")
    parser.add_argument("bag", help="bag 디렉터리 경로")
    parser.add_argument("--out", default=None, help="매니페스트 JSON 출력 경로")
    parser.add_argument("--min-hz", type=float, default=15.0, help="권장 최소 FPS (기본 15)")
    parser.add_argument("--max-gap", type=float, default=1.0, help="경고 최대 gap(초) (기본 1.0)")
    args = parser.parse_args()

    bag_dir = os.path.abspath(args.bag)
    if not os.path.isdir(bag_dir):
        print(f"❌ [validate_bag] bag 디렉터리 없음: {bag_dir}")
        sys.exit(1)

    t0 = time.time()
    type_map, topic_counts, topic_stamps = read_bag(bag_dir)
    if not type_map:
        print(f"❌ [validate_bag] bag 에 토픽이 없습니다 (판독 실패 가능성): {bag_dir}")
        sys.exit(1)

    # ── 토픽별 통계 ──
    topics = {}
    for topic, ttype in type_map.items():
        stats = _gap_stats(topic_stamps[topic])
        topics[topic] = {
            "type": ttype,
            "count": topic_counts[topic],
            **({"stats": stats} if stats else {}),
        }

    # ── 카테고리 집계 ──
    by_role = {}
    for topic, ttype in type_map.items():
        role = classify_topic(topic)
        by_role.setdefault(role, []).append(topic)

    # ── RGB↔Depth sync delta (근접 프레임 매칭) ──
    sync = {}
    rgb_topics = [t for t in by_role.get("rgb", []) if t in topic_stamps and topic_stamps[t]]
    depth_topics = [t for t in by_role.get("depth", []) if t in topic_stamps and topic_stamps[t]]
    if rgb_topics and depth_topics:
        rgb_stamps = np.asarray(topic_stamps[rgb_topics[0]], dtype=np.float64)
        depth_stamps = np.asarray(topic_stamps[depth_topics[0]], dtype=np.float64)
        idx = np.searchsorted(rgb_stamps, depth_stamps)
        idx = np.clip(idx, 0, len(rgb_stamps) - 1)
        deltas = depth_stamps - rgb_stamps[idx]
        sync = {
            "rgb_topic": rgb_topics[0],
            "depth_topic": depth_topics[0],
            "n": int(len(deltas)),
            "mean_s": round(float(np.abs(deltas).mean()), 4),
            "p50_s": round(float(np.percentile(np.abs(deltas), 50)), 4),
            "p95_s": round(float(np.percentile(np.abs(deltas), 95)), 4),
            "max_s": round(float(np.abs(deltas).max()), 4),
        }

    # ── 필수 체크 ──
    checks = {}
    fail = False
    for role, name in [("rgb", "RGB"), ("depth", "Depth"), ("camera_info", "CameraInfo"),
                       ("imu", "IMU"), ("tf_static", "TF_static")]:
        topics_in = by_role.get(role, [])
        ok = len(topics_in) > 0
        if ok:
            n = sum(topic_counts[t] for t in topics_in)
            ok = n > 0
            checks[name] = {"pass": ok, "count": n, "topics": topics_in}
        else:
            checks[name] = {"pass": False, "count": 0, "topics": []}
        fail = fail or not ok

    # 단조성 / zero stamp / max gap
    for role in ("rgb", "depth", "imu", "camera_info"):
        for topic in by_role.get(role, []):
            st = topics[topic].get("stats")
            if not st:
                continue
            if st["monotonic_violations"] > 0:
                checks[f"monotonic:{topic}"] = {
                    "pass": False, "violations": st["monotonic_violations"]}
                fail = True
            if st["zero_stamps"] > 0:
                checks[f"zero_stamp:{topic}"] = {"pass": False, "count": st["zero_stamps"]}
                fail = True

    # 경고 (실패 아님): 저 FPS / 큰 gap
    warnings = []
    for topic, info in topics.items():
        st = info.get("stats")
        if not st:
            continue
        if st["hz"] is not None and st["hz"] < args.min_hz:
            warnings.append(f"{topic}: {st['hz']}Hz < 권장 {args.min_hz}Hz")
        if st["max_gap_s"] > args.max_gap:
            warnings.append(f"{topic}: 최대 gap {st['max_gap_s']:.2f}s > {args.max_gap}s")

    # ── 매니페스트 ──
    commit, dirty = _git_commit()
    manifest = {
        "dataset_id": os.path.basename(bag_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bag": {
            "path": bag_dir,
            "size_bytes": sum(os.path.getsize(os.path.join(bag_dir, f))
                              for f in os.listdir(bag_dir)
                              if os.path.isfile(os.path.join(bag_dir, f))),
            "message_count": sum(topic_counts.values()),
        },
        "env": {
            "ros_distro": os.environ.get("ROS_DISTRO", ""),
            "rmw": os.environ.get("RMW_IMPLEMENTATION", ""),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", ""),
            "cyclonedds_uri": os.environ.get("CYCLONEDDS_URI", ""),
            "camera_profile": os.environ.get("CAMERA_PROFILE", ""),
            "camera_mode": os.environ.get("CAMERA_MODE", ""),
            "git_commit": commit,
            "git_dirty": dirty,
        },
        "topics": topics,
        "sync_rgb_depth": sync,
        "checks": checks,
        "warnings": warnings,
        "duration_s": round(time.time() - t0, 1),
    }

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        print(f"📄 매니페스트 저장: {args.out}")

    # ── 콘솔 요약 ──
    print("=" * 60)
    print(" 🧪 데이터셋 검증 결과")
    print("=" * 60)
    for topic, info in sorted(topics.items()):
        st = info.get("stats")
        hz = f"{st['hz']}Hz" if st and st["hz"] else "-"
        cnt = f"{info['count']:,} msgs"
        gap = f"max_gap={st['max_gap_s']:.2f}s" if st else ""
        print(f"  {'✅' if info['count'] else '❌'} {topic} [{info['type'].split('/')[-1]}] {cnt} {hz} {gap}")
    if sync:
        print(f"\n 🔀 RGB↔Depth sync delta: mean={sync['mean_s']*1000:.1f}ms "
              f"p95={sync['p95_s']*1000:.1f}ms max={sync['max_s']*1000:.1f}ms (n={sync['n']})")
    print("\n [필수 체크]")
    for name, c in checks.items():
        print(f"  {'✅' if c['pass'] else '❌'} {name}")
    for w in warnings:
        print(f"  ⚠️  {w}")
    print("=" * 60)
    if fail:
        print(" ❌ [검증 실패] 필수 항목 미달")
        sys.exit(1)
    if warnings:
        print(" ⚠️  [통과] 경고 존재 (데이터는 기록됨)")
    else:
        print(" ✅ [검증 통과]")
    sys.exit(0)


if __name__ == "__main__":
    main()
