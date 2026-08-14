#!/usr/bin/env python3
"""
analyze_session.py - 촬영 세션 통합 분석 요약 생성 (2026-08-12 신규)

촬영 완료 후 파이프라인 로그 + capture_guard JSON + RTAB-Map DB 를 파싱해
세션 단위 구조화 요약(session_<ts>.summary.json)을 만든다.

포함 내용 (모두 컴팩트):
  - VO 품질: quality 분포 / quality=0 횟수 / 최장 연속 끊김 / delay·update time 통계
  - 매핑: 등록 실패·이미지 무시·프레임 폐기·루프클로저·맵 노드 수
  - TSDF: 추출/적분 프레임·skip 원인·mesh 통계·소요 시간
  - 단계별 소요 시간 (capture / mesh)
  - 환경 스냅샷: 버전(rtabmap/realsense/ROS/open3d/CUDA)·CPU·RAM·config 해시

사용법:
  python3 analyze_session.py --db session.db --log pipeline_xxx.log \
      [--guard-json capture_guard_xxx.json] --out session_xxx.summary.json

로그 원문 전체를 저장하지 않고 집계값만 남겨 파일이 비대해지지 않게 한다.
"""

import os
import re
import sys
import json
import hashlib
import argparse
import subprocess
from datetime import datetime

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ────────────────────────────── 로그 파싱 ──────────────────────────────

RE_ODOM = re.compile(
    r"\[(\d\d:\d\d:\d\d)\].*Odom: quality=(\d+), .*update time=([\d.]+)s delay=([-\d.]+)s"
)
RE_TS = re.compile(r"\[(\d\d:\d\d:\d\d)\]")

COUNT_PATTERNS = {
    "registration_failed": "Registration failed",
    "no_odometry_ignored": "no odometry is provided",
    "frame_dropped": "Dropping image/scan",
    "intermediate_node_added": "Intermediate node added",
    "loop_closure_total": "Loop closure",
    "loop_closure_rejected": "Rejected loop closure",
    "map_correction_error": "Map correction should be identity",
    "whole_cloud_regenerated": "Graph has changed",
}


def parse_pipeline_log(log_path: str) -> dict:
    out = {k: 0 for k in COUNT_PATTERNS}
    odom_quality = []
    delays = []
    update_times = []
    rtabmap_times = []
    tsdf = {}
    mesh = {}
    stage_times = {}
    max_rtabmap = 0.0
    max_delay = 0.0

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            for key, pat in COUNT_PATTERNS.items():
                if pat in line:
                    out[key] += 1

            m = RE_ODOM.search(line)
            if m:
                odom_quality.append(int(m.group(2)))
                update_times.append(float(m.group(3)))
                delays.append(float(m.group(4)))

            rm = re.search(r"RTAB-Map=([\d.]+)s", line)
            if rm:
                v = float(rm.group(1))
                rtabmap_times.append(v)
                max_rtabmap = max(max_rtabmap, v)

            dm = re.search(r"delay=([\d.]+)s", line)
            if dm:
                max_delay = max(max_delay, float(dm.group(1)))

            if "exported" in line:
                mm = re.search(r"exported (\d+) RGB-D frames", line)
                if mm:
                    tsdf["frames_exported"] = int(mm.group(1))
            if "통합 완료" in line:
                mm = re.search(r"통합 완료: (\d+) 프레임 \(skip (\d+)\)", line)
                if mm:
                    tsdf["frames_integrated"] = int(mm.group(1))
                    tsdf["frames_skipped"] = int(mm.group(2))
            if "mesh:" in line:
                mm = re.search(r"mesh: ([\d,]+) vertices / ([\d,]+) triangles", line)
                if mm:
                    mesh["vertices"] = int(mm.group(1).replace(",", ""))
                    mesh["triangles"] = int(mm.group(2).replace(",", ""))
            if "Mesh generation complete" in line:
                mm = re.search(r"complete in ([\d.]+)s", line)
                if mm:
                    mesh["poisson_duration_s"] = float(mm.group(1))
            if "저장:" in line and ".obj" in line:
                mm = re.search(r"\(([\d.]+)s\)", line)
                if mm and "tsdf" not in tsdf:
                    tsdf["reconstruct_duration_s"] = float(mm.group(1))

            # 단계별 시작/종료 타임스탬프
            ts = RE_TS.match(line)
            if ts and ("[STEP 1]" in line or "SLAM 실시간 데이터 수집 시작" in line):
                stage_times.setdefault("capture_start", ts.group(1))
            if ts and "Saving database" in line and "done" in line:
                stage_times["capture_end"] = ts.group(1)
            if ts and "[STEP 2-1]" in line:
                stage_times.setdefault("mesh_start", ts.group(1))
            if ts and "Mesh 무결성 및 구조 검사 통과" in line:
                stage_times["mesh_end"] = ts.group(1)
            if ts and "촬영 품질 모니터링 보고서" in line:
                stage_times.setdefault("capture_end", ts.group(1))

    result = {
        "counts": out,
        "vo": {
            "odom_frames": len(odom_quality),
            "quality_zero": sum(1 for q in odom_quality if q == 0),
            "quality_below_10": sum(1 for q in odom_quality if 0 < q < 10),
            "quality_avg": round(sum(odom_quality) / len(odom_quality), 1) if odom_quality else 0,
            "quality_min": min(odom_quality) if odom_quality else 0,
            "quality_max": max(odom_quality) if odom_quality else 0,
            "longest_zero_run": _longest_zero_run(odom_quality),
            "update_time_avg_s": _avg(update_times),
            "update_time_max_s": _max(update_times),
            "delay_avg_s": _avg(delays),
            "delay_max_s": _max(delays),
        },
        "mapping": {
            "max_rtabmap_time_s": round(max_rtabmap, 3),
            "max_delay_s": round(max_delay, 3),
        },
        "tsdf": tsdf,
        "mesh": mesh,
        "stages": stage_times,
    }
    return result


def _longest_zero_run(qs):
    best = cur = 0
    for q in qs:
        cur = cur + 1 if q == 0 else 0
        best = max(best, cur)
    return best


def _avg(vals):
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _max(vals):
    return round(max(vals), 1) if vals else 0.0


# ────────────────────────────── DB 통계 ──────────────────────────────

def db_stats(db_path: str) -> dict:
    if not os.path.exists(db_path):
        return {"error": "db not found"}
    try:
        import sqlite3
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        nodes = cur.execute("SELECT COUNT(*) FROM Node").fetchone()[0]
        links = cur.execute("SELECT COUNT(*) FROM Link").fetchone()[0]
        data = cur.execute("SELECT COUNT(*) FROM Data").fetchone()[0]
        weight_rows = cur.execute("SELECT weight, COUNT(*) FROM Node GROUP BY weight ORDER BY weight").fetchall()
        con.close()
        return {
            "nodes": nodes,
            "links": links,
            "data_rows": data,
            "weight_distribution": {w: c for w, c in weight_rows},
            "size_mb": round(os.path.getsize(db_path) / 1e6, 1),
        }
    except Exception as e:
        return {"error": str(e)}


# ────────────────────────────── 환경 스냅샷 ──────────────────────────────

def _run(cmd, timeout=5):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except Exception:
        return ""


def _md5(path):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return ""


def env_snapshot() -> dict:
    cpu = _run(["lscpu"]).splitlines()
    cpu_model = next((l.split(":", 1)[1].strip() for l in cpu if "Model name" in l), "")
    cpu_cores = next((l.split(":", 1)[1].strip() for l in cpu if "CPU(s):" in l), "")
    try:
        with open("/proc/meminfo") as f:
            mem_kb = int([l for l in f if "MemTotal" in l][0].split()[1])
        ram_gb = round(mem_kb / 1024 / 1024, 1)
    except Exception:
        ram_gb = 0
    shm = _run(["df", "-m", "/dev/shm"]).splitlines()
    shm_free = next((l.split()[3] for l in shm[1:] if l.split()), "") if len(shm) > 1 else ""

    def pkg_ver(pattern):
        out = _run(["dpkg", "-l"], timeout=8)
        for line in out.splitlines():
            if pattern in line:
                parts = line.split()
                if len(parts) >= 3:
                    return f"{parts[1]} {parts[2]}"
        return ""

    cfg_dir = os.path.join(PROJECT_DIR, "src", "auto_mobility")
    src_dir = os.path.join(cfg_dir, "launch")
    return {
        "ros_distro": os.environ.get("ROS_DISTRO", ""),
        "kernel": _run(["uname", "-r"]),
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "ram_gb": ram_gb,
        "shm_free_mb": shm_free,
        "gpu": _run(["nvidia-smi", "-L"]) or "n/a",
        "rtabmap": pkg_ver("rtabmap"),
        "librealsense": pkg_ver("librealsense"),
        "realsense_ros": pkg_ver("realsense2-camera"),
        "open3d": _run(["python3", "-c", "import open3d; print(open3d.__version__)"]) or "",
        "numpy": _run(["python3", "-c", "import numpy; print(numpy.__version__)"]) or "",
        "cv2": _run(["python3", "-c", "import cv2; print(cv2.__version__)"]) or "",
        "config_hashes": {
            "config.py": _md5(os.path.join(cfg_dir, "config.py")),
            "launch_common.py": _md5(os.path.join(src_dir, "launch_common.py")),
            "topics.yaml": _md5(os.path.join(PROJECT_DIR, "config", "topics.yaml")),
            "camera.launch.py": _md5(os.path.join(PROJECT_DIR, "launch", "camera.launch.py")),
            "reconstruct_tsdf.py": _md5(os.path.join(cfg_dir, "mesh", "reconstruct_tsdf.py")),
        },
    }


# ────────────────────────────── main ──────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="촬영 세션 통합 분석 요약 생성")
    parser.add_argument("--db", required=True, help="RTAB-Map DB 경로")
    parser.add_argument("--log", default=None, help="pipeline_*.log 경로")
    parser.add_argument("--guard-json", default=None, help="capture_guard_*.json 경로 (선택)")
    parser.add_argument("--out", required=True, help="출력 summary JSON 경로")
    args = parser.parse_args()

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "db": db_stats(args.db),
        "env": env_snapshot(),
    }
    if args.log and os.path.exists(args.log):
        summary["pipeline"] = parse_pipeline_log(args.log)
    if args.guard_json and os.path.exists(args.guard_json):
        try:
            with open(args.guard_json, encoding="utf-8") as f:
                guard = json.load(f)
            summary["guard"] = {
                "duration_s": guard.get("duration_s"),
                "usb": guard.get("usb"),
                "samples": guard.get("samples", []),
            }
        except Exception as e:
            summary["guard"] = {"error": str(e)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    size = os.path.getsize(args.out) / 1024
    print(f"✅ 세션 분석 요약 저장: {args.out} ({size:.1f}KB)")
    vo = summary.get("pipeline", {}).get("vo", {})
    if vo:
        print(f"   odom: {vo['odom_frames']} 프레임, quality=0 {vo['quality_zero']}회, "
              f"최장 끊김 {vo['longest_zero_run']}프레임")


if __name__ == "__main__":
    main()
