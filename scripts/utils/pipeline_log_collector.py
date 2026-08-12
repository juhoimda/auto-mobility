#!/usr/bin/env python3
"""
pipeline_log_collector.py - 파이프라인 중요 로그 자동 수집기 (2026-08-10 신규)

실시간 Real-to-Sim 파이프라인(run_pipeline_all.sh 등)의 전체 출력을
그대로 터미널에 전달하면서, 품질/성능 분석에 필요한 라인만 필터링해
타임스탬프와 함께 로그 파일에 기록한다.

불필요한 일반 로그는 제외해 파일 용량이 커지는 것을 방지한다.
  - ERROR  : 오류/차단/예외 라인 (항상 저장)
  - WARN   : 경고/주의/저하/리소스 부족 라인 (항상 저장)
  - METRIC : Hz/FPS/CPU/RAM/검증/요약 등 성능·품질 수치 (저장)
  - CONTEXT: 중요 라인 직전의 맥락 라인 (분석 편의, 기본 3줄)
  - 중복 라인은 (xN) 으로 압축해 폭주 방지
  - ANSI 색상 코드 제거 → 로그 파일은 순수 텍스트

사용법(셸에서):
  exec 3>&1 4>&2
  python3 pipeline_log_collector.py LOG_FILE < FIFO >&3 &
  exec > FIFO 2>&1
"""

import os
import re
import sys
import signal
import argparse
from datetime import datetime
from collections import deque

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
MAX_LINE_LEN = 500

# 라인 분류 정규식
ERROR_RE = re.compile(
    r"❌|\[ERROR\]|\[FATAL\]|\[CRITICAL\]|\bError\b|Exception|Traceback|"
    r"failed to|FAILED|오류|차단|중단|미발행|미감지|실패|"
    r"not found|No such file|does not exist|cannot open"
)
WARN_RE = re.compile(
    r"⚠|\[WARN\]|\bWarning\b|\bwarn\b|경고|주의|저하|느립니다|부족|낮게|초과|"
    r"degraded|timeout|Timeout|drop|Drop|skipping|falling back|권장"
)
METRIC_RE = re.compile(
    r"Hz|FPS|odom|Odom|CPU|RAM|MB|GB|KB|vertices|triangles|검증|보고서|"
    r"시작|종료|완료|저장|경과|STEP|GATEWAY|Barrier|BARRIER"
)

# 저장 대상이 아닌 순수 구분선/빈 줄 (CONTEXT 후보에서 제외)
SEPARATOR_RE = re.compile(r"^[\s=—\-─━|#*.\s]*$")


def classify(line):
    """라인을 'error'/'warn'/'metric'/'info' 로 분류한다."""
    if ERROR_RE.search(line):
        return "error"
    if WARN_RE.search(line):
        return "warn"
    if METRIC_RE.search(line):
        return "metric"
    return "info"


def main():
    ap = argparse.ArgumentParser(description="파이프라인 중요 로그 수집기")
    ap.add_argument("logfile", help="저장할 로그 파일 경로")
    ap.add_argument("--context", type=int, default=3,
                    help="중요 라인 직전에 함께 저장할 맥락 라인 수 (기본 3)")
    ap.add_argument("--max-lines", type=int, default=50000,
                    help="최대 저장 라인 수 (기본 50000). 초과분은 저장하지 않고 건수만 집계 — 파일 비대화 방지")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logfile = args.logfile
    context_n = max(0, args.context)
    max_lines = max(1000, args.max_lines)
    os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)

    now = datetime.now()
    counts = {"error": 0, "warn": 0, "metric": 0, "context": 0, "info": 0}
    stored = 0
    skipped = 0
    truncated_note_written = False

    # 중복 압축: 마지막으로 저장한 라인과 반복 횟수
    prev_plain = None
    prev_class = None
    prev_count = 0

    pending = deque(maxlen=context_n)

    def flush_prev(f):
        nonlocal prev_plain, prev_count, stored, skipped, truncated_note_written
        if prev_plain is None:
            return
        if stored >= max_lines:
            # 용량 상한 도달 → 세부 라인 저장 중단, 건수만 집계
            skipped += prev_count
            prev_plain = None
            prev_count = 0
            return
        ts = datetime.now().strftime("%H:%M:%S")
        suffix = f"  (x{prev_count})" if prev_count > 1 else ""
        f.write(f"[{ts}] [{prev_class.upper():<7}] {prev_plain}{suffix}\n")
        counts[prev_class] += 1
        stored += 1
        prev_plain = None
        prev_count = 0

    with open(logfile, "w", encoding="utf-8") as f:
        f.write(f"# 파이프라인 중요 로그 (자동 수집)\n")
        f.write(f"# 시작: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# 분류: ERROR / WARN / METRIC / CONTEXT\n")
        f.write("# " + "=" * 60 + "\n")

        for line in sys.stdin:
            line = line.rstrip("\n")
            plain = ANSI_RE.sub("", line)[:MAX_LINE_LEN]

            # 콘솔에는 원본 그대로 전달
            print(line, flush=True)

            if not plain.strip():
                continue

            cls = classify(plain)

            # 중복 압축 (연속 동일 라인 + 동일 분류)
            if plain == prev_plain:
                prev_count += 1
                continue

            flush_prev(f)

            if cls == "info":
                pending.append(plain)
                continue

            # 중요 라인 발견 → 맥락 라인 먼저 저장
            if stored < max_lines:
                for ctx in pending:
                    if ctx and not SEPARATOR_RE.match(ctx):
                        ts = datetime.now().strftime("%H:%M:%S")
                        f.write(f"[{ts}] [CONTEXT ] {ctx}\n")
                        counts["context"] += 1
                        stored += 1
            pending.clear()

            # 중요 라인 저장 (압축 단위로 flush_prev 로 처리)
            prev_plain = plain
            prev_class = cls
            prev_count = 1

        flush_prev(f)

        # 요약 저장 (상한 도달 시 잘린 건수 명시)
        total_kept = counts["error"] + counts["warn"] + counts["metric"] + counts["context"]
        f.write("# " + "=" * 60 + "\n")
        f.write(f"# 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 요약: ERROR {counts['error']} / WARN {counts['warn']} / "
                f"METRIC {counts['metric']} / CONTEXT {counts['context']} "
                f"(저장 라인 {total_kept}줄")
        if skipped > 0:
            f.write(f", 상한 {max_lines} 도달로 {skipped}건 미저장")
        f.write(")\n")

    print(f"\n📝 [로그 수집] 분석용 중요 로그 저장: {logfile}")
    print(f"   ERROR {counts['error']} / WARN {counts['warn']} / "
          f"METRIC {counts['metric']} / CONTEXT {counts['context']} "
          f"(저장 {total_kept}줄)", flush=True)


if __name__ == "__main__":
    main()
