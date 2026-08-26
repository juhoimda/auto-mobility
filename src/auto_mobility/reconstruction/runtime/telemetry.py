"""Stage telemetry for runtime/regression analysis.

Records wall time, CPU time, peak RSS/VRAM, and item counters per stage.
Aggregation is bounded (one record per stage invocation).

Complexity: O(stages). Memory: O(stages).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any, Optional


@dataclass
class StageRecord:
    stage: str
    wall_s: float = 0.0
    cpu_s: float = 0.0
    peak_rss_mb: float = 0.0
    peak_vram_mb: float = 0.0
    items_processed: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "wall_s": round(self.wall_s, 3),
            "cpu_s": round(self.cpu_s, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "items_processed": self.items_processed,
            "extra": self.extra,
        }


class TelemetryCollector:
    def __init__(self, process_name: Optional[str] = None):
        import psutil

        self._proc = psutil.Process()
        self.records: dict[str, StageRecord] = {}

    def start_stage(self, stage: str) -> StageTimer:
        return StageTimer(stage, self, self._proc)

    def add(self, record: StageRecord) -> None:
        existing = self.records.get(record.stage)
        if existing is not None:
            existing.wall_s += record.wall_s
            existing.cpu_s += record.cpu_s
            existing.peak_rss_mb = max(existing.peak_rss_mb, record.peak_rss_mb)
            existing.peak_vram_mb = max(existing.peak_vram_mb, record.peak_vram_mb)
            existing.items_processed += record.items_processed
            existing.extra.update(record.extra)
        else:
            self.records[record.stage] = record

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stages": {k: v.to_dict() for k, v in sorted(self.records.items())},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(path)


class StageTimer:
    def __init__(self, stage: str, collector: TelemetryCollector, proc):
        self._stage = stage
        self._collector = collector
        self._proc = proc
        self.items_processed = 0
        self.extra: dict[str, Any] = {}
        self._t0 = 0.0
        self._c0 = 0.0
        self._peak_rss = 0.0

    def __enter__(self) -> "StageTimer":
        import psutil

        self._t0 = time.monotonic()
        self._c0 = self._proc.cpu_times() and sum(self._proc.cpu_times()[:2]) or 0.0
        self._peak_rss = self._proc.memory_info().rss / (1024 * 1024)
        return self

    def tick_items(self, count: int) -> None:
        self.items_processed += count

    def note(self, **kwargs: Any) -> None:
        self.extra.update(kwargs)

    def sample_peak_rss(self) -> None:
        try:
            rss = self._proc.memory_info().rss / (1024 * 1024)
            self._peak_rss = max(self._peak_rss, rss)
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb) -> None:
        wall = time.monotonic() - self._t0
        cpu_now = sum(self._proc.cpu_times()[:2])
        rec = StageRecord(
            stage=self._stage,
            wall_s=max(0.0, wall),
            cpu_s=max(0.0, cpu_now - self._c0),
            peak_rss_mb=self._peak_rss,
            items_processed=self.items_processed,
            extra=self.extra,
        )
        self._collector.add(rec)
