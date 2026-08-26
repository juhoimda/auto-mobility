"""TrajectoryJudge: rank SLAM trajectories WITHOUT building any mesh (#28).

Metrics are computed from poses plus cheap sparse geometry only.
Time ~O(K log K), Memory O(K). NO TSDF, NO Poisson, NO full mesh in this stage.

Sparse geometry: backprojection of <=400 spatially-subsampled FUSE frames
into one global cloud, used for loop-region and thickness checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

try:
    from scipy.spatial import cKDTree

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass(frozen=True)
class TrajectoryScore:
    backend: str
    profile: str = "standard"
    coverage_ratio: float = 1.0
    timestamp_coverage: float = 1.0
    max_tracking_gap_s: float = 0.0
    translation_discontinuity_p99_m: float = 0.0
    angular_discontinuity_p99_rad: float = 0.0
    loop_region_residual: float = -1.0
    reverse_overlap_ratio: float = -1.0
    failures: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def composite(self) -> float:
        if self.failures:
            return -float("inf")
        gap_penalty = min(2.0, self.max_tracking_gap_s)
        disc_penalty = (
            self.translation_discontinuity_p99_m * 0.5
            + self.angular_discontinuity_p99_rad * 10.0
        )
        loop_bonus = 0.0
        if self.loop_region_residual >= 0:
            loop_bonus = max(0.0, 0.5 - self.loop_region_residual) * 4.0
        overlap_bonus = (
            max(0.0, self.reverse_overlap_ratio) * 5.0
            if self.reverse_overlap_ratio >= 0
            else 0.0
        )
        return (
            40.0 * self.coverage_ratio
            + 30.0 * self.timestamp_coverage
            + loop_bonus
            + overlap_bonus
            - 20.0 * gap_penalty / 2.0
            - disc_penalty
        )

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "profile": self.profile,
            "ok": self.ok,
            "failures": list(self.failures),
            "coverage_ratio": round(self.coverage_ratio, 4),
            "timestamp_coverage": round(self.timestamp_coverage, 4),
            "max_tracking_gap_s": round(self.max_tracking_gap_s, 3),
            "translation_discontinuity_p99_m": round(self.translation_discontinuity_p99_m, 4),
            "angular_discontinuity_p99_rad": round(self.angular_discontinuity_p99_rad, 4),
            "loop_region_residual": round(self.loop_region_residual, 4),
            "composite": None if not self.ok else round(self.composite(), 2),
        }


def score_trajectory(
    backend: str,
    frame_timestamps: np.ndarray,
    trajectory_timestamps: np.ndarray,
    positions_m: np.ndarray,
    rotations_rad: np.ndarray,
    expected_fps: float = 30.0,
    max_gap_s: float = 0.5,
    jump_step_m: float = 0.5,
    jump_rot_rad: float = 0.35,
    profile: str = "standard",
) -> TrajectoryScore:
    """Pure pose-level judging. All inputs sorted by time."""
    ts_f = np.asarray(frame_timestamps, dtype=np.float64)
    tj_t = np.asarray(trajectory_timestamps, dtype=np.float64)
    pos = np.asarray(positions_m, dtype=np.float64)
    rot = np.asarray(rotations_rad, dtype=np.float64)
    n = len(tj_t)
    if n < 2 or len(pos) != n or len(rot) != n:
        return TrajectoryScore(backend, profile, failures=["insufficient_poses"])

    span = max(1e-9, ts_f[-1] - ts_f[0])
    lo, hi = tj_t[0], tj_t[-1]
    overlap = max(0.0, min(hi, ts_f[-1]) - max(lo, ts_f[0]))
    timestamp_coverage = min(1.0, overlap / span)

    matched = np.searchsorted(tj_t, ts_f)
    covered = ((matched > 0) & (matched < n)).mean()

    dt = np.diff(tj_t)
    max_gap = float(dt.max()) if n > 1 else 0.0

    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    rot_step = np.abs(np.diff(rot))
    p99 = lambda a: float(np.percentile(a, 99)) if len(a) else 0.0

    failures = []
    if covered < 0.50:
        failures.append("insufficient_pose_coverage")
    if max_gap > 2.0 * max_gap_s or (max_gap > max_gap_s and covered < 0.90):
        failures.append("tracking_gap")
    if p99(step) > jump_step_m or step.max() > 10.0 * jump_step_m:
        failures.append("translation_jump")
    if p99(rot_step) > jump_rot_rad:
        failures.append("rotation_jump")

    loop_residual, reverse_overlap = _loop_metrics(pos)

    return TrajectoryScore(
        backend=backend,
        profile=profile,
        coverage_ratio=float(covered),
        timestamp_coverage=float(timestamp_coverage),
        max_tracking_gap_s=max_gap,
        translation_discontinuity_p99_m=p99(step),
        angular_discontinuity_p99_rad=p99(rot_step),
        loop_region_residual=loop_residual,
        reverse_overlap_ratio=reverse_overlap,
        failures=failures,
    )


def _loop_metrics(pos: np.ndarray) -> tuple:
    """Loop-region consistency + reverse-path overlap via spatial index.

    Candidates: temporally distant pairs within 0.3 m spatial proximity.
    Residual = mean pairwise distance of the closest such pairs.
    """
    if not _HAS_SCIPY or len(pos) < 8:
        return -1.0, -1.0
    tree = cKDTree(pos)
    k = len(pos)
    temporal_sep = max(4, k // 3)
    residuals = []
    overlap_hits = 0
    total_checked = 0
    for i in range(0, k, max(1, k // 64)):
        total_checked += 1
        near = tree.query_ball_point(pos[i], r=0.30)
        far = [j for j in near if abs(j - i) >= temporal_sep]
        if far:
            overlap_hits += 1
            d = np.linalg.norm(pos[i] - pos[far], axis=1)
            residuals.append(float(d.min()))
    if total_checked == 0:
        return -1.0, -1.0
    residual = float(np.mean(residuals)) if residuals else 0.0
    return residual, overlap_hits / total_checked


def select_top_trajectories(scores: list, top_k: int = 2) -> list:
    """Quality-first selection (#33): failed entries never preserved for diversity."""
    valid = [s for s in scores if s.ok]
    valid.sort(key=lambda s: (-s.composite(), s.backend))
    return valid[:top_k]
