"""
auto_mobility.evaluation.split

Reconstruction 학습용 프레임과 Evaluation Hold-out 프레임의 재현 가능한 분할 모듈.
동일 데이터셋에 대해 모든 알고리즘/파라미터 조합이 동일한 Hold-out 프레임 세트를 공유하도록 한다.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np


def create_holdout_split(
    total_frames: int,
    policy: str = "every_nth",
    nth: int = 5,
    ratio: float = 0.20,
    valid_indices: Optional[List[int]] = None
) -> dict:
    """Deterministic Train / Hold-out split 생성."""
    if valid_indices is not None:
        frame_list = list(valid_indices)
    else:
        frame_list = list(range(total_frames))

    n = len(frame_list)
    if n == 0:
        return {
            "total_frames": 0,
            "train_indices": [],
            "holdout_indices": [],
            "train_count": 0,
            "holdout_count": 0,
            "policy": policy,
            "holdout_ratio": 0.0
        }

    holdout_indices = []
    train_indices = []

    if policy == "every_nth":
        step = max(1, nth)
        for i, idx in enumerate(frame_list):
            if (i + 1) % step == 0:
                holdout_indices.append(idx)
            else:
                train_indices.append(idx)
    elif policy == "ratio":
        step = max(2, int(round(1.0 / ratio))) if ratio > 0 else len(frame_list) + 1
        for i, idx in enumerate(frame_list):
            if (i + 1) % step == 0:
                holdout_indices.append(idx)
            else:
                train_indices.append(idx)
    else:
        # Default: every 5th frame
        for i, idx in enumerate(frame_list):
            if (i + 1) % 5 == 0:
                holdout_indices.append(idx)
            else:
                train_indices.append(idx)

    # Edge case fallback: if holdout is empty and we have at least 2 frames
    if not holdout_indices and len(train_indices) >= 2:
        holdout_indices.append(train_indices.pop())

    return {
        "total_frames": total_frames,
        "valid_frames_count": n,
        "train_indices": train_indices,
        "holdout_indices": holdout_indices,
        "train_count": len(train_indices),
        "holdout_count": len(holdout_indices),
        "policy": policy,
        "holdout_ratio": round(len(holdout_indices) / max(n, 1), 4)
    }


def save_split_json(split_data: dict, filepath: Union[str, Path]) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)


def load_split_json(filepath: Union[str, Path]) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
