"""
Shared utilities for the EA assignment.

Goals:
 - Reproducibility (seed everything deterministically)
 - Consistent stdout reporting (per-run + aggregated)
 - Small helpers used by GA/ES/tuning/plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import os
import random
import time

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_global_seed(seed: int) -> None:
    """
    Best-effort reproducible seeding.

    Notes:
      - PYTHONHASHSEED affects hash randomization and should ideally be set
        before the Python process starts. We still set it here if absent,
        which is usually fine for this assignment.
      - We also seed numpy + random.
    """
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class RunSummary:
    run_index: int
    seed: int
    best: float
    best_eval: int
    final_best: float
    evals_used: int
    seconds: float


def summarize_final(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def format_summary_line(label: str, stats: dict) -> str:
    return (
        f"{label}: n={stats['n']} | "
        f"min={stats['min']:.6g} | max={stats['max']:.6g} | "
        f"mean={stats['mean']:.6g} | median={stats['median']:.6g} | std={stats['std']:.6g}"
    )


def print_run_header(title: str) -> None:
    bar = "=" * max(12, len(title))
    print(bar)
    print(title)
    print(bar)


def print_run_summary(rs: RunSummary, maximize: bool) -> None:
    direction = "maximize" if maximize else "minimize"
    print(
        f"[run {rs.run_index:02d}] seed={rs.seed} ({direction}) | "
        f"best={rs.best:.6g} @ eval={rs.best_eval} | "
        f"final_best={rs.final_best:.6g} | evals={rs.evals_used} | time={rs.seconds:.2f}s"
    )


def walltime() -> float:
    return time.perf_counter()


def best_update(current_best: float, new_value: float, maximize: bool, eps: float = 1e-12) -> bool:
    if maximize:
        return new_value > current_best + eps
    return new_value < current_best - eps


def choose_extreme(a: float, b: float, maximize: bool) -> float:
    return max(a, b) if maximize else min(a, b)


def finite_or(x: float, fallback: float) -> float:
    return x if np.isfinite(x) else fallback
