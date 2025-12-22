#!/usr/bin/env python3
"""
plot_ioh_results.py

Robust plotting for IOH Analyzer results across IOH JSON schema variants.

Works with the schema your files use (keys like):
  - function_id, function_name, maximization, scenarios, runs
and also with alternative schemas (problem/problem_meta_data).

Scans a root directory (default: ./data) for IOHprofiler_*.json and matching .dat,
then produces:
  - spaghetti + mean
  - mean ± std
  - heatmap
  - histogram of final best
  - ECDF of final best

Usage:
  python plot_ioh_results.py
  python plot_ioh_results.py --data-root submission/data --out plots
  python plot_ioh_results.py --budget 50000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DatasetKey:
    fid: int
    fname: str
    dim: int
    maximize: bool


@dataclass
class DatasetRecord:
    key: DatasetKey
    algo: str
    json_path: Path
    dat_path: Path


def _safe_read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_meta_any_schema(obj: dict) -> Optional[Tuple[int, str, int, bool, str]]:
    """
    Returns (fid, fname, dim, maximize, algo) if possible, else None.

    Supports two common IOH JSON layouts:
    A) Course/Analyzer schema:
       - function_id, function_name, maximization, scenarios[0].dimension, algorithm_name/algorithm
    B) Alternative schema:
       - problem/problem_meta_data + optimization_type
    """
    # ---- A) Course/Analyzer schema ----
    if "function_id" in obj and "function_name" in obj and "scenarios" in obj:
        try:
            fid = int(obj["function_id"])
            fname = str(obj["function_name"])
            maximize = bool(obj.get("maximization", True))
            dim = int(obj["scenarios"][0]["dimension"])
            algo = str(obj.get("algorithm_name", obj.get("algorithm", "unknown")))
            return fid, fname, dim, maximize, algo
        except Exception:
            pass

    # ---- B) Alternative schema ----
    meta = obj.get("problem", obj.get("problem_meta_data", None))
    if isinstance(meta, dict):
        try:
            fid = int(meta.get("problem_id", meta.get("fid", 0)))
            fname = str(meta.get("name", meta.get("function_name", f"F{fid}")))
            dim = int(meta.get("n_variables", meta.get("dimension", 0)))

            opt_type = meta.get("optimization_type", meta.get("optimization", ""))
            if isinstance(opt_type, dict):
                opt_type = opt_type.get("name", "")
            maximize = "max" in str(opt_type).lower()

            algo = str(obj.get("algorithm", obj.get("algorithm_name", "unknown")))
            return fid, fname, dim, maximize, algo
        except Exception:
            pass

    return None


def _find_matching_dat(json_path: Path, fid: int, dim: int) -> Optional[Path]:
    # Your structure: data/<FOLDER>/data_fXX_NAME/IOHprofiler_fXX_DIM*.dat
    candidates = list(json_path.parent.rglob("IOHprofiler_*.dat"))
    if not candidates:
        return None
    # Prefer exact match
    for dp in candidates:
        name = dp.name.lower()
        if f"f{fid}" in name and f"dim{dim}" in name:
            return dp
    return candidates[0]


def _find_datasets(data_root: Path) -> Tuple[List[DatasetRecord], List[Path]]:
    json_files = list(data_root.rglob("IOHprofiler_*.json"))
    records: List[DatasetRecord] = []

    for jp in json_files:
        obj = _safe_read_json(jp)
        if not obj:
            continue
        meta = _extract_meta_any_schema(obj)
        if meta is None:
            continue
        fid, fname, dim, maximize, algo = meta

        dat_path = _find_matching_dat(jp, fid=fid, dim=dim)
        if dat_path is None:
            continue

        key = DatasetKey(fid=fid, fname=fname, dim=dim, maximize=maximize)
        records.append(DatasetRecord(key=key, algo=algo, json_path=jp, dat_path=dat_path))

    return records, json_files


def _parse_dat_runs(dat_path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Your .dat files: blocks separated by header line "evaluations raw_y"
    and then lines: "<eval> <raw_y>"
    """
    runs: List[List[Tuple[int, float]]] = []
    cur: List[Tuple[int, float]] = []

    for line in dat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s == "...":
            continue
        if s.lower().startswith("evaluations"):
            if cur:
                runs.append(cur)
            cur = []
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            ev = int(float(parts[0]))
            y = float(parts[1])
        except Exception:
            continue
        cur.append((ev, y))

    if cur:
        runs.append(cur)

    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for r in runs:
        evs = np.array([t[0] for t in r], dtype=int)
        ys = np.array([t[1] for t in r], dtype=float)
        idx = np.argsort(evs)
        out.append((evs[idx], ys[idx]))
    return out


def _best_so_far(ys: np.ndarray, maximize: bool) -> np.ndarray:
    return np.maximum.accumulate(ys) if maximize else np.minimum.accumulate(ys)


def _resample_step(evs: np.ndarray, bs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.empty(grid.size, dtype=float)
    j = 0
    last = np.nan
    for i, g in enumerate(grid):
        while j < evs.size and evs[j] <= g:
            last = bs[j]
            j += 1
        out[i] = last
    return out


def _make_grid_matrix(runs: List[Tuple[np.ndarray, np.ndarray]], maximize: bool, budget: int) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.arange(1, budget + 1, dtype=int)
    Y = np.full((len(runs), grid.size), np.nan, dtype=float)

    for i, (evs, ys) in enumerate(runs):
        bs = _best_so_far(ys, maximize=maximize)
        Y[i] = _resample_step(evs, bs, grid)

    return grid, Y


def _plot_spaghetti(outdir: Path, title: str, grid: np.ndarray, Y: np.ndarray, mean: np.ndarray, ylabel: str, fname: str):
    plt.figure(figsize=(10, 5))
    for r in Y:
        plt.plot(grid, r, alpha=0.25, linewidth=1)
    plt.plot(grid, mean, linewidth=2.6, label="Mean best-so-far")
    plt.xlabel("Function evaluations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=220)
    plt.close()


def _plot_mean_std(outdir: Path, title: str, grid: np.ndarray, mean: np.ndarray, std: np.ndarray, ylabel: str, fname: str):
    plt.figure(figsize=(10, 5))
    plt.plot(grid, mean, linewidth=2.6, label="Mean best-so-far")
    plt.fill_between(grid, mean - std, mean + std, alpha=0.2, label="± 1 std")
    plt.xlabel("Function evaluations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=220)
    plt.close()


def _plot_heatmap(outdir: Path, title: str, Y: np.ndarray, ylabel: str, fname: str):
    plt.figure(figsize=(10, 5))
    im = plt.imshow(Y, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label=ylabel)
    plt.xlabel("Evaluation index (resampled 1..budget)")
    plt.ylabel("Run")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outdir / fname, dpi=220)
    plt.close()


def _plot_hist_ecdf(outdir: Path, title: str, finals: np.ndarray, xlabel: str, fname_hist: str, fname_ecdf: str):
    finals = finals[np.isfinite(finals)]
    if finals.size == 0:
        return

    plt.figure(figsize=(8, 4.5))
    plt.hist(finals, bins=12)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outdir / fname_hist, dpi=220)
    plt.close()

    xs = np.sort(finals)
    ys = np.arange(1, xs.size + 1) / xs.size
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, linewidth=2.6)
    plt.xlabel(xlabel)
    plt.ylabel("Fraction of runs ≤ value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outdir / fname_ecdf, dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--out", type=str, default="plots")
    ap.add_argument("--budget", type=int, default=0, help="Override budget (0 = infer from max eval in .dat)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    records, json_files = _find_datasets(data_root)

    if not json_files:
        print(f"No IOHprofiler_*.json files found under: {data_root}")
        return

    if not records:
        print(f"Found {len(json_files)} IOHprofiler JSON files under {data_root}, but could not parse metadata.")
        print("This usually means the plot script does not match your IOH JSON schema.")
        print("Example file:", json_files[0])
        return

    grouped: Dict[DatasetKey, List[DatasetRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.key, []).append(rec)

    for key, items in sorted(grouped.items(), key=lambda kv: (kv[0].fid, kv[0].dim, kv[0].fname)):
        # infer budget if not provided
        if args.budget > 0:
            budget = int(args.budget)
        else:
            max_ev = 0
            for rec in items:
                runs = _parse_dat_runs(rec.dat_path)
                for evs, _ in runs:
                    if evs.size:
                        max_ev = max(max_ev, int(np.max(evs)))
            budget = max(1, max_ev)

        ylabel = "Best-so-far (higher is better)" if key.maximize else "Best-so-far (lower is better)"
        finals_label = "Final best-so-far"

        for rec in items:
            runs = _parse_dat_runs(rec.dat_path)
            if not runs:
                print(f"[WARN] No runs parsed from {rec.dat_path}")
                continue

            grid, Y = _make_grid_matrix(runs, maximize=key.maximize, budget=budget)
            mean = np.nanmean(Y, axis=0)
            std = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
            finals = Y[:, -1]

            safe_algo = rec.algo.replace(" ", "_").replace("/", "_")
            prefix = f"F{key.fid}_d{key.dim}_{safe_algo}"

            title_base = f"{key.fname} (F{key.fid}), d={key.dim} | algo={rec.algo} | runs={Y.shape[0]} | budget={budget}"

            _plot_spaghetti(outdir, title_base + "\nSpaghetti (all runs) + mean",
                            grid, Y, mean, ylabel, prefix + "_spaghetti.png")
            _plot_mean_std(outdir, title_base + "\nMean ± std of best-so-far",
                           grid, mean, std, ylabel, prefix + "_mean_std.png")
            _plot_heatmap(outdir, title_base + "\nRun-by-evaluation heatmap",
                          Y, ylabel, prefix + "_heatmap.png")
            _plot_hist_ecdf(outdir, title_base + "\nFinal best distribution",
                            finals, finals_label, prefix + "_hist.png", prefix + "_ecdf.png")

            print(f"[OK] {prefix}: wrote plots to {outdir}")

    print(f"\nAll plots saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
