#!/usr/bin/env python3
"""
plot_es_strategy_compare.py

Plot ES results from IOHprofiler JSON + DAT files, focusing on strategy comparisons.

Generates (per function_id + dimension):
  1) Mean best-so-far curves (±std) over evaluation fraction for each strategy label
  2) Final best distribution (boxplot per strategy) + best-found stars
  3) Final mean±std per strategy (+ best-found stars)
  4) AUC mean±std per strategy (+ best-found stars)

Also writes a JSON summary including mean/std and best-found per strategy.

Usage:
  python plot_es_strategy_compare.py --data-root data --outdir plots_es
  python plot_es_strategy_compare.py --data-root data --algo-prefix s4822285_ES_
  python plot_es_strategy_compare.py --only-strategies "(1,10)" "(1+10)"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

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
        return json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _extract_meta_any_schema(obj: dict) -> Optional[Tuple[int, str, int, bool, str]]:
    if "function_id" in obj and "function_name" in obj and "scenarios" in obj:
        try:
            fid = int(obj["function_id"])
            fname = str(obj["function_name"])
            maximize = bool(obj.get("maximization", False))  # BBOB usually minimization
            dim = int(obj["scenarios"][0]["dimension"])
            algo_obj = obj.get("algorithm_name", obj.get("algorithm", "unknown"))
            algo = str(algo_obj.get("name", "unknown")) if isinstance(algo_obj, dict) else str(algo_obj)
            return fid, fname, dim, maximize, algo
        except Exception:
            pass

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
            algo_obj = obj.get("algorithm", obj.get("algorithm_name", "unknown"))
            algo = str(algo_obj.get("name", "unknown")) if isinstance(algo_obj, dict) else str(algo_obj)
            return fid, fname, dim, maximize, algo
        except Exception:
            pass

    return None


def _find_matching_dat(json_path: Path, fid: int, dim: int) -> Optional[Path]:
    cands = list(json_path.parent.rglob("IOHprofiler_*.dat"))
    if not cands:
        return None
    for dp in cands:
        name = dp.name.lower()
        if f"f{fid}" in name and f"dim{dim}" in name:
            return dp
    return cands[0]


def _find_datasets(data_root: Path, algo_prefix: str) -> List[DatasetRecord]:
    json_files = list(data_root.rglob("IOHprofiler_*.json"))
    recs: List[DatasetRecord] = []
    for jp in json_files:
        obj = _safe_read_json(jp)
        if not obj:
            continue
        meta = _extract_meta_any_schema(obj)
        if meta is None:
            continue
        fid, fname, dim, maximize, algo = meta
        if algo_prefix and (not str(algo).startswith(algo_prefix)):
            continue
        dp = _find_matching_dat(jp, fid=fid, dim=dim)
        if dp is None or not dp.exists():
            continue
        recs.append(DatasetRecord(
            key=DatasetKey(fid=fid, fname=fname, dim=dim, maximize=maximize),
            algo=str(algo),
            json_path=jp,
            dat_path=dp,
        ))
    return recs


def _parse_dat_runs(dat_path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
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
    last = bs[0] if bs.size else np.nan
    for i, g in enumerate(grid):
        while j < evs.size and evs[j] <= g:
            last = bs[j]
            j += 1
        out[i] = last
    return out


def _resample_to_fraction_grid(evs: np.ndarray, ys: np.ndarray, maximize: bool, grid_n: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    bs = _best_so_far(ys, maximize=maximize)
    budget = int(np.max(evs)) if evs.size else 1
    frac = np.linspace(0.0, 1.0, grid_n)
    eval_grid = np.clip(np.round(frac * budget).astype(int), 1, budget)
    y_grid = _resample_step(evs, bs, eval_grid)
    return frac, y_grid


def _auc(y: np.ndarray, maximize: bool) -> float:
    """Return higher-is-better AUC on a normalized x-axis in [0,1]."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz", None)
    if integrate is None:
        raise RuntimeError("Neither np.trapezoid nor np.trapz is available.")
    dx = 1.0 / max(1, (y.size - 1))
    return float(integrate(y if maximize else -y, dx=dx))


def _infer_strategy_label(algo: str) -> str:
    m = re.search(r"\((\d+)\s*([,+])\s*(\d+)\)", algo)
    if m:
        return f"({int(m.group(1))}{m.group(2)}{int(m.group(3))})"
    m2 = re.search(r"(\d+)\s*([,+])\s*(\d+)", algo)
    if m2:
        return f"({int(m2.group(1))}{m2.group(2)}{int(m2.group(3))})"
    return algo


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _plot_curve_overlay(outdir: Path, title: str, frac: np.ndarray, mean_by_strat: Dict[str, np.ndarray], std_by_strat: Dict[str, np.ndarray], ylabel: str, outname: str):
    plt.figure(figsize=(10, 5))
    for strat in sorted(mean_by_strat.keys()):
        m = mean_by_strat[strat]
        s = std_by_strat[strat]
        plt.plot(frac, m, linewidth=2.2, label=strat)
        plt.fill_between(frac, m - s, m + s, alpha=0.15)
    plt.xlabel("Evaluation fraction (t / budget)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / outname, dpi=220)
    plt.close()


def _plot_metric_with_star(
        outdir: Path,
        title: str,
        metric_by_strat: Dict[str, Tuple[float, float]],
        best_by_strat: Dict[str, float],
        ylabel: str,
        outname: str,
):
    labels = sorted(metric_by_strat.keys())
    means = [metric_by_strat[k][0] for k in labels]
    stds = [metric_by_strat[k][1] for k in labels]
    bests = [best_by_strat[k] for k in labels]
    xs = np.arange(len(labels))

    plt.figure(figsize=(10, 4.8))
    plt.errorbar(xs, means, yerr=stds, fmt="o", capsize=4, linewidth=2.0)
    plt.scatter(xs, bests, marker="*", s=160, zorder=5, label="Best found")
    plt.xticks(xs, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / outname, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--outdir", type=str, default="plots_es")
    ap.add_argument("--algo-prefix", type=str, default="s4822285_ES_", help="only include algorithms whose name starts with this")
    ap.add_argument("--only-strategies", nargs="*", default=None, help="only include inferred strategy labels like '(1,10)' '(1+10)'")
    ap.add_argument("--grid-n", type=int, default=250)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    recs = _find_datasets(data_root, algo_prefix=args.algo_prefix)
    if not recs:
        raise SystemExit(f"No ES datasets found under {data_root} with algo-prefix={args.algo_prefix}")

    grouped: DefaultDict[Tuple[int, str, int, bool], List[DatasetRecord]] = defaultdict(list)
    for r in recs:
        grouped[(r.key.fid, r.key.fname, r.key.dim, r.key.maximize)].append(r)

    only = set(args.only_strategies) if args.only_strategies else None

    for (fid, fname, dim, maximize), items in sorted(grouped.items(), key=lambda t: (t[0][0], t[0][2], t[0][1])):
        runs_by_strat: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
        finals_by_strat: DefaultDict[str, List[float]] = defaultdict(list)
        aucs_by_strat: DefaultDict[str, List[float]] = defaultdict(list)

        frac = np.linspace(0.0, 1.0, args.grid_n)

        for rec in items:
            strat = _infer_strategy_label(rec.algo)
            if only is not None and strat not in only:
                continue
            runs = _parse_dat_runs(rec.dat_path)
            for evs, ys in runs:
                _, y_grid = _resample_to_fraction_grid(evs, ys, maximize=maximize, grid_n=args.grid_n)
                runs_by_strat[strat].append(y_grid)
                finals_by_strat[strat].append(float(y_grid[-1]))
                aucs_by_strat[strat].append(_auc(y_grid, maximize=maximize))

        if not runs_by_strat:
            continue

        mean_by_strat: Dict[str, np.ndarray] = {}
        std_by_strat: Dict[str, np.ndarray] = {}
        final_stats: Dict[str, Tuple[float, float]] = {}
        auc_stats: Dict[str, Tuple[float, float]] = {}
        final_best_found: Dict[str, float] = {}
        auc_best_found: Dict[str, float] = {}

        for strat, Ys in runs_by_strat.items():
            Y = np.vstack(Ys)
            mean_by_strat[strat] = np.nanmean(Y, axis=0)
            std_by_strat[strat] = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros(Y.shape[1])

            finals = np.asarray(finals_by_strat[strat], dtype=float)
            aucs = np.asarray(aucs_by_strat[strat], dtype=float)

            final_stats[strat] = (
                float(np.nanmean(finals)),
                float(np.nanstd(finals, ddof=1)) if finals.size > 1 else 0.0,
            )
            auc_stats[strat] = (
                float(np.nanmean(aucs)),
                float(np.nanstd(aucs, ddof=1)) if aucs.size > 1 else 0.0,
            )

            # best-found per strategy
            final_best_found[strat] = float(np.nanmax(finals) if maximize else np.nanmin(finals))
            auc_best_found[strat] = float(np.nanmax(aucs) if maximize else np.nanmin(aucs))

        ylabel = "Best-so-far (higher is better)" if maximize else "Best-so-far (lower is better)"
        safe = _safe(f"F{fid}_{fname}_d{dim}")
        title = f"{fname} (F{fid}) d={dim} | ES strategies (n={sum(len(v) for v in runs_by_strat.values())})"

        _plot_curve_overlay(outdir, title + "\nMean best-so-far (±std)", frac, mean_by_strat, std_by_strat, ylabel, f"{safe}_curves.png")

        # Boxplot + stars
        labels = sorted(finals_by_strat.keys())
        data = [np.asarray(finals_by_strat[k], dtype=float) for k in labels]
        data = [d[np.isfinite(d)] for d in data]
        keep = [i for i, d in enumerate(data) if d.size > 0]
        labels2 = [labels[i] for i in keep]
        data2 = [data[i] for i in keep]

        if data2:
            plt.figure(figsize=(10, 5))
            plt.boxplot(data2, labels=labels2, showfliers=False)
            stars = [final_best_found[k] for k in labels2]
            plt.scatter(np.arange(1, len(labels2) + 1), stars, marker="*", s=140, zorder=5, label="Best found")
            plt.xticks(rotation=25, ha="right")
            plt.ylabel("Final best-so-far")
            plt.title(title + "\nFinal distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"{safe}_final_boxplot.png", dpi=220)
            plt.close()

        _plot_metric_with_star(
            outdir,
            title + "\nFinal mean±std (+ best found)",
            final_stats,
            final_best_found,
            "Final best-so-far",
            f"{safe}_final_mean_std.png",
            )
        _plot_metric_with_star(
            outdir,
            title + "\nAUC mean±std (+ best found)",
            auc_stats,
            auc_best_found,
            "AUC (higher better)" if maximize else "AUC (higher better via sign flip)",
            f"{safe}_auc_mean_std.png",
            )

        summary = {
            "fid": fid,
            "function": fname,
            "dimension": dim,
            "maximize": maximize,
            "strategies": sorted(final_stats.keys()),
            "final_mean": {k: final_stats[k][0] for k in final_stats},
            "final_std": {k: final_stats[k][1] for k in final_stats},
            "auc_mean": {k: auc_stats[k][0] for k in auc_stats},
            "auc_std": {k: auc_stats[k][1] for k in auc_stats},
            "final_best_found": final_best_found,
            "auc_best_found": auc_best_found,
            "n_runs": {k: len(runs_by_strat[k]) for k in runs_by_strat},
        }
        (outdir / f"{safe}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        print(f"[OK] ES plots for F{fid} d={dim}: {outdir}")

    print(f"\nAll ES plots saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
