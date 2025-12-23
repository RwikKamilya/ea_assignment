#!/usr/bin/env python3
"""
plot_ga_dim_effects.py

Purpose (final-report oriented):
  Show the *effect of dimension* on GA performance (not 5 plots per dim).

What it generates (per problem + algorithm):
  1) Overlay of mean best-so-far curves across dimensions (x = evaluation fraction).
  2) Heatmap: dimension (y) × evaluation fraction (x) of mean best-so-far.
  3) Trend plots vs dimension:
       - Final best (mean ± std) + best-found stars
       - AUC (mean ± std) + best-found stars
  4) Distribution plot vs dimension (boxplot) for final best + best-found stars.
  5) Optional (N-Queens): success rate vs dimension (final best == sqrt(dim)).

Additionally:
  - Writes a JSON summary including mean/std and best-found vectors.

Usage:
  python plot_ga_dim_effects.py --data-root data --out plots_ga_dims
  python plot_ga_dim_effects.py --data-root data --algo-prefix s4822285_GA
  python plot_ga_dim_effects.py --budget 5000
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
class RecordKey:
    fid: int
    fname: str
    algo: str
    maximize: bool


@dataclass
class Record:
    key: RecordKey
    dim: int
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

    Supports:
      A) Analyzer schema: function_id, function_name, maximization, scenarios[0].dimension
      B) Alternative schema: problem/problem_meta_data + optimization_type
    """
    if "function_id" in obj and "function_name" in obj and "scenarios" in obj:
        try:
            fid = int(obj["function_id"])
            fname = str(obj["function_name"])
            maximize = bool(obj.get("maximization", True))
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
    candidates = list(json_path.parent.rglob("IOHprofiler_*.dat"))
    if not candidates:
        return None
    for dp in candidates:
        name = dp.name.lower()
        if f"f{fid}" in name and f"dim{dim}" in name:
            return dp
    return candidates[0]


def _parse_dat_runs(dat_path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    .dat blocks separated by line starting with "evaluations".
    Each data line: "<eval> <raw_y>"
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


def _make_Y_on_fraction_grid(
        runs: List[Tuple[np.ndarray, np.ndarray]],
        maximize: bool,
        budget: int,
        npoints: int = 201,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      frac_grid in [0,1]
      Y: [n_runs, npoints'] best-so-far resampled at eval=ceil(frac*budget)
    """
    frac = np.linspace(0.0, 1.0, npoints, dtype=float)
    eval_grid = np.unique(np.clip(np.ceil(frac * budget).astype(int), 1, budget))
    frac_grid = (eval_grid / float(budget)).astype(float)

    Y = np.full((len(runs), eval_grid.size), np.nan, dtype=float)
    for i, (evs, ys) in enumerate(runs):
        if evs.size == 0:
            continue
        bs = _best_so_far(ys, maximize=maximize)
        Y[i] = _resample_step(evs, bs, eval_grid)
    return frac_grid, Y


def _auc_per_run(Y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    AUC of best-so-far curve on x-grid.
    Normalized by (x[-1]-x[0]) so it behaves like an average value.

    Uses np.trapezoid when available; falls back to np.trapz on older NumPy.
    """
    if Y.ndim != 2 or Y.shape[1] < 2:
        return np.nanmean(Y, axis=1)
    integ = getattr(np, "trapezoid", None)
    if integ is None:
        integ = getattr(np, "trapz", None)
    if integ is None:
        raise RuntimeError("Neither np.trapezoid nor np.trapz is available.")
    area = integ(Y, x=x, axis=1)
    denom = float(max(1e-12, x[-1] - x[0]))
    return area / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--out", type=str, default="plots_ga_dims")
    ap.add_argument("--algo-prefix", type=str, default="s4822285_GA", help="only include algorithms whose name starts with this")
    ap.add_argument("--budget", type=int, default=0, help="override budget (0 = infer from .dat)")
    ap.add_argument("--npoints", type=int, default=201, help="points on evaluation-fraction grid for overlays/heatmaps")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    json_files = list(data_root.rglob("IOHprofiler_*.json"))
    if not json_files:
        print(f"No IOHprofiler_*.json under {data_root}")
        return

    records: List[Record] = []
    for jp in json_files:
        obj = _safe_read_json(jp)
        if not obj:
            continue
        meta = _extract_meta_any_schema(obj)
        if meta is None:
            continue
        fid, fname, dim_json, maximize, algo = meta
        if not str(algo).startswith(args.algo_prefix):
            continue

        dats = list(jp.parent.rglob("IOHprofiler_*.dat"))
        fid_str = f"f{int(fid)}"
        matched = [dp for dp in dats if fid_str in dp.name.lower()]

        if not matched:
            dp = _find_matching_dat(jp, fid=int(fid), dim=int(dim_json))
            if dp is not None:
                matched = [dp]

        key = RecordKey(fid=int(fid), fname=str(fname), algo=str(algo), maximize=bool(maximize))

        import re as _re
        for dp in sorted(set(matched)):
            m = _re.search(r"DIM(\d+)", dp.name, flags=_re.IGNORECASE)
            dim_dat = int(m.group(1)) if m else int(dim_json)
            records.append(Record(key=key, dim=dim_dat, json_path=jp, dat_path=dp))

    if not records:
        print(f"No datasets found for algo-prefix={args.algo_prefix} under {data_root}")
        return

    grouped: Dict[RecordKey, List[Record]] = {}
    for r in records:
        grouped.setdefault(r.key, []).append(r)

    for key, recs in sorted(grouped.items(), key=lambda kv: (kv[0].fid, kv[0].algo)):
        recs = sorted(recs, key=lambda r: r.dim)

        dim_to_recs: Dict[int, List[Record]] = {}
        for rr in recs:
            dim_to_recs.setdefault(int(rr.dim), []).append(rr)
        dims_all = sorted(dim_to_recs.keys())

        curves_mean = []
        curves_std = []
        frac_common = None

        finals_by_dim = []
        aucs_by_dim = []
        run_counts = []

        is_nqueens = (key.fid == 23) or ("queen" in key.fname.lower())
        success_rate = []
        nqueens_target = []

        kept_dims = []

        for d in dims_all:
            rec_list = dim_to_recs[d]

            runs: List[Tuple[np.ndarray, np.ndarray]] = []
            for rr in rec_list:
                runs.extend(_parse_dat_runs(rr.dat_path))
            if not runs:
                print(f"[WARN] no runs in {[str(rr.dat_path) for rr in rec_list]}")
                continue

            if args.budget > 0:
                budget = int(args.budget)
            else:
                budget = 1
                for evs, _ in runs:
                    if evs.size:
                        budget = max(budget, int(np.max(evs)))

            frac, Y = _make_Y_on_fraction_grid(runs, maximize=key.maximize, budget=budget, npoints=args.npoints)
            mean = np.nanmean(Y, axis=0)
            std = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)

            if frac_common is None:
                frac_common = frac
            else:
                if frac.shape != frac_common.shape or np.any(frac != frac_common):
                    mean = np.interp(frac_common, frac, mean)
                    std = np.interp(frac_common, frac, std)
                    frac = frac_common

            curves_mean.append(mean)
            curves_std.append(std)

            finals = Y[:, -1]
            finals_by_dim.append(finals)

            aucs = _auc_per_run(Y, x=frac)
            aucs_by_dim.append(aucs)

            run_counts.append(Y.shape[0])
            kept_dims.append(d)

            n = int(round(np.sqrt(d)))
            if n * n == d:
                tgt = float(n)
                nqueens_target.append(tgt)
                if is_nqueens:
                    sr = float(np.mean(np.isfinite(finals) & (np.abs(finals - tgt) < 1e-9)))
                    success_rate.append(sr)
                else:
                    success_rate.append(np.nan)
            else:
                nqueens_target.append(np.nan)
                success_rate.append(np.nan)

        if frac_common is None or not curves_mean:
            continue

        dims = kept_dims
        safe_name = key.fname.replace(" ", "_").replace("/", "_")
        safe_algo = key.algo.replace(" ", "_").replace("/", "_")

        ylabel = "Best-so-far (higher is better)" if key.maximize else "Best-so-far (lower is better)"
        title_base = f"{key.fname} (F{key.fid}) | algo={key.algo}"

        # 1) Overlay
        plt.figure(figsize=(10, 5))
        for d, mean in zip(dims, curves_mean):
            plt.plot(frac_common, mean, linewidth=2.0, label=f"d={d}")
        plt.xlabel("Evaluation fraction (t / budget)")
        plt.ylabel(ylabel)
        plt.title(title_base + "\nMean best-so-far across dimensions")
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir / f"F{key.fid}_{safe_name}_{safe_algo}_dims_overlay.png", dpi=220)
        plt.close()

        # 2) Heatmap
        M = np.vstack(curves_mean)
        plt.figure(figsize=(10, 5))
        im = plt.imshow(
            M,
            aspect="auto",
            interpolation="nearest",
            extent=[frac_common[0], frac_common[-1], 0, len(dims)],
        )
        plt.colorbar(im, label=ylabel)
        plt.yticks(np.arange(len(dims)) + 0.5, [str(d) for d in dims])
        plt.xlabel("Evaluation fraction (t / budget)")
        plt.ylabel("Dimension")
        plt.title(title_base + "\nMean best-so-far heatmap (dimension × budget fraction)")
        plt.tight_layout()
        plt.savefig(outdir / f"F{key.fid}_{safe_name}_{safe_algo}_dims_heatmap.png", dpi=220)
        plt.close()

        # 3) Trends + best-found
        final_means = np.array([np.nanmean(x) for x in finals_by_dim], dtype=float)
        final_stds = np.array([np.nanstd(x, ddof=1) if np.isfinite(x).sum() > 1 else 0.0 for x in finals_by_dim], dtype=float)
        auc_means = np.array([np.nanmean(x) for x in aucs_by_dim], dtype=float)
        auc_stds = np.array([np.nanstd(x, ddof=1) if np.isfinite(x).sum() > 1 else 0.0 for x in aucs_by_dim], dtype=float)

        final_best_found = np.array(
            [np.nanmax(v) if key.maximize else np.nanmin(v) for v in finals_by_dim],
            dtype=float,
        )
        auc_best_found = np.array(
            [np.nanmax(v) if key.maximize else np.nanmin(v) for v in aucs_by_dim],
            dtype=float,
        )

        plt.figure(figsize=(8, 4.8))
        plt.errorbar(dims, final_means, yerr=final_stds, marker="o", linewidth=2.0, capsize=4)
        plt.scatter(dims, final_best_found, marker="*", s=160, zorder=5, label="Best found")
        plt.legend()
        plt.xlabel("Dimension")
        plt.ylabel("Final best-so-far (mean ± std)")
        plt.title(title_base + "\nFinal performance vs dimension")
        plt.tight_layout()
        plt.savefig(outdir / f"F{key.fid}_{safe_name}_{safe_algo}_trend_final_vs_dim.png", dpi=220)
        plt.close()

        plt.figure(figsize=(8, 4.8))
        plt.errorbar(dims, auc_means, yerr=auc_stds, marker="o", linewidth=2.0, capsize=4)
        plt.scatter(dims, auc_best_found, marker="*", s=160, zorder=5, label="Best found")
        plt.legend()
        plt.xlabel("Dimension")
        plt.ylabel("AUC of best-so-far (mean ± std)")
        plt.title(title_base + "\nSearch efficiency (AUC) vs dimension")
        plt.tight_layout()
        plt.savefig(outdir / f"F{key.fid}_{safe_name}_{safe_algo}_trend_auc_vs_dim.png", dpi=220)
        plt.close()

        # 4) Boxplot + stars
        plt.figure(figsize=(10, 5))
        plt.boxplot([x[np.isfinite(x)] for x in finals_by_dim], labels=[str(d) for d in dims], showfliers=False)
        plt.scatter(np.arange(1, len(dims) + 1), final_best_found, marker="*", s=140, zorder=5)
        plt.xlabel("Dimension")
        plt.ylabel("Final best-so-far")
        plt.title(title_base + "\nFinal best distribution across dimensions (runs merged from logs)")
        for tick in plt.gca().get_xticklabels():
            tick.set_rotation(25)
            tick.set_ha("right")
        plt.tight_layout()
        plt.savefig(outdir / f"F{key.fid}_{safe_name}_{safe_algo}_final_boxplot.png", dpi=220)
        plt.close()

        # 5) Success rate
        if is_nqueens and np.isfinite(np.array(success_rate)).any():
            plt.figure(figsize=(8, 4.8))
            plt.plot(dims, success_rate, marker="o", linewidth=2.0)
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Dimension")
            plt.ylabel("Success rate (final == sqrt(dim))")
            plt.title(title_base + "\nN-Queens solved-rate vs dimension")
            plt.tight_layout()
            plt.savefig(outdir / f"F{key.fid}_{safe_name}_{safe_algo}_success_rate_vs_dim.png", dpi=220)
            plt.close()

        summary = {
            "fid": key.fid,
            "function": key.fname,
            "algorithm": key.algo,
            "maximize": key.maximize,
            "dims": dims,
            "runs_per_dim": run_counts,
            "final_best_mean": final_means.tolist(),
            "final_best_std": final_stds.tolist(),
            "auc_mean": auc_means.tolist(),
            "auc_std": auc_stds.tolist(),
            "final_best_found": final_best_found.tolist(),
            "auc_best_found": auc_best_found.tolist(),
            "nqueens_target": nqueens_target,
            "success_rate": success_rate,
        }
        (outdir / f"F{key.fid}_{safe_name}_{safe_algo}_dim_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        print(f"[OK] {title_base}: wrote dim-effect plots to {outdir}")

    print(f"\nAll dimension-effect plots saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
