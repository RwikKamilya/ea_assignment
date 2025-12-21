#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_dat_files(root: Path) -> list[Path]:
    return sorted(root.rglob("IOHprofiler_f*_DIM*.dat"))

def find_matching_json(dat_path: Path) -> Path | None:
    # Example: IOHprofiler_f18_DIM50.dat -> try IOHprofiler_f18_*.json in same folder or parent
    folder = dat_path.parent
    candidates = list(folder.glob("IOHprofiler_f*.json")) + list(folder.parent.glob("IOHprofiler_f*.json"))
    if not candidates:
        return None
    # Prefer same function id
    dat_name = dat_path.name
    fid = dat_name.split("_")[1]  # "f18"
    for c in candidates:
        if fid in c.name:
            return c
    return candidates[0]

def load_meta(json_path: Path | None) -> dict:
    # Defaults if json missing
    meta = {"maximize": True, "function": "unknown", "fid": "unknown", "dim": "unknown"}
    if json_path is None:
        return meta
    j = json.loads(json_path.read_text(encoding="utf-8"))
    meta["maximize"] = bool(j.get("maximization", True))
    meta["fid"] = j.get("function_id", "unknown")
    meta["function"] = j.get("function_name", "unknown")
    meta["dim"] = j.get("scenarios", [{}])[0].get("dimension", "unknown")
    return meta

def parse_runs(dat_path: Path) -> list[list[tuple[int, float]]]:
    """
    Your .dat files separate runs by repeating the header:
    'evaluations raw_y'
    (2 columns).
    """
    runs = []
    cur = []
    for line in dat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line == "...":
            continue
        if line.lower().startswith("evaluations"):
            if cur:
                runs.append(cur)
            cur = []
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            e = int(float(parts[0]))
            y = float(parts[1])
        except ValueError:
            continue
        cur.append((e, y))
    if cur:
        runs.append(cur)
    return [sorted(r, key=lambda t: t[0]) for r in runs if r]

def best_so_far_curve(run_pts: list[tuple[int, float]], budget: int, maximize: bool) -> np.ndarray:
    """
    Build best-so-far curve length=budget for a run.
    We first forward-fill raw_y between recorded evals, then cummax/cummin.
    """
    curve = np.full(budget + 1, np.nan, float)
    for e, y in run_pts:
        if 1 <= e <= budget:
            curve[e] = y

    last = -np.inf if maximize else np.inf
    for e in range(1, budget + 1):
        if not np.isnan(curve[e]):
            last = curve[e]
        else:
            curve[e] = last

    # If we never observed anything, fall back to 0s
    if not np.isfinite(curve[1]):
        curve[1:] = 0.0

    # IMPORTANT: don’t include curve[0] (NaN) in accumulate
    out = np.empty_like(curve)
    out[0] = curve[0]
    if maximize:
        out[1:] = np.maximum.accumulate(curve[1:])
    else:
        out[1:] = np.minimum.accumulate(curve[1:])
    return out[1:]  # 1..budget

def plot_bundle(curves: np.ndarray, outdir: Path, title: str, maximize: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, curves.shape[1] + 1)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    finals = curves[:, -1]

    # 1) Spaghetti + mean
    plt.figure(figsize=(9, 5))
    for r in range(curves.shape[0]):
        plt.plot(x, curves[r], alpha=0.25)
    plt.plot(x, mean, linewidth=2.8)
    plt.title(title + "\nBest-so-far traces (all runs) + mean")
    plt.xlabel("Evaluations")
    plt.ylabel("Best-so-far (maximize)" if maximize else "Best-so-far (minimize)")
    plt.tight_layout()
    plt.savefig(outdir / "spaghetti_mean.png", dpi=200)
    plt.close()

    # 2) Mean ± std
    plt.figure(figsize=(9, 5))
    plt.plot(x, mean, linewidth=2.8)
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    plt.title(title + "\nMean ± std best-so-far")
    plt.xlabel("Evaluations")
    plt.ylabel("Best-so-far (maximize)" if maximize else "Best-so-far (minimize)")
    plt.tight_layout()
    plt.savefig(outdir / "mean_std.png", dpi=200)
    plt.close()

    # 3) Heatmap (very pictorial)
    plt.figure(figsize=(9, 5))
    plt.imshow(curves, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Best-so-far value")
    plt.title(title + "\nRun-by-evaluation heatmap (each row = one run)")
    plt.xlabel("Evaluation index (1..budget)")
    plt.ylabel("Run")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap.png", dpi=200)
    plt.close()

    # 4) Histogram of final values
    plt.figure(figsize=(7, 5))
    plt.hist(finals, bins=12)
    plt.title(title + "\nDistribution of final best-so-far over runs")
    plt.xlabel("Final best-so-far value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "hist_final.png", dpi=200)
    plt.close()

    # 5) ECDF of final values
    plt.figure(figsize=(7, 5))
    xs = np.sort(finals)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.plot(xs, ys, linewidth=2.8)
    plt.title(title + "\nECDF of final best-so-far over runs")
    plt.xlabel("Final best-so-far value")
    plt.ylabel("Fraction of runs ≤ value")
    plt.tight_layout()
    plt.savefig(outdir / "ecdf_final.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="data", help="Root folder containing IOHprofiler logs")
    ap.add_argument("--outdir", type=str, default="plots", help="Where to write plots")
    ap.add_argument("--budget", type=int, required=True, help="Evaluation budget per run (GA=5000, ES=5000)")
    args = ap.parse_args()

    root = Path(args.logdir)
    outroot = Path(args.outdir)
    dat_files = find_dat_files(root)
    if not dat_files:
        raise SystemExit(f"No IOHprofiler_f*_DIM*.dat found under {root.resolve()}")

    for dat in dat_files:
        meta = load_meta(find_matching_json(dat))
        runs = parse_runs(dat)
        if not runs:
            print(f"[WARN] No runs parsed from {dat}")
            continue

        curves = np.stack([best_so_far_curve(r, args.budget, meta["maximize"]) for r in runs], axis=0)
        title = f"{dat.name} | f{meta['fid']} {meta['function']} dim={meta['dim']} | runs={curves.shape[0]} | budget={args.budget}"
        outdir = outroot / dat.parent.name / dat.stem
        plot_bundle(curves, outdir, title, meta["maximize"])
        print(f"[OK] Wrote plots -> {outdir}")

if __name__ == "__main__":
    main()
