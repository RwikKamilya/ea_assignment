#!/usr/bin/env python3
"""
scan_ioh_data_report.py

Scans an IOH Analyzer output directory and prints a Markdown report.

- Recursively finds IOHprofiler_*.json
- Supports the schema your files use:
    function_id, function_name, maximization, scenarios, runs
- Extracts final best per run:
    Prefer JSON runs[*].best.y
    Fallback to parsing matching .dat and computing final best-so-far

Usage:
  python scan_ioh_data_report.py --data-root data
  python scan_ioh_data_report.py --data-root submission/data
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# Shared schema helpers (match plot_ioh_results.py)
# ----------------------------

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
    dat_path: Optional[Path]


def _safe_read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_meta_any_schema(obj: dict) -> Optional[Tuple[int, str, int, bool, str]]:
    """
    Returns (fid, fname, dim, maximize, algo) if possible, else None.

    Supports:
    A) Analyzer schema:
       - function_id, function_name, maximization, scenarios[0].dimension, algorithm_name/algorithm
    B) Alternative schema:
       - problem/problem_meta_data + optimization_type
    """
    # A) Analyzer schema
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

    # B) Alternative schema
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
    fid_s = f"f{fid}"
    dim_s = f"dim{dim}"
    for dp in candidates:
        name = dp.name.lower()
        if fid_s in name and dim_s in name:
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

        key = DatasetKey(fid=fid, fname=fname, dim=dim, maximize=maximize)
        records.append(DatasetRecord(key=key, algo=algo, json_path=jp, dat_path=dat_path))

    return records, json_files


# ----------------------------
# Run extraction
# ----------------------------

@dataclass
class RunFinal:
    final_best: float
    instance: Optional[int] = None
    seed: Optional[int] = None  # may not exist in IOH JSON
    best_eval: Optional[int] = None


def _extract_finals_from_json(obj: dict) -> List[RunFinal]:
    """
    Your JSON stores runs inside scenarios[*].runs[*].best.y etc.
    We'll gather all finals across all scenarios.
    """
    out: List[RunFinal] = []
    scenarios = obj.get("scenarios", [])
    if not isinstance(scenarios, list):
        return out

    for sc in scenarios:
        runs = sc.get("runs", [])
        if not isinstance(runs, list):
            continue
        for r in runs:
            best = r.get("best", {})
            if not isinstance(best, dict):
                continue
            y = best.get("y", None)
            if y is None:
                continue
            try:
                y = float(y)
            except Exception:
                continue
            inst = r.get("instance", None)
            try:
                inst = int(inst) if inst is not None else None
            except Exception:
                inst = None
            be = best.get("evals", None)
            try:
                be = int(be) if be is not None else None
            except Exception:
                be = None
            out.append(RunFinal(final_best=y, instance=inst, best_eval=be))
    return out


def _parse_dat_runs(dat_path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Parses .dat as blocks separated by header line 'evaluations ...'
    Then reads first two columns as: eval, raw_y.
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


def _extract_finals_from_dat(dat_path: Path, maximize: bool) -> List[RunFinal]:
    runs = _parse_dat_runs(dat_path)
    out: List[RunFinal] = []
    for evs, ys in runs:
        if ys.size == 0:
            continue
        bs = _best_so_far(ys, maximize=maximize)
        out.append(RunFinal(final_best=float(bs[-1])))
    return out


# ----------------------------
# Reporting
# ----------------------------

def _summary_stats(vals: np.ndarray) -> Dict[str, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return dict(n=0, min=np.nan, max=np.nan, mean=np.nan, median=np.nan, std=np.nan)
    return dict(
        n=int(vals.size),
        min=float(np.min(vals)),
        max=float(np.max(vals)),
        mean=float(np.mean(vals)),
        median=float(np.median(vals)),
        std=float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    records, json_files = _find_datasets(data_root)

    print("# IOH Results Scan Report\n")
    print(f"Data root: `{data_root}`")
    print(f"Found JSON files: {len(json_files)}\n")

    if not json_files:
        print("No IOHprofiler_*.json found.")
        return

    if not records:
        print("Found JSONs, but could not parse metadata from them.")
        print("First JSON:", json_files[0])
        return

    grouped: Dict[DatasetKey, List[DatasetRecord]] = {}
    for r in records:
        grouped.setdefault(r.key, []).append(r)

    for key, items in sorted(grouped.items(), key=lambda kv: (kv[0].fid, kv[0].dim, kv[0].fname)):
        direction = "maximization" if key.maximize else "minimization"
        better = "higher" if key.maximize else "lower"
        print(f"## {key.fname} (F{key.fid}), d={key.dim}\n")
        print(f"- Optimization: **{direction}** (better = {better})\n")

        for rec in items:
            obj = _safe_read_json(rec.json_path)
            if not obj:
                print(f"- Algo: `{rec.algo}` | JSON unreadable: {rec.json_path}\n")
                continue

            finals_json = _extract_finals_from_json(obj)
            finals: List[RunFinal] = finals_json

            used = "JSON"
            if len(finals) == 0 and rec.dat_path is not None and rec.dat_path.exists():
                finals = _extract_finals_from_dat(rec.dat_path, maximize=key.maximize)
                used = ".dat fallback"

            vals = np.array([r.final_best for r in finals], dtype=float)
            s = _summary_stats(vals)

            print(f"### Algo: `{rec.algo}`")
            print(f"- JSON: `{rec.json_path.name}`")
            if rec.dat_path:
                print(f"- DAT: `{rec.dat_path.relative_to(data_root)}`")
            print(f"- Finals source: **{used}**")
            print(
                f"- Final best summary: n={s['n']} | better={better} | "
                f"min={s['min']:.6g} | max={s['max']:.6g} | mean={s['mean']:.6g} | "
                f"median={s['median']:.6g} | std={s['std']:.6g}"
            )

            if s["n"] > 0:
                # best/worst depending on direction
                if key.maximize:
                    best_i = int(np.nanargmax(vals))
                    worst_i = int(np.nanargmin(vals))
                else:
                    best_i = int(np.nanargmin(vals))
                    worst_i = int(np.nanargmax(vals))

                best = finals[best_i]
                worst = finals[worst_i]

                def _fmt_run(rf: RunFinal) -> str:
                    bits = [f"final_best={rf.final_best:.6g}"]
                    if rf.best_eval is not None:
                        bits.append(f"best@eval={rf.best_eval}")
                    if rf.instance is not None:
                        bits.append(f"instance={rf.instance}")
                    if rf.seed is not None:
                        bits.append(f"seed={rf.seed}")
                    return ", ".join(bits)

                print(f"- Best run:  {_fmt_run(best)}")
                print(f"- Worst run: {_fmt_run(worst)}")

            # Direction sanity note for common gotcha
            if "nqueens" in key.fname.lower():
                # NQueens in your JSON is maximization=true; flag if someone might be minimizing.
                if key.maximize:
                    print("- Note: This problem is marked `maximization: true` in IOH JSON. Ensure your GA fitness/direction matches this.")

            print("")  # blank line between algos

        print("")  # blank line between datasets


if __name__ == "__main__":
    main()
