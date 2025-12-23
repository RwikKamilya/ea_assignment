#!/usr/bin/env python3
"""
s4822285_GA.py

GA runner for Part 1 (PBO):
  - F18 (LABS)
  - F23 (N-Queens)

Key goals for *your workflow*:
  1) Run tuning -> write ga_best_params.json
  2) Run GA on a *suite of dimensions* (trend experiments)
  3) Keep IOH logs separated:
       data_root/GA_LABS/...   (F18)
       data_root/GA_NQUEENS/... (F23)

Submission-friendly defaults:
  - F18 dimension=50
  - F23 dimension=64
  - budget=5000, runs=20

Examples:
  # 1) tuning
  python s4822285_tuning.py

  # 2) submission-like GA run
  python s4822285_GA.py

  # 3) multi-dimension trend experiment
  python s4822285_GA.py --multi-dim

  # 4) custom dimensions
  python s4822285_GA.py --dims-f18 20,50,100,200 --dims-f23 16,25,36,49,64,81,100,121,144
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List
import argparse
import json
import math

import numpy as np

import ioh
from ioh import get_problem, logger, ProblemClass

from ea_utils import (
    RunSummary,
    best_update,
    format_summary_line,
    print_run_header,
    print_run_summary,
    set_global_seed,
    summarize_final,
    walltime,
    ensure_dir,
)

DEFAULT_BUDGET = 5_000
DEFAULT_RUNS = 20
DEFAULT_BASE_SEED = 1_234_567

# Trend dimension lists you requested
DEFAULT_DIMS_F18 = [20, 50, 100, 200]
DEFAULT_DIMS_F23 = [16, 25, 36, 49, 64, 81, 100, 121, 144]


@dataclass
class GAParams:
    pop_size: int = 200
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 1.0 / 50.0  # bitstrings: per-bit flip probability
    elite_frac: float = 0.02

    def clamp(self) -> "GAParams":
        self.pop_size = int(max(10, min(400, self.pop_size)))
        self.tournament_k = int(max(2, min(8, self.tournament_k)))
        self.crossover_rate = float(min(1.0, max(0.0, self.crossover_rate)))
        self.mutation_rate = float(max(0.0, min(0.5, self.mutation_rate)))
        self.elite_frac = float(min(0.2, max(0.0, self.elite_frac)))
        return self


def _is_maximization(problem: ioh.Problem) -> bool:
    md = getattr(problem, "meta_data", None)
    if md is None:
        return True
    for attr in ("optimization_type", "optimization", "is_minimization", "is_maximization"):
        if hasattr(md, attr):
            val = getattr(md, attr)
            if isinstance(val, bool):
                if attr == "is_minimization":
                    return not val
                return val
            s = str(val).lower()
            if "min" in s:
                return False
            if "max" in s:
                return True
    return True


def _looks_like_nqueens(problem: ioh.Problem) -> Tuple[bool, int]:
    # N-Queens in PBO has n*n variables (bitboard). We'll use permutation encoding internally.
    n_bits = int(problem.meta_data.n_variables)
    n = int(round(math.sqrt(n_bits)))
    if n * n != n_bits:
        return False, n
    name = str(getattr(problem.meta_data, "name", "")).lower()
    if "queen" in name:
        return True, n
    if int(getattr(problem.meta_data, "problem_id", getattr(problem.meta_data, "problem_index", -1))) == 23:
        return True, n
    return False, n


# ---------- operators (bitstrings) ----------

def _uniform_crossover_bits(rng: np.random.Generator, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = rng.random(a.size) < 0.5
    c = a.copy()
    c[mask] = b[mask]
    return c


def _mutate_bits(rng: np.random.Generator, x: np.ndarray, p_flip: float) -> np.ndarray:
    m = rng.random(x.size) < p_flip
    if np.any(m):
        x = x.copy()
        x[m] = 1 - x[m]
    return x


# ---------- operators (permutations) ----------

def _ox1_crossover(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    n = p1.size
    i, j = sorted(rng.integers(0, n, size=2).tolist())
    if i == j:
        return p1.copy()
    child = np.full(n, -1, dtype=int)
    child[i:j] = p1[i:j]
    fill = [g for g in p2.tolist() if g not in set(child[i:j].tolist())]
    k = 0
    for idx in list(range(0, i)) + list(range(j, n)):
        child[idx] = fill[k]
        k += 1
    return child


def _mutate_perm_swap(rng: np.random.Generator, perm: np.ndarray, mutation_rate: float) -> np.ndarray:
    n = perm.size
    # convert a "per-bit" style rate to a swap probability
    p_swap = min(1.0, mutation_rate * n)
    if rng.random() >= p_swap:
        return perm
    i, j = rng.integers(0, n, size=2)
    if i == j:
        return perm
    y = perm.copy()
    y[i], y[j] = y[j], y[i]
    return y


def _decode_perm_to_board(perm: np.ndarray, n: int) -> np.ndarray:
    board = np.zeros(n * n, dtype=int)
    for col in range(n):
        row = int(perm[col])
        board[row * n + col] = 1
    return board


# ---------- selection ----------

def _tournament_select(rng: np.random.Generator, fitness: np.ndarray, k: int, maximize: bool) -> int:
    idx = rng.integers(0, fitness.size, size=k)
    best = int(idx[0])
    for j in idx[1:]:
        j = int(j)
        if best_update(fitness[best], fitness[j], maximize=maximize):
            best = j
    return int(best)


def _load_best_params(path: Path) -> Optional[GAParams]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return GAParams(**obj).clamp()
    except Exception:
        return None


# ---------- core GA (called by tuning) ----------

def s4822285_GA(problem: ioh.Problem, budget: int, seed: int, params: GAParams) -> RunSummary:
    """One GA run on a single IOH problem instance."""
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    maximize = _is_maximization(problem)
    use_perm, n = _looks_like_nqueens(problem)

    t0 = walltime()
    best_val = -float("inf") if maximize else float("inf")
    best_eval = 0

    if use_perm:
        # permutation encoding (size n), decode to bitboard for evaluation
        pop = np.array([rng.permutation(n) for _ in range(params.pop_size)], dtype=int)
        fitness = np.empty(pop.shape[0], dtype=float)

        for i in range(pop.shape[0]):
            y = float(problem(_decode_perm_to_board(pop[i], n)))
            fitness[i] = y
            ev = int(problem.state.evaluations)
            if best_update(best_val, y, maximize=maximize):
                best_val, best_eval = y, ev
            if ev >= budget:
                pop, fitness = pop[: i + 1], fitness[: i + 1]
                break

        while int(problem.state.evaluations) < budget:
            elite_n = int(round(params.elite_frac * pop.shape[0]))
            elite_n = int(max(0, min(pop.shape[0], elite_n)))

            if elite_n > 0:
                order = np.argsort(fitness)[::-1] if maximize else np.argsort(fitness)
                elite_idx = order[:elite_n]
                elites = pop[elite_idx].copy()
                elite_fit = fitness[elite_idx].copy()
            else:
                elites = np.empty((0, n), dtype=int)
                elite_fit = np.empty((0,), dtype=float)

            offspring, off_fit = [], []
            while len(offspring) + elite_n < params.pop_size and int(problem.state.evaluations) < budget:
                p1 = pop[_tournament_select(rng, fitness, params.tournament_k, maximize)]
                p2 = pop[_tournament_select(rng, fitness, params.tournament_k, maximize)]
                c = _ox1_crossover(rng, p1, p2) if rng.random() < params.crossover_rate else p1.copy()
                c = _mutate_perm_swap(rng, c, params.mutation_rate)

                y = float(problem(_decode_perm_to_board(c, n)))
                ev = int(problem.state.evaluations)
                if best_update(best_val, y, maximize=maximize):
                    best_val, best_eval = y, ev

                offspring.append(c)
                off_fit.append(y)

            if not offspring:
                break

            child_pop = np.asarray(offspring, dtype=int)
            child_fit = np.asarray(off_fit, dtype=float)
            pop = np.vstack([elites, child_pop]) if elite_n > 0 else child_pop
            fitness = np.concatenate([elite_fit, child_fit]) if elite_n > 0 else child_fit

    else:
        # bitstring encoding
        n_bits = int(problem.meta_data.n_variables)
        pop = rng.integers(0, 2, size=(params.pop_size, n_bits), dtype=int)
        fitness = np.empty(pop.shape[0], dtype=float)

        for i in range(pop.shape[0]):
            y = float(problem(pop[i]))
            fitness[i] = y
            ev = int(problem.state.evaluations)
            if best_update(best_val, y, maximize=maximize):
                best_val, best_eval = y, ev
            if ev >= budget:
                pop, fitness = pop[: i + 1], fitness[: i + 1]
                break

        while int(problem.state.evaluations) < budget:
            elite_n = int(round(params.elite_frac * pop.shape[0]))
            elite_n = int(max(0, min(pop.shape[0], elite_n)))

            if elite_n > 0:
                order = np.argsort(fitness)[::-1] if maximize else np.argsort(fitness)
                elites = pop[order[:elite_n]].copy()
                elite_fit = fitness[order[:elite_n]].copy()
            else:
                elites = np.empty((0, pop.shape[1]), dtype=int)
                elite_fit = np.empty((0,), dtype=float)

            offspring, off_fit = [], []
            while len(offspring) + elite_n < params.pop_size and int(problem.state.evaluations) < budget:
                i1 = _tournament_select(rng, fitness, params.tournament_k, maximize)
                i2 = _tournament_select(rng, fitness, params.tournament_k, maximize)
                p1, p2 = pop[i1], pop[i2]

                c = _uniform_crossover_bits(rng, p1, p2) if rng.random() < params.crossover_rate else p1.copy()
                c = _mutate_bits(rng, c, params.mutation_rate)

                y = float(problem(c))
                ev = int(problem.state.evaluations)
                if best_update(best_val, y, maximize=maximize):
                    best_val, best_eval = y, ev

                offspring.append(c)
                off_fit.append(y)

            if not offspring:
                break

            pop = np.vstack([elites, np.asarray(offspring, dtype=int)]) if elite_n > 0 else np.asarray(offspring, dtype=int)
            fitness = np.concatenate([elite_fit, np.asarray(off_fit, dtype=float)]) if elite_n > 0 else np.asarray(off_fit, dtype=float)

    t1 = walltime()
    return RunSummary(
        run_index=-1,
        seed=seed,
        best=float(best_val),
        best_eval=int(best_eval),
        final_best=float(best_val),
        evals_used=int(problem.state.evaluations),
        seconds=float(t1 - t0),
    )


# ---------- experiment runner ----------

def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _problem_root(base: Path, fid: int, split_outputs: bool) -> Path:
    if not split_outputs:
        return base
    if fid == 18:
        return base / "GA_LABS"
    if fid == 23:
        return base / "GA_NQUEENS"
    return base / f"GA_F{fid}"


def _run_suite(fid: int, dim: int, budget: int, runs: int, base_seed: int, data_root: Path, params: GAParams) -> None:
    problem = get_problem(fid, dimension=dim, instance=1, problem_class=ProblemClass.PBO)
    maximize = _is_maximization(problem)

    algo_name = "s4822285_GA"
    folder_name = f"GA_F{fid}_d{dim}_data_f{fid}_{problem.meta_data.name}"

    lgr = logger.Analyzer(
        root=str(data_root),
        folder_name=folder_name,
        algorithm_name=algo_name,
        algorithm_info="algorithm_info",
    )
    problem.attach_logger(lgr)

    print_run_header(f"GA on PBO F{fid} (dim={dim}) | runs={runs} | budget={budget} | maximize={maximize}")
    print("Params:", asdict(params))
    finals: List[float] = []

    for r in range(runs):
        seed = base_seed + r
        problem.reset()
        rs = s4822285_GA(problem, budget=budget, seed=seed, params=params)
        rs.run_index = r
        print_run_summary(rs, maximize=maximize)
        finals.append(rs.final_best)

    stats = summarize_final(finals)
    label = f"F{fid} d={dim} final_best"
    print("\n" + format_summary_line(label, stats))

    problem.detach_logger()
    try:
        lgr.close()
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)

    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--params", type=str, default="ga_best_params.json")
    ap.add_argument("--split-outputs", action="store_true", default=True,
                    help="Write F18 logs under GA_LABS/ and F23 logs under GA_NQUEENS/ (default: on).")

    ap.add_argument("--multi-dim", action="store_true", help="Run the default multi-dimension lists for F18 and F23.")
    ap.add_argument("--dims-f18", type=str, default="", help="Comma-separated dims for F18 (e.g., 20,50,100,200). ")
    ap.add_argument("--dims-f23", type=str, default="", help="Comma-separated dims for F23 (perfect squares recommended).")

    args = ap.parse_args()

    params = _load_best_params(Path(args.params)) or GAParams().clamp()

    base = ensure_dir(args.data_root)
    # If user passes --split-outputs False is not possible via argparse boolean; keep flag for clarity.
    split_outputs = bool(args.split_outputs)

    if args.multi_dim:
        dims_f18 = _parse_int_list(args.dims_f18) if args.dims_f18 else list(DEFAULT_DIMS_F18)
        dims_f23 = _parse_int_list(args.dims_f23) if args.dims_f23 else list(DEFAULT_DIMS_F23)
    else:
        dims_f18 = _parse_int_list(args.dims_f18) if args.dims_f18 else [50]
        dims_f23 = _parse_int_list(args.dims_f23) if args.dims_f23 else [64]

    # F18 (LABS)
    for i, d in enumerate(dims_f18):
        outdir = ensure_dir(_problem_root(base, fid=18, split_outputs=split_outputs))
        _run_suite(
            fid=18,
            dim=int(d),
            budget=int(args.budget),
            runs=int(args.runs),
            base_seed=int(args.base_seed) + i * 1000,
            data_root=outdir,
            params=params,
        )

    # F23 (N-Queens): separate seed stream from F18
    for i, d in enumerate(dims_f23):
        outdir = ensure_dir(_problem_root(base, fid=23, split_outputs=split_outputs))
        _run_suite(
            fid=23,
            dim=int(d),
            budget=int(args.budget),
            runs=int(args.runs),
            base_seed=int(args.base_seed) + 10_000 + i * 1000,
            data_root=outdir,
            params=params,
        )


if __name__ == "__main__":
    main()
