"""s4822285_GA_v2.py

Genetic Algorithm for IOH PBO:
  - F18: LABS (dimension 50)
  - F23: NQueens (dimension 64)

Submission-ready properties:
  - Deterministic reproducibility via fixed per-run seeds
  - stdout report per run + aggregated over 20 runs
  - Parameter loading from ga_best_params.json (optional)
  - IOH Analyzer logging to data/GA_F*/...

Run:
  python s4822285_GA_v2.py

Optional:
  python s4822285_GA_v2.py --budget 5000 --runs 20 --base-seed 123456 --params ga_best_params.json
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import argparse
import json
import math
import os

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
)

DEFAULT_BUDGET = 5_000
DEFAULT_RUNS = 20
DEFAULT_BASE_SEED = 1_234_567


@dataclass
class GAParams:
    pop_size: int = 200
    tournament_k: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 1.0 / 50.0
    elite_frac: float = 0.02

    def clamp(self) -> "GAParams":
        self.pop_size = int(max(10, self.pop_size))
        self.tournament_k = int(max(2, self.tournament_k))
        self.crossover_rate = float(min(1.0, max(0.0, self.crossover_rate)))
        self.mutation_rate = float(min(1.0, max(0.0, self.mutation_rate)))
        self.elite_frac = float(min(0.2, max(0.0, self.elite_frac)))
        return self


def _load_params(path: str) -> Optional[GAParams]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return GAParams(**obj).clamp()
    except Exception:
        return None


def _tournament_select(rng: np.random.Generator, fitness: np.ndarray, k: int, maximize: bool) -> int:
    n = fitness.size
    cand = rng.integers(0, n, size=k)
    if maximize:
        return int(cand[np.argmax(fitness[cand])])
    return int(cand[np.argmin(fitness[cand])])


def _one_point_crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = a.size
    if n <= 1:
        return a.copy(), b.copy()
    p = int(rng.integers(1, n))
    c1 = np.concatenate([a[:p], b[p:]])
    c2 = np.concatenate([b[:p], a[p:]])
    return c1, c2


def _bitflip_mutation(rng: np.random.Generator, x: np.ndarray, p: float) -> np.ndarray:
    if p <= 0:
        return x
    mask = rng.random(x.size) < p
    x[mask] = 1 - x[mask]
    return x


def _init_population(rng: np.random.Generator, pop_size: int, n_bits: int) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, n_bits), dtype=np.int8)


def s4822285_GA(problem: ioh.problem.PBO, budget: int, seed: int, params: GAParams) -> RunSummary:
    """
    Run one GA run on the given PBO problem.
    """
    t0 = walltime()
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    maximize = bool(problem.meta_data.optimization_type.name.lower() == "maximization")
    n = int(problem.meta_data.n_variables)

    pop_size = int(params.pop_size)
    elite_n = int(max(0, round(params.elite_frac * pop_size)))

    # init
    pop = _init_population(rng, pop_size, n)

    # evaluate initial population
    fitness = np.empty(pop_size, dtype=float)
    best = -float("inf") if maximize else float("inf")
    best_eval = 0

    evals_used = 0

    for i in range(pop_size):
        if problem.state.evaluations >= budget:
            break
        y = float(problem(pop[i].tolist()))
        fitness[i] = y
        evals_used = int(problem.state.evaluations)
        if best_update(best, y, maximize=maximize):
            best = y
            best_eval = evals_used

    # main loop
    while problem.state.evaluations < budget:
        remaining = budget - int(problem.state.evaluations)
        if remaining <= 0:
            break

        # elitism (copy indices)
        if elite_n > 0:
            elite_idx = np.argsort(fitness)
            elite_idx = elite_idx[::-1] if maximize else elite_idx
            elites = pop[elite_idx[:elite_n]].copy()
        else:
            elites = np.empty((0, n), dtype=np.int8)

        # generate offspring
        offspring: List[np.ndarray] = []
        while len(offspring) < pop_size - elite_n and problem.state.evaluations < budget:
            i1 = _tournament_select(rng, fitness, params.tournament_k, maximize)
            i2 = _tournament_select(rng, fitness, params.tournament_k, maximize)
            p1 = pop[i1]
            p2 = pop[i2]

            if rng.random() < params.crossover_rate:
                c1, c2 = _one_point_crossover(rng, p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = _bitflip_mutation(rng, c1, params.mutation_rate)
            offspring.append(c1)

            if len(offspring) < pop_size - elite_n:
                c2 = _bitflip_mutation(rng, c2, params.mutation_rate)
                offspring.append(c2)

        new_pop = np.vstack([elites, np.asarray(offspring, dtype=np.int8)])
        new_pop = new_pop[:pop_size]

        # evaluate new population (full generational replacement)
        pop = new_pop
        fitness = np.empty(pop_size, dtype=float)
        for i in range(pop_size):
            if problem.state.evaluations >= budget:
                fitness = fitness[:i]
                pop = pop[:i]
                break
            y = float(problem(pop[i].tolist()))
            fitness[i] = y
            evals_used = int(problem.state.evaluations)
            if best_update(best, y, maximize=maximize):
                best = y
                best_eval = evals_used

        if pop.shape[0] < 2:
            break

        # if population got truncated due to budget, stop
        if int(problem.state.evaluations) >= budget:
            break

    # final
    final_best = float(best)
    seconds = walltime() - t0
    return RunSummary(
        run_index=-1,
        seed=seed,
        best=float(best),
        best_eval=int(best_eval),
        final_best=float(final_best),
        evals_used=int(problem.state.evaluations),
        seconds=float(seconds),
    )


def _attach_logger(problem, data_root: str, folder_name: str):
    l = logger.Analyzer(
        root=data_root,
        folder_name=folder_name,
        algorithm_name="s4822285_GA",
        store_positions=False,
    )
    problem.attach_logger(l)
    return l


def _run_suite(fid: int, dim: int, budget: int, runs: int, base_seed: int, data_root: str, params: GAParams) -> None:
    problem = get_problem(fid, dimension=dim, instance=1, problem_class=ProblemClass.PBO)
    log = _attach_logger(problem, data_root=data_root, folder_name=f"GA_F{fid}")

    maximize = bool(problem.meta_data.optimization_type.name.lower() == "maximization")

    print_run_header(f"GA on PBO F{fid} (dim={dim}) | runs={runs} | budget={budget} | maximize={maximize}")
    print(f"Params: {asdict(params)}\n")

    finals: List[float] = []
    run_summaries: List[RunSummary] = []

    for r in range(runs):
        seed = base_seed + r
        rs = s4822285_GA(problem, budget=budget, seed=seed, params=params)
        rs.run_index = r
        run_summaries.append(rs)
        finals.append(rs.final_best)

        print_run_summary(rs, maximize=maximize)
        problem.reset()

    log.close()

    label = "FINAL BEST (higher is better)" if maximize else "FINAL BEST (lower is better)"
    stats = summarize_final(finals)
    print("\n" + format_summary_line(label, stats))
    print()


def _is_maximization(problem: ioh.ProblemType) -> bool:
    """
    Robustly infer whether the IOH problem is a maximization problem.
    IOH JSON uses 'maximization: true/false' and the Python API exposes similar metadata.
    """
    # Try common IOH python metadata fields
    md = getattr(problem, "meta_data", None)
    if md is not None:
        # Some versions expose md.maximization directly
        if hasattr(md, "maximization"):
            try:
                return bool(getattr(md, "maximization"))
            except Exception:
                pass

        # Some expose an enum-like optimization_type
        if hasattr(md, "optimization_type"):
            try:
                s = str(getattr(md, "optimization_type")).lower()
                if "max" in s:
                    return True
                if "min" in s:
                    return False
            except Exception:
                pass

        # Fallback: sometimes 'optimization' exists as string/dict
        if hasattr(md, "optimization"):
            try:
                s = str(getattr(md, "optimization")).lower()
                if "max" in s:
                    return True
                if "min" in s:
                    return False
            except Exception:
                pass

    # Default: assume maximization for PBO problems, but this is a safe fallback
    return True




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--params", type=str, default="ga_best_params.json")
    args = ap.parse_args()

    params = _load_params(args.params) or GAParams()
    params = params.clamp()

    # F18: LABS dim=50
    _run_suite(
        fid=18,
        dim=50,
        budget=int(args.budget),
        runs=int(args.runs),
        base_seed=int(args.base_seed),
        data_root=str(args.data_root),
        params=params,
    )

    # F23: NQueens dim=64 (perfect square)
    _run_suite(
        fid=23,
        dim=64,
        budget=int(args.budget),
        runs=int(args.runs),
        base_seed=int(args.base_seed) + 10_000,  # separate seed stream
        data_root=str(args.data_root),
        params=params,
    )


if __name__ == "__main__":
    main()
