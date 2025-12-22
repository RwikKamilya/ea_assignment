"""s4822285_tuning_v2.py

Small, reproducible hyperparameter tuning for the GA.

This script runs a *random search* over a compact parameter space and
selects the configuration that maximizes the **median** final best value
over several seeds (robust to outliers).

It writes the winning configuration to ga_best_params.json so that
`s4822285_GA_v2.py` can load it automatically.

Run:
  python s4822285_tuning_v2.py
Optional:
  python s4822285_tuning_v2.py --fid 18 --dim 50 --tune-budget 20000 --eval-budget 2000 --trials 40
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import List

import numpy as np

import ioh
from ioh import get_problem, ProblemClass

from ea_utils import set_global_seed, summarize_final, format_summary_line
from s4822285_GA_v2 import GAParams, s4822285_GA


def sample_params(rng: np.random.Generator, dim: int) -> GAParams:
    # Keep the search space small + sensible
    pop_size = int(rng.choice([50, 80, 120, 160, 200, 240, 300]))
    tournament_k = int(rng.choice([2, 3, 4, 5]))
    crossover_rate = float(rng.uniform(0.6, 0.98))
    mutation_rate = float(rng.uniform(0.2 / dim, 3.0 / dim))
    elite_frac = float(rng.choice([0.0, 0.01, 0.02, 0.03, 0.05]))
    return GAParams(
        pop_size=pop_size,
        tournament_k=tournament_k,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_frac=elite_frac,
    ).clamp()


def evaluate_params(
        fid: int,
        dim: int,
        eval_budget: int,
        seeds: List[int],
        params: GAParams,
) -> float:
    # Fresh problem each evaluation (no IOH logger during tuning)
    problem = get_problem(fid, dimension=dim, instance=1, problem_class=ProblemClass.PBO)
    finals: List[float] = []
    for s in seeds:
        rs = s4822285_GA(problem, budget=eval_budget, seed=s, params=params)
        finals.append(rs.final_best)
        problem.reset()
    # We optimize the median performance
    return float(np.median(np.asarray(finals, dtype=float)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", type=int, default=18)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--tune-budget", type=int, default=20_000, help="Total tuning evaluations (rough guideline)")
    ap.add_argument("--eval-budget", type=int, default=2_000, help="Budget per trial evaluation run")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seeds-per-trial", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=777_000)
    ap.add_argument("--out", type=str, default="ga_best_params.json")
    args = ap.parse_args()

    fid = int(args.fid)
    dim = int(args.dim)
    trials = int(args.trials)
    seeds_per_trial = int(args.seeds_per_trial)
    eval_budget = int(args.eval_budget)

    set_global_seed(int(args.base_seed))
    rng = np.random.default_rng(int(args.base_seed))

    # Fixed seed pool so tuning is deterministic
    seed_pool = [int(args.base_seed) + i for i in range(10_000)]

    best_score = -float("inf")
    best_params: GAParams | None = None

    print(f"Tuning GA on PBO F{fid} dim={dim}")
    print(f"Trials={trials} | seeds/trial={seeds_per_trial} | eval_budget={eval_budget}\n")

    for t in range(trials):
        params = sample_params(rng, dim=dim)
        seeds = seed_pool[t * seeds_per_trial : (t + 1) * seeds_per_trial]
        score = evaluate_params(fid=fid, dim=dim, eval_budget=eval_budget, seeds=seeds, params=params)

        if score > best_score:
            best_score = score
            best_params = params
            mark = "  <-- best"
        else:
            mark = ""

        print(f"[trial {t:02d}] median_final={score:.6g} | {asdict(params)}{mark}")

    if best_params is None:
        raise RuntimeError("No parameters were evaluated.")

    # Save best
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(asdict(best_params), f, indent=2, sort_keys=True)

    print("\n" + "=" * 60)
    print(f"Best median_final={best_score:.6g}")
    print(f"Saved to: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
