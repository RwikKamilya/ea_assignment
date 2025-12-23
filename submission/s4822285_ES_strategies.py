#!/usr/bin/env python3
"""
s4822285_ES_strategies.py

Purpose (your experiment workflow):
  Run a *suite* of classic ES strategies:
    - (mu, lambda)-ES  (comma selection)
    - (mu+lambda)-ES   (plus selection)
  with different (mu, lambda) pairs, log each config as a separate IOH algorithm,
  and print aggregated stats.

This complements your CMA-ES runner. It is meant for analysis plots + report story:
  - Why (mu+lambda) can be more elitist/stable
  - Why larger lambda improves exploration
  - How mu controls selection pressure / noise reduction via recombination

BBOB Part 2:
  - Function: F23 (Katsuura)
  - Dimension: 10 (fixed by assignment)
  - Budget: 50,000 evaluations (assignment uses 50,0000 in slides; adjust if needed)

Usage:
  # Run a single baseline strategy (submission-like quick check)
  python s4822285_ES_strategies.py

  # Run a whole comparison suite
  python s4822285_ES_strategies.py --compare

  # Custom pairs (runs both comma and plus by default)
  python s4822285_ES_strategies.py --pairs 1:10 3:30 5:50 --both

Outputs (IOH logs):
  data_root/ES_STRATEGIES/ESSTRAT_.../IOHprofiler_*.json + .dat

Then plot comparisons:
  python plot_es_strategy_compare.py --data-root data --outdir plots_es --algo-prefix s4822285_ES_
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import argparse
import math
from pathlib import Path


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

DEFAULT_BUDGET = 50_000
DEFAULT_RUNS = 20
DEFAULT_BASE_SEED = 7_654_321
DEFAULT_DIM = 10
DEFAULT_FID = 23
DEFAULT_BOUNDS = (-5.0, 5.0)


@dataclass(frozen=True)
class ESConfig:
    mu: int
    lam: int
    plus: bool  # True => (mu+lambda), False => (mu,lambda)
    sigma0: float = 2.0
    stagnation_gens: int = 80
    max_restarts: int = 4

    @property
    def strategy_label(self) -> str:
        return f"({self.mu}{'+' if self.plus else ','}{self.lam})"


def _clip(X: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(X, lo, hi)


def s4822285_ES_classic(problem: ioh.problem.BBOB, budget: int, seed: int, cfg: ESConfig) -> RunSummary:
    """A simple (mu,lambda)/(mu+lambda) ES with global sigma adaptation (1/5 success-ish)."""
    t0 = walltime()
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    n = int(problem.meta_data.n_variables)
    lo, hi = DEFAULT_BOUNDS

    mu = int(max(1, cfg.mu))
    lam = int(max(1, cfg.lam))
    sigma = float(cfg.sigma0)

    # parents
    parents = rng.uniform(lo, hi, size=(mu, n))
    f_par = np.empty(mu, dtype=float)
    best_f = float("inf")
    best_eval = 0

    for i in range(mu):
        f_par[i] = float(problem(_clip(parents[i], lo, hi).tolist()))
        ev = int(problem.state.evaluations)
        if best_update(best_f, f_par[i], maximize=False):
            best_f, best_eval = float(f_par[i]), ev
        if ev >= budget:
            break

    no_improve_gens = 0
    restarts = 0

    # learning rate for sigma adaptation
    lr = 0.2
    target_sr = 0.2  # 1/5 success rule target

    while int(problem.state.evaluations) < budget:
        remaining = budget - int(problem.state.evaluations)
        if remaining <= 0:
            break

        lam_gen = int(min(lam, remaining))
        if lam_gen <= 0:
            break

        # recombination: mean of current parents (after sorting by fitness)
        order = np.argsort(f_par)
        parents = parents[order]
        f_par = f_par[order]
        x_mean = np.mean(parents, axis=0)

        best_parent_f = float(f_par[0])

        # sample offspring
        Z = rng.standard_normal((lam_gen, n))
        offspring = x_mean[None, :] + sigma * Z
        offspring = _clip(offspring, lo, hi)

        f_off = np.empty(lam_gen, dtype=float)
        for i in range(lam_gen):
            f_off[i] = float(problem(offspring[i].tolist()))
            ev = int(problem.state.evaluations)
            if best_update(best_f, f_off[i], maximize=False):
                best_f, best_eval = float(f_off[i]), ev

        # success ratio: offspring better than best parent (before selection)
        sr = float(np.mean(f_off < best_parent_f))

        # selection
        if cfg.plus:
            pool_x = np.vstack([parents, offspring])
            pool_f = np.concatenate([f_par, f_off])
        else:
            pool_x = offspring
            pool_f = f_off

        sel = np.argsort(pool_f)[:mu]
        parents = pool_x[sel].copy()
        f_par = pool_f[sel].copy()

        # sigma adaptation
        # increase sigma if sr > target, decrease otherwise
        sigma *= math.exp(lr * (sr - target_sr) / max(1e-9, (1.0 - target_sr)))
        sigma = float(np.clip(sigma, 1e-12, 5.0))

        if best_f < best_parent_f - 1e-12:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if (no_improve_gens >= cfg.stagnation_gens) and (restarts < cfg.max_restarts):
            restarts += 1
            no_improve_gens = 0
            sigma = float(cfg.sigma0)
            parents = rng.uniform(lo, hi, size=(mu, n))
            f_par = np.empty(mu, dtype=float)
            for i in range(mu):
                if int(problem.state.evaluations) >= budget:
                    break
                f_par[i] = float(problem(_clip(parents[i], lo, hi).tolist()))
                ev = int(problem.state.evaluations)
                if best_update(best_f, f_par[i], maximize=False):
                    best_f, best_eval = float(f_par[i]), ev

    seconds = walltime() - t0
    return RunSummary(
        run_index=-1,
        seed=int(seed),
        best=float(best_f),
        best_eval=int(best_eval),
        final_best=float(best_f),
        evals_used=int(problem.state.evaluations),
        seconds=float(seconds),
    )


def _run_config(cfg: ESConfig, budget: int, runs: int, base_seed: int, data_root: Path, instance: int) -> None:
    fid = DEFAULT_FID
    dim = DEFAULT_DIM
    algo_name = f"s4822285_ES_{cfg.strategy_label}"
    folder_name = f"ESSTRAT_F{fid}_d{dim}_mu{cfg.mu}_{'plus' if cfg.plus else 'comma'}_lam{cfg.lam}_sig{cfg.sigma0}"

    problem = get_problem(fid, dimension=dim, instance=instance, problem_class=ProblemClass.BBOB)
    lgr = logger.Analyzer(
        root=str(data_root),
        folder_name=folder_name,
        algorithm_name=algo_name,
        store_positions=False,
    )
    problem.attach_logger(lgr)

    print_run_header(
        f"Classic ES on BBOB F{fid} (Katsuura) d={dim} | cfg={cfg.strategy_label} | runs={runs} | budget={budget}"
    )
    print(f"Config: {cfg}")

    finals: List[float] = []
    for r in range(runs):
        seed = base_seed + r
        rs = s4822285_ES_classic(problem, budget=budget, seed=seed, cfg=cfg)
        rs.run_index = r
        finals.append(rs.final_best)
        print_run_summary(rs, maximize=False)
        problem.reset()

    try:
        lgr.close()
    except Exception:
        pass

    stats = summarize_final(finals)
    print("\n" + format_summary_line("FINAL BEST (lower is better)", stats))
    print("")


def _parse_pairs(tokens: List[str]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for t in tokens:
        if ":" not in t:
            continue
        a, b = t.split(":", 1)
        out.append((int(a), int(b)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--instance", type=int, default=1)

    ap.add_argument("--compare", action="store_true", help="Run the default comparison suite.")
    ap.add_argument("--pairs", nargs="*", default=None, help="List like: 1:10 3:30 5:50")
    ap.add_argument("--sigma0", type=float, default=2.0)

    ap.add_argument("--comma", action="store_true", help="Only run (mu,lambda)" )
    ap.add_argument("--plus", action="store_true", help="Only run (mu+lambda)" )
    ap.add_argument("--both", action="store_true", help="Run both (mu,lambda) and (mu+lambda)" )

    args = ap.parse_args()

    data_root = ensure_dir(Path(args.data_root) / "ES_STRATEGIES")

    # Default suite: a few classic comparisons
    default_pairs = [(1, 10), (1, 30), (3, 30), (5, 50), (10, 50)]
    if args.compare:
        pairs = default_pairs
    elif args.pairs:
        pairs = _parse_pairs(args.pairs)
    else:
        pairs = [(1, 10)]  # single baseline

    # selection of strategies
    if args.both or (not args.comma and not args.plus):
        modes = [False, True]  # comma, plus
    elif args.comma:
        modes = [False]
    else:
        modes = [True]

    cfgs: List[ESConfig] = []
    for mu, lam in pairs:
        for plus in modes:
            cfgs.append(ESConfig(mu=mu, lam=lam, plus=plus, sigma0=float(args.sigma0)))

    # run configs with separated seed streams
    for i, cfg in enumerate(cfgs):
        _run_config(
            cfg=cfg,
            budget=int(args.budget),
            runs=int(args.runs),
            base_seed=int(args.base_seed) + i * 1000,
            data_root=data_root,
            instance=int(args.instance),
        )


if __name__ == "__main__":
    main()
