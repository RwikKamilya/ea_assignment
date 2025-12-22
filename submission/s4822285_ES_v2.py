"""s4822285_ES_v2.py

CMA-ES for IOH BBOB (F23: Katsuura), dimension 10.

What this script provides (submission-ready):
  - (μ/μ_w, λ)-CMA-ES with CSA step-size adaptation
  - Safe handling of the last partial generation near the evaluation budget
  - Deterministic reproducibility via fixed per-run seeds
  - Clear stdout report per run + aggregated over 20 runs
  - IOH Analyzer logging to data/ES_F23/...

Run:
  python s4822285_ES_v2.py

Optional:
  python s4822285_ES_v2.py --budget 50000 --runs 20 --base-seed 7654321
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional
import argparse
import math
import os
import random
import time

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

DEFAULT_BUDGET = 50_000
DEFAULT_RUNS = 20
DEFAULT_BASE_SEED = 7_654_321


@dataclass
class CMAESConfig:
    lam: int = 20
    sigma0: float = 2.0
    bounds: Tuple[float, float] = (-5.0, 5.0)
    stagnation_gens: int = 60
    max_restarts: int = 4


def _cmaes_setup(n: int, lam: int):
    mu = max(1, lam // 2)
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
    w = w / np.sum(w)
    mueff = (np.sum(w) ** 2) / np.sum(w**2)

    c_sigma = (mueff + 2.0) / (n + mueff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    c_mu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))

    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
    return mu, w, mueff, c_sigma, d_sigma, c_c, c1, c_mu, chi_n


def _eig(C: np.ndarray):
    C = (C + C.T) / 2.0
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-20)
    return vecs, np.sqrt(vals)


def _ask(rng: np.random.Generator, m: np.ndarray, sigma: float, B: np.ndarray, D: np.ndarray, lam: int):
    n = m.size
    Z = rng.standard_normal((lam, n))
    BD = B * D
    Y = Z @ BD.T
    X = m + sigma * Y
    return X, Z


def _repair_clip(X: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(X, lo, hi)


def s4822285_ES(problem: ioh.problem.BBOB, budget: int, seed: int) -> RunSummary:
    """
    Run one CMA-ES run up to `budget` evaluations on the given IOH problem.

    Returns a RunSummary with best and final best values.
    """
    t0 = walltime()
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    n = problem.meta_data.n_variables
    cfg = CMAESConfig()

    lo, hi = cfg.bounds
    lam = int(cfg.lam)
    sigma = float(cfg.sigma0)

    # initial mean
    m = rng.uniform(lo, hi, size=n)

    mu, w, mueff, c_sigma, d_sigma, c_c, c1, c_mu, chi_n = _cmaes_setup(n, lam)
    C = np.eye(n)
    B = np.eye(n)
    D = np.ones(n)
    ps = np.zeros(n)
    pc = np.zeros(n)

    best_f = float("inf")
    best_eval = 0
    no_improve_gens = 0
    restarts = 0
    gen = 0

    # We stop only by budget
    while problem.state.evaluations < budget:
        remaining = budget - int(problem.state.evaluations)
        if remaining <= 0:
            break

        # If the remaining budget is extremely small, do not attempt CMA updates
        lam_gen = int(min(lam, remaining))
        if lam_gen < 2:
            break

        gen += 1
        m_old = m.copy()

        # sample candidates
        X, Z = _ask(rng, m, sigma, B, D, lam_gen)
        Xr = _repair_clip(X, lo, hi)

        # evaluate
        f = np.array([problem(xx.tolist()) for xx in Xr], dtype=float)

        # sort
        idx = np.argsort(f)
        Xr = Xr[idx]
        X = X[idx]
        Z = Z[idx]
        f = f[idx]

        # update best tracking
        cur_best = float(f[0])
        if best_update(best_f, cur_best, maximize=False):
            best_f = cur_best
            best_eval = int(problem.state.evaluations)
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        # recombination with safe mu for partial generation
        mu_eff = int(min(mu, lam_gen))
        if mu_eff < 1:
            break

        w_eff = w[:mu_eff].copy()
        w_eff /= np.sum(w_eff)
        mueff_eff = (np.sum(w_eff) ** 2) / np.sum(w_eff**2)

        x_mu = Xr[:mu_eff]
        m = np.sum(x_mu * w_eff[:, None], axis=0)

        # evolution path updates
        y = (m - m_old) / max(1e-30, sigma)

        invsqrtC = B @ np.diag(1.0 / np.maximum(1e-30, D)) @ B.T
        ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mueff_eff) * (invsqrtC @ y)

        norm_ps = float(np.linalg.norm(ps))
        hsig = norm_ps / math.sqrt(1 - (1 - c_sigma) ** (2 * gen)) < (1.4 + 2 / (n + 1)) * chi_n

        pc = (1 - c_c) * pc + (hsig * math.sqrt(c_c * (2 - c_c) * mueff_eff)) * y

        # rank-mu update
        y_k = (X[:mu_eff] - m_old) / max(1e-30, sigma)
        rank_mu = np.zeros((n, n))
        for i in range(mu_eff):
            yi = y_k[i][:, None]
            rank_mu += w_eff[i] * (yi @ yi.T)

        C = (1 - c1 - c_mu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * rank_mu

        # CSA sigma update
        sigma *= math.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1.0))

        # periodic eigendecomposition
        if gen % (n // 2 + 1) == 0:
            B, D = _eig(C)

        # restart conditions
        cond = float(np.max(D) / max(1e-30, np.min(D)))
        if (no_improve_gens >= cfg.stagnation_gens) or (sigma < 1e-12) or (cond > 1e14):
            restarts += 1
            if restarts > cfg.max_restarts:
                break

            # IPOP-style: increase population, reset state
            lam = int(lam * 2)
            mu, w, mueff, c_sigma, d_sigma, c_c, c1, c_mu, chi_n = _cmaes_setup(n, lam)
            sigma = float(cfg.sigma0)
            m = rng.uniform(lo, hi, size=n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            gen = 0
            no_improve_gens = 0

    final_best = float(best_f)
    evals_used = int(problem.state.evaluations)
    seconds = walltime() - t0
    return RunSummary(
        run_index=-1,  # filled by caller
        seed=seed,
        best=float(best_f),
        best_eval=int(best_eval),
        final_best=float(final_best),
        evals_used=int(evals_used),
        seconds=float(seconds),
    )


def create_problem(fid: int, dimension: int, instance: int, data_root: str, folder_name: str):
    problem = get_problem(fid, dimension=dimension, instance=instance, problem_class=ProblemClass.BBOB)
    l = logger.Analyzer(
        root=data_root,
        folder_name=folder_name,
        algorithm_name="s4822285_ES",
        store_positions=False,
    )
    problem.attach_logger(l)
    return problem, l


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--instance", type=int, default=1)
    args = ap.parse_args()

    budget = int(args.budget)
    runs = int(args.runs)
    base_seed = int(args.base_seed)

    print_run_header(f"ES (CMA-ES) on BBOB F23 (Katsuura), d=10 | runs={runs} | budget={budget}")

    # F23 Katsuura, dim=10 (assignment requirement)
    problem, log = create_problem(
        fid=23,
        dimension=10,
        instance=int(args.instance),
        data_root=str(args.data_root),
        folder_name="ES_F23",
    )

    run_summaries: List[RunSummary] = []
    finals: List[float] = []

    for r in range(runs):
        seed = base_seed + r
        # Run
        rs = s4822285_ES(problem, budget=budget, seed=seed)
        rs.run_index = r
        run_summaries.append(rs)
        finals.append(rs.final_best)

        print_run_summary(rs, maximize=False)

        # reset IOH problem for next run
        problem.reset()

    log.close()

    stats = summarize_final(finals)
    print("\n" + format_summary_line("FINAL BEST (lower is better)", stats))


if __name__ == "__main__":
    main()
