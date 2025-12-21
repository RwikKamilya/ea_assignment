from __future__ import annotations

"""
CMA-ES for IOH BBOB F23 (Katsuura), dimension 10.

Assignment requirements:
- Implement an ES for BBOB F23, dimension fixed to 10. fileciteturn1file10
- Budget: 50,000 evaluations. fileciteturn1file9
- 20 independent runs, reproducible via fixed seeds. fileciteturn1file9

We implement a clean (μ/μ_w, λ)-CMA-ES with CSA, plus a light IPOP-style restart.
Box constraints for BBOB are typically [-5, 5]^d; we use clipping repair. citeturn0search2
"""

from dataclasses import dataclass
from typing import Tuple, List
import os
import glob
import math
import random

import numpy as np

import ioh
from ioh import get_problem, logger, ProblemClass

budget = 5000
BASE_SEED = 7654321


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    # os.environ.setdefault("PYTHONHASHSEED", str(seed))


@dataclass
class CMAESConfig:
    lam: int = 20
    sigma0: float = 2.0
    bounds: Tuple[float, float] = (-5.0, 5.0)
    # restart
    stagnation_gens: int = 50
    max_restarts: int = 4


def _cmaes_setup(n: int, lam: int):
    mu = lam // 2
    # log weights
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / np.sum(w)
    mueff = (np.sum(w) ** 2) / np.sum(w ** 2)

    # strategy params (Hansen's defaults)
    c_sigma = (mueff + 2) / (n + mueff + 5)
    d_sigma = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    c_mu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    return mu, w, mueff, c_sigma, d_sigma, c_c, c1, c_mu, chi_n


def _eig(C: np.ndarray):
    # symmetric eigendecomposition with jitter for numerical stability
    C = (C + C.T) / 2.0
    vals, vecs = np.linalg.eigh(C)
    vals = np.maximum(vals, 1e-20)
    return vecs, np.sqrt(vals)


def _ask(m: np.ndarray, sigma: float, B: np.ndarray, D: np.ndarray, lam: int) -> Tuple[np.ndarray, np.ndarray]:
    n = m.size
    Z = np.random.randn(lam, n)
    # transform: Y = Z @ (B * D)^T  => equivalently (B @ diag(D) @ z)
    BD = B * D  # scales columns of B
    Y = Z @ BD.T
    X = m + sigma * Y
    return X, Z


def _repair_clip(X: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(X, lo, hi)


def studentnumber1_studentnumber2_ES(problem: ioh.problem.BBOB) -> None:
    n = problem.meta_data.n_variables
    cfg = CMAESConfig()

    lo, hi = cfg.bounds
    lam = int(cfg.lam)
    sigma = float(cfg.sigma0)

    # Initial mean sampled in the (clipped) domain
    m = np.random.uniform(lo, hi, size=n)

    # Initialize CMA-ES state
    mu, w, mueff, c_sigma, d_sigma, c_c, c1, c_mu, chi_n = _cmaes_setup(n, lam)
    C = np.eye(n)
    B = np.eye(n)
    D = np.ones(n)
    invsqrtC = np.eye(n)
    ps = np.zeros(n)
    pc = np.zeros(n)

    best_f = float("inf")
    best_eval = int(problem.state.evaluations)
    no_improve_gens = 0
    restarts = 0

    gen = 0
    # CMA-ES loop; stop by evaluation budget
    while problem.state.evaluations < budget:
        gen += 1
        m_old = m.copy()

        # sample
        X, Z = _ask(m, sigma, B, D, lam)
        Xr = _repair_clip(X, lo, hi)

        # evaluate (minimization)
        remaining = budget - int(problem.state.evaluations)
        if remaining <= 0:
            break

        # if we're close to budget, evaluate only what we can
        lam_eff = min(lam, remaining)
        X = X[:lam_eff]
        Z = Z[:lam_eff]
        Xr = Xr[:lam_eff]

        # evaluate (minimization)
        f = np.array([problem(xx.tolist()) for xx in Xr], dtype=float)

        # sort
        idx = np.argsort(f)
        Xr = Xr[idx]
        X = X[idx]
        Z = Z[idx]
        f = f[idx]

        # IMPORTANT: effective μ and weights must match how many we evaluated
        mu_eff = min(mu, lam_eff)
        w_eff = w[:mu_eff].copy()
        w_eff /= np.sum(w_eff)

        # recombination (mean)
        x_mu = Xr[:mu_eff]
        m = np.sum(x_mu * w_eff[:, None], axis=0)

        # y = (m - m_old) / sigma in coordinate space of current C
        y = (m - m_old) / max(1e-30, sigma)

        # update invsqrtC from eigendecomposition
        invsqrtC = B @ np.diag(1 / D) @ B.T

        # update ps
        ps = (1 - c_sigma) * ps + math.sqrt(c_sigma * (2 - c_sigma) * mueff) * (invsqrtC @ y)

        # hsig
        norm_ps = np.linalg.norm(ps)
        hsig = norm_ps / math.sqrt(1 - (1 - c_sigma) ** (2 * gen)) < (1.4 + 2 / (n + 1)) * chi_n

        # update pc
        pc = (1 - c_c) * pc + (hsig * math.sqrt(c_c * (2 - c_c) * mueff)) * y

        # rank-mu update term: sum w_i * y_i y_i^T
        # Use y_i in the *unrepaired* steps to retain adaptation signal:
        y_k = (X[:mu_eff] - m_old) / max(1e-30, sigma)
        rank_mu = np.zeros((n, n))
        for i in range(mu_eff):
            yi = y_k[i][:, None]
            rank_mu += w_eff[i] * (yi @ yi.T)

        # update C
        C = (1 - c1 - c_mu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * rank_mu

        # update sigma (CSA)
        sigma *= math.exp((c_sigma / d_sigma) * (norm_ps / chi_n - 1))

        # occasional eigen update
        if gen % (n // 2 + 1) == 0:
            B, D = _eig(C)

        # restart conditions (stagnation or numerical issues)
        cond = float(np.max(D) / max(1e-30, np.min(D)))
        if (no_improve_gens >= cfg.stagnation_gens) or (sigma < 1e-12) or (cond > 1e14):
            restarts += 1
            if restarts > cfg.max_restarts:
                break
            # IPOP: increase population size, reset sigma and state
            lam = int(lam * 2)
            mu, w, mueff, c_sigma, d_sigma, c_c, c1, c_mu, chi_n = _cmaes_setup(n, lam)
            sigma = float(cfg.sigma0)
            m = np.random.uniform(lo, hi, size=n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            ps = np.zeros(n)
            pc = np.zeros(n)
            gen = 0
            no_improve_gens = 0


# ----------------------------
# IOH plumbing (keep as-is)
# ----------------------------

def create_problem(fid: int):
    # Katsuura is BBOB F23, dimension 10 (as required) fileciteturn1file10
    problem = get_problem(fid, dimension=10, instance=1, problem_class=ProblemClass.BBOB)
    l = logger.Analyzer(
        root="data",
        folder_name=f"ES_F{fid}",
        algorithm_name="studentnumber1_studentnumber2_ES",
        store_positions=False,
    )
    problem.attach_logger(l)
    return problem, l


# ----------------------------
# Plot helpers for your report
# ----------------------------

def _read_ioh_dat(path: str) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                rows.append((int(float(parts[0])), float(parts[2])))
    return np.array(rows, dtype=float)  # eval, best_y


def load_runs_best_so_far(folder: str) -> List[np.ndarray]:
    dats = glob.glob(os.path.join(folder, "**", "*.dat"), recursive=True)
    runs = []
    for p in dats:
        arr = _read_ioh_dat(p)
        if arr.size:
            runs.append(arr)
    return runs


def aggregate_convergence(runs: List[np.ndarray], max_eval: int):
    grid = np.arange(1, max_eval + 1)
    Ys = np.empty((len(runs), grid.size), dtype=float)

    for i, r in enumerate(runs):
        y = np.full(grid.size, np.nan, dtype=float)
        e = r[:, 0].astype(int)
        b = r[:, 1]
        last = np.nan
        j = 0
        for t, ev in enumerate(grid):
            while j < e.size and e[j] <= ev:
                last = b[j]
                j += 1
            y[t] = last
        Ys[i] = y

    med = np.nanmedian(Ys, axis=0)
    q25 = np.nanpercentile(Ys, 25, axis=0)
    q75 = np.nanpercentile(Ys, 75, axis=0)
    return grid, med, (q25, q75)


if __name__ == "__main__":
    # 20 independent runs, fixed seeds for reproducibility fileciteturn1file9
    F23, log = create_problem(23)
    for run in range(20):
        seed_everything(BASE_SEED + run)
        studentnumber1_studentnumber2_ES(F23)
        F23.reset()
    log.close()

    # Optional local convergence plot
    try:
        import matplotlib.pyplot as plt
        runs = load_runs_best_so_far("data/ES_F23")
        xs, med, (q25, q75) = aggregate_convergence(runs, budget)
        plt.figure()
        plt.plot(xs, med)
        plt.fill_between(xs, q25, q75, alpha=0.2)
        plt.xlabel("Function evaluations")
        plt.ylabel("Best-so-far f(x)")
        plt.title("ES convergence (median/IQR) - BBOB F23 (Katsuura), d=10")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass
