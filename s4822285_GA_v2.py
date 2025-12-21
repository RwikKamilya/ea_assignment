from __future__ import annotations

"""
Genetic Algorithm for IOH PBO problems F18 (LABS) and F23 (N-Queens).

Key design goals for this practical assignment:
- Strong performance under a tight evaluation budget (5000 evals). fileciteturn1file4
- One single hyperparameter setting that works for BOTH problems. fileciteturn1file5
- Reproducibility: deterministic given fixed seeds. fileciteturn1file4

Notes:
- PBO problems are maximization in IOH (higher is better), so we use argmax selections.
- The tuning script can set GA.TUNED_PARAM and temporarily override GA.budget.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Dict, Any
import os
import glob
import json
import random

import numpy as np

import ioh
from ioh import get_problem, logger, ProblemClass

# Default evaluation budget for the GA (tuning may temporarily override this).
budget = 5000

# If set (by tuning), GA will read hyperparams from here.
TUNED_PARAM = None  # type: ignore[assignment]

# Optional: tuning may write a JSON file; GA will load it if present and TUNED_PARAM is None.
PARAM_JSON = os.path.join(os.path.dirname(__file__), "ga_best_params.json")

BASE_SEED = 1234567  # constant base; external scripts can override via np.random.seed


# ----------------------------
# Reproducibility helpers
# ----------------------------

def seed_everything(seed: int) -> None:
    """Seed numpy + python random for fully reproducible runs."""
    np.random.seed(seed)
    random.seed(seed)
    # Some libs use PYTHONHASHSEED; setting it here won't affect current process hashing,
    # but we still keep it for completeness if user runs as a subprocess.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# ----------------------------
# GA configuration
# ----------------------------

@dataclass(frozen=True)
class GAConfig:
    pop_size: int = 64
    offsprings: int = 64
    tourn_k: int = 3
    p_mut_scale: float = 1.0      # p_mut = p_mut_scale / n
    p_cross: float = 0.9
    elitism: int = 2

    # light-weight self-adaptation / memetic spice (generic, not problem-specific)
    adapt_mut: bool = True
    mut_c: float = 0.85           # 1/5-ish rule multiplier
    p_mut_min: float = 1e-4
    p_mut_max: float = 0.5

    # tiny local search budget (evaluations count!)
    ls_elites: int = 1            # how many top individuals to hill-climb
    ls_steps: int = 3             # max improving flips per elite per generation


def _load_cfg_from_json() -> Optional[GAConfig]:
    try:
        if os.path.exists(PARAM_JSON):
            with open(PARAM_JSON, "r", encoding="utf-8") as f:
                d = json.load(f)
            return GAConfig(**d)
    except Exception:
        return None
    return None


def _get_cfg(n: int) -> GAConfig:
    """Priority: TUNED_PARAM (from tuning) > JSON file > defaults."""
    global TUNED_PARAM
    if TUNED_PARAM is not None:
        # tuning.Param might have slightly different fields; map conservatively
        d = TUNED_PARAM if isinstance(TUNED_PARAM, dict) else getattr(TUNED_PARAM, "__dict__", {})
        # Support the Param class from the provided tuning template fileciteturn1file12
        pop_size = int(d.get("pop_size", 64))
        offsprings = int(d.get("offsprings", pop_size))
        tourn_k = int(d.get("tourn_k", 3))
        p_mut_scale = float(d.get("p_mut_scale", 1.0))
        p_cross = float(d.get("p_cross", 0.9))
        elitism = int(d.get("elitism", 2))
        adapt_mut = bool(d.get("adapt_mut", True))
        ls_elites = int(d.get("ls_elites", 1))
        ls_steps = int(d.get("ls_steps", 3))
        return GAConfig(
            pop_size=pop_size,
            offsprings=offsprings,
            tourn_k=tourn_k,
            p_mut_scale=p_mut_scale,
            p_cross=p_cross,
            elitism=elitism,
            adapt_mut=adapt_mut,
            ls_elites=ls_elites,
            ls_steps=ls_steps,
        )

    jcfg = _load_cfg_from_json()
    if jcfg is not None:
        return jcfg

    return GAConfig()


# ----------------------------
# Core operators
# ----------------------------

def _tournament(fit: np.ndarray, k: int) -> int:
    idx = np.random.randint(0, fit.size, size=k)
    return int(idx[np.argmax(fit[idx])])


def _uniform_crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = np.random.rand(a.size) < 0.5
    c = a.copy()
    c[mask] = b[mask]
    return c


def _bitflip(x: np.ndarray, p: float) -> np.ndarray:
    y = x.copy()
    flips = np.random.rand(x.size) < p
    # XOR for bits
    y[flips] ^= 1
    return y


def _init_pop(n: int, pop_size: int) -> np.ndarray:
    return (np.random.rand(pop_size, n) < 0.5).astype(np.int8)


def _evaluate(problem: ioh.problem.PBO, X: np.ndarray) -> np.ndarray:
    # IOH counts evaluations when calling problem(x)
    return np.array([problem(x.tolist()) for x in X], dtype=float)


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _make_offspring(pop: np.ndarray, fit: np.ndarray, cfg: GAConfig, p_mut: float) -> np.ndarray:
    """
    Offspring creation:
      - tournament selection
      - uniform crossover
      - bitflip mutation
    """
    n = pop.shape[1]
    off = np.empty((cfg.offsprings, n), dtype=np.int8)

    for i in range(cfg.offsprings):
        p1 = pop[_tournament(fit, cfg.tourn_k)]
        p2 = pop[_tournament(fit, cfg.tourn_k)]

        child = p1.copy()
        if np.random.rand() < cfg.p_cross:
            child = _uniform_crossover(p1, p2)

        child = _bitflip(child, p_mut)
        off[i] = child

    return off


def _survivor_selection(pop: np.ndarray, fit: np.ndarray,
                        off: np.ndarray, off_fit: np.ndarray,
                        cfg: GAConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    (mu+lambda) with elitism and a tiny diversity tie-breaker:

    - Combine parents + offspring
    - Sort primarily by fitness (desc)
    - Tie-break: prefer larger Hamming distance to current best (diversity)
    """
    comb = np.vstack([pop, off])
    comb_fit = np.concatenate([fit, off_fit])

    # current best in combined pool (for diversity tie-break)
    best_idx = int(np.argmax(comb_fit))
    best = comb[best_idx]

    # lexsort uses last key as primary; we want: fitness desc, diversity desc
    div = np.array([_hamming(ind, best) for ind in comb], dtype=int)
    order = np.lexsort((-div, -comb_fit))
    next_pop = comb[order[:cfg.pop_size]].copy()
    next_fit = comb_fit[order[:cfg.pop_size]].copy()
    return next_pop, next_fit


def _local_search(problem: ioh.problem.PBO, x: np.ndarray, fx: float, steps: int) -> Tuple[np.ndarray, float]:
    """
    Very small first-improvement hill-climber:
    - Try random bit flips; accept if improves fitness.
    Each attempt costs 1 function evaluation.
    """
    n = x.size
    cur = x.copy()
    cur_fx = float(fx)

    for _ in range(steps):
        j = int(np.random.randint(0, n))
        cand = cur.copy()
        cand[j] ^= 1
        f_cand = float(problem(cand.tolist()))
        if f_cand > cur_fx:
            cur, cur_fx = cand, f_cand

        # Stop early if budget reached
        if problem.state.evaluations >= budget:
            break

    return cur, cur_fx


# ----------------------------
# Main GA entrypoint (called by grader)
# ----------------------------

def studentnumber1_studentnumber2_GA(problem: ioh.problem.PBO) -> None:
    n = problem.meta_data.n_variables
    cfg = _get_cfg(n)

    # Derived per-bit mutation probability
    p_mut = float(np.clip(cfg.p_mut_scale / max(1, n), cfg.p_mut_min, cfg.p_mut_max))

    pop = _init_pop(n, cfg.pop_size)
    fit = _evaluate(problem, pop)

    # GA loop: stop by evaluation budget
    while problem.state.evaluations < budget:
        # offspring
        off = _make_offspring(pop, fit, cfg, p_mut)

        # If remaining budget is smaller than len(off), truncate offspring evaluation
        remaining = budget - int(problem.state.evaluations)
        if remaining <= 0:
            break
        if remaining < off.shape[0]:
            off = off[:remaining]

        off_fit = _evaluate(problem, off)

        # survivor selection
        pop, fit = _survivor_selection(pop, fit, off, off_fit, cfg)

        # Local search on elites (tiny)
        if cfg.ls_elites > 0 and cfg.ls_steps > 0 and problem.state.evaluations < budget:
            elite_order = np.argsort(-fit)[: cfg.ls_elites]
            for idx in elite_order:
                pop[idx], fit[idx] = _local_search(problem, pop[idx], fit[idx], cfg.ls_steps)
                if problem.state.evaluations >= budget:
                    break

        # Generic (problem-agnostic) mutation adaptation based on offspring success
        if cfg.adapt_mut and off_fit.size > 0:
            # success = proportion of offspring that beats median parent fitness
            thr = float(np.median(fit))
            succ = float(np.mean(off_fit > thr))
            if succ > 0.2:
                p_mut = max(cfg.p_mut_min, p_mut * cfg.mut_c)
            else:
                p_mut = min(cfg.p_mut_max, p_mut / cfg.mut_c)


# ----------------------------
# IOH plumbing (keep compatible with template)
# ----------------------------

def create_problem(fid: int):
    # Template uses dimension=50. Keep that consistent unless assignment specifies otherwise.
    dim = 64 if fid == 23 else 50
    problem = get_problem(fid, dimension=dim, instance=1, problem_class=ProblemClass.PBO)
    l = logger.Analyzer(
        root="data",
        folder_name=f"GA_F{fid}",
        algorithm_name="studentnumber1_studentnumber2_GA",
        store_positions=False,
    )
    problem.attach_logger(l)
    return problem, l


# ----------------------------
# Plot helpers for your report (unchanged)
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
    # Example execution (20 independent runs), fixed seeds for reproducibility fileciteturn1file4
    for fid in (18, 23):
        prob, log = create_problem(fid)
        for run in range(20):
            seed_everything(BASE_SEED + fid * 10_000 + run)
            studentnumber1_studentnumber2_GA(prob)
            prob.reset()
        log.close()
