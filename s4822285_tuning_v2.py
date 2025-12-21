from __future__ import annotations

"""
Hyper-parameter tuning for the GA on BOTH PBO problems F18 and F23.

Assignment requirements:
- Total tuning budget <= 100,000 function evaluations across both problems. fileciteturn1file4
- Output ONE hyperparameter setting that works for BOTH problems (no per-problem params). fileciteturn1file5
- Tuning code must call your GA implementation. fileciteturn1file5
- Reproducible: fixed seeds. fileciteturn1file4

We implement a simple, robust two-stage racing / successive-halving approach:
Stage 1: 40 random configs, budget=500 evals, 1 trial per problem.
Stage 2: top 10 configs, budget=1500 evals, 2 trials per problem.

Total evals:
  Stage1: 40 * 2 * 500 = 40,000
  Stage2: 10 * 2 * 2 * 1500 = 60,000
  Total: 100,000 (exactly)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import json
import os
import numpy as np

import ioh
from ioh import get_problem, ProblemClass

import s4822285_GA_v2 as GA_mod


TUNING_BUDGET = 100_000

BASE_SEED = 20251221  # fixed seed so tuning outcome is reproducible


@dataclass(frozen=True)
class Param:
    pop_size: int
    offsprings: int
    tourn_k: int
    p_mut_scale: float     # p_mut = p_mut_scale / n
    p_cross: float
    elitism: int
    adapt_mut: bool
    ls_elites: int
    ls_steps: int


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_params(rng: np.random.Generator, k: int) -> List[Param]:
    params: List[Param] = []
    for _ in range(k):
        pop = int(rng.choice([24, 32, 40, 48, 64, 80, 96]))
        off = int(pop)  # keep λ = μ for evaluation efficiency
        tk = int(rng.integers(2, 6))  # 2..5
        pms = float(rng.choice([0.25, 0.5, 1.0, 1.5, 2.0, 3.0]))
        pc = float(rng.choice([0.6, 0.7, 0.8, 0.9, 0.95]))
        el = int(rng.choice([1, 2, 3]))
        adapt = bool(rng.choice([True, True, False]))  # bias towards True
        ls_el = int(rng.choice([0, 1, 1, 2]))          # bias towards 1
        ls_st = int(rng.choice([0, 2, 3, 4]))
        params.append(Param(pop, off, tk, pms, pc, el, adapt, ls_el, ls_st))
    return params


def make_problem(fid: int):
    # No logger during tuning to keep output light
    dim = 64 if fid == 23 else 50
    return get_problem(fid, dimension=dim, instance=1, problem_class=ProblemClass.PBO)


def eval_config(param: Param, fid: int, run_seed: int, budget_override: int) -> float:
    """
    Runs GA with the given params on one problem instance and returns end best-so-far fitness.
    Higher is better for PBO.
    """
    np.random.seed(run_seed)

    # Patch params into GA module (GA reads GA_mod.TUNED_PARAM)
    GA_mod.TUNED_PARAM = param

    # Temporarily override GA budget
    old_budget = GA_mod.budget
    GA_mod.budget = int(budget_override)
    try:
        prob = make_problem(fid)
        GA_mod.studentnumber1_studentnumber2_GA(prob)
        score = float(prob.state.current_best.y)
        prob.reset()
    finally:
        GA_mod.budget = old_budget

    return score


def score_param(param: Param, budget_override: int, trials: int, seed_offset: int) -> float:
    """
    Average performance across BOTH problems (F18,F23) and multiple trials.
    """
    vals: List[float] = []
    for fid in (18, 23):
        for t in range(trials):
            seed = BASE_SEED + seed_offset + fid * 10_000 + t
            vals.append(eval_config(param, fid, seed, budget_override))
    return float(np.mean(vals))


def tune() -> Dict[str, Any]:
    rng = _rng(BASE_SEED)

    # Stage 1
    stage1_budget = 500
    stage1_trials = 1
    stage1_k = 40

    candidates = sample_params(rng, stage1_k)

    stage1_scores: List[Tuple[float, Param]] = []
    for i, p in enumerate(candidates):
        s = score_param(p, budget_override=stage1_budget, trials=stage1_trials, seed_offset=i * 100)
        stage1_scores.append((s, p))

    stage1_scores.sort(key=lambda x: x[0], reverse=True)
    survivors = [p for (_, p) in stage1_scores[:10]]

    # Stage 2
    stage2_budget = 1500
    stage2_trials = 2

    stage2_scores: List[Tuple[float, Param]] = []
    for i, p in enumerate(survivors):
        s = score_param(p, budget_override=stage2_budget, trials=stage2_trials, seed_offset=50_000 + i * 1000)
        stage2_scores.append((s, p))

    stage2_scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_param = stage2_scores[0]

    out = {
        "best_score": best_score,
        "best_param": asdict(best_param),
        "stage1_top10": [(float(s), asdict(p)) for (s, p) in stage1_scores[:10]],
        "stage2_all": [(float(s), asdict(p)) for (s, p) in stage2_scores],
        "evals_used": stage1_k * 2 * stage1_budget * stage1_trials + 10 * 2 * stage2_budget * stage2_trials,
    }
    return out


def save_best(param_dict: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(param_dict, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    res = tune()
    print("Tuning evals used:", res["evals_used"], "/", TUNING_BUDGET)
    print("Best score:", res["best_score"])
    print("Best param:", res["best_param"])

    # write best params for GA to consume (GA loads ga_best_params.json if present)
    here = os.path.dirname(__file__)
    out_path = os.path.join(here, "ga_best_params.json")
    save_best(res["best_param"], out_path)
    print("Wrote:", out_path)
