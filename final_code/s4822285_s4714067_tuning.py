
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ioh import get_problem, ProblemClass

from ea_utils import set_global_seed
from s4822285_s4714067_GA import GAParams, s4822285_s7414067_GA


# Assignment tasks for Part 1
TASKS: List[Tuple[int, int]] = [
    (18, 50),  # LABS
    (23, 64),  # NQueens
]


def sample_params(rng: np.random.Generator, dim_for_mut: int) -> GAParams:
    """Small but meaningful random search space."""
    pop_size = int(rng.choice([50, 80, 120, 160, 200, 240, 300]))
    tournament_k = int(rng.choice([2, 3, 4, 5]))
    crossover_rate = float(rng.uniform(0.6, 0.98))
    mutation_rate = float(rng.uniform(0.2 / dim_for_mut, 3.0 / dim_for_mut))
    elite_frac = float(rng.choice([0.0, 0.01, 0.02, 0.03, 0.05]))
    return GAParams(
        pop_size=pop_size,
        tournament_k=tournament_k,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_frac=elite_frac,
    ).clamp()


def _median_final_for_task(
    fid: int,
    dim: int,
    eval_budget: int,
    seeds: List[int],
    params: GAParams,
) -> float:
    problem = get_problem(fid, dimension=dim, instance=1, problem_class=ProblemClass.PBO)
    finals = []
    for s in seeds:
        problem.reset()
        rs = s4822285_s7414067_GA(problem, budget=eval_budget, seed=s, params=params)
        finals.append(float(rs.final_best))
    return float(np.median(np.asarray(finals, dtype=float)))


def _zscore(xs: np.ndarray) -> np.ndarray:
    mu = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0
    if sd <= 0.0:
        return np.zeros_like(xs)
    return (xs - mu) / sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=100_000, help="Max tuning evaluations (assignment: 100000)")
    ap.add_argument("--stage1-eval", type=int, default=1_000)
    ap.add_argument("--stage1-seeds", type=int, default=2)
    ap.add_argument("--stage1-trials", type=int, default=15)
    ap.add_argument("--stage2-eval", type=int, default=5_000)
    ap.add_argument("--stage2-seeds", type=int, default=2)
    ap.add_argument("--stage2-topk", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=777_000)
    ap.add_argument("--out", type=str, default="ga_best_params.json")
    args = ap.parse_args()

    cap = int(args.cap)
    set_global_seed(int(args.base_seed))
    rng = np.random.default_rng(int(args.base_seed))

    seed_pool = [int(args.base_seed) + i for i in range(50_000)]

    per_trial_cost = int(args.stage1_eval) * int(args.stage1_seeds) * len(TASKS)
    max_trials = cap // max(1, per_trial_cost)
    trials = int(min(args.stage1_trials, max_trials))
    if trials < 1:
        raise RuntimeError("Tuning cap too small for the requested stage1 settings.")

    dim_for_mut = max(d for _, d in TASKS)
    stage1_params: List[GAParams] = []
    stage1_medians: Dict[Tuple[int, int], List[float]] = {t: [] for t in TASKS}

    print("=" * 72)
    print(f"Stage 1: random search | trials={trials} | eval={args.stage1_eval} | seeds={args.stage1_seeds}")
    print(f"Per-trial cost: {per_trial_cost} evals across {len(TASKS)} tasks")
    print(f"Cap: {cap} evals | Stage1 budget used: {trials * per_trial_cost}")
    print("=" * 72)

    for t in range(trials):
        params = sample_params(rng, dim_for_mut=dim_for_mut)
        stage1_params.append(params)

        seeds = seed_pool[t * int(args.stage1_seeds) : (t + 1) * int(args.stage1_seeds)]
        for (fid, dim) in TASKS:
            med = _median_final_for_task(
                fid=fid,
                dim=dim,
                eval_budget=int(args.stage1_eval),
                seeds=seeds,
                params=params,
            )
            stage1_medians[(fid, dim)].append(med)

        parts = [f"trial {t:02d}"]
        for (fid, dim) in TASKS:
            parts.append(f"F{fid}d{dim} med={stage1_medians[(fid, dim)][-1]:.6g}")
        print(" | ".join(parts) + f" | {asdict(params)}")

    zsum = np.zeros(trials, dtype=float)
    for task in TASKS:
        arr = np.asarray(stage1_medians[task], dtype=float)
        zsum += _zscore(arr)

    stage1_order = np.argsort(zsum)[::-1]  # high zsum best
    topk = int(min(int(args.stage2_topk), len(stage1_order)))
    top_idx = stage1_order[:topk].tolist()

    stage1_budget_used = trials * per_trial_cost
    remaining = cap - stage1_budget_used
    per_cfg_cost = int(args.stage2_eval) * int(args.stage2_seeds) * len(TASKS)
    max_cfg = remaining // max(1, per_cfg_cost)
    top_idx = top_idx[: int(max_cfg)]

    if not top_idx:
        best_idx = int(stage1_order[0])
        best_params = stage1_params[best_idx]
        Path(args.out).write_text(json.dumps(asdict(best_params), indent=2, sort_keys=True), encoding="utf-8")
        print("\n[WARN] No budget left for stage 2; saved best stage-1 params.")
        print(f"Saved to: {args.out}")
        return

    print("\n" + "=" * 72)
    print(f"Stage 2: refine top-{len(top_idx)} | eval={args.stage2_eval} | seeds={args.stage2_seeds}")
    print(f"Remaining budget: {remaining} | Per-config cost: {per_cfg_cost} | Stage2 budget used: {len(top_idx) * per_cfg_cost}")
    print("=" * 72)

    stage2_details: List[List[float]] = []

    stage1_seed_end = trials * int(args.stage1_seeds)


    for rank, idx in enumerate(top_idx):
        params = stage1_params[idx]
        start = stage1_seed_end + rank * int(args.stage2_seeds)
        end = start + int(args.stage2_seeds)
        seeds = seed_pool[start:end]
        if len(seeds) < int(args.stage2_seeds):
            raise RuntimeError(
                f"Not enough seeds for stage2: need {args.stage2_seeds}, got {len(seeds)}. "
                f"Increase seed_pool or reduce stage2-seeds/topk."
            )

        meds = []
        for (fid, dim) in TASKS:
            meds.append(
                _median_final_for_task(
                    fid=fid,
                    dim=dim,
                    eval_budget=int(args.stage2_eval),
                    seeds=seeds,
                    params=params,
                )
            )
        stage2_details.append(meds)
        print(f"[cand {rank}] " + " | ".join(
            [f"F{TASKS[i][0]}d{TASKS[i][1]} med={meds[i]:.6g}" for i in range(len(TASKS))]
        ))

    arr = np.asarray(stage2_details, dtype=float)  
    z = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        z[:, j] = _zscore(arr[:, j])
    zsum2 = np.sum(z, axis=1)
    best_rank = int(np.argmax(zsum2))
    best_params = stage1_params[top_idx[best_rank]]

    Path(args.out).write_text(json.dumps(asdict(best_params), indent=2, sort_keys=True), encoding="utf-8")
    print("\n" + "=" * 72)
    print("Selected best params (saved):")
    print(json.dumps(asdict(best_params), indent=2, sort_keys=True))
    print("=" * 72)


if __name__ == "__main__":
    main()
