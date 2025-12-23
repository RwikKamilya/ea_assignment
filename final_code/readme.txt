(venv) rwik@PROENG93:~/leiden_projects/ea_assignment/final$ python s4822285_tuning.py
========================================================================
Stage 1: random search | trials=15 | eval=1000 | seeds=2
Per-trial cost: 4000 evals across 2 tasks
Cap: 100000 evals | Stage1 budget used: 60000
========================================================================
trial 00 | F18d50 med=2.45959 | F23d64 med=8 | {'pop_size': 160, 'tournament_k': 4, 'crossover_rate': 0.9057433964072521, 'mutation_rate': 0.02393144635271269, 'elite_frac': 0.0}
trial 01 | F18d50 med=2.50042 | F23d64 med=8 | {'pop_size': 240, 'tournament_k': 2, 'crossover_rate': 0.7753966974195534, 'mutation_rate': 0.036875621678275294, 'elite_frac': 0.0}
trial 02 | F18d50 med=2.94222 | F23d64 med=8 | {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
trial 03 | F18d50 med=3.53737 | F23d64 med=0 | {'pop_size': 50, 'tournament_k': 3, 'crossover_rate': 0.7936680491295396, 'mutation_rate': 0.007341791719977668, 'elite_frac': 0.01}
trial 04 | F18d50 med=2.72813 | F23d64 med=8 | {'pop_size': 300, 'tournament_k': 3, 'crossover_rate': 0.6177009281687551, 'mutation_rate': 0.029680986803868624, 'elite_frac': 0.0}
trial 05 | F18d50 med=2.5177 | F23d64 med=4 | {'pop_size': 240, 'tournament_k': 3, 'crossover_rate': 0.6504288215757827, 'mutation_rate': 0.01639556970853366, 'elite_frac': 0.02}
trial 06 | F18d50 med=2.39923 | F23d64 med=8 | {'pop_size': 300, 'tournament_k': 2, 'crossover_rate': 0.8577622881128169, 'mutation_rate': 0.022195391735518148, 'elite_frac': 0.01}
trial 07 | F18d50 med=2.49899 | F23d64 med=8 | {'pop_size': 300, 'tournament_k': 2, 'crossover_rate': 0.95361225794889, 'mutation_rate': 0.04360177897607923, 'elite_frac': 0.03}
trial 08 | F18d50 med=3.35468 | F23d64 med=8 | {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8978874049340569, 'mutation_rate': 0.008533291969990264, 'elite_frac': 0.05}
trial 09 | F18d50 med=2.67478 | F23d64 med=8 | {'pop_size': 50, 'tournament_k': 2, 'crossover_rate': 0.7483735802091891, 'mutation_rate': 0.026939363342401757, 'elite_frac': 0.05}
trial 10 | F18d50 med=2.69535 | F23d64 med=8 | {'pop_size': 80, 'tournament_k': 3, 'crossover_rate': 0.8854652702359547, 'mutation_rate': 0.042728943400457874, 'elite_frac': 0.02}
trial 11 | F18d50 med=2.63556 | F23d64 med=8 | {'pop_size': 160, 'tournament_k': 5, 'crossover_rate': 0.799635054124125, 'mutation_rate': 0.021760208362653997, 'elite_frac': 0.01}
trial 12 | F18d50 med=2.60163 | F23d64 med=4 | {'pop_size': 200, 'tournament_k': 4, 'crossover_rate': 0.6392188328467618, 'mutation_rate': 0.003697772263166828, 'elite_frac': 0.01}
trial 13 | F18d50 med=2.51574 | F23d64 med=8 | {'pop_size': 300, 'tournament_k': 3, 'crossover_rate': 0.8339341587028605, 'mutation_rate': 0.03573233977546439, 'elite_frac': 0.0}
trial 14 | F18d50 med=2.52556 | F23d64 med=8 | {'pop_size': 160, 'tournament_k': 4, 'crossover_rate': 0.7639028600603968, 'mutation_rate': 0.007646307857602739, 'elite_frac': 0.01}

========================================================================
Stage 2: refine top-2 | eval=5000 | seeds=2
Remaining budget: 40000 | Per-config cost: 20000 | Stage2 budget used: 40000
========================================================================
[cand 0] F18d50 med=3.631 | F23d64 med=4
[cand 1] F18d50 med=4.30553 | F23d64 med=8

========================================================================
Selected best params (saved):
{
  "crossover_rate": 0.8359334351205238,
  "elite_frac": 0.01,
  "mutation_rate": 0.021702729161426414,
  "pop_size": 80,
  "tournament_k": 5
}
========================================================================
(venv) rwik@PROENG93:~/leiden_projects/ea_assignment/final$ python s4822285_GA.py --multi-dim
==============================================================
GA on PBO F18 (dim=20) | runs=20 | budget=5000 | maximize=True
==============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1234567 (maximize) | best=4.34783 @ eval=464 | final_best=4.34783 | evals=5000 | time=0.09s
[run 01] seed=1234568 (maximize) | best=4 @ eval=415 | final_best=4 | evals=5000 | time=0.07s
[run 02] seed=1234569 (maximize) | best=3.44828 @ eval=112 | final_best=3.44828 | evals=5000 | time=0.07s
[run 03] seed=1234570 (maximize) | best=3.7037 @ eval=464 | final_best=3.7037 | evals=5000 | time=0.07s
[run 04] seed=1234571 (maximize) | best=4.7619 @ eval=319 | final_best=4.7619 | evals=5000 | time=0.07s
[run 05] seed=1234572 (maximize) | best=4 @ eval=348 | final_best=4 | evals=5000 | time=0.07s
[run 06] seed=1234573 (maximize) | best=4 @ eval=1114 | final_best=4 | evals=5000 | time=0.07s
[run 07] seed=1234574 (maximize) | best=4 @ eval=255 | final_best=4 | evals=5000 | time=0.07s
[run 08] seed=1234575 (maximize) | best=5.88235 @ eval=704 | final_best=5.88235 | evals=5000 | time=0.07s
[run 09] seed=1234576 (maximize) | best=4.34783 @ eval=4772 | final_best=4.34783 | evals=5000 | time=0.07s
[run 10] seed=1234577 (maximize) | best=4 @ eval=368 | final_best=4 | evals=5000 | time=0.07s
[run 11] seed=1234578 (maximize) | best=4 @ eval=782 | final_best=4 | evals=5000 | time=0.07s
[run 12] seed=1234579 (maximize) | best=3.7037 @ eval=587 | final_best=3.7037 | evals=5000 | time=0.07s
[run 13] seed=1234580 (maximize) | best=3.7037 @ eval=647 | final_best=3.7037 | evals=5000 | time=0.07s
[run 14] seed=1234581 (maximize) | best=3.44828 @ eval=429 | final_best=3.44828 | evals=5000 | time=0.07s
[run 15] seed=1234582 (maximize) | best=5.26316 @ eval=3447 | final_best=5.26316 | evals=5000 | time=0.07s
[run 16] seed=1234583 (maximize) | best=3.7037 @ eval=233 | final_best=3.7037 | evals=5000 | time=0.08s
[run 17] seed=1234584 (maximize) | best=4.7619 @ eval=557 | final_best=4.7619 | evals=5000 | time=0.08s
[run 18] seed=1234585 (maximize) | best=3.44828 @ eval=791 | final_best=3.44828 | evals=5000 | time=0.08s
[run 19] seed=1234586 (maximize) | best=4 @ eval=713 | final_best=4 | evals=5000 | time=0.08s

F18 d=20 final_best: n=20 | min=3.44828 | max=5.88235 | mean=4.12623 | median=4 | std=0.627756
==============================================================
GA on PBO F18 (dim=50) | runs=20 | budget=5000 | maximize=True
==============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1235567 (maximize) | best=3.79939 @ eval=1293 | final_best=3.79939 | evals=5000 | time=0.09s
[run 01] seed=1235568 (maximize) | best=3.7092 @ eval=3922 | final_best=3.7092 | evals=5000 | time=0.09s
[run 02] seed=1235569 (maximize) | best=3.99361 @ eval=3135 | final_best=3.99361 | evals=5000 | time=0.08s
[run 03] seed=1235570 (maximize) | best=4.20875 @ eval=3458 | final_best=4.20875 | evals=5000 | time=0.08s
[run 04] seed=1235571 (maximize) | best=3.99361 @ eval=4856 | final_best=3.99361 | evals=5000 | time=0.09s
[run 05] seed=1235572 (maximize) | best=3.4626 @ eval=1961 | final_best=3.4626 | evals=5000 | time=0.09s
[run 06] seed=1235573 (maximize) | best=3.99361 @ eval=1410 | final_best=3.99361 | evals=5000 | time=0.08s
[run 07] seed=1235574 (maximize) | best=3.99361 @ eval=3104 | final_best=3.99361 | evals=5000 | time=0.09s
[run 08] seed=1235575 (maximize) | best=3.79939 @ eval=2225 | final_best=3.79939 | evals=5000 | time=0.09s
[run 09] seed=1235576 (maximize) | best=4.4484 @ eval=1157 | final_best=4.4484 | evals=5000 | time=0.08s
[run 10] seed=1235577 (maximize) | best=3.89408 @ eval=4015 | final_best=3.89408 | evals=5000 | time=0.09s
[run 11] seed=1235578 (maximize) | best=4.86381 @ eval=1122 | final_best=4.86381 | evals=5000 | time=0.09s
[run 12] seed=1235579 (maximize) | best=4.09836 @ eval=3504 | final_best=4.09836 | evals=5000 | time=0.08s
[run 13] seed=1235580 (maximize) | best=3.99361 @ eval=2743 | final_best=3.99361 | evals=5000 | time=0.08s
[run 14] seed=1235581 (maximize) | best=3.62319 @ eval=2092 | final_best=3.62319 | evals=5000 | time=0.09s
[run 15] seed=1235582 (maximize) | best=3.89408 @ eval=1442 | final_best=3.89408 | evals=5000 | time=0.09s
[run 16] seed=1235583 (maximize) | best=4.20875 @ eval=1672 | final_best=4.20875 | evals=5000 | time=0.09s
[run 17] seed=1235584 (maximize) | best=4.4484 @ eval=3378 | final_best=4.4484 | evals=5000 | time=0.09s
[run 18] seed=1235585 (maximize) | best=3.7092 @ eval=2446 | final_best=3.7092 | evals=5000 | time=0.09s
[run 19] seed=1235586 (maximize) | best=4.57875 @ eval=2666 | final_best=4.57875 | evals=5000 | time=0.09s

F18 d=50 final_best: n=20 | min=3.4626 | max=4.86381 | mean=4.03572 | median=3.99361 | std=0.344705
===============================================================
GA on PBO F18 (dim=100) | runs=20 | budget=5000 | maximize=True
===============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1236567 (maximize) | best=3.33778 @ eval=4567 | final_best=3.33778 | evals=5000 | time=0.10s
[run 01] seed=1236568 (maximize) | best=3.36474 @ eval=4701 | final_best=3.36474 | evals=5000 | time=0.10s
[run 02] seed=1236569 (maximize) | best=3.5868 @ eval=4962 | final_best=3.5868 | evals=5000 | time=0.10s
[run 03] seed=1236570 (maximize) | best=3.79363 @ eval=3822 | final_best=3.79363 | evals=5000 | time=0.10s
[run 04] seed=1236571 (maximize) | best=2.9036 @ eval=4776 | final_best=2.9036 | evals=5000 | time=0.10s
[run 05] seed=1236572 (maximize) | best=3.20102 @ eval=2800 | final_best=3.20102 | evals=5000 | time=0.10s
[run 06] seed=1236573 (maximize) | best=3.3557 @ eval=4427 | final_best=3.3557 | evals=5000 | time=0.10s
[run 07] seed=1236574 (maximize) | best=3.72578 @ eval=4062 | final_best=3.72578 | evals=5000 | time=0.10s
[run 08] seed=1236575 (maximize) | best=3.12891 @ eval=4349 | final_best=3.12891 | evals=5000 | time=0.10s
[run 09] seed=1236576 (maximize) | best=3.18471 @ eval=3938 | final_best=3.18471 | evals=5000 | time=0.10s
[run 10] seed=1236577 (maximize) | best=3.44828 @ eval=4661 | final_best=3.44828 | evals=5000 | time=0.10s
[run 11] seed=1236578 (maximize) | best=3.5461 @ eval=4632 | final_best=3.5461 | evals=5000 | time=0.10s
[run 12] seed=1236579 (maximize) | best=3.72578 @ eval=4286 | final_best=3.72578 | evals=5000 | time=0.10s
[run 13] seed=1236580 (maximize) | best=3.69276 @ eval=4871 | final_best=3.69276 | evals=5000 | time=0.10s
[run 14] seed=1236581 (maximize) | best=3.74813 @ eval=4406 | final_best=3.74813 | evals=5000 | time=0.10s
[run 15] seed=1236582 (maximize) | best=3.25098 @ eval=4205 | final_best=3.25098 | evals=5000 | time=0.10s
[run 16] seed=1236583 (maximize) | best=4.06504 @ eval=4058 | final_best=4.06504 | evals=5000 | time=0.10s
[run 17] seed=1236584 (maximize) | best=3.78215 @ eval=4389 | final_best=3.78215 | evals=5000 | time=0.10s
[run 18] seed=1236585 (maximize) | best=3.55619 @ eval=4975 | final_best=3.55619 | evals=5000 | time=0.10s
[run 19] seed=1236586 (maximize) | best=4.30293 @ eval=4825 | final_best=4.30293 | evals=5000 | time=0.10s

F18 d=100 final_best: n=20 | min=2.9036 | max=4.30293 | mean=3.53505 | median=3.55114 | std=0.334902
===============================================================
GA on PBO F18 (dim=200) | runs=20 | budget=5000 | maximize=True
===============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1237567 (maximize) | best=2.22816 @ eval=4344 | final_best=2.22816 | evals=5000 | time=0.14s
[run 01] seed=1237568 (maximize) | best=2.61506 @ eval=3817 | final_best=2.61506 | evals=5000 | time=0.13s
[run 02] seed=1237569 (maximize) | best=2.56674 @ eval=4774 | final_best=2.56674 | evals=5000 | time=0.13s
[run 03] seed=1237570 (maximize) | best=2.69542 @ eval=4157 | final_best=2.69542 | evals=5000 | time=0.13s
[run 04] seed=1237571 (maximize) | best=2.37079 @ eval=3115 | final_best=2.37079 | evals=5000 | time=0.14s
[run 05] seed=1237572 (maximize) | best=2.14869 @ eval=4897 | final_best=2.14869 | evals=5000 | time=0.14s
[run 06] seed=1237573 (maximize) | best=2.62605 @ eval=4770 | final_best=2.62605 | evals=5000 | time=0.13s
[run 07] seed=1237574 (maximize) | best=2.22916 @ eval=2706 | final_best=2.22916 | evals=5000 | time=0.13s
[run 08] seed=1237575 (maximize) | best=2.50376 @ eval=4508 | final_best=2.50376 | evals=5000 | time=0.13s
[run 09] seed=1237576 (maximize) | best=2.33209 @ eval=4506 | final_best=2.33209 | evals=5000 | time=0.14s
[run 10] seed=1237577 (maximize) | best=2.3855 @ eval=3438 | final_best=2.3855 | evals=5000 | time=0.15s
[run 11] seed=1237578 (maximize) | best=2.54065 @ eval=4057 | final_best=2.54065 | evals=5000 | time=0.14s
[run 12] seed=1237579 (maximize) | best=2.39808 @ eval=4365 | final_best=2.39808 | evals=5000 | time=0.14s
[run 13] seed=1237580 (maximize) | best=2.56279 @ eval=4331 | final_best=2.56279 | evals=5000 | time=0.14s
[run 14] seed=1237581 (maximize) | best=2.30627 @ eval=3812 | final_best=2.30627 | evals=5000 | time=0.14s
[run 15] seed=1237582 (maximize) | best=2.42718 @ eval=4955 | final_best=2.42718 | evals=5000 | time=0.14s
[run 16] seed=1237583 (maximize) | best=2.49377 @ eval=4905 | final_best=2.49377 | evals=5000 | time=0.14s
[run 17] seed=1237584 (maximize) | best=2.35849 @ eval=3641 | final_best=2.35849 | evals=5000 | time=0.14s
[run 18] seed=1237585 (maximize) | best=2.22124 @ eval=3383 | final_best=2.22124 | evals=5000 | time=0.14s
[run 19] seed=1237586 (maximize) | best=2.54323 @ eval=4267 | final_best=2.54323 | evals=5000 | time=0.15s

F18 d=200 final_best: n=20 | min=2.14869 | max=2.69542 | mean=2.42766 | median=2.41263 | std=0.15462
==============================================================
GA on PBO F23 (dim=16) | runs=20 | budget=5000 | maximize=True
==============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1244567 (maximize) | best=4 @ eval=11 | final_best=4 | evals=5000 | time=0.08s
[run 01] seed=1244568 (maximize) | best=4 @ eval=7 | final_best=4 | evals=5000 | time=0.08s
[run 02] seed=1244569 (maximize) | best=4 @ eval=18 | final_best=4 | evals=5000 | time=0.08s
[run 03] seed=1244570 (maximize) | best=4 @ eval=7 | final_best=4 | evals=5000 | time=0.07s
[run 04] seed=1244571 (maximize) | best=4 @ eval=2 | final_best=4 | evals=5000 | time=0.08s
[run 05] seed=1244572 (maximize) | best=4 @ eval=6 | final_best=4 | evals=5000 | time=0.08s
[run 06] seed=1244573 (maximize) | best=4 @ eval=22 | final_best=4 | evals=5000 | time=0.08s
[run 07] seed=1244574 (maximize) | best=4 @ eval=13 | final_best=4 | evals=5000 | time=0.08s
[run 08] seed=1244575 (maximize) | best=4 @ eval=28 | final_best=4 | evals=5000 | time=0.08s
[run 09] seed=1244576 (maximize) | best=4 @ eval=19 | final_best=4 | evals=5000 | time=0.08s
[run 10] seed=1244577 (maximize) | best=4 @ eval=2 | final_best=4 | evals=5000 | time=0.08s
[run 11] seed=1244578 (maximize) | best=4 @ eval=45 | final_best=4 | evals=5000 | time=0.07s
[run 12] seed=1244579 (maximize) | best=4 @ eval=13 | final_best=4 | evals=5000 | time=0.08s
[run 13] seed=1244580 (maximize) | best=4 @ eval=6 | final_best=4 | evals=5000 | time=0.08s
[run 14] seed=1244581 (maximize) | best=4 @ eval=2 | final_best=4 | evals=5000 | time=0.08s
[run 15] seed=1244582 (maximize) | best=4 @ eval=9 | final_best=4 | evals=5000 | time=0.08s
[run 16] seed=1244583 (maximize) | best=4 @ eval=2 | final_best=4 | evals=5000 | time=0.09s
[run 17] seed=1244584 (maximize) | best=4 @ eval=7 | final_best=4 | evals=5000 | time=0.08s
[run 18] seed=1244585 (maximize) | best=4 @ eval=3 | final_best=4 | evals=5000 | time=0.08s
[run 19] seed=1244586 (maximize) | best=4 @ eval=16 | final_best=4 | evals=5000 | time=0.07s

F23 d=16 final_best: n=20 | min=4 | max=4 | mean=4 | median=4 | std=0
==============================================================
GA on PBO F23 (dim=25) | runs=20 | budget=5000 | maximize=True
==============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1245567 (maximize) | best=5 @ eval=17 | final_best=5 | evals=5000 | time=0.08s
[run 01] seed=1245568 (maximize) | best=5 @ eval=19 | final_best=5 | evals=5000 | time=0.08s
[run 02] seed=1245569 (maximize) | best=5 @ eval=8 | final_best=5 | evals=5000 | time=0.09s
[run 03] seed=1245570 (maximize) | best=5 @ eval=35 | final_best=5 | evals=5000 | time=0.08s
[run 04] seed=1245571 (maximize) | best=5 @ eval=12 | final_best=5 | evals=5000 | time=0.09s
[run 05] seed=1245572 (maximize) | best=5 @ eval=23 | final_best=5 | evals=5000 | time=0.08s
[run 06] seed=1245573 (maximize) | best=5 @ eval=2 | final_best=5 | evals=5000 | time=0.08s
[run 07] seed=1245574 (maximize) | best=5 @ eval=38 | final_best=5 | evals=5000 | time=0.08s
[run 08] seed=1245575 (maximize) | best=5 @ eval=15 | final_best=5 | evals=5000 | time=0.08s
[run 09] seed=1245576 (maximize) | best=5 @ eval=1 | final_best=5 | evals=5000 | time=0.08s
[run 10] seed=1245577 (maximize) | best=5 @ eval=20 | final_best=5 | evals=5000 | time=0.08s
[run 11] seed=1245578 (maximize) | best=5 @ eval=2 | final_best=5 | evals=5000 | time=0.08s
[run 12] seed=1245579 (maximize) | best=5 @ eval=1 | final_best=5 | evals=5000 | time=0.08s
[run 13] seed=1245580 (maximize) | best=5 @ eval=1 | final_best=5 | evals=5000 | time=0.09s
[run 14] seed=1245581 (maximize) | best=5 @ eval=3 | final_best=5 | evals=5000 | time=0.09s
[run 15] seed=1245582 (maximize) | best=5 @ eval=11 | final_best=5 | evals=5000 | time=0.09s
[run 16] seed=1245583 (maximize) | best=5 @ eval=51 | final_best=5 | evals=5000 | time=0.08s
[run 17] seed=1245584 (maximize) | best=5 @ eval=6 | final_best=5 | evals=5000 | time=0.08s
[run 18] seed=1245585 (maximize) | best=5 @ eval=2 | final_best=5 | evals=5000 | time=0.08s
[run 19] seed=1245586 (maximize) | best=5 @ eval=14 | final_best=5 | evals=5000 | time=0.08s

F23 d=25 final_best: n=20 | min=5 | max=5 | mean=5 | median=5 | std=0
==============================================================
GA on PBO F23 (dim=36) | runs=20 | budget=5000 | maximize=True
==============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1246567 (maximize) | best=6 @ eval=145 | final_best=6 | evals=5000 | time=0.09s
[run 01] seed=1246568 (maximize) | best=6 @ eval=201 | final_best=6 | evals=5000 | time=0.09s
[run 02] seed=1246569 (maximize) | best=6 @ eval=239 | final_best=6 | evals=5000 | time=0.09s
[run 03] seed=1246570 (maximize) | best=6 @ eval=118 | final_best=6 | evals=5000 | time=0.09s
[run 04] seed=1246571 (maximize) | best=6 @ eval=207 | final_best=6 | evals=5000 | time=0.09s
[run 05] seed=1246572 (maximize) | best=6 @ eval=70 | final_best=6 | evals=5000 | time=0.09s
[run 06] seed=1246573 (maximize) | best=6 @ eval=82 | final_best=6 | evals=5000 | time=0.09s
[run 07] seed=1246574 (maximize) | best=6 @ eval=319 | final_best=6 | evals=5000 | time=0.09s
[run 08] seed=1246575 (maximize) | best=6 @ eval=321 | final_best=6 | evals=5000 | time=0.09s
[run 09] seed=1246576 (maximize) | best=6 @ eval=236 | final_best=6 | evals=5000 | time=0.09s
[run 10] seed=1246577 (maximize) | best=0 @ eval=14 | final_best=0 | evals=5000 | time=0.09s
[run 11] seed=1246578 (maximize) | best=6 @ eval=251 | final_best=6 | evals=5000 | time=0.08s
[run 12] seed=1246579 (maximize) | best=6 @ eval=159 | final_best=6 | evals=5000 | time=0.08s
[run 13] seed=1246580 (maximize) | best=6 @ eval=39 | final_best=6 | evals=5000 | time=0.08s
[run 14] seed=1246581 (maximize) | best=6 @ eval=49 | final_best=6 | evals=5000 | time=0.08s
[run 15] seed=1246582 (maximize) | best=6 @ eval=1964 | final_best=6 | evals=5000 | time=0.09s
[run 16] seed=1246583 (maximize) | best=6 @ eval=11 | final_best=6 | evals=5000 | time=0.09s
[run 17] seed=1246584 (maximize) | best=6 @ eval=12 | final_best=6 | evals=5000 | time=0.09s
[run 18] seed=1246585 (maximize) | best=6 @ eval=13 | final_best=6 | evals=5000 | time=0.09s
[run 19] seed=1246586 (maximize) | best=6 @ eval=6 | final_best=6 | evals=5000 | time=0.09s

F23 d=36 final_best: n=20 | min=0 | max=6 | mean=5.7 | median=6 | std=1.34164
==============================================================
GA on PBO F23 (dim=49) | runs=20 | budget=5000 | maximize=True
==============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1247567 (maximize) | best=7 @ eval=139 | final_best=7 | evals=5000 | time=0.09s
[run 01] seed=1247568 (maximize) | best=7 @ eval=57 | final_best=7 | evals=5000 | time=0.09s
[run 02] seed=1247569 (maximize) | best=7 @ eval=537 | final_best=7 | evals=5000 | time=0.09s
[run 03] seed=1247570 (maximize) | best=7 @ eval=100 | final_best=7 | evals=5000 | time=0.10s
[run 04] seed=1247571 (maximize) | best=7 @ eval=233 | final_best=7 | evals=5000 | time=0.10s
[run 05] seed=1247572 (maximize) | best=7 @ eval=17 | final_best=7 | evals=5000 | time=0.09s
[run 06] seed=1247573 (maximize) | best=7 @ eval=6 | final_best=7 | evals=5000 | time=0.09s
[run 07] seed=1247574 (maximize) | best=7 @ eval=91 | final_best=7 | evals=5000 | time=0.10s
[run 08] seed=1247575 (maximize) | best=7 @ eval=116 | final_best=7 | evals=5000 | time=0.09s
[run 09] seed=1247576 (maximize) | best=7 @ eval=54 | final_best=7 | evals=5000 | time=0.09s
[run 10] seed=1247577 (maximize) | best=7 @ eval=190 | final_best=7 | evals=5000 | time=0.09s
[run 11] seed=1247578 (maximize) | best=7 @ eval=163 | final_best=7 | evals=5000 | time=0.09s
[run 12] seed=1247579 (maximize) | best=7 @ eval=107 | final_best=7 | evals=5000 | time=0.09s
[run 13] seed=1247580 (maximize) | best=7 @ eval=335 | final_best=7 | evals=5000 | time=0.09s
[run 14] seed=1247581 (maximize) | best=7 @ eval=6 | final_best=7 | evals=5000 | time=0.09s
[run 15] seed=1247582 (maximize) | best=7 @ eval=184 | final_best=7 | evals=5000 | time=0.09s
[run 16] seed=1247583 (maximize) | best=7 @ eval=114 | final_best=7 | evals=5000 | time=0.09s
[run 17] seed=1247584 (maximize) | best=7 @ eval=60 | final_best=7 | evals=5000 | time=0.09s
[run 18] seed=1247585 (maximize) | best=7 @ eval=99 | final_best=7 | evals=5000 | time=0.09s
[run 19] seed=1247586 (maximize) | best=7 @ eval=308 | final_best=7 | evals=5000 | time=0.09s

F23 d=49 final_best: n=20 | min=7 | max=7 | mean=7 | median=7 | std=0
==============================================================
GA on PBO F23 (dim=64) | runs=20 | budget=5000 | maximize=True
==============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1248567 (maximize) | best=8 @ eval=369 | final_best=8 | evals=5000 | time=0.09s
[run 01] seed=1248568 (maximize) | best=8 @ eval=139 | final_best=8 | evals=5000 | time=0.09s
[run 02] seed=1248569 (maximize) | best=8 @ eval=889 | final_best=8 | evals=5000 | time=0.09s
[run 03] seed=1248570 (maximize) | best=8 @ eval=2677 | final_best=8 | evals=5000 | time=0.09s
[run 04] seed=1248571 (maximize) | best=8 @ eval=151 | final_best=8 | evals=5000 | time=0.09s
[run 05] seed=1248572 (maximize) | best=8 @ eval=167 | final_best=8 | evals=5000 | time=0.09s
[run 06] seed=1248573 (maximize) | best=8 @ eval=2475 | final_best=8 | evals=5000 | time=0.09s
[run 07] seed=1248574 (maximize) | best=8 @ eval=16 | final_best=8 | evals=5000 | time=0.09s
[run 08] seed=1248575 (maximize) | best=8 @ eval=151 | final_best=8 | evals=5000 | time=0.09s
[run 09] seed=1248576 (maximize) | best=8 @ eval=22 | final_best=8 | evals=5000 | time=0.09s
[run 10] seed=1248577 (maximize) | best=8 @ eval=282 | final_best=8 | evals=5000 | time=0.09s
[run 11] seed=1248578 (maximize) | best=8 @ eval=353 | final_best=8 | evals=5000 | time=0.09s
[run 12] seed=1248579 (maximize) | best=8 @ eval=203 | final_best=8 | evals=5000 | time=0.10s
[run 13] seed=1248580 (maximize) | best=8 @ eval=4552 | final_best=8 | evals=5000 | time=0.10s
[run 14] seed=1248581 (maximize) | best=8 @ eval=274 | final_best=8 | evals=5000 | time=0.09s
[run 15] seed=1248582 (maximize) | best=8 @ eval=156 | final_best=8 | evals=5000 | time=0.09s
[run 16] seed=1248583 (maximize) | best=8 @ eval=340 | final_best=8 | evals=5000 | time=0.09s
[run 17] seed=1248584 (maximize) | best=8 @ eval=338 | final_best=8 | evals=5000 | time=0.09s
[run 18] seed=1248585 (maximize) | best=0 @ eval=134 | final_best=0 | evals=5000 | time=0.11s
[run 19] seed=1248586 (maximize) | best=8 @ eval=141 | final_best=8 | evals=5000 | time=0.10s

F23 d=64 final_best: n=20 | min=0 | max=8 | mean=7.6 | median=8 | std=1.78885
==============================================================
GA on PBO F23 (dim=81) | runs=20 | budget=5000 | maximize=True
==============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1249567 (maximize) | best=9 @ eval=308 | final_best=9 | evals=5000 | time=0.11s
[run 01] seed=1249568 (maximize) | best=0 @ eval=152 | final_best=0 | evals=5000 | time=0.11s
[run 02] seed=1249569 (maximize) | best=9 @ eval=252 | final_best=9 | evals=5000 | time=0.11s
[run 03] seed=1249570 (maximize) | best=9 @ eval=107 | final_best=9 | evals=5000 | time=0.11s
[run 04] seed=1249571 (maximize) | best=9 @ eval=2327 | final_best=9 | evals=5000 | time=0.11s
[run 05] seed=1249572 (maximize) | best=9 @ eval=350 | final_best=9 | evals=5000 | time=0.11s
[run 06] seed=1249573 (maximize) | best=9 @ eval=472 | final_best=9 | evals=5000 | time=0.11s
[run 07] seed=1249574 (maximize) | best=0 @ eval=31 | final_best=0 | evals=5000 | time=0.11s
[run 08] seed=1249575 (maximize) | best=9 @ eval=556 | final_best=9 | evals=5000 | time=0.11s
[run 09] seed=1249576 (maximize) | best=9 @ eval=319 | final_best=9 | evals=5000 | time=0.11s
[run 10] seed=1249577 (maximize) | best=9 @ eval=193 | final_best=9 | evals=5000 | time=0.11s
[run 11] seed=1249578 (maximize) | best=9 @ eval=558 | final_best=9 | evals=5000 | time=0.11s
[run 12] seed=1249579 (maximize) | best=9 @ eval=281 | final_best=9 | evals=5000 | time=0.11s
[run 13] seed=1249580 (maximize) | best=9 @ eval=166 | final_best=9 | evals=5000 | time=0.11s
[run 14] seed=1249581 (maximize) | best=9 @ eval=305 | final_best=9 | evals=5000 | time=0.10s
[run 15] seed=1249582 (maximize) | best=9 @ eval=234 | final_best=9 | evals=5000 | time=0.10s
[run 16] seed=1249583 (maximize) | best=9 @ eval=85 | final_best=9 | evals=5000 | time=0.10s
[run 17] seed=1249584 (maximize) | best=9 @ eval=501 | final_best=9 | evals=5000 | time=0.10s
[run 18] seed=1249585 (maximize) | best=9 @ eval=607 | final_best=9 | evals=5000 | time=0.10s
[run 19] seed=1249586 (maximize) | best=0 @ eval=85 | final_best=0 | evals=5000 | time=0.10s

F23 d=81 final_best: n=20 | min=0 | max=9 | mean=7.65 | median=9 | std=3.29713
===============================================================
GA on PBO F23 (dim=100) | runs=20 | budget=5000 | maximize=True
===============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1250567 (maximize) | best=0 @ eval=91 | final_best=0 | evals=5000 | time=0.11s
[run 01] seed=1250568 (maximize) | best=10 @ eval=357 | final_best=10 | evals=5000 | time=0.11s
[run 02] seed=1250569 (maximize) | best=0 @ eval=75 | final_best=0 | evals=5000 | time=0.11s
[run 03] seed=1250570 (maximize) | best=10 @ eval=59 | final_best=10 | evals=5000 | time=0.11s
[run 04] seed=1250571 (maximize) | best=10 @ eval=3130 | final_best=10 | evals=5000 | time=0.11s
[run 05] seed=1250572 (maximize) | best=0 @ eval=205 | final_best=0 | evals=5000 | time=0.11s
[run 06] seed=1250573 (maximize) | best=0 @ eval=101 | final_best=0 | evals=5000 | time=0.11s
[run 07] seed=1250574 (maximize) | best=0 @ eval=142 | final_best=0 | evals=5000 | time=0.11s
[run 08] seed=1250575 (maximize) | best=0 @ eval=117 | final_best=0 | evals=5000 | time=0.11s
[run 09] seed=1250576 (maximize) | best=10 @ eval=363 | final_best=10 | evals=5000 | time=0.11s
[run 10] seed=1250577 (maximize) | best=10 @ eval=330 | final_best=10 | evals=5000 | time=0.11s
[run 11] seed=1250578 (maximize) | best=10 @ eval=312 | final_best=10 | evals=5000 | time=0.11s
[run 12] seed=1250579 (maximize) | best=10 @ eval=354 | final_best=10 | evals=5000 | time=0.11s
[run 13] seed=1250580 (maximize) | best=10 @ eval=410 | final_best=10 | evals=5000 | time=0.11s
[run 14] seed=1250581 (maximize) | best=10 @ eval=2021 | final_best=10 | evals=5000 | time=0.11s
[run 15] seed=1250582 (maximize) | best=0 @ eval=90 | final_best=0 | evals=5000 | time=0.11s
[run 16] seed=1250583 (maximize) | best=10 @ eval=259 | final_best=10 | evals=5000 | time=0.11s
[run 17] seed=1250584 (maximize) | best=0 @ eval=41 | final_best=0 | evals=5000 | time=0.11s
[run 18] seed=1250585 (maximize) | best=0 @ eval=232 | final_best=0 | evals=5000 | time=0.10s
[run 19] seed=1250586 (maximize) | best=10 @ eval=334 | final_best=10 | evals=5000 | time=0.10s

F23 d=100 final_best: n=20 | min=0 | max=10 | mean=5.5 | median=10 | std=5.10418
===============================================================
GA on PBO F23 (dim=121) | runs=20 | budget=5000 | maximize=True
===============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1251567 (maximize) | best=0 @ eval=200 | final_best=0 | evals=5000 | time=0.11s
[run 01] seed=1251568 (maximize) | best=11 @ eval=936 | final_best=11 | evals=5000 | time=0.11s
[run 02] seed=1251569 (maximize) | best=11 @ eval=2212 | final_best=11 | evals=5000 | time=0.11s
[run 03] seed=1251570 (maximize) | best=0 @ eval=172 | final_best=0 | evals=5000 | time=0.11s
[run 04] seed=1251571 (maximize) | best=11 @ eval=417 | final_best=11 | evals=5000 | time=0.11s
[run 05] seed=1251572 (maximize) | best=11 @ eval=541 | final_best=11 | evals=5000 | time=0.11s
[run 06] seed=1251573 (maximize) | best=0 @ eval=258 | final_best=0 | evals=5000 | time=0.11s
[run 07] seed=1251574 (maximize) | best=11 @ eval=1887 | final_best=11 | evals=5000 | time=0.11s
[run 08] seed=1251575 (maximize) | best=11 @ eval=757 | final_best=11 | evals=5000 | time=0.11s
[run 09] seed=1251576 (maximize) | best=0 @ eval=274 | final_best=0 | evals=5000 | time=0.11s
[run 10] seed=1251577 (maximize) | best=0 @ eval=363 | final_best=0 | evals=5000 | time=0.11s
[run 11] seed=1251578 (maximize) | best=11 @ eval=878 | final_best=11 | evals=5000 | time=0.11s
[run 12] seed=1251579 (maximize) | best=11 @ eval=493 | final_best=11 | evals=5000 | time=0.11s
[run 13] seed=1251580 (maximize) | best=11 @ eval=230 | final_best=11 | evals=5000 | time=0.11s
[run 14] seed=1251581 (maximize) | best=11 @ eval=969 | final_best=11 | evals=5000 | time=0.11s
[run 15] seed=1251582 (maximize) | best=0 @ eval=76 | final_best=0 | evals=5000 | time=0.11s
[run 16] seed=1251583 (maximize) | best=0 @ eval=168 | final_best=0 | evals=5000 | time=0.11s
[run 17] seed=1251584 (maximize) | best=11 @ eval=1941 | final_best=11 | evals=5000 | time=0.11s
[run 18] seed=1251585 (maximize) | best=11 @ eval=501 | final_best=11 | evals=5000 | time=0.11s
[run 19] seed=1251586 (maximize) | best=0 @ eval=200 | final_best=0 | evals=5000 | time=0.11s

F23 d=121 final_best: n=20 | min=0 | max=11 | mean=6.6 | median=11 | std=5.52887
===============================================================
GA on PBO F23 (dim=144) | runs=20 | budget=5000 | maximize=True
===============================================================
Params: {'pop_size': 80, 'tournament_k': 5, 'crossover_rate': 0.8359334351205238, 'mutation_rate': 0.021702729161426414, 'elite_frac': 0.01}
[run 00] seed=1252567 (maximize) | best=12 @ eval=1050 | final_best=12 | evals=5000 | time=0.12s
[run 01] seed=1252568 (maximize) | best=0 @ eval=129 | final_best=0 | evals=5000 | time=0.12s
[run 02] seed=1252569 (maximize) | best=0 @ eval=1503 | final_best=0 | evals=5000 | time=0.12s
[run 03] seed=1252570 (maximize) | best=0 @ eval=1169 | final_best=0 | evals=5000 | time=0.12s
[run 04] seed=1252571 (maximize) | best=0 @ eval=448 | final_best=0 | evals=5000 | time=0.12s
[run 05] seed=1252572 (maximize) | best=12 @ eval=764 | final_best=12 | evals=5000 | time=0.12s
[run 06] seed=1252573 (maximize) | best=0 @ eval=387 | final_best=0 | evals=5000 | time=0.12s
[run 07] seed=1252574 (maximize) | best=12 @ eval=612 | final_best=12 | evals=5000 | time=0.12s
[run 08] seed=1252575 (maximize) | best=0 @ eval=301 | final_best=0 | evals=5000 | time=0.12s
[run 09] seed=1252576 (maximize) | best=0 @ eval=270 | final_best=0 | evals=5000 | time=0.12s
[run 10] seed=1252577 (maximize) | best=12 @ eval=472 | final_best=12 | evals=5000 | time=0.12s
[run 11] seed=1252578 (maximize) | best=0 @ eval=545 | final_best=0 | evals=5000 | time=0.12s
[run 12] seed=1252579 (maximize) | best=0 @ eval=837 | final_best=0 | evals=5000 | time=0.12s
[run 13] seed=1252580 (maximize) | best=12 @ eval=1191 | final_best=12 | evals=5000 | time=0.12s
[run 14] seed=1252581 (maximize) | best=0 @ eval=326 | final_best=0 | evals=5000 | time=0.12s
[run 15] seed=1252582 (maximize) | best=12 @ eval=739 | final_best=12 | evals=5000 | time=0.12s
[run 16] seed=1252583 (maximize) | best=0 @ eval=608 | final_best=0 | evals=5000 | time=0.12s
[run 17] seed=1252584 (maximize) | best=12 @ eval=82 | final_best=12 | evals=5000 | time=0.12s
[run 18] seed=1252585 (maximize) | best=12 @ eval=339 | final_best=12 | evals=5000 | time=0.11s
[run 19] seed=1252586 (maximize) | best=12 @ eval=1201 | final_best=12 | evals=5000 | time=0.12s

F23 d=144 final_best: n=20 | min=0 | max=12 | mean=5.4 | median=0 | std=6.12501
(venv) rwik@PROENG93:~/leiden_projects/ea_assignment/final$ python s4822285_ES_strategies.py --pairs 1:10 3:30 5:50 --both
============================================================================
Classic ES on BBOB F23 (Katsuura) d=10 | cfg=(1,10) | runs=20 | budget=50000
============================================================================
Config: ESConfig(mu=1, lam=10, plus=False, sigma0=2.0, stagnation_gens=80, max_restarts=4)
[run 00] seed=7654321 (minimize) | best=7.02311 @ eval=49792 | final_best=7.02311 | evals=50000 | time=0.16s
[run 01] seed=7654322 (minimize) | best=7.4228 @ eval=48459 | final_best=7.4228 | evals=50000 | time=0.16s
[run 02] seed=7654323 (minimize) | best=8.19899 @ eval=49780 | final_best=8.19899 | evals=50000 | time=0.16s
[run 03] seed=7654324 (minimize) | best=7.0428 @ eval=49800 | final_best=7.0428 | evals=50000 | time=0.16s
[run 04] seed=7654325 (minimize) | best=7.25051 @ eval=49478 | final_best=7.25051 | evals=50000 | time=0.16s
[run 05] seed=7654326 (minimize) | best=7.06969 @ eval=49980 | final_best=7.06969 | evals=50000 | time=0.16s
[run 06] seed=7654327 (minimize) | best=7.10257 @ eval=49949 | final_best=7.10257 | evals=50000 | time=0.16s
[run 07] seed=7654328 (minimize) | best=7.25479 @ eval=49968 | final_best=7.25479 | evals=50000 | time=0.16s
[run 08] seed=7654329 (minimize) | best=7.12104 @ eval=39875 | final_best=7.12104 | evals=50000 | time=0.16s
[run 09] seed=7654330 (minimize) | best=7.08715 @ eval=49895 | final_best=7.08715 | evals=50000 | time=0.16s
[run 10] seed=7654331 (minimize) | best=7.16142 @ eval=49849 | final_best=7.16142 | evals=50000 | time=0.16s
[run 11] seed=7654332 (minimize) | best=7.19432 @ eval=49986 | final_best=7.19432 | evals=50000 | time=0.16s
[run 12] seed=7654333 (minimize) | best=7.57533 @ eval=49836 | final_best=7.57533 | evals=50000 | time=0.16s
[run 13] seed=7654334 (minimize) | best=7.56391 @ eval=49495 | final_best=7.56391 | evals=50000 | time=0.16s
[run 14] seed=7654335 (minimize) | best=7.13968 @ eval=32166 | final_best=7.13968 | evals=50000 | time=0.16s
[run 15] seed=7654336 (minimize) | best=7.44813 @ eval=49971 | final_best=7.44813 | evals=50000 | time=0.16s
[run 16] seed=7654337 (minimize) | best=6.96008 @ eval=33602 | final_best=6.96008 | evals=50000 | time=0.16s
[run 17] seed=7654338 (minimize) | best=7.40083 @ eval=49591 | final_best=7.40083 | evals=50000 | time=0.16s
[run 18] seed=7654339 (minimize) | best=7.3909 @ eval=49968 | final_best=7.3909 | evals=50000 | time=0.16s
[run 19] seed=7654340 (minimize) | best=6.95692 @ eval=49766 | final_best=6.95692 | evals=50000 | time=0.16s

FINAL BEST (lower is better): n=20 | min=6.95692 | max=8.19899 | mean=7.26825 | median=7.17787 | std=0.289729

============================================================================
Classic ES on BBOB F23 (Katsuura) d=10 | cfg=(1+10) | runs=20 | budget=50000
============================================================================
Config: ESConfig(mu=1, lam=10, plus=True, sigma0=2.0, stagnation_gens=80, max_restarts=4)
[run 00] seed=7655321 (minimize) | best=7.15035 @ eval=19089 | final_best=7.15035 | evals=50000 | time=0.17s
[run 01] seed=7655322 (minimize) | best=7.23484 @ eval=7227 | final_best=7.23484 | evals=50000 | time=0.17s
[run 02] seed=7655323 (minimize) | best=7.07319 @ eval=7117 | final_best=7.07319 | evals=50000 | time=0.17s
[run 03] seed=7655324 (minimize) | best=7.1243 @ eval=14982 | final_best=7.1243 | evals=50000 | time=0.17s
[run 04] seed=7655325 (minimize) | best=7.34339 @ eval=14994 | final_best=7.34339 | evals=50000 | time=0.18s
[run 05] seed=7655326 (minimize) | best=6.99259 @ eval=10202 | final_best=6.99259 | evals=50000 | time=0.18s
[run 06] seed=7655327 (minimize) | best=7.04909 @ eval=18966 | final_best=7.04909 | evals=50000 | time=0.18s
[run 07] seed=7655328 (minimize) | best=7.21055 @ eval=14244 | final_best=7.21055 | evals=50000 | time=0.17s
[run 08] seed=7655329 (minimize) | best=7.06756 @ eval=33388 | final_best=7.06756 | evals=50000 | time=0.17s
[run 09] seed=7655330 (minimize) | best=7.1851 @ eval=7838 | final_best=7.1851 | evals=50000 | time=0.17s
[run 10] seed=7655331 (minimize) | best=7.1529 @ eval=7185 | final_best=7.1529 | evals=50000 | time=0.17s
[run 11] seed=7655332 (minimize) | best=7.19095 @ eval=6723 | final_best=7.19095 | evals=50000 | time=0.17s
[run 12] seed=7655333 (minimize) | best=7.18314 @ eval=6591 | final_best=7.18314 | evals=50000 | time=0.17s
[run 13] seed=7655334 (minimize) | best=7.27487 @ eval=6429 | final_best=7.27487 | evals=50000 | time=0.18s
[run 14] seed=7655335 (minimize) | best=7.07992 @ eval=15694 | final_best=7.07992 | evals=50000 | time=0.18s
[run 15] seed=7655336 (minimize) | best=7.04063 @ eval=16453 | final_best=7.04063 | evals=50000 | time=0.18s
[run 16] seed=7655337 (minimize) | best=7.32343 @ eval=7647 | final_best=7.32343 | evals=50000 | time=0.18s
[run 17] seed=7655338 (minimize) | best=7.07435 @ eval=7264 | final_best=7.07435 | evals=50000 | time=0.19s
[run 18] seed=7655339 (minimize) | best=7.2828 @ eval=8959 | final_best=7.2828 | evals=50000 | time=0.20s
[run 19] seed=7655340 (minimize) | best=7.04279 @ eval=28959 | final_best=7.04279 | evals=50000 | time=0.19s

FINAL BEST (lower is better): n=20 | min=6.99259 | max=7.34339 | mean=7.15384 | median=7.15163 | std=0.102006

============================================================================
Classic ES on BBOB F23 (Katsuura) d=10 | cfg=(3,30) | runs=20 | budget=50000
============================================================================
Config: ESConfig(mu=3, lam=30, plus=False, sigma0=2.0, stagnation_gens=80, max_restarts=4)
[run 00] seed=7656321 (minimize) | best=6.9436 @ eval=49756 | final_best=6.9436 | evals=50000 | time=0.13s
[run 01] seed=7656322 (minimize) | best=6.97727 @ eval=49932 | final_best=6.97727 | evals=50000 | time=0.13s
[run 02] seed=7656323 (minimize) | best=6.89092 @ eval=41347 | final_best=6.89092 | evals=50000 | time=0.12s
[run 03] seed=7656324 (minimize) | best=6.9264 @ eval=49278 | final_best=6.9264 | evals=50000 | time=0.12s
[run 04] seed=7656325 (minimize) | best=7.00554 @ eval=48460 | final_best=7.00554 | evals=50000 | time=0.12s
[run 05] seed=7656326 (minimize) | best=6.88141 @ eval=49622 | final_best=6.88141 | evals=50000 | time=0.12s
[run 06] seed=7656327 (minimize) | best=6.88695 @ eval=49515 | final_best=6.88695 | evals=50000 | time=0.12s
[run 07] seed=7656328 (minimize) | best=7.01014 @ eval=49864 | final_best=7.01014 | evals=50000 | time=0.11s
[run 08] seed=7656329 (minimize) | best=6.87063 @ eval=45042 | final_best=6.87063 | evals=50000 | time=0.11s
[run 09] seed=7656330 (minimize) | best=6.94885 @ eval=49864 | final_best=6.94885 | evals=50000 | time=0.12s
[run 10] seed=7656331 (minimize) | best=6.90811 @ eval=48348 | final_best=6.90811 | evals=50000 | time=0.12s
[run 11] seed=7656332 (minimize) | best=6.87709 @ eval=36959 | final_best=6.87709 | evals=50000 | time=0.12s
[run 12] seed=7656333 (minimize) | best=7.04769 @ eval=47501 | final_best=7.04769 | evals=50000 | time=0.12s
[run 13] seed=7656334 (minimize) | best=6.93288 @ eval=38409 | final_best=6.93288 | evals=50000 | time=0.12s
[run 14] seed=7656335 (minimize) | best=6.96806 @ eval=49445 | final_best=6.96806 | evals=50000 | time=0.12s
[run 15] seed=7656336 (minimize) | best=7.0614 @ eval=34798 | final_best=7.0614 | evals=50000 | time=0.12s
[run 16] seed=7656337 (minimize) | best=6.87626 @ eval=49909 | final_best=6.87626 | evals=50000 | time=0.12s
[run 17] seed=7656338 (minimize) | best=6.92156 @ eval=49006 | final_best=6.92156 | evals=50000 | time=0.12s
[run 18] seed=7656339 (minimize) | best=6.92142 @ eval=49665 | final_best=6.92142 | evals=50000 | time=0.11s
[run 19] seed=7656340 (minimize) | best=6.93827 @ eval=36287 | final_best=6.93827 | evals=50000 | time=0.11s

FINAL BEST (lower is better): n=20 | min=6.87063 | max=7.0614 | mean=6.93972 | median=6.92964 | std=0.0567766

============================================================================
Classic ES on BBOB F23 (Katsuura) d=10 | cfg=(3+30) | runs=20 | budget=50000
============================================================================
Config: ESConfig(mu=3, lam=30, plus=True, sigma0=2.0, stagnation_gens=80, max_restarts=4)
[run 00] seed=7657321 (minimize) | best=8.29392 @ eval=2829 | final_best=8.29392 | evals=50000 | time=0.13s
[run 01] seed=7657322 (minimize) | best=8.89933 @ eval=12026 | final_best=8.89933 | evals=50000 | time=0.13s
[run 02] seed=7657323 (minimize) | best=8.29195 @ eval=6170 | final_best=8.29195 | evals=50000 | time=0.13s
[run 03] seed=7657324 (minimize) | best=9.09183 @ eval=6034 | final_best=9.09183 | evals=50000 | time=0.13s
[run 04] seed=7657325 (minimize) | best=8.76874 @ eval=6162 | final_best=8.76874 | evals=50000 | time=0.12s
[run 05] seed=7657326 (minimize) | best=8.82212 @ eval=1131 | final_best=8.82212 | evals=50000 | time=0.12s
[run 06] seed=7657327 (minimize) | best=8.39405 @ eval=5710 | final_best=8.39405 | evals=50000 | time=0.12s
[run 07] seed=7657328 (minimize) | best=8.71615 @ eval=2401 | final_best=8.71615 | evals=50000 | time=0.13s
[run 08] seed=7657329 (minimize) | best=8.35956 @ eval=835 | final_best=8.35956 | evals=50000 | time=0.13s
[run 09] seed=7657330 (minimize) | best=8.04255 @ eval=637 | final_best=8.04255 | evals=50000 | time=0.13s
[run 10] seed=7657331 (minimize) | best=8.53905 @ eval=5483 | final_best=8.53905 | evals=50000 | time=0.13s
[run 11] seed=7657332 (minimize) | best=8.74388 @ eval=5749 | final_best=8.74388 | evals=50000 | time=0.13s
[run 12] seed=7657333 (minimize) | best=8.25182 @ eval=968 | final_best=8.25182 | evals=50000 | time=0.13s
[run 13] seed=7657334 (minimize) | best=8.21683 @ eval=3583 | final_best=8.21683 | evals=50000 | time=0.13s
[run 14] seed=7657335 (minimize) | best=8.96808 @ eval=8876 | final_best=8.96808 | evals=50000 | time=0.12s
[run 15] seed=7657336 (minimize) | best=8.49594 @ eval=4556 | final_best=8.49594 | evals=50000 | time=0.12s
[run 16] seed=7657337 (minimize) | best=9.11023 @ eval=2338 | final_best=9.11023 | evals=50000 | time=0.12s
[run 17] seed=7657338 (minimize) | best=8.88524 @ eval=4994 | final_best=8.88524 | evals=50000 | time=0.13s
[run 18] seed=7657339 (minimize) | best=9.41505 @ eval=335 | final_best=9.41505 | evals=50000 | time=0.13s
[run 19] seed=7657340 (minimize) | best=8.67147 @ eval=7070 | final_best=8.67147 | evals=50000 | time=0.13s

FINAL BEST (lower is better): n=20 | min=8.04255 | max=9.41505 | mean=8.64889 | median=8.69381 | std=0.357214

============================================================================
Classic ES on BBOB F23 (Katsuura) d=10 | cfg=(5,50) | runs=20 | budget=50000
============================================================================
Config: ESConfig(mu=5, lam=50, plus=False, sigma0=2.0, stagnation_gens=80, max_restarts=4)
[run 00] seed=7658321 (minimize) | best=6.89544 @ eval=48834 | final_best=6.89544 | evals=50000 | time=0.11s
[run 01] seed=7658322 (minimize) | best=6.88708 @ eval=49821 | final_best=6.88708 | evals=50000 | time=0.11s
[run 02] seed=7658323 (minimize) | best=6.88555 @ eval=49180 | final_best=6.88555 | evals=50000 | time=0.11s
[run 03] seed=7658324 (minimize) | best=6.88875 @ eval=49954 | final_best=6.88875 | evals=50000 | time=0.10s
[run 04] seed=7658325 (minimize) | best=6.9466 @ eval=39146 | final_best=6.9466 | evals=50000 | time=0.11s
[run 05] seed=7658326 (minimize) | best=7.00354 @ eval=41123 | final_best=7.00354 | evals=50000 | time=0.11s
[run 06] seed=7658327 (minimize) | best=6.88802 @ eval=47724 | final_best=6.88802 | evals=50000 | time=0.11s
[run 07] seed=7658328 (minimize) | best=6.87834 @ eval=39916 | final_best=6.87834 | evals=50000 | time=0.11s
[run 08] seed=7658329 (minimize) | best=6.88825 @ eval=48633 | final_best=6.88825 | evals=50000 | time=0.11s
[run 09] seed=7658330 (minimize) | best=6.87374 @ eval=49799 | final_best=6.87374 | evals=50000 | time=0.11s
[run 10] seed=7658331 (minimize) | best=6.95905 @ eval=41872 | final_best=6.95905 | evals=50000 | time=0.11s
[run 11] seed=7658332 (minimize) | best=6.97472 @ eval=36132 | final_best=6.97472 | evals=50000 | time=0.10s
[run 12] seed=7658333 (minimize) | best=6.88793 @ eval=44709 | final_best=6.88793 | evals=50000 | time=0.10s
[run 13] seed=7658334 (minimize) | best=6.88491 @ eval=49865 | final_best=6.88491 | evals=50000 | time=0.11s
[run 14] seed=7658335 (minimize) | best=6.90544 @ eval=48643 | final_best=6.90544 | evals=50000 | time=0.11s
[run 15] seed=7658336 (minimize) | best=6.94125 @ eval=49765 | final_best=6.94125 | evals=50000 | time=0.11s
[run 16] seed=7658337 (minimize) | best=6.89778 @ eval=40907 | final_best=6.89778 | evals=50000 | time=0.11s
[run 17] seed=7658338 (minimize) | best=6.94458 @ eval=49610 | final_best=6.94458 | evals=50000 | time=0.11s
[run 18] seed=7658339 (minimize) | best=6.88881 @ eval=39993 | final_best=6.88881 | evals=50000 | time=0.11s
[run 19] seed=7658340 (minimize) | best=6.90092 @ eval=49505 | final_best=6.90092 | evals=50000 | time=0.11s

FINAL BEST (lower is better): n=20 | min=6.87374 | max=7.00354 | mean=6.91103 | median=6.89212 | std=0.0367876

============================================================================
Classic ES on BBOB F23 (Katsuura) d=10 | cfg=(5+50) | runs=20 | budget=50000
============================================================================
Config: ESConfig(mu=5, lam=50, plus=True, sigma0=2.0, stagnation_gens=80, max_restarts=4)
[run 00] seed=7659321 (minimize) | best=8.93564 @ eval=4898 | final_best=8.93564 | evals=50000 | time=0.12s
[run 01] seed=7659322 (minimize) | best=7.87579 @ eval=16086 | final_best=7.87579 | evals=50000 | time=0.11s
[run 02] seed=7659323 (minimize) | best=8.99985 @ eval=2781 | final_best=8.99985 | evals=50000 | time=0.11s
[run 03] seed=7659324 (minimize) | best=8.52849 @ eval=11240 | final_best=8.52849 | evals=50000 | time=0.11s
[run 04] seed=7659325 (minimize) | best=8.12074 @ eval=16862 | final_best=8.12074 | evals=50000 | time=0.11s
[run 05] seed=7659326 (minimize) | best=8.5291 @ eval=2851 | final_best=8.5291 | evals=50000 | time=0.11s
[run 06] seed=7659327 (minimize) | best=8.7059 @ eval=237 | final_best=8.7059 | evals=50000 | time=0.11s
[run 07] seed=7659328 (minimize) | best=7.91057 @ eval=13718 | final_best=7.91057 | evals=50000 | time=0.11s
[run 08] seed=7659329 (minimize) | best=8.48969 @ eval=3265 | final_best=8.48969 | evals=50000 | time=0.12s
[run 09] seed=7659330 (minimize) | best=8.94845 @ eval=1767 | final_best=8.94845 | evals=50000 | time=0.12s
[run 10] seed=7659331 (minimize) | best=8.36875 @ eval=27807 | final_best=8.36875 | evals=50000 | time=0.12s
[run 11] seed=7659332 (minimize) | best=8.67101 @ eval=3368 | final_best=8.67101 | evals=50000 | time=0.12s
[run 12] seed=7659333 (minimize) | best=8.57658 @ eval=14411 | final_best=8.57658 | evals=50000 | time=0.11s
[run 13] seed=7659334 (minimize) | best=8.71306 @ eval=22 | final_best=8.71306 | evals=50000 | time=0.12s
[run 14] seed=7659335 (minimize) | best=8.81889 @ eval=9529 | final_best=8.81889 | evals=50000 | time=0.11s
[run 15] seed=7659336 (minimize) | best=8.72223 @ eval=2120 | final_best=8.72223 | evals=50000 | time=0.11s
[run 16] seed=7659337 (minimize) | best=8.85384 @ eval=8053 | final_best=8.85384 | evals=50000 | time=0.11s
[run 17] seed=7659338 (minimize) | best=8.55168 @ eval=2571 | final_best=8.55168 | evals=50000 | time=0.11s
[run 18] seed=7659339 (minimize) | best=9.11482 @ eval=3656 | final_best=9.11482 | evals=50000 | time=0.11s
[run 19] seed=7659340 (minimize) | best=7.9169 @ eval=9764 | final_best=7.9169 | evals=50000 | time=0.11s

FINAL BEST (lower is better): n=20 | min=7.87579 | max=9.11482 | mean=8.5676 | median=8.62379 | std=0.367717

(venv) rwik@PROENG93:~/leiden_projects/ea_assignment/final$ python plot_dim_ga.py --data-root data --out plots_ga_dims --algo-prefix s4822285_GA
/home/rwik/leiden_projects/ea_assignment/final/plot_dim_ga.py:420: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([x[np.isfinite(x)] for x in finals_by_dim], labels=[str(d) for d in dims], showfliers=False)
[OK] LABS (F18) | algo=s4822285_GA: wrote dim-effect plots to plots_ga_dims
/home/rwik/leiden_projects/ea_assignment/final/plot_dim_ga.py:420: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([x[np.isfinite(x)] for x in finals_by_dim], labels=[str(d) for d in dims], showfliers=False)
[OK] NQueens (F23) | algo=s4822285_GA: wrote dim-effect plots to plots_ga_dims

All dimension-effect plots saved in: /home/rwik/leiden_projects/ea_assignment/final/plots_ga_dims
(venv) rwik@PROENG93:~/leiden_projects/ea_assignment/final$ python plot_es_strategy_compare.py --data-root data --outdir plots_es --algo-prefix s4822285_ES_
/home/rwik/leiden_projects/ea_assignment/final/plot_es_strategy_compare.py:339: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data2, labels=labels2, showfliers=False)
[OK] ES plots for F23 d=10: plots_es

All ES plots saved in: /home/rwik/leiden_projects/ea_assignment/final/plots_es
(venv) rwik@PROENG93:~/leiden_projects/ea_assignment/final$


Final scripts to run:

  429  python s4822285_tuning.py
  430  python s4822285_GA.py --multi-dim
  431  python s4822285_ES_strategies.py --pairs 1:10 3:30 5:50 --both
  432  python plot_dim_ga.py --data-root data --out plots_ga_dims --algo-prefix s4822285_GA
  433  python plot_es_strategy_compare.py --data-root data --outdir plots_es --algo-prefix s4822285_ES_