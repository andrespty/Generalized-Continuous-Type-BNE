# Generalized-Continuous-Type-BNE

PyTorch code for neural approximation of generalized Bayesian Nash equilibria (GBNE)
with continuous types and actions, using a Fischer-Burmeister merit function.
Constraints may depend on own type and on rivals' strategies (Tao & Xu, 2025).
Plain BNE problems (Ho Lee & Li, 2026) are the special case with box constraints only.

## Layout

```
gbne/
├── games/        Game interface, constraints, type distributions, concrete games, registry
├── models/       strategy, multiplier (capped sigmoid / softplus), slack networks, StrategyProfile
├── merit/        FB function, opponent-expectation estimators (L0/L1/L2/quadrature), generic merit
├── training/     samplers, optimizers, trainer, checkpoints
├── evaluation/   best response, certification (C4), bounds (C2/C3), metrics, ground truth
├── reporting/    tables and figures
└── utils/        seeding, device, config I/O
configs/          one YAML per experiment
scripts/          train.py, evaluate.py, sweep.py
tests/            sanity checks
notebooks/        call into gbne only
outputs/          runs, tables, images (gitignored)
```

Adding a problem: subclass `gbne.games.base.Game`, register it, add its ground truth.
