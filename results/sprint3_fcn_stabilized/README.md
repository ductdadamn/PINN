# Task 0 Verification — FCN + Stabilized Loss (No LSTM Yet)

**Purpose:** confirm the Task 0 loss-stabilization hotfix (warm-up + clamping, `src/losses.py`) actually fixes the Sprint 2 degenerate collapse, using the *same* point-wise FCN architecture as Sprint 2 — before moving on to the LSTM. This is a verification artifact, not the Sprint 3 final result.

**Reproduce:** `git checkout develop` (once `feature/leader/loss-stabilization-hotfix` is merged), then:
```bash
.venv/bin/python scripts/train_fcn.py --epochs 20 --batch-size 32 --lr 1e-3 --checkpoint-path outputs/checkpoints/fcn_cho2022_stabilized.pth
.venv/bin/python scripts/evaluate.py --checkpoint-path outputs/checkpoints/fcn_cho2022_stabilized.pth
```

## Result

```
RMSE: 0.5639  MAE: 0.5300
```

## ⚠️ Read the plot before reading the number

`0.56°C` is *below* the 0.6°C academic target set for the LSTM — but don't read that as "the FCN alone already solves this." Looking at `fcn_stabilized_predicted_vs_true.png`: the predicted line is still nearly **flat**, hovering around 27.75–27.8°C, while the true temperature swings 26.8–28.0°C with real structure (visible drive-cycle-correlated bumps). The good RMSE is largely an artifact of FUDS's true temperature staying in a narrow band with its mean/mode pulled toward the upper end — a near-constant prediction near that value scores deceptively well on RMSE/MAE without actually tracking any dynamics.

**What this does confirm:** the hotfix fixed the *worst* failure mode — total collapse to the wrong constant (25.00°C = `T_amb`, unrelated to the data at all). It does **not** confirm the point-wise FCN can track FUDS's actual dynamics. If anything, this strengthens the case for Sprint 3: even with a stable loss, a model that only sees one timestep at a time (no history) has no mechanism to predict a trajectory-dependent quantity like temperature response — which is exactly the gap `BatteryPINN_Shen` (the sequence-wise LSTM) is meant to close.

Compare against:
- `results/sprint2_fcn_baseline/` — the original, more severe collapse (pre-hotfix, RMSE ~2.27, flat at 25.00°C).
- `results/sprint3_lstm/` (once available) — the actual Sprint 3 deliverable, expected to show real tracking, not just a better-placed flat line.
