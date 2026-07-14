# Handoff: Sprint 3 — train_lstm.py (implemented by leader, Cam unavailable)

**Branch:** `feature/cam/train-lstm`
**Date:** 2026-07-11

## What was built

`scripts/train_lstm.py` is fully implemented — the training loop, `compute_epoch_metrics`, and `save_checkpoint`, per the spec already documented in `handoffs/sprint3_tasks.md` §3. Cam wasn't available, so I (leader) implemented it directly on `feature/cam/train-lstm` to keep Sprint 3 moving, rather than leaving it blocked.

## How to run

```bash
.venv/bin/python scripts/train_lstm.py --epochs 25 --batch-size 32 --lr 1e-4
```
(all CLI defaults match the spec: `lr-step-size=3000`, `lr-gamma=0.9`, `sequence-length=50`)

## Dependencies / merge order

This branch requires `feature/leader/fix-3d-physics-loss` merged into `develop` first (or physics_loss/initial_loss will be silently wrong — see that branch's commit message and `src/losses.py`'s module docstring). It does **not** require Kieu's `BatteryPINN_Shen` to be finished to merge — the code imports and calls it correctly by contract, it just can't be run for real (with meaningful results) until that model's `forward()`/`shared_parameters()` are implemented.

## Verification notes (important for Cam to know)

Since `BatteryPINN_Shen` isn't implemented yet, I verified this against a structurally-correct dummy model (a real `nn.LSTM`, not a shortcut) standing in for it — confirmed: no crashes, `alpha`/`beta` correctly hardcode to `0.0`/`1.0` during the epoch<5 warm-up (proves `epoch` is threaded through `loss_fn` correctly), the LR schedule decays consistent with per-iteration `StepLR` stepping, and checkpoints save/reload correctly.

**This has not been run against the real `BatteryPINN_Shen`** — once Kieu's model is done, please do a real training run and sanity-check the RMSE/MAE trend and final numbers before trusting them for the pitching proof. The mechanics are verified; the actual LSTM's learning behavior is not.

## Integration notes

- `train()`'s per-epoch metrics use an extra `torch.no_grad()` forward pass on `model(x)` to get `y_pred`, since `AdaptivePINNLoss.forward()` only returns `(total_loss, log_dict)`, not predictions directly. This was a deliberate choice to avoid touching `src/losses.py`'s return signature — if you'd prefer a different approach (e.g. extending the log_dict), raise it with the leader rather than changing it solo.
- `compute_epoch_metrics` reuses `scripts/evaluate.py`'s `compute_metrics` — if you change that function's signature, this will need updating too.
