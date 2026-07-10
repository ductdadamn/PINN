# Sprint 2 — FCN Baseline (Cho 2022) Skeleton

**Goal:** train a plain FCN (per Cho et al. 2022) on the DST drive cycle and evaluate it on FUDS, to demonstrate that a point-wise FCN fails to generalize to an unseen dynamic-load profile — motivating later work (sequence models / proper physics constraints).

**Base branch:** `develop` (already contains Task 1's `src/data_loader.py`, merged from `feature/leader/data-loader-ocv`).

Each person should branch off `develop` and work independently; the files below don't overlap, so there should be no merge conflicts if everyone stays inside their own file.

| File | Owner | Branch to create |
|---|---|---|
| `src/losses.py` | Leader | `feature/leader/adaptive-pinn-loss` |
| `src/models/fcn_cho2022.py` | Kieu | `feature/kieu/fcn-cho2022` |
| `scripts/train_fcn.py` | Cam | `feature/cam/train-loop` |
| `scripts/evaluate.py` | Kem | `feature/kem/evaluate` |

All four files currently exist as **skeletons**: fully-typed signatures, detailed docstrings, and CLI wiring (where relevant) are already in place and verified working (imports resolve, models/losses construct, `--help` runs). The actual computation inside each method is `raise NotImplementedError(...)` — that's what each of you fills in.

## Integration contract (the shapes/types everyone must respect)

This is what lets all 4 pieces plug together without renegotiating interfaces mid-sprint:

- **Features** (from `src.data_loader.BatteryDataset`): `Tensor` shape `(batch, 4)`, order `[Time, Current, Voltage, OCV_Estimated]`, Min-Max scaled to `[0,1]`.
- **Target**: `Tensor` shape `(batch, 1)`, Temperature, Min-Max scaled to `[0,1]` (scaler fit on DST train set only — see `src/data_loader.py`'s handoff for why).
- **`BatteryPINN_Cho2022.forward(x)`**: input `(batch, 4)` → output `(batch, 1)`. Same shape as the target tensor, so it can be compared directly.
- **`AdaptivePINNLoss.forward(y_pred, y_true, *physics_inputs)`**: returns `(total_loss: Tensor, log_dict: Dict[str, float])`. `log_dict` includes at least `data_loss`, `physics_loss`, `alpha`, `beta` — `train_fcn.py` should log these per epoch.
- **Checkpoint**: `scripts/train_fcn.py` saves to `outputs/checkpoints/fcn_cho2022.pth` (via `model.state_dict()`); `scripts/evaluate.py` loads from the same default path. Both are `outputs/` (gitignored) — checkpoints are local artifacts, not committed.

## Per-file detail

### `src/losses.py` — Leader

`AdaptivePINNLoss(nn.Module)`:
- `compute_data_loss(y_pred, y_true) -> Tensor` — differentiable scalar.
- `compute_physics_loss(*physics_inputs) -> Tensor` — differentiable scalar; exact signature still open pending the PDE residual spec from the Strategic Planner.
- `_max_abs_grad(loss, params) -> Tensor` — `max(|∇θ loss|)` over given params.
- `update_weights(data_loss, physics_loss) -> (alpha, beta)` — EMA-based gradient-magnitude balancing, decay = `ema_decay`.
- `forward(y_pred, y_true, *physics_inputs) -> (total_loss, log_dict)`.

**Not implementing the exact weighting formula yet** — per project rule, core loss math needs explicit instruction before being written; will follow in a dedicated task once the PDE term is specified.

### `src/models/fcn_cho2022.py` — Kieu

`BatteryPINN_Cho2022(nn.Module)`, architecture per spec:
```
Input(4) -> [Current -> Sin branch] + [Time,Voltage,OCV_Estimated -> Exp branch]
         -> Concat -> 4x FC(145) -> FC(1) output
```
`SinActivation` / `ExpActivation` stub classes are provided for the two branches. Pre-layer output widths (before concat) aren't specified beyond the activation assignment — confirm exact dims with the leader/paper before finalizing `__init__`, since that's an architecture decision, not something to guess independently.

### `scripts/train_fcn.py` — Cam

`train(model, train_loader, loss_fn, optimizer, epochs, device) -> model` — standard loop: `zero_grad → forward → loss_fn(...) → backward → step`, per the step-by-step TODO in the docstring. `save_checkpoint(model, path)` — `torch.save(model.state_dict(), path)`.
CLI already wired (`--epochs`, `--batch-size`, `--lr`, `--sequence-length`, `--checkpoint-path`, `--device`) — verified via `--help`.
Uses `build_datasets()` from Task 1 for the DST split — no changes needed there.

### `scripts/evaluate.py` — Kem

`load_model(checkpoint_path, device) -> model` (eval mode). `run_inference(model, test_loader, device) -> (y_true, y_pred)` — **remember to inverse-transform both from the Min-Max scaled space back to real °C** using the target scaler from the FUDS `BatteryDataset` (fit on DST, reused on FUDS — do not re-fit). `compute_metrics(y_true, y_pred) -> {"rmse": ..., "mae": ...}`. `plot_predictions(y_true, y_pred, save_path)` — saves to `outputs/figures/fcn_predicted_vs_true.png` by default.
CLI already wired and verified via `--help`.

## Workflow reminder

1. `git checkout develop && git pull origin develop`
2. `git checkout -b <your branch from the table above>`
3. Implement only inside your file's `NotImplementedError` bodies (don't touch other teammates' files or `src/data_loader.py`).
4. Push, open a PR into `develop`.
5. When your task is done, I'll (or you can ask me to) generate your `handoffs/handoff_<name>.md`.
