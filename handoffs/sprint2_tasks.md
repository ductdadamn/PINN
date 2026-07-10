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
- **`BatteryPINN_Cho2022.shared_parameters()`** *(new requirement)*: returns only the parameters of the shared 4x145 FC + output stack (not the Sin/Exp pre-layer branches). Required by `AdaptivePINNLoss.update_weights()` — see below.
- **`AdaptivePINNLoss.forward(x, y_true)`** *(updated from the original skeleton — see below)*: returns `(total_loss: Tensor, log_dict: Dict[str, float])`. `log_dict` includes `data_loss`, `physics_loss`, `initial_loss`, `alpha`, `beta`, `lambda1`, `lambda2` — `train_fcn.py` should log these per epoch.

> **Update since PR #2 merged:** the physics equation and the alpha/beta weighting formula are now both confirmed, which changed `AdaptivePINNLoss`'s interface from the original skeleton:
> - Constructor now requires `feature_scaler` and `target_scaler` (pass `train_dataset.feature_scaler` / `.target_scaler`), since the physics residual needs to unscale back to real units.
> - `forward(x, y_true)` now takes the **raw scaled input batch `x`**, not a precomputed `y_pred` — it runs `model(x)` internally so it can autograd `dT/dt` w.r.t. the input. **Do not call `model(x)` yourself and pass the output in.**
> - Loss formula is **`Loss_total = Loss_data + alpha*Loss_PDE + beta*Loss_initial`** (Cho 2022 Adaptive Normalization) — note `Loss_data` itself is unweighted, unlike the original skeleton's `alpha*data + beta*physics` guess.
> - The optimizer must include `loss_fn.parameters()` too (owns trainable `lambda1`, `lambda2`): `torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=...)`.
> - `scripts/train_fcn.py`'s `main()` and docstrings are already updated to match — Cam just needs to fill in the loop body per the updated TODO.
- **Checkpoint**: `scripts/train_fcn.py` saves to `outputs/checkpoints/fcn_cho2022.pth` (via `model.state_dict()`); `scripts/evaluate.py` loads from the same default path. Both are `outputs/` (gitignored) — checkpoints are local artifacts, not committed.

## Per-file detail

### `src/losses.py` — Leader

`AdaptivePINNLoss(nn.Module)` — **implemented**, except `compute_initial_loss`:
- `compute_data_loss(y_pred, y_true) -> Tensor` — MSE, done.
- `compute_physics_loss(x, T_pred_scaled) -> Tensor` — Lumped Capacitance Model residual `f = dT/dt + lambda1*(V-V_ocv)*I + lambda2*(T_amb-T)`, `Loss = mean(f**2)`, done. Unscales all quantities back to real units before evaluating (see module docstring for why). `lambda1`/`lambda2` are trainable `nn.Parameter`s.
- `_max_abs_grad` / `_mean_abs_grad(loss, params) -> Tensor` — done.
- `update_weights(data_loss, physics_loss, initial_loss) -> (alpha, beta)` — **done**. Cho 2022 Adaptive Normalization: `alpha_hat = max(|∇Loss_data|)/mean(|∇Loss_PDE|)`, `beta_hat = max(|∇Loss_data|)/mean(|∇Loss_initial|)`, both computed w.r.t. `model.shared_parameters()`, then EMA'd with `gamma=0.9`.
- `compute_initial_loss(x, y_true, T_pred_scaled) -> Tensor` — **still `NotImplementedError`**. Exact definition (which sample(s) count as "initial", what's compared) not yet confirmed. `forward()` will raise here until it's filled in — everything else (including `update_weights`) is independently testable in the meantime.
- `forward(x, y_true) -> (total_loss, log_dict)` — `Loss_total = Loss_data + alpha*Loss_PDE + beta*Loss_initial`, runs `model(x)` internally (see contract note above).

Validated with a dummy stand-in model exposing `shared_parameters()` (since `BatteryPINN_Cho2022` isn't implemented yet): gradients correctly flow into `lambda1`/`lambda2` and the model's shared-layer parameters, and `alpha`/`beta` move correctly across successive `update_weights()` calls.

### `src/models/fcn_cho2022.py` — Kieu

`BatteryPINN_Cho2022(nn.Module)`, architecture per spec:
```
Input(4) -> [Current -> Sin branch] + [Time,Voltage,OCV_Estimated -> Exp branch]
         -> Concat -> 4x FC(145) -> FC(1) output
```
`SinActivation` / `ExpActivation` stub classes are provided for the two branches. Pre-layer output widths (before concat) aren't specified beyond the activation assignment — confirm exact dims with the leader/paper before finalizing `__init__`, since that's an architecture decision, not something to guess independently.

**New requirement from `src/losses.py`:** keep the shared 4x145 FC + output stack in its own submodule (e.g. `self.shared_fc = nn.Sequential(...)`), separate from the Sin/Exp pre-layer branches, and implement `shared_parameters()` to return `self.shared_fc.parameters()`. `AdaptivePINNLoss`'s gradient-balancing scheme is defined specifically w.r.t. these shared weights (per Cho 2022) — it will raise `NotImplementedError` until this method returns real parameters.

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
