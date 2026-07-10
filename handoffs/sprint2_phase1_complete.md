# Sprint 2 Phase 1 Complete — `AdaptivePINNLoss` (Leader)

**Status:** `src/losses.py` is fully implemented and unit-validated with a dummy stand-in network (since the real `BatteryPINN_Cho2022` doesn't exist yet). No `NotImplementedError` remains in this file. This is the gradient/loss foundation the rest of Sprint 2 builds on top of.

**Branch:** `feature/leader/adaptive-pinn-loss` → merge into `develop` before Kieu/Cam/Kem branch off for their pieces.

This document is the authoritative contract for all four Sprint 2 files. If anything here conflicts with an older note in `handoffs/sprint2_tasks.md`, **this document wins** — it reflects the final, validated state.

---

## 1. `AdaptivePINNLoss` — input/output contract

```python
from src.losses import AdaptivePINNLoss

loss_fn = AdaptivePINNLoss(
    model,                                    # your BatteryPINN_Cho2022 instance
    feature_scaler=train_dataset.feature_scaler,
    target_scaler=train_dataset.target_scaler,
    alpha_init=1.0,       # optional
    beta_init=1.0,        # optional
    gamma=0.9,             # optional, Cho 2022's moving-average weight
    ambient_temp_c=25.0,   # optional, T_amb
)
```

- `model` is stored as `loss_fn.model` (a registered submodule). **`loss_fn.parameters()` already includes every model parameter, plus the two trainable Lumped Capacitance coefficients `lambda1`/`lambda2`.** Build your optimizer as:
  ```python
  optimizer = torch.optim.Adam(loss_fn.parameters(), lr=...)
  ```
  **Do NOT** do `list(model.parameters()) + list(loss_fn.parameters())` — that double-counts every model parameter. (This was a real bug caught in Phase 1 testing and is now fixed everywhere in this repo, but watch for it if you write new training code.)

- **Call signature:**
  ```python
  total_loss, log_dict = loss_fn(x, y_true, is_initial_step)
  ```
  | Arg | Shape | Notes |
  |---|---|---|
  | `x` | `(batch, 4)` | SCALED (Min-Max [0,1]) features, order `[Time, Current, Voltage, OCV_Estimated]`. **Pass the raw batch straight from the DataLoader — do NOT call `model(x)` yourself first.** `loss_fn` runs `model(x)` internally because it needs `x`'s autograd graph intact to differentiate the model's output w.r.t. its own input (for `dT/dt`). If you pass a precomputed/detached `y_pred` instead, the physics loss will silently be wrong or crash. |
  | `y_true` | `(batch, 1)` | SCALED target (Temperature), from `BatteryDataset`. |
  | `is_initial_step` | `(batch,)` | See §3 below. |

- **Returns** `(total_loss: Tensor, log_dict: Dict[str, float])`.
  - `total_loss` — call `.backward()` on this, nothing else.
  - `log_dict` keys: `data_loss`, `physics_loss`, `initial_loss`, `alpha`, `beta`, `lambda1`, `lambda2`. Log all of these per step/epoch — see §4 for what to expect from them.

- **Loss formula** (Cho 2022 Adaptive Normalization — confirmed by Strategic Planner, not the more common single-weight scheme):
  ```
  Loss_total = Loss_data + alpha * Loss_PDE + beta * Loss_initial
  ```
  `Loss_data` is unweighted. `alpha`/`beta` adapt every call via:
  ```
  alpha_hat = max(|grad Loss_data|) / mean(|grad Loss_PDE|)
  beta_hat  = max(|grad Loss_data|) / mean(|grad Loss_initial|)
  alpha = (1-gamma)*alpha_prev + gamma*alpha_hat
  beta  = (1-gamma)*beta_prev  + gamma*beta_hat
  ```
  gradients taken w.r.t. `model.shared_parameters()` only — see §2.

- **Physics loss (`Loss_PDE`)**: Lumped Capacitance Model, `f = dT/dt + lambda1*(V-V_ocv)*I + lambda2*(T_amb-T)`, `Loss_PDE = mean(f**2)`. All quantities are unscaled back to real physical units (seconds, Volts, Amps, °C) internally before evaluating `f` — you don't need to do anything about scaling yourself, just pass the scaled `x`/model output as normal.

- **Initial-condition loss (`Loss_initial`)**: `MSE(T_pred(t_start), T_amb)`, evaluated only on the sample(s) flagged by `is_initial_step`. If none are flagged in a batch, this term is a differentiable zero (not an error) — but see §3, you shouldn't normally hit that case if you use the required sampler.

---

## 2. `shared_parameters()` requirement — **for Kieu**

`update_weights()` computes its gradient ratios **only w.r.t. the model's shared last-layer weights** (the concat → 4×145 FC → output stack), **not** the input-specific Sin/Exp pre-layer branches. Per Cho 2022.

Your `BatteryPINN_Cho2022` **must** implement:

```python
def shared_parameters(self) -> Iterator[nn.Parameter]:
    return self.shared_fc.parameters()
```

Practically: keep the 4×145 FC + output stack in its own submodule (e.g. `self.shared_fc = nn.Sequential(...)`), separate from `self.current_branch` / `self.other_branch` (or whatever you name the Sin/Exp pre-layers), so `shared_parameters()` can return exactly that subset.

**This is not optional** — `AdaptivePINNLoss.update_weights()` calls `self.model.shared_parameters()` directly and will raise `AttributeError`/`NotImplementedError` if it's missing or returns the wrong thing. The stub is already in `src/models/fcn_cho2022.py` with this exact docstring; you just need to fill it in once `self.shared_fc` exists.

Also required (already in the skeleton, restating for clarity): `forward(x)` takes `(batch, 4)` and returns `(batch, 1)` in the **same scaled space** as `BatteryDataset`'s target — `AdaptivePINNLoss` unscales internally, you don't need to.

---

## 3. DataLoader structure — **for Cam**

### 3-tuple, not 2-tuple

`BatteryDataset.__getitem__` (and therefore every `DataLoader` built on it) yields **`(x, y, is_initial_step)`**, not `(x, y)`. This changed after Task 1 was originally merged — if you find old code/docs referencing `(x, y)`, they're stale.

- `is_initial_step`: shape `(batch,)`, float 0.0/1.0. Marks the single row per DST/FUDS trajectory that represents `t_start` — the sample right after the pre-drive-cycle rest, i.e. SOC=1.0, no current yet. Needed for `Loss_initial` (§1).

### `AnchorInclusiveBatchSampler` — use this instead of `shuffle=True`

Each trajectory has **exactly one** `is_initial_step==1` row. With plain `shuffle=True` and `batch_size=32` over ~7400 samples, that row lands in roughly 1-in-230 batches — so `mean(|grad Loss_initial|)` was **exactly zero** on almost every batch, and `update_weights()`'s `beta_hat = max(|grad L_data|) / (0 + eps)` blew up to ~1e12 within a single epoch. (This was caught and confirmed empirically during Phase 1 testing — not a hypothetical.)

Fix: use `AnchorInclusiveBatchSampler` (in `src/data_loader.py`), which guarantees the anchor row is present in every batch:

```python
from src.data_loader import AnchorInclusiveBatchSampler, build_datasets

train_dataset, test_dataset, meta = build_datasets(raw_dir="data/raw", sequence_length=args.sequence_length)

train_sampler = AnchorInclusiveBatchSampler(
    len(train_dataset), batch_size=args.batch_size, anchor_indices=(0,), shuffle=True
)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
```

Note the `DataLoader` takes `batch_sampler=`, not `batch_size=`/`shuffle=` — those are mutually exclusive with `batch_sampler` in PyTorch's API, so don't pass all three.

This is **already wired into `scripts/train_fcn.py`'s `main()`** — if you're extending that file, the sampler is set up for you. If you're writing new training code from scratch, replicate this pattern; don't fall back to plain `shuffle=True` for the training loader.

`scripts/evaluate.py` (FUDS, Kem) does **not** need this sampler — plain `DataLoader(test_dataset, batch_size=..., shuffle=False)` is correct there, since inference never calls `update_weights()`. Just unpack and discard the third element: `for x, y, _ in test_loader:`.

### Full loop shape

```python
model = BatteryPINN_Cho2022().to(device)
loss_fn = AdaptivePINNLoss(model, feature_scaler=..., target_scaler=...).to(device)
optimizer = torch.optim.Adam(loss_fn.parameters(), lr=...)  # see §1

for epoch in range(epochs):
    model.train()
    for x, y, is_initial_step in train_loader:
        x, y, is_initial_step = x.to(device), y.to(device), is_initial_step.to(device)
        optimizer.zero_grad()
        loss, log_dict = loss_fn(x, y, is_initial_step)
        loss.backward()
        optimizer.step()
```

---

## 4. Expected loss behavior in the first few epochs — **for Kem**

From the Phase 1 trial run (dummy placeholder network, real DST data, `AnchorInclusiveBatchSampler` in place — real numbers will differ once Kieu's actual architecture replaces the placeholder, but the *shape* of the behavior should carry over):

| Quantity | Trial-run behavior | What to expect / watch for |
|---|---|---|
| `data_loss` | ~3–5 early (scaled-space MSE) | Should trend down as training progresses. This is the metric most directly tied to "is the model learning anything" — the others are physics/regularization terms, not predictive accuracy. |
| `physics_loss` | Very small (~1e-5–1e-6) | Small values here are **not** necessarily a good sign early on — an untrained/undertrained network can trivially have a near-flat output, which trivially satisfies the PDE residual without the model having learned real dynamics. Don't read "low physics_loss" as "good physics fit" in isolation, especially in epoch 1. |
| `initial_loss` | Nonzero every batch (the sampler guarantees the anchor row is always present — see §3) | Should trend down as the model learns to predict ~T_amb at `t_start`. |
| `alpha` | Can spike very high early (1e4–1e10+ in the trial) | **Expected**, not a bug — confirmed by the Strategic Planner: caused by `mean(|grad Loss_physics|)` being tiny for an undertrained/placeholder network. Should shrink via the EMA once Kieu's real non-linear architecture (Sin/Exp pre-layers) is in place and starts producing a physics gradient with real magnitude. If `alpha` is still enormous after several epochs with the *real* architecture, that's worth flagging back to the leader — but don't be alarmed by large `alpha` in the first few epochs specifically. |
| `beta` | Bounded, ~1–1e5 in the trial (post-sampler-fix) | Should stay in a sane range throughout, thanks to `AnchorInclusiveBatchSampler`. If you see `beta` exploding into the 1e9+ range again, that's a regression — check the sampler is actually wired into whatever `DataLoader` is being used. |
| `total_loss` | Can look large/erratic early, dominated by `alpha * physics_loss` even though `physics_loss` itself is tiny (because `alpha` is huge) | **Don't use raw `total_loss` magnitude as your primary training-health signal early on** — it's an artifact of the adaptive weighting, not a direct measure of model quality. Prefer watching `data_loss` (scaled) or, better, the real-unit RMSE/MAE from `evaluate.py` on held-out data for judging actual predictive performance. |

**Sprint 2's actual goal**, restated for evaluation design: this FCN baseline is *expected* to train reasonably on DST but **generalize poorly to FUDS** — that's the whole point (demonstrating a plain FCN's failure on an unseen dynamic-load profile, motivating later work). So in `scripts/evaluate.py`, a large train/test performance gap is the expected, successful outcome of this sprint, not a bug to chase.

---

## 5. Bugs found & fixed during Phase 1 (for context, not action)

1. **Optimizer double-counting** — `train_fcn.py` originally built the optimizer from `model.parameters() + loss_fn.parameters()`; fixed to `loss_fn.parameters()` alone (§1).
2. **`beta` exact-zero-division blowup** — fixed via `AnchorInclusiveBatchSampler` (§3).
3. Cho 2022 eq. 6's `Loss_initial = |f(t=0) - T_amb|²` has a dimensional-mismatch typo (residual `f` compared directly to a temperature) — implemented per the Strategic Planner's correction: `MSE(T_pred(t_start), T_amb)`.

---

## Status: HALTING per Strategic Planner instruction

The gradient/loss foundation (`src/losses.py`) is complete and validated. Passing the baton to Kieu (`src/models/fcn_cho2022.py`), Cam (`scripts/train_fcn.py`), and Kem (`scripts/evaluate.py`) to implement against this contract. No further changes planned from this side until the team's implementations are ready for integration review.
