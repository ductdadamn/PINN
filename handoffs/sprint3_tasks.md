# Sprint 3 — Phase 2: Shen 2025 Sequence-wise LSTM

**Goal:** fix the Sprint 2 FCN's degenerate collapse (Task 0, done), then build and train a stacked-LSTM sequence model expected to generalize far better to the unseen FUDS drive cycle (academic target: RMSE ≤ 0.6°C), and produce a "pitching proof" figure demonstrating the improvement over the Sprint 2 FCN baseline.

**Base branch:** `develop`, once the following are merged (in this order — they build on each other):
1. `feature/leader/loss-stabilization-hotfix` (Task 0)
2. `feature/leader/sequence-dataloader` (3D sliding-window data pipeline)
3. `feature/leader/sprint3-skeleton` (the 3 skeleton files this doc describes)

| File | Owner | Branch to create |
|---|---|---|
| `src/losses.py` (Task 0 hotfix) | Leader | `feature/leader/loss-stabilization-hotfix` (done) |
| `src/data_loader.py` (sequence refactor) | Leader | `feature/leader/sequence-dataloader` (done) |
| `src/models/lstm_shen2025.py` | Kieu | `feature/kieu/lstm-model` |
| `scripts/train_lstm.py` | Cam | `feature/cam/train-lstm` |
| `scripts/evaluate_lstm.py` | Kem | `feature/kem/evaluate-lstm` |

All three teammate files are **skeletons**: verified to import, construct, and run their CLI (`--help`) correctly. Logic is `raise NotImplementedError(...)` for each of you to fill in, per the detailed docstrings in each file.

---

## 0. Task 0: Loss stabilization hotfix — DONE, read this before touching `AdaptivePINNLoss`

`AdaptivePINNLoss.forward()` and `.update_weights()` now **require an `epoch: int` argument**:
```python
loss, log_dict = loss_fn(x, y, is_initial_step, epoch)
```
- **Epochs 0–4** (`warmup_epochs=5`, configurable): `alpha=0.0, beta=1.0` hardcoded, no adaptive update at all — `Loss_total` reduces to just `Loss_data` so the network learns real data-fitting dynamics before physics/initial terms compete.
- **Epoch 5+**: the original Cho 2022 EMA update runs unchanged, then the result is clamped to `[1e-3, 1e3]`.

Verified with a real 20-epoch FCN run: the Sprint 2 collapse (flat 25.00°C prediction) is gone — predictions now vary meaningfully and track the true range. Do not modify this formula further without syncing with the leader.

---

## 1. Sequence-wise data pipeline — DONE, read this before writing `train_lstm.py`

`BatteryDataset` (`src/data_loader.py`) now **defaults** to `sequence_length=50` (`SEQUENCE_LENGTH_DEFAULT`), yielding 3D windows. `__getitem__` returns `(x, y, is_initial_step)`:
- `x`: shape `(sequence_length, 4)` per item → a `DataLoader` batches these into **`(Batch, Seq_Len, Features)`** — exactly what `BatteryPINN_Shen` expects.
- `y`: shape `(1,)` — Temperature **at the window's last timestep** (many-to-one).
- `is_initial_step`: `1.0` for the window starting at row 0 of the segment (the `t_start` anchor), `0.0` otherwise — same semantics as Sprint 2, just now indexed by window-start position.

`AnchorInclusiveBatchSampler` needed **no changes** — its `anchor_indices=(0,)` already refers to window-start index 0, which is exactly the anchor window. Use it exactly as in `train_fcn.py`:
```python
train_sampler = AnchorInclusiveBatchSampler(len(train_dataset), batch_size=..., anchor_indices=(0,), shuffle=True)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
```

Sprint 2's FCN scripts are unaffected — they explicitly pass `sequence_length=1`.

---

## 2. `src/models/lstm_shen2025.py` — Kieu

`BatteryPINN_Shen(nn.Module)`, architecture per spec:
```
Input (Batch, Seq_Len=50, 4)
    -> 4 separate single-layer nn.LSTM(hidden_size=64, batch_first=True), stacked
    -> h_T (last timestep) of LSTM layer 4 only
    -> nn.Linear(64, 1) -> scalar predicted Temperature
```
Declared as **4 separate `nn.LSTM` instances** (`self.lstm1`...`self.lstm4`), not one `nn.LSTM(num_layers=4)` — this is so `shared_parameters()` can cleanly return just layer 4 + the output layer, per the same contract as `BatteryPINN_Cho2022.shared_parameters()` from Sprint 2.

**⚠️ Known integration gap, not yet resolved:** `AdaptivePINNLoss.compute_physics_loss`/`compute_initial_loss` (`src/losses.py`) currently index their input assuming it's 2D (`x[:, time_idx:time_idx+1]`). For this model's 3D input, that indexing is **wrong but won't crash** — it'll silently slice the wrong dimension. `data_loss` (the primary training signal) is unaffected. **Do not fix `src/losses.py` yourself** — flag it to the leader before trusting `physics_loss`/`initial_loss` numbers from a real training run. This needs to be resolved before Sprint 3 training is fully trustworthy; tracked, not forgotten.

---

## 3. `scripts/train_lstm.py` — Cam

New requirements vs. Sprint 2's `train_fcn.py`:
- **Optimizer:** `Adam(loss_fn.parameters(), lr=1e-4)` (same double-counting caveat as Sprint 2 — see the file's docstring).
- **LR schedule:** `StepLR(optimizer, step_size=3000, gamma=0.9)`, stepped **once per iteration/batch**, NOT once per epoch — a common mistake with `StepLR`, called out explicitly in the skeleton.
- **25 epochs**, training on DST.
- **Must pass `epoch` to `loss_fn`:** `loss_fn(x, y, is_initial_step, epoch)` — the current 0-indexed epoch number, needed for the Task 0 warm-up/clamp logic.
- **Log RMSE, MAE, alpha, beta per epoch** (not just raw loss like Sprint 2) — `compute_epoch_metrics` stub provided for the RMSE/MAE half; consider reusing `scripts/evaluate.py`'s `compute_metrics` instead of duplicating the sklearn calls.

---

## 4. `scripts/evaluate_lstm.py` — Kem

Two new things beyond Sprint 2's `evaluate.py`:

**a) Dual-model evaluation.** Load both the FCN checkpoint (Sprint 2) and the LSTM checkpoint (Sprint 3), run both on FUDS, report both RMSE/MAE (plus a pass/fail line against the 0.6°C academic target for the LSTM). Reuses `scripts/evaluate.py`'s `load_model`/`run_inference`/`compute_metrics` directly — `run_inference` and `compute_metrics` are model-agnostic, no duplication needed.

**b) The alignment problem — read carefully before writing `generate_pitching_plot`.** FCN and LSTM predictions on FUDS have **different lengths** and are not aligned 1:1:
- FCN (point-wise): length `== len(df_fuds)`, index `i` ↔ `df_fuds` row `i`.
- LSTM (windowed): length `== len(df_fuds) - sequence_length + 1`, index `i` ↔ `df_fuds` row `i + sequence_length - 1` (the window's **last** row).

To plot all 3 subplots (Current, Voltage vs OCV_Estimated, Temperature) on one correctly-aligned shared time axis: slice `df_fuds` and the FCN predictions to `[sequence_length - 1:]`; the LSTM predictions need no slicing. Full detail is in the skeleton's module docstring and `generate_pitching_plot`'s docstring — this is an easy off-by-N bug to introduce, so read it before coding.

**Output:** `outputs/figures/pitching_proof.png`, 3 subplots sharing the X-axis (Time):
1. True dynamic Current profile (A)
2. True Voltage (V) vs `OCV_Estimated` (V)
3. Temperature: True Measured vs FCN Baseline (flat/failed) vs Shen LSTM (predicted)

---

## Workflow reminder

1. `git checkout develop && git pull origin develop` (after the 3 leader branches above are merged)
2. `git checkout -b <your branch from the table>`
3. Implement only inside your file's `NotImplementedError` bodies.
4. If you hit the `src/losses.py` 3D-indexing gap (§2) or anything else in a shared file, **ping the leader — do not resolve it yourself**, especially not by resolving a merge conflict solo in the GitHub web UI (see the repo hygiene notes below; this is how `develop` broke once in Sprint 2).
5. Push, open a PR into `develop`.
