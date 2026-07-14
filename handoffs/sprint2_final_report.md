# Sprint 2 Final Report — FCN Baseline (Cho 2022)

**To:** Strategic Commander
**From:** Lead Execution Developer
**Status:** Infrastructure complete and merged. Empirical result surfaced a real training-stability issue that needs a decision before Sprint 3.

---

## 1. Code status: all four pieces complete, merged, verified

| File | Owner | Status |
|---|---|---|
| `src/data_loader.py` | Leader | Done (Task 1) + `AnchorInclusiveBatchSampler` added in Sprint 2 |
| `src/losses.py` | Leader | Done — `AdaptivePINNLoss` fully implemented (data/PDE/initial losses, Cho 2022 Adaptive Normalization) |
| `src/models/fcn_cho2022.py` | Kieu | Done — architecture matches spec exactly |
| `scripts/train_fcn.py` | Cam | Done — verified with a real training run |
| `scripts/evaluate.py` | Kem | Done — verified with real inference |

All merged into `develop` (PRs #2–#8, #10). Every file compiles clean; the full pipeline (`build_datasets` → train → checkpoint → evaluate → plot) runs end-to-end without errors.

**One process incident during integration, now resolved:** a merge-conflict resolution on Kieu's branch briefly reverted the `shared_parameters()` fix after it had already been applied, which reached `develop` via PR #7 and broke it for everyone. Caught during this final check, root-caused via git history, fixed via hotfix PR #10. `develop` has been broken-then-fixed, but is confirmed working now (verified by a full real training run, not just a diff read).

---

## 2. Empirical result: the model collapses to a constant, degenerate solution

Ran a real 20-epoch training pass on DST, then evaluated on FUDS, to close out Sprint 2 properly.

**Finding: the trained model predicts a flat constant ≈25.00°C for every input — on both the training set (DST) and the test set (FUDS).**

```
Predictions on TRAIN (DST): min=24.9994  max=24.9995  mean=24.9994  std=0.000003
True DST temperature range: 26.7269 to 27.8407

FUDS evaluation: RMSE=2.2616  MAE=2.2528
```

This is **not** ordinary overfitting or poor generalization (a model that fits DST but fails on FUDS) — it never learned to use its inputs at all, even on the data it was trained on. 25.00°C is exactly `ambient_temp_c` (`T_amb`), the constant target of `Loss_initial`.

**Root cause:** `alpha` (weight on `Loss_PDE`) and `beta` (weight on `Loss_initial`) swing to extreme, unstable magnitudes under Cho 2022's Adaptive Normalization — in this run, `alpha` ranged from ~34,000 to ~6×10¹⁰ and `beta` from ~6 to ~6.7×10⁵, changing by orders of magnitude epoch to epoch. Since `Loss_total = Loss_data + alpha·Loss_PDE + beta·Loss_initial` and `Loss_data` carries no adaptive weight of its own (coefficient fixed at 1), these swings mean `Loss_data`'s gradient contribution is regularly dwarfed by orders of magnitude. A constant output equal to `T_amb` trivially satisfies both `Loss_PDE` (≈0 derivative) and `Loss_initial` (already equals `T_amb`), so gradient descent converges to that degenerate solution instead of learning real dynamics. The 20-epoch loss curve is correspondingly erratic and non-monotonic (205 → 526 → 232 → 30 → 513 → ... → 1836), tracking `alpha`'s instability rather than showing real convergence.

This was flagged as a *possible* concern earlier (when `alpha` ran large against a placeholder network) and attributed to the placeholder being untrained. **That explanation no longer holds** — this result is from the real `BatteryPINN_Cho2022` architecture, and the collapse is total, not a transient early-training artifact.

**I have not attempted to fix this.** Per standing instruction, the adaptive weighting formula is core math and changing it (e.g. clamping alpha/beta, adding a warm-up period before adaptive weighting kicks in, switching normalization scheme) needs your call, not mine.

---

## 3. Reframing Sprint 2's stated goal

The sprint's goal was "chứng minh sự thất bại của FCN trước dữ liệu tải động" (demonstrate the FCN's failure on dynamic-load data). This result **does** show failure, but via a different and more fundamental mechanism than the DST→FUDS generalization gap that was likely intended: the model fails to learn *any* signal from data at all, due to loss-weighting instability, not architecture limitations under distribution shift. Worth knowing which failure mode you want documented as Sprint 2's conclusion, since they imply different next steps.

## 4. Recommendation for a decision (not a proposal to act on unilaterally)

Options to stabilize training, for you/the team to choose from — I can implement whichever is decided:
- **Warm-up period:** hold `alpha`/`beta` fixed (e.g. at their init values) for the first N steps/epochs before adaptive weighting kicks in, so `Loss_data` gets a chance to pull weights away from the trivial solution first.
- **Clamp `alpha`/`beta`** to a bounded range each update.
- **Log-space or bounded normalization** instead of raw gradient-magnitude ratios.
- **Re-verify the formula** against the paper once more — possible we're missing a normalization/clipping detail Cho 2022 specifies elsewhere.

## 5. Not yet done / explicitly out of scope this sprint

- `Loss_initial`'s definition assumes a single anchor row per trajectory (row 0). Not revisited.
- US06 segment (extracted but discarded in Task 1) still unused.
- No LSTM/sequence-length>1 path has been exercised yet (`BatteryDataset` supports it, untested).
- `main` has **not** been updated — recommend against promoting this milestone to `main` until the training-collapse question is resolved, since `main` is supposed to reflect a working state.

---

## Repo hygiene notes (unrelated to Sprint 2 code, worth your attention separately)

- Kieu's branch name (`feature/gaiparisxinhvaicalon/fcn`) contains inappropriate language and doesn't follow the `feature/<name>/<task>` convention — worth a word.
- A merge-conflict resolution broke `develop` once already this sprint (see §1) — suggest anyone resolving a non-trivial conflict pings the leader first, rather than resolving solo in the GitHub web UI.

---

## Standing by for Sprint 3 direction

Infrastructure is solid and the team executed their assigned files correctly — the one real issue is a training-dynamics question that needs your decision before further training runs are worth doing. Ready to implement whichever stabilization approach you choose, or move to a different next task if you'd rather revisit this later.
