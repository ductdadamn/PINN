# Sprint 2 FCN (Cho 2022) Baseline — Milestone Artifacts

**Reproduce this exact result:** `git checkout sprint2-fcn-baseline` (tag), then:
```bash
.venv/bin/python scripts/train_fcn.py --epochs 20 --batch-size 32 --lr 1e-3 --checkpoint-path outputs/checkpoints/fcn_cho2022.pth
.venv/bin/python scripts/evaluate.py --checkpoint-path outputs/checkpoints/fcn_cho2022.pth
```
This tag is the commit right **before** the Task 0 stabilization hotfix (warm-up + clamping) was added to `src/losses.py` — training on any later commit will not reproduce this collapse, since the hotfix changes the adaptive weighting behavior.

## Files

- `fcn_cho2022.pth` — trained checkpoint (state_dict) reproducing the collapse below.
- `fcn_predicted_vs_true.png` — Predicted vs. True Temperature on the FUDS test set.

## Result: degenerate collapse to a flat constant prediction

The model predicts a flat constant ≈25.00°C for every input — on both the training set (DST) and the test set (FUDS), not a generalization gap.

```
RMSE: 2.2756  MAE: 2.2668   (regeneration run; original documented run: RMSE 2.2616, MAE 2.2528 — 
                              same qualitative collapse, small numeric difference since no random 
                              seed is fixed in the training pipeline)
True FUDS temperature range: 26.79 to 27.98 degC
Predicted range: essentially constant at ~25.00 degC (= ambient_temp_c, ~0 variance)
```

**Root cause:** `alpha` (weight on `Loss_PDE`) and `beta` (weight on `Loss_initial`) swing across orders of magnitude under Cho 2022's Adaptive Normalization (alpha reached ~1.6×10¹¹ in this run), drowning out the unweighted `Loss_data` term. A constant output equal to `T_amb` trivially satisfies both `Loss_PDE` and `Loss_initial`, so gradient descent converges there instead of learning real dynamics.

Full narrative and root-cause analysis: `handoffs/sprint2_final_report.md`. The fix (warm-up + clamping) is documented in `handoffs/sprint3_tasks.md` §0 and lives in `src/losses.py` on `develop` (not in this tagged snapshot, by design).
