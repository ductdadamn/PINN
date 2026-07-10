# Handoff: Task 1 — Data Engineering & DataLoader Configuration

**Author:** Lead Execution Developer (Claude)
**Branch:** feature/leader/data-loader-ocv
**Date:** 2026-07-10

## What was built

- [src/data_loader.py](src/data_loader.py) — full data pipeline:
  - `coulomb_counting_soc()` — discrete coulomb counting: `SOC_k = SOC_{k-1} + I_k*dt_k/(3600*Q_nom)`, clipped to `[0,1]`.
  - `estimate_nominal_capacity()` — auto-detects the slow (C/25-style) constant-current discharge step in the OCV file and estimates `Q_nom` from its total discharge capacity (**1.0636 Ah** for A1-007).
  - `extract_ocv_reference_curve()` — Bước 1: builds the `SOC_static → OCV_static` reference curve from that slow-discharge step (V ≈ OCV at near-zero current), sorted + de-duplicated for monotonic interpolation.
  - `split_dst_fuds()` — auto-detects the 3 oscillating-current drive-cycle steps in the dynamic file and returns the first (DST, train) and last (FUDS, test) by step order, matching the filename's `DST-US06-FUDS` sequence. US06 (middle) is currently dropped since Task 1 only asked for DST/FUDS.
  - `add_ocv_estimated()` — Bước 2 (per-segment coulomb counting) + Bước 3 (`np.interp` 1D linear interpolation) → adds `OCV_Estimated` (and `SOC_Estimated`) columns.
  - `BatteryDataset(torch.utils.data.Dataset)` — features `[Time, Current, Voltage, OCV_Estimated]`, target `Temperature`, Min-Max scaled (`sklearn.preprocessing.MinMaxScaler`). `sequence_length=1` → point-wise 2D `(n_features,)` samples for the current FCN; `sequence_length>1` → sliding-window 3D `(seq_len, n_features)` samples, same class/interface, ready for LSTM later.
  - `build_datasets()` — one-call orchestration: raw CSVs → OCV curve → per-segment `OCV_Estimated` → `BatteryDataset` for train (DST) and test (FUDS). Scalers are **fit on train (DST) only** and reused (transform-only) on test (FUDS) to avoid data leakage.
- [scripts/test_loader.py](scripts/test_loader.py) — verification script: loads a batch, prints tensor shapes, and saves the OCV mapping plot.
- [outputs/figures/ocv_mapping_sample.png](outputs/figures/ocv_mapping_sample.png) — verification plot (Voltage + OCV_Estimated vs Time, Current on secondary axis, FUDS segment).
- [requirements.txt](requirements.txt) — pinned versions of the packages this module depends on.

## Important bug caught during verification (worth knowing)

The raw dynamic file is **not one continuous discharge** — it's 3 repeated blocks (CC-CV charge back to full → drive cycle) for DST, US06, FUDS respectively. My first version ran coulomb counting continuously from the start of the whole file with a single `SOC0=1.0`, which overshot past SOC=1 during the pre-DST charging phase and got stuck at the clip ceiling for the rest of the file — `OCV_Estimated` came out flat. Fixed by splitting into segments **first**, then resetting `SOC0=1.0` at the start of each segment independently (each drive cycle genuinely starts from a fresh full charge). Confirmed via the verification plot and by checking `SOC_Estimated` now spans `~1.0 → ~0.03` across both DST and FUDS, as expected for a full discharge cycle.

## How to run

```bash
cd /home/ductlord/project/PINN
.venv/bin/pip install -r requirements.txt   # if setting up fresh
.venv/bin/python scripts/test_loader.py
```

Expected output:
```
Estimated Q_nom: 1.0636 Ah
Train (DST) samples: 7368
Test  (FUDS) samples: 7372
Batch features shape: (32, 4)
Batch target shape:   (32, 1)
Saved verification plot to outputs/figures/ocv_mapping_sample.png
```

Or use directly in code:
```python
from src.data_loader import build_datasets
from torch.utils.data import DataLoader

train_ds, test_ds, meta = build_datasets(raw_dir="data/raw", sequence_length=1)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
```

## Dependencies

See [requirements.txt](requirements.txt): `pandas`, `numpy`, `torch` (`+cu121` build — teammates without CUDA 12.1 should install a matching torch build for their machine instead of pinning this exact wheel), `scikit-learn`, `matplotlib`. All already installed in `.venv/`.

## Integration notes

- `build_datasets()` is the single entry point — no need to call the lower-level functions directly unless customizing.
- Feature order is fixed: `[Test_Time(s), Current(A), Voltage(V), OCV_Estimated]`; target is `Temperature (C)_1`. Both are Min-Max scaled to `[0,1]`.
- `meta` dict returned by `build_datasets()` also exposes `q_nom_ah`, `soc_static`/`ocv_static` (reference curve), and the un-scaled `df_dst`/`df_fuds` DataFrames (useful for plotting/debugging, as in `test_loader.py`).
- To upgrade to sequence data for an LSTM later: just pass `sequence_length=N` to `build_datasets()` / `BatteryDataset` — no other code changes needed.
- `find_raw_files()` locates files by glob pattern (`*OCV*.csv` vs the other `*.csv`), not hardcoded filenames — safe as long as `data/raw/` still contains exactly one OCV file and one dynamic-cycle file.

## Known limitations / TODOs

- US06 segment is detected but discarded — not wired into the pipeline yet (easy to add a `df_us06` output from `split_dst_fuds()` if a future task needs it).
- Coulombic efficiency `η` is assumed `= 1` (per spec's ideal/linear-approximation assumption) — not configurable yet.
- `_find_slow_discharge_step` / drive-cycle detection use heuristics (row count, current sign/CV thresholds) tuned to this dataset's structure; if other CALCE cell files have a different step layout, these thresholds may need revisiting.
