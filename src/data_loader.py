"""
Data pipeline for the CALCE A1-007 battery dataset.

Coulomb Counting (discrete form, per spec):
    SOC_k = SOC_{k-1} + I_k * dt_k / (3600 * Q_nom)

Sign convention: this dataset follows the Arbin convention where
Current(A) > 0 is charge and Current(A) < 0 is discharge. The equation
above is applied to the raw Current(A) column unchanged -- no sign flip.
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

TIME_COL = "Test_Time(s)"
CURRENT_COL = "Current(A)"
VOLTAGE_COL = "Voltage(V)"
TEMP_COL = "Temperature (C)_1"
STEP_COL = "Step_Index"
DISCHARGE_CAP_COL = "Discharge_Capacity(Ah)"

FEATURE_COLS = [TIME_COL, CURRENT_COL, VOLTAGE_COL, "OCV_Estimated"]
TARGET_COL = TEMP_COL


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_raw_files(raw_dir="data/raw"):
    """Locate the static OCV file and the dynamic drive-cycle file by name pattern."""
    ocv_matches = glob.glob(os.path.join(raw_dir, "*OCV*.csv"))
    dynamic_matches = [
        f for f in glob.glob(os.path.join(raw_dir, "*.csv")) if f not in ocv_matches
    ]
    if len(ocv_matches) != 1:
        raise FileNotFoundError(f"Expected exactly one *OCV*.csv file in {raw_dir}, found {ocv_matches}")
    if len(dynamic_matches) != 1:
        raise FileNotFoundError(f"Expected exactly one dynamic-cycle csv in {raw_dir}, found {dynamic_matches}")
    return ocv_matches[0], dynamic_matches[0]


def load_raw_csv(path):
    df = pd.read_csv(path)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Bước 1: Coulomb counting + reference OCV curve extraction
# ---------------------------------------------------------------------------

def coulomb_counting_soc(current, time_s, soc0, q_nom_ah):
    """Discrete rectangular-integration coulomb counting.

    SOC_k = SOC_{k-1} + I_k * dt_k / (3600 * Q_nom), clipped to [0, 1].
    dt_k = t_k - t_{k-1} (first sample has dt=0, i.e. SOC_0 = soc0).
    """
    current = np.asarray(current, dtype=np.float64)
    time_s = np.asarray(time_s, dtype=np.float64)
    dt = np.diff(time_s, prepend=time_s[0])
    dt[0] = 0.0

    delta_soc = current * dt / (3600.0 * q_nom_ah)
    soc = soc0 + np.cumsum(delta_soc)
    return np.clip(soc, 0.0, 1.0)


def _find_slow_discharge_step(ocv_df, min_points=1000, cv_threshold=0.05):
    """Identify the C/25-style quasi-static discharge step used as the OCV reference.

    Physical criterion (per spec): a step with very small, effectively constant
    discharge current (Current(A) < 0 under the charge-positive convention),
    sustained over many samples. Selected automatically (by row count) instead
    of hardcoding a step index, so this keeps working if the schedule shifts.
    """
    candidates = []
    for step_id, g in ocv_df.groupby(STEP_COL):
        if len(g) < min_points:
            continue
        cur = g[CURRENT_COL].to_numpy()
        mean_i = cur.mean()
        std_i = cur.std()
        if mean_i >= 0:
            continue  # not a discharge step
        cv = std_i / abs(mean_i) if mean_i != 0 else np.inf
        if cv > cv_threshold:
            continue  # not constant-current enough
        candidates.append((len(g), step_id, g))

    if not candidates:
        raise ValueError("Could not auto-detect a slow constant-current discharge step in the OCV file")

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][2]  # the longest matching step


def estimate_nominal_capacity(ocv_df):
    """Q_nom estimated as the total discharge capacity accumulated over the
    slow (C/25-style) discharge step -- the closest empirical measurement to
    nominal capacity available in this dataset."""
    step_df = _find_slow_discharge_step(ocv_df)
    q_nom = step_df[DISCHARGE_CAP_COL].iloc[-1] - step_df[DISCHARGE_CAP_COL].iloc[0]
    return float(q_nom)


def extract_ocv_reference_curve(ocv_df, q_nom_ah, soc0=1.0):
    """Bước 1: build the SOC_static -> OCV_static reference curve (f_ref).

    1. Take the slow discharge step (V approx= OCV since I is near zero).
    2. Run discrete coulomb counting over it starting from soc0 (fully charged
       going into the discharge, per spec).
    3. OCV_static = Voltage(V) on those same rows.
    4. Clean: sort by SOC and drop duplicate SOC values so the curve is
       strictly monotonic for interpolation.
    """
    step_df = _find_slow_discharge_step(ocv_df)

    soc_static = coulomb_counting_soc(
        step_df[CURRENT_COL].to_numpy(),
        step_df[TIME_COL].to_numpy(),
        soc0=soc0,
        q_nom_ah=q_nom_ah,
    )
    ocv_static = step_df[VOLTAGE_COL].to_numpy()

    order = np.argsort(soc_static)
    soc_static = soc_static[order]
    ocv_static = ocv_static[order]

    soc_static, unique_idx = np.unique(soc_static, return_index=True)
    ocv_static = ocv_static[unique_idx]

    return soc_static, ocv_static


# ---------------------------------------------------------------------------
# Bước 2 + 3: dynamic SOC + OCV interpolation
# ---------------------------------------------------------------------------

def add_ocv_estimated(segment_df, soc_static, ocv_static, q_nom_ah, soc0=1.0):
    """Bước 2 (SOC_dynamic via coulomb counting) + Bước 3 (1D linear interpolation
    of OCV_static(SOC_static) evaluated at SOC_dynamic). Adds OCV_Estimated column.

    Must be called on a single drive-cycle segment (e.g. DST or FUDS alone), not
    the raw multi-block file -- see split_dst_fuds for why.
    """
    df = segment_df.copy()
    soc_dynamic = coulomb_counting_soc(
        df[CURRENT_COL].to_numpy(),
        df[TIME_COL].to_numpy(),
        soc0=soc0,
        q_nom_ah=q_nom_ah,
    )
    df["SOC_Estimated"] = soc_dynamic
    df["OCV_Estimated"] = np.interp(soc_dynamic, soc_static, ocv_static)
    return df


# ---------------------------------------------------------------------------
# Drive-cycle split (DST = train, FUDS = test)
# ---------------------------------------------------------------------------

def split_dst_fuds(dynamic_df, min_points=1000):
    """Auto-detect the oscillating-current drive-cycle steps and split them
    into DST (train) and FUDS (test).

    Assumption: the file name ("...DST-US06-FUDS...") gives the execution
    order of the three drive-cycle blocks (DST, then US06, then FUDS), which
    matches the CALCE test protocol for this cell. The middle block (US06) is
    detected and dropped since the spec only asks for DST/FUDS. This is an
    inference from step ordering, not a labeled column in the raw data --
    flagged here since a mislabeled schedule would silently mis-split.

    Each returned segment is reset to its own local time/index but keeps the
    original absolute Test_Time(s) values. Coulomb counting must be run
    separately per segment (see build_datasets) because the raw file is not
    one continuous discharge: each drive cycle is preceded by its own CC-CV
    charge back to a fresh full charge, so a single running SOC integrated
    across the whole file overshoots past SOC=1 during those charge phases
    and gets stuck at the clip ceiling for the rest of the file.
    """
    drive_steps = []
    for step_id, g in dynamic_df.groupby(STEP_COL):
        if len(g) < min_points:
            continue
        cur = g[CURRENT_COL].to_numpy()
        if cur.min() < 0 < cur.max():  # oscillating charge/discharge = drive cycle
            drive_steps.append(step_id)

    drive_steps = sorted(drive_steps)
    if len(drive_steps) < 2:
        raise ValueError(f"Expected >=2 drive-cycle steps (DST/.../FUDS), found {drive_steps}")

    dst_step = drive_steps[0]
    fuds_step = drive_steps[-1]

    df_dst = dynamic_df[dynamic_df[STEP_COL] == dst_step].reset_index(drop=True)
    df_fuds = dynamic_df[dynamic_df[STEP_COL] == fuds_step].reset_index(drop=True)
    return df_dst, df_fuds


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BatteryDataset(Dataset):
    """Battery drive-cycle dataset.

    Features: [Time, Current, Voltage, OCV_Estimated] (Min-Max scaled).
    Target: Temperature.

    sequence_length=1 returns point-wise 2D samples: (n_features,) per item,
    suitable for the current FCN. sequence_length>1 returns sliding-window 3D
    samples: (sequence_length, n_features), ready for an LSTM later -- same
    class, no interface change needed when upgrading.

    __getitem__ returns a 3-tuple (x, y, is_initial_step) -- NOT (x, y). The
    third element flags whether this sample is the trajectory's initial
    condition anchor (row 0 of the DST/FUDS segment -- the first sample after
    the pre-drive-cycle rest, i.e. t_start with SOC=1.0), needed by
    AdaptivePINNLoss.compute_initial_loss (src/losses.py) per Cho 2022's
    Loss_initial = MSE(T_pred(t_start), T_amb). Each segment has exactly one
    such row, so with shuffle=True most batches will have is_initial_step
    all-zero -- that's expected, not a bug.
    """

    def __init__(
        self,
        df,
        feature_cols=None,
        target_col=TARGET_COL,
        sequence_length=1,
        feature_scaler=None,
        target_scaler=None,
        fit_scalers=True,
    ):
        self.feature_cols = feature_cols or FEATURE_COLS
        self.target_col = target_col
        self.sequence_length = sequence_length

        features = df[self.feature_cols].to_numpy(dtype=np.float64)
        target = df[[self.target_col]].to_numpy(dtype=np.float64)

        self.feature_scaler = feature_scaler or MinMaxScaler()
        self.target_scaler = target_scaler or MinMaxScaler()

        if fit_scalers:
            features = self.feature_scaler.fit_transform(features)
            target = self.target_scaler.fit_transform(target)
        else:
            features = self.feature_scaler.transform(features)
            target = self.target_scaler.transform(target)

        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)

        # Row 0 of the (already-segmented, reset_index'd) df is the
        # trajectory's t_start / initial-condition anchor -- see class
        # docstring. Indexed the same way as __getitem__'s window-start
        # `idx`, so this works unchanged for both point-wise and windowed
        # (sequence_length>1) sampling.
        self.is_initial_step = np.zeros(len(df), dtype=np.float32)
        self.is_initial_step[0] = 1.0

    def __len__(self):
        if self.sequence_length == 1:
            return len(self.features)
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        if self.sequence_length == 1:
            x = self.features[idx]  # (n_features,)
            y = self.target[idx]  # (1,)
        else:
            x = self.features[idx: idx + self.sequence_length]  # (seq_len, n_features)
            y = self.target[idx + self.sequence_length - 1]  # target at window end
        is_initial = self.is_initial_step[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(is_initial, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_datasets(raw_dir="data/raw", sequence_length=1, soc0=1.0):
    """End-to-end pipeline: load raw CSVs -> OCV reference curve -> OCV_Estimated
    -> DST/FUDS split -> BatteryDataset for train (DST) and test (FUDS).

    The feature/target scalers are fit on the train set (DST) only and reused
    (transform-only) on the test set (FUDS) to avoid data leakage.
    """
    ocv_path, dynamic_path = find_raw_files(raw_dir)
    ocv_df = load_raw_csv(ocv_path)
    dynamic_df = load_raw_csv(dynamic_path)

    q_nom_ah = estimate_nominal_capacity(ocv_df)
    soc_static, ocv_static = extract_ocv_reference_curve(ocv_df, q_nom_ah, soc0=soc0)

    # Split into segments BEFORE coulomb counting: each drive cycle starts from
    # its own fresh full charge, so SOC0=1.0 must be reset per segment.
    df_dst_raw, df_fuds_raw = split_dst_fuds(dynamic_df)
    df_dst = add_ocv_estimated(df_dst_raw, soc_static, ocv_static, q_nom_ah, soc0=soc0)
    df_fuds = add_ocv_estimated(df_fuds_raw, soc_static, ocv_static, q_nom_ah, soc0=soc0)

    train_dataset = BatteryDataset(df_dst, sequence_length=sequence_length, fit_scalers=True)
    test_dataset = BatteryDataset(
        df_fuds,
        sequence_length=sequence_length,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
        fit_scalers=False,
    )

    meta = {
        "q_nom_ah": q_nom_ah,
        "soc_static": soc_static,
        "ocv_static": ocv_static,
        "df_dst": df_dst,
        "df_fuds": df_fuds,
    }
    return train_dataset, test_dataset, meta
