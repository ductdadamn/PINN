"""
Evaluation & pitching-proof script for the Shen 2025 LSTM.

Owner: Kem.

Goal: run inference with a trained BatteryPINN_Shen checkpoint on the FUDS
(test) split, compute final RMSE/MAE (academic target: RMSE <= 0.6 degC),
and generate the "pitching proof" overlay figure comparing the Sprint 2 FCN
baseline (flat/failed) against the Sprint 3 LSTM (predicted) alongside the
true dynamic load profile.

Run (once implemented):
    .venv/bin/python scripts/evaluate_lstm.py \\
        --fcn-checkpoint-path outputs/checkpoints/fcn_cho2022.pth \\
        --lstm-checkpoint-path outputs/checkpoints/lstm_shen2025.pth

SKELETON ONLY -- inference/plotting logic intentionally left unimplemented.

REUSE OPPORTUNITY: scripts/evaluate.py (your own Sprint 2 file) already has
load_model (FCN-specific), run_inference (model-agnostic -- works for the
LSTM too, since it just calls model(x) and doesn't care about x's shape),
and compute_metrics (fully generic). Importing and reusing these for the FCN
side (and run_inference/compute_metrics for the LSTM side too) avoids
duplicating that logic -- see the imports below. Only genuinely new code
needed: a load_lstm_model analog, and the pitching plot itself.

CRITICAL ALIGNMENT NOTE (read before implementing the plot): the FCN
(sequence_length=1) and LSTM (sequence_length=N, default 50) predictions on
FUDS have DIFFERENT lengths and are NOT aligned 1:1 with each other or with
raw df_fuds rows by the same index:
  - FCN predictions: length == len(df_fuds) (point-wise, index i <-> df_fuds
    row i directly).
  - LSTM predictions: length == len(df_fuds) - sequence_length + 1 (each
    window i covers df_fuds rows [i, i+sequence_length-1], and its
    prediction corresponds to the LAST row of that window, i.e. df_fuds row
    (i + sequence_length - 1)).
  To plot all three temperature series (true, FCN, LSTM) AND the
  current/voltage subplots on one shared, correctly-aligned time axis: slice
  everything to start at df_fuds row (sequence_length - 1) --
  i.e. `df_fuds.iloc[sequence_length - 1:]` for the true data/time axis, and
  `fcn_preds[sequence_length - 1:]` for the FCN predictions -- the LSTM
  predictions need NO slicing (already exactly that length/alignment). Get
  `sequence_length` from whatever value was used to build the LSTM's test
  dataset (default src.data_loader.SEQUENCE_LENGTH_DEFAULT=50), not
  hardcoded, in case it's changed via --sequence-length.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import SEQUENCE_LENGTH_DEFAULT, build_datasets  # noqa: E402
from src.models.fcn_cho2022 import BatteryPINN_Cho2022  # noqa: E402
from src.models.lstm_shen2025 import BatteryPINN_Shen  # noqa: E402
from scripts.evaluate import compute_metrics, load_model as load_fcn_model, run_inference  # noqa: E402

DEFAULT_FCN_CHECKPOINT_PATH = "outputs/checkpoints/fcn_cho2022.pth"
DEFAULT_LSTM_CHECKPOINT_PATH = "outputs/checkpoints/lstm_shen2025.pth"
DEFAULT_PLOT_PATH = "outputs/figures/pitching_proof.png"
ACADEMIC_TARGET_RMSE_C = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Shen 2025 LSTM on FUDS and generate the pitching proof plot")
    parser.add_argument("--fcn-checkpoint-path", type=str, default=DEFAULT_FCN_CHECKPOINT_PATH)
    parser.add_argument("--lstm-checkpoint-path", type=str, default=DEFAULT_LSTM_CHECKPOINT_PATH)
    parser.add_argument("--plot-path", type=str, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH_DEFAULT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_lstm_model(checkpoint_path: str, device: str) -> BatteryPINN_Shen:
    """Instantiate BatteryPINN_Shen and load weights from `checkpoint_path`.
    Mirrors scripts/evaluate.py's load_model, just for the LSTM class.

    Returns
    -------
    BatteryPINN_Shen
        Model in eval() mode, on `device`.
    """
    raise NotImplementedError("Kem: implement checkpoint loading (torch.load + load_state_dict), same pattern as scripts/evaluate.py's load_model")


def generate_pitching_plot(
    df_fuds,
    true_temp_real,
    fcn_pred_real,
    lstm_pred_real,
    sequence_length: int,
    save_path: str,
) -> None:
    """Generate the 3-subplot 'pitching proof' figure, sharing the X-axis (Time).

    Subplot 1: True dynamic Current profile (A).
    Subplot 2: True Voltage (V) vs OCV_Estimated (V).
    Subplot 3: Temperature comparison -- True Measured vs FCN Baseline
               (flat/failed) vs Shen LSTM (predicted).

    Parameters
    ----------
    df_fuds : pd.DataFrame
        The FUDS segment dataframe (from build_datasets()'s meta["df_fuds"]),
        UNSLICED -- this function is responsible for the row-49-style
        alignment slicing described in the module docstring, so all 3
        subplots share exactly the same time axis and are visually
        comparable (e.g. seeing a current spike line up with a temperature
        response).
    true_temp_real : array-like
        Ground-truth Temperature, UNSLICED, same length as df_fuds (i.e.
        the FCN-aligned length) -- inverse-transformed, real degC.
    fcn_pred_real : array-like
        FCN baseline predictions, UNSLICED, same length as df_fuds -- real
        degC. Expected to look flat/near-constant per the Sprint 2 finding
        (even after the Task 0 hotfix, it's the LSTM that's meant to fix
        this, not the FCN retroactively -- plot whatever the FCN checkpoint
        actually produces).
    lstm_pred_real : array-like
        LSTM predictions, length == len(df_fuds) - sequence_length + 1,
        ALREADY the correct alignment -- no slicing needed on this one.
    sequence_length : int
        The sequence_length used for the LSTM's test dataset -- needed to
        slice df_fuds/true_temp_real/fcn_pred_real to match lstm_pred_real's
        alignment (see module docstring's ALIGNMENT NOTE).
    save_path : str
        Where to save the figure (creates parent dirs as needed).
    """
    raise NotImplementedError(
        "Kem: slice df_fuds/true_temp_real/fcn_pred_real to [sequence_length-1:], "
        "then plt.subplots(3, 1, sharex=True, figsize=...) with Current, "
        "Voltage+OCV_Estimated, and Temperature (3 lines: true/FCN/LSTM) subplots"
    )


def main() -> None:
    args = parse_args()

    # Two BatteryDataset builds: point-wise (sequence_length=1) for the FCN,
    # sequence-wise for the LSTM. Both refit scalers on DST independently,
    # but since it's the same underlying DST data both times, the resulting
    # scalers are numerically equivalent -- no correctness issue, just two
    # redundant (cheap) passes over the raw CSVs, acceptable for an
    # evaluation script that runs once.
    _train_fcn, test_dataset_fcn, meta = build_datasets(raw_dir="data/raw", sequence_length=1)
    _train_lstm, test_dataset_lstm, _meta_lstm = build_datasets(
        raw_dir="data/raw", sequence_length=args.sequence_length
    )

    fcn_loader = DataLoader(test_dataset_fcn, batch_size=args.batch_size, shuffle=False)
    lstm_loader = DataLoader(test_dataset_lstm, batch_size=args.batch_size, shuffle=False)

    fcn_model = load_fcn_model(args.fcn_checkpoint_path, args.device)
    lstm_model = load_lstm_model(args.lstm_checkpoint_path, args.device)

    true_temp_real, fcn_pred_real = run_inference(fcn_model, fcn_loader, args.device)
    true_temp_lstm_aligned, lstm_pred_real = run_inference(lstm_model, lstm_loader, args.device)

    lstm_metrics = compute_metrics(true_temp_lstm_aligned, lstm_pred_real)
    fcn_metrics = compute_metrics(true_temp_real, fcn_pred_real)

    print(f"FCN  baseline -- RMSE: {fcn_metrics['rmse']:.4f}  MAE: {fcn_metrics['mae']:.4f}")
    print(f"LSTM (Shen)   -- RMSE: {lstm_metrics['rmse']:.4f}  MAE: {lstm_metrics['mae']:.4f}")
    status = "PASS" if lstm_metrics["rmse"] <= ACADEMIC_TARGET_RMSE_C else "FAIL"
    print(f"Academic target RMSE <= {ACADEMIC_TARGET_RMSE_C} degC: {status}")

    generate_pitching_plot(
        meta["df_fuds"],
        true_temp_real,
        fcn_pred_real,
        lstm_pred_real,
        args.sequence_length,
        args.plot_path,
    )
    print(f"Saved pitching proof plot to {args.plot_path}")


if __name__ == "__main__":
    main()
