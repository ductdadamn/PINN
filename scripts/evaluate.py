"""
Evaluation script for the Cho2022 FCN baseline.

Owner: Kem.

Goal: run inference with a trained BatteryPINN_Cho2022 checkpoint on the
FUDS (test) split, compute RMSE/MAE, and save a Predicted-vs-True Temperature
plot. This is the script that should demonstrate the FCN's failure to
generalize to the unseen dynamic-load profile (per Sprint 2's goal).

Run (once implemented):
    .venv/bin/python scripts/evaluate.py --checkpoint-path outputs/checkpoints/fcn_cho2022.pth

SKELETON ONLY -- inference/metric/plotting logic intentionally left
unimplemented.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import build_datasets  # noqa: E402
from src.models.fcn_cho2022 import BatteryPINN_Cho2022  # noqa: E402

DEFAULT_CHECKPOINT_PATH = "outputs/checkpoints/fcn_cho2022.pth"
DEFAULT_PLOT_PATH = "outputs/figures/fcn_predicted_vs_true.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Cho2022 FCN baseline on FUDS data")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--plot-path", type=str, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> BatteryPINN_Cho2022:
    """Instantiate BatteryPINN_Cho2022 and load weights from `checkpoint_path`.

    Returns
    -------
    BatteryPINN_Cho2022
        Model in eval() mode, on `device`.
    """
    raise NotImplementedError("Kem: implement checkpoint loading (torch.load + load_state_dict)")


def run_inference(model: BatteryPINN_Cho2022, test_loader: DataLoader, device: str):
    """Run the model over the full FUDS test set.

    Parameters
    ----------
    model : BatteryPINN_Cho2022
        Trained model in eval() mode.
    test_loader : DataLoader
        Yields (x, y) batches from the FUDS BatteryDataset:
        x shape (batch, 4), y shape (batch, 1). Note: y is Min-Max SCALED
        (same scaler fit on DST train set, see src/data_loader.build_datasets) --
        inverse-transform before computing RMSE/MAE in real-world units (deg C).
    device : str
        "cuda" or "cpu".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (y_true, y_pred), both shape (n_samples,), in ORIGINAL (unscaled)
        temperature units.

    TODO for Kem
    ------------
    1. model.eval(), torch.no_grad()
    2. iterate test_loader, collect y_pred and y_true
    3. inverse-transform both using the target scaler from build_datasets()'s
       meta / the BatteryDataset instance (test_dataset.target_scaler)
    """
    raise NotImplementedError("Kem: implement inference loop")


def compute_metrics(y_true, y_pred) -> dict:
    """Compute RMSE and MAE between y_true and y_pred (both 1D arrays, same
    units).

    Returns
    -------
    dict
        {"rmse": float, "mae": float}
    """
    raise NotImplementedError("Kem: implement RMSE/MAE computation")


def plot_predictions(y_true, y_pred, save_path: str) -> None:
    """Plot Predicted vs True Temperature and save to `save_path`.

    Suggested: a line plot over sample index (or time, if available) with
    both series overlaid, plus axis labels and a legend. Create parent dirs
    as needed (see scripts/test_loader.py for the os.makedirs pattern used
    in Task 1).
    """
    raise NotImplementedError("Kem: implement prediction plot")


def main() -> None:
    args = parse_args()

    _train_dataset, test_dataset, _meta = build_datasets(
        raw_dir="data/raw", sequence_length=args.sequence_length
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.checkpoint_path, args.device)
    y_true, y_pred = run_inference(model, test_loader, args.device)

    metrics = compute_metrics(y_true, y_pred)
    print(f"RMSE: {metrics['rmse']:.4f}  MAE: {metrics['mae']:.4f}")

    plot_predictions(y_true, y_pred, args.plot_path)
    print(f"Saved plot to {args.plot_path}")


if __name__ == "__main__":
    main()
