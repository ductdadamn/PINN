"""
Training script for the Shen 2025 stacked-LSTM sequence model.

Owner: Cam. Implemented by the leader while Cam was unavailable -- see
handoffs/ for the corresponding note.

Goal: train BatteryPINN_Shen on the DST (train) split using sequence-wise
(sliding-window) data and the hotfixed AdaptivePINNLoss, logging RMSE/MAE/
alpha/beta per epoch, and checkpoint the trained model for
scripts/evaluate_lstm.py to consume.

Run:
    .venv/bin/python scripts/train_lstm.py --epochs 25 --batch-size 32 --lr 1e-4

The 3D-indexing gap in AdaptivePINNLoss.compute_physics_loss that was
flagged when this file was still a skeleton (it assumed 2D input, silently
mis-indexing this script's 3D sequence input) has been fixed in
src/losses.py -- see that module's docstring for details. data_loss was
never affected; physics_loss/initial_loss are now correct too.
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import AnchorInclusiveBatchSampler, SEQUENCE_LENGTH_DEFAULT, build_datasets  # noqa: E402
from src.losses import AdaptivePINNLoss  # noqa: E402
from src.models.lstm_shen2025 import BatteryPINN_Shen  # noqa: E402
from scripts.evaluate import compute_metrics  # noqa: E402

DEFAULT_CHECKPOINT_PATH = "outputs/checkpoints/lstm_shen2025.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Shen 2025 LSTM on DST data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=3000, help="decay LR every N iterations (batches), NOT epochs")
    parser.add_argument("--lr-gamma", type=float, default=0.9, help="multiplicative LR decay factor (0.9 = -10%%)")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH_DEFAULT)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compute_epoch_metrics(y_true_scaled, y_pred_scaled, target_scaler) -> dict:
    """Inverse-transform scaled predictions/targets accumulated over an
    epoch and compute RMSE/MAE in real-world units (degC).

    Parameters
    ----------
    y_true_scaled, y_pred_scaled : array-like, shape (n_samples, 1)
        Concatenated across all batches in the epoch (scaled space, as
        yielded by BatteryDataset / produced by the model).
    target_scaler : sklearn MinMaxScaler
        `train_dataset.target_scaler` -- inverts the scaling.

    Returns
    -------
    dict
        {"rmse": float, "mae": float}, in degC. Reuses
        scripts/evaluate.py's compute_metrics rather than duplicating the
        sklearn calls.
    """
    y_true_scaled = np.asarray(y_true_scaled).reshape(-1, 1)
    y_pred_scaled = np.asarray(y_pred_scaled).reshape(-1, 1)
    y_true_real = target_scaler.inverse_transform(y_true_scaled).ravel()
    y_pred_real = target_scaler.inverse_transform(y_pred_scaled).ravel()
    return compute_metrics(y_true_real, y_pred_real)


def train(
    model: BatteryPINN_Shen,
    train_loader: DataLoader,
    loss_fn: AdaptivePINNLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    device: str,
) -> BatteryPINN_Shen:
    """Standard PyTorch training loop, extended with per-epoch RMSE/MAE.

    Parameters
    ----------
    model : BatteryPINN_Shen
        Model to train (already moved to `device` by caller or here).
    train_loader : DataLoader
        Yields (x, y, is_initial_step) 3-tuples from the DST BatteryDataset,
        sequence-wise: x shape (batch, seq_len, 4), y shape (batch, 1),
        is_initial_step shape (batch,). Built with
        batch_sampler=AnchorInclusiveBatchSampler (see main()), NOT plain
        shuffle=True -- see that sampler's docstring for why.
    loss_fn : AdaptivePINNLoss
        loss_fn.forward(x, y, is_initial_step, epoch) runs `model(x)`
        internally -- pass the raw batch straight through, do not call
        model(x) yourself. `epoch` (0-indexed) drives the Task 0
        warm-up/clamping hotfix -- MUST be the current epoch number, not a
        running iteration count.
    optimizer : torch.optim.Optimizer
        Built from `loss_fn.parameters()` (NOT `model.parameters()` +
        `loss_fn.parameters()` -- see src/losses.py's class docstring for
        why that double-counts).
    scheduler : torch.optim.lr_scheduler.LRScheduler
        StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma).
        IMPORTANT: step_size counts ITERATIONS (batches), not epochs -- called
        once per batch/iteration inside the inner loop below, NOT once per
        epoch (a common mistake with StepLR).
    epochs : int
        Number of training epochs (25 by default, per spec).
    device : str
        "cuda" or "cpu".

    Returns
    -------
    BatteryPINN_Shen
        The trained model (same object, mutated in place).
    """
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        last_log_dict = {}
        y_true_epoch = []
        y_pred_epoch = []

        for x, y, is_initial_step in train_loader:
            x = x.to(device)
            y = y.to(device)
            is_initial_step = is_initial_step.to(device)

            optimizer.zero_grad()
            loss, log_dict = loss_fn(x, y, is_initial_step, epoch)
            loss.backward()
            optimizer.step()
            scheduler.step()  # per-ITERATION, not per-epoch -- see docstring above

            epoch_loss += loss.item()
            n_batches += 1
            last_log_dict = log_dict

            # loss_fn.forward() only returns total_loss/log_dict, not y_pred
            # directly -- a separate no_grad() forward pass for per-epoch
            # metrics is the simplest way to get it without touching
            # src/losses.py's return signature.
            with torch.no_grad():
                y_pred = model(x)
            y_true_epoch.append(y.detach().cpu().numpy())
            y_pred_epoch.append(y_pred.detach().cpu().numpy())

        avg_loss = epoch_loss / n_batches
        y_true_epoch = np.concatenate(y_true_epoch, axis=0)
        y_pred_epoch = np.concatenate(y_pred_epoch, axis=0)
        metrics = compute_epoch_metrics(y_true_epoch, y_pred_epoch, train_loader.dataset.target_scaler)

        alpha = last_log_dict.get("alpha", "N/A")
        beta = last_log_dict.get("beta", "N/A")
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}, "
            f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
            f"Alpha: {alpha}, Beta: {beta}"
        )

    return model


def save_checkpoint(model: BatteryPINN_Shen, path: str) -> None:
    """Save model state_dict to `path`, creating parent dirs as needed.
    Same pattern as scripts/train_fcn.py's save_checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved to {path}")


def main() -> None:
    args = parse_args()

    train_dataset, _test_dataset, _meta = build_datasets(
        raw_dir="data/raw", sequence_length=args.sequence_length
    )
    train_sampler = AnchorInclusiveBatchSampler(
        len(train_dataset), batch_size=args.batch_size, anchor_indices=(0,), shuffle=True
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    model = BatteryPINN_Shen().to(args.device)
    loss_fn = AdaptivePINNLoss(
        model,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
    ).to(args.device)
    # loss_fn.parameters() already includes model's parameters (model is a
    # registered submodule of loss_fn) plus lambda1/lambda2 -- do not also
    # pass model.parameters(), that double-counts them.
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model = train(model, train_loader, loss_fn, optimizer, scheduler, args.epochs, args.device)
    save_checkpoint(model, args.checkpoint_path)


if __name__ == "__main__":
    main()
