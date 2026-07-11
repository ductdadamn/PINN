"""
Training script for the Shen 2025 stacked-LSTM sequence model.

Owner: Cam.

Goal: train BatteryPINN_Shen on the DST (train) split using sequence-wise
(sliding-window) data and the hotfixed AdaptivePINNLoss, logging RMSE/MAE/
alpha/beta per epoch, and checkpoint the trained model for
scripts/evaluate_lstm.py to consume.

Run (once implemented):
    .venv/bin/python scripts/train_lstm.py --epochs 25 --batch-size 32 --lr 1e-4

SKELETON ONLY -- the training loop body is intentionally left unimplemented.
Do not modify src/losses.py's math, src/data_loader.py's pipeline, or
src/models/lstm_shen2025.py's architecture while wiring this up; import and
use them as-is.

IMPORTANT KNOWN GAP (flagged by the leader, not yet resolved): AdaptivePINNLoss.
compute_physics_loss/compute_initial_loss (src/losses.py) currently index
their input tensor `x` assuming it's 2D (Batch, Features) -- e.g.
`x[:, time_idx:time_idx+1]`. For this script's 3D (Batch, Seq_Len, Features)
sequence input, that indexing is WRONG (it would slice along the sequence
dimension, not features) and will silently produce incorrect physics/initial
loss values rather than crashing. Do not attempt to fix src/losses.py
yourself -- ping the leader to resolve this before trusting physics_loss/
initial_loss numbers from a real training run. data_loss (the primary
metric) is unaffected, since compute_data_loss doesn't index into x.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import AnchorInclusiveBatchSampler, SEQUENCE_LENGTH_DEFAULT, build_datasets  # noqa: E402
from src.losses import AdaptivePINNLoss  # noqa: E402
from src.models.lstm_shen2025 import BatteryPINN_Shen  # noqa: E402

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
        {"rmse": float, "mae": float}, in degC. Mirrors
        scripts/evaluate.py's compute_metrics -- consider importing/reusing
        that instead of duplicating the sklearn calls, if convenient.
    """
    raise NotImplementedError("Cam: inverse-transform both arrays via target_scaler, compute RMSE/MAE")


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
        running iteration count. See the module docstring above re: the
        known 3D-indexing gap in compute_physics_loss/compute_initial_loss.
    optimizer : torch.optim.Optimizer
        Built from `loss_fn.parameters()` (NOT `model.parameters()` +
        `loss_fn.parameters()` -- see src/losses.py's class docstring for
        why that double-counts).
    scheduler : torch.optim.lr_scheduler.LRScheduler
        StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma).
        IMPORTANT: step_size counts ITERATIONS (batches), not epochs -- call
        scheduler.step() once per batch/iteration inside the inner loop, NOT
        once per epoch (a common mistake with StepLR).
    epochs : int
        Number of training epochs (25 by default, per spec).
    device : str
        "cuda" or "cpu".

    Returns
    -------
    BatteryPINN_Shen
        The trained model (same object, mutated in place).

    Expected steps (TODO for Cam)
    ------------------------------
    1. model.train()
    2. for epoch in range(epochs):
         for each (x, y, is_initial_step) batch:
           - move x, y, is_initial_step to device
           - optimizer.zero_grad()
           - loss, log_dict = loss_fn(x, y, is_initial_step, epoch)
           - loss.backward()
           - optimizer.step()
           - scheduler.step()              # per-ITERATION, see above
           - accumulate y (unscaled-space y_true) and the model's y_pred
             (re-run or capture from inside loss_fn.forward if you refactor
             to expose it -- currently loss_fn.forward doesn't return
             y_pred directly, only total_loss/log_dict; simplest is an
             extra no_grad() forward pass for metrics, or track loss_fn's
             internal T_pred_scaled if you extend the return value -- your
             call, just don't touch src/losses.py's math while doing it)
         after the epoch: compute_epoch_metrics(...) for RMSE/MAE, and pull
         alpha/beta from the last batch's log_dict (same pattern as
         scripts/train_fcn.py), print/log all four numbers for this epoch
    3. return model
    """
    raise NotImplementedError("Cam: implement the training loop")


def save_checkpoint(model: BatteryPINN_Shen, path: str) -> None:
    """Save model state_dict to `path`, creating parent dirs as needed.
    Same pattern as scripts/train_fcn.py's save_checkpoint."""
    raise NotImplementedError("Cam: implement checkpoint saving (torch.save(model.state_dict(), path))")


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
