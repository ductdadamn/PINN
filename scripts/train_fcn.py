"""
Training script for the Cho2022 FCN baseline.

Owner: Cam.


Goal: train BatteryPINN_Cho2022 on the DST (train) split, using
AdaptivePINNLoss, and checkpoint the trained model for scripts/evaluate.py
to consume.

Run (once implemented):
    .venv/bin/python scripts/train_fcn.py --epochs 100 --batch-size 32 --lr 1e-3
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import AnchorInclusiveBatchSampler, build_datasets  # noqa: E402
from src.losses import AdaptivePINNLoss  # noqa: E402
from src.models.fcn_cho2022 import BatteryPINN_Cho2022  # noqa: E402

DEFAULT_CHECKPOINT_PATH = "outputs/checkpoints/fcn_cho2022.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Cho2022 FCN baseline on DST data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train(
    model: BatteryPINN_Cho2022,
    train_loader: DataLoader,
    loss_fn: AdaptivePINNLoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
) -> BatteryPINN_Cho2022:
    """Standard PyTorch training loop.

    Parameters
    ----------
    model : BatteryPINN_Cho2022
        Model to train (already moved to `device` by caller or here).
    train_loader : DataLoader
        Yields (x, y, is_initial_step) 3-tuples from the DST BatteryDataset
        (NOT (x, y) -- see src/data_loader.py's BatteryDataset docstring):
        x shape (batch, 4), y shape (batch, 1), is_initial_step shape (batch,).
        Built with batch_sampler=AnchorInclusiveBatchSampler (see main()),
        NOT plain shuffle=True -- guarantees the t_start anchor row is in
        every batch, which Loss_initial's gradient-ratio weighting needs
        (see AnchorInclusiveBatchSampler's docstring for why).
    loss_fn : AdaptivePINNLoss
        Combines data + physics (PDE) + initial-condition loss. IMPORTANT:
        loss_fn.forward(x, y, is_initial_step) runs `model(x)` internally
        (needed so it can autograd dT/dt w.r.t. the raw input) -- do NOT
        call model(x) yourself and pass y_pred in; pass the raw batch
        straight through. See src/losses.py's forward() docstring.
    optimizer : torch.optim.Optimizer
        Built from `loss_fn.parameters()` (NOT `model.parameters()` +
        `loss_fn.parameters()` -- since `model` is a registered submodule of
        `loss_fn`, the latter already includes every model parameter plus
        the trainable lambda1/lambda2; concatenating both double-counts
        model params, which caused a real "duplicate parameters" bug caught
        during a trial run):
            optimizer = torch.optim.Adam(loss_fn.parameters(), lr=...)
        (see main() below, already wired this way)
    epochs : int
        Number of training epochs.
    device : str
        "cuda" or "cpu".

    Returns
    -------
    BatteryPINN_Cho2022
        The trained model (same object, mutated in place).
    """
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        last_log_dict = {}
        
        for x, y, is_initial_step in train_loader:
            x = x.to(device)
            y = y.to(device)
            is_initial_step = is_initial_step.to(device)
            
            optimizer.zero_grad()
            
            loss, log_dict = loss_fn(x, y, is_initial_step)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            last_log_dict = log_dict
            
        avg_loss = epoch_loss / len(train_loader)
        alpha = last_log_dict.get('alpha', 'N/A')
        beta = last_log_dict.get('beta', 'N/A')
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}, Alpha: {alpha}, Beta: {beta}")
        
    return model


def save_checkpoint(model: BatteryPINN_Cho2022, path: str) -> None:
    """Save model state_dict to `path`, creating parent dirs as needed."""
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

    model = BatteryPINN_Cho2022().to(args.device)
    loss_fn = AdaptivePINNLoss(
        model,
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
    ).to(args.device)
    # loss_fn.parameters() already includes model's parameters (model is a
    # registered submodule of loss_fn) plus lambda1/lambda2 -- do not also
    # pass model.parameters(), that double-counts them.
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=args.lr)

    model = train(model, train_loader, loss_fn, optimizer, args.epochs, args.device)
    save_checkpoint(model, args.checkpoint_path)


if __name__ == "__main__":
    main()
