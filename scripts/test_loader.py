"""Verification script for src/data_loader.py.

Run from the project root:
    .venv/bin/python scripts/test_loader.py
"""
import os
import sys

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import build_datasets, TIME_COL, VOLTAGE_COL, CURRENT_COL  # noqa: E402


def main():
    train_dataset, test_dataset, meta = build_datasets(raw_dir="data/raw", sequence_length=1)

    print(f"Estimated Q_nom: {meta['q_nom_ah']:.4f} Ah")
    print(f"Train (DST) samples: {len(train_dataset)}")
    print(f"Test  (FUDS) samples: {len(test_dataset)}")

    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    x_batch, y_batch = next(iter(loader))
    print(f"Batch features shape: {tuple(x_batch.shape)}  (expected: [batch, 4] for sequence_length=1)")
    print(f"Batch target shape:   {tuple(y_batch.shape)}  (expected: [batch, 1])")

    df_fuds = meta["df_fuds"]
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.plot(df_fuds[TIME_COL], df_fuds[VOLTAGE_COL], label="Voltage (measured)", color="tab:blue", linewidth=1)
    ax1.plot(df_fuds[TIME_COL], df_fuds["OCV_Estimated"], label="OCV_Estimated", color="tab:orange", linewidth=1)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Current (A)")
    ax2.plot(df_fuds[TIME_COL], df_fuds[CURRENT_COL], label="Current", color="tab:green", alpha=0.4, linewidth=0.8)
    ax2.legend(loc="upper right")

    plt.title("OCV Interpolation Verification (FUDS segment)")
    fig.tight_layout()

    out_dir = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ocv_mapping_sample.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved verification plot to {out_path}")


if __name__ == "__main__":
    main()
