"""
Sequence-wise stacked LSTM, per Shen 2025.

Owner: Kieu.

Purpose (Sprint 3 Phase 2): replace the Sprint 2 point-wise FCN baseline
(which collapsed to a degenerate flat prediction, then was stabilized but
still only sees one timestep at a time) with a sequence model that can learn
temporal dynamics -- expected to generalize far better to the unseen FUDS
drive cycle (academic target: RMSE <= 0.6 degC, see scripts/evaluate_lstm.py).

Architecture (as specified):
    Input (Batch, Seq_Len=50, 4 features: Time, Current, Voltage, OCV_Estimated)
        -> 4 stacked nn.LSTM layers, hidden_size=64, batch_first=True
        -> take h_T (last timestep's hidden state) from LSTM layer 4 only
        -> nn.Linear(64, 1) -> scalar predicted Temperature (current/last
           timestep of the window, matching BatteryDataset's target -- see
           src/data_loader.py's BatteryDataset docstring)

SKELETON ONLY -- layer wiring/forward pass intentionally left unimplemented.

IMPORTANT for src/losses.py integration: same contract as
BatteryPINN_Cho2022 (Sprint 2) -- AdaptivePINNLoss.update_weights() calls
model.shared_parameters(), which here must return exactly the LAST (4th)
LSTM layer's parameters plus the output Linear layer's parameters (NOT
layers 1-3). Stack the 4 LSTM layers as 4 SEPARATE nn.LSTM instances (each
single-layer, hidden_size=64, batch_first=True; layer 1 takes the raw
4-feature input, layers 2-4 each take the previous layer's 64-dim hidden
sequence as input) rather than one nn.LSTM(num_layers=4, ...) -- this makes
"layer 4's parameters" directly addressable as self.lstm4.parameters()
instead of having to filter a combined module's per-layer-named parameters.
"""
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

INPUT_DIM = 4  # [Time, Current, Voltage, OCV_Estimated]
HIDDEN_SIZE = 64
N_LSTM_LAYERS = 4
OUTPUT_DIM = 1  # Temperature


class BatteryPINN_Shen(nn.Module):
    """Stacked-LSTM sequence model per Shen 2025.

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep (default 4).
        Feature order MUST match src.data_loader.FEATURE_COLS.
    hidden_size : int
        Hidden state size of each LSTM layer (default 64, per spec).
    n_lstm_layers : int
        Number of stacked single-layer LSTMs (default 4, per spec).
    output_dim : int
        Output width (default 1: Temperature).

    Expected forward() contract
    ----------------------------
    Input:  Tensor, shape (Batch, Seq_Len, input_dim) -- sequence-wise
            samples from BatteryDataset (sequence_length=50 by default, see
            src/data_loader.py's SEQUENCE_LENGTH_DEFAULT).
    Output: Tensor, shape (Batch, output_dim) -- predicted Temperature AT
            THE LAST TIMESTEP of each input window, same shape as
            BatteryDataset's target tensor, so it can be compared directly
            against y_true in AdaptivePINNLoss.

    Note for AdaptivePINNLoss.compute_physics_loss (src/losses.py): that
    method differentiates the model's output w.r.t. its raw input tensor `x`
    (for dT/dt) via torch.autograd.grad(..., inputs=x, ...). For a 3D
    sequence input this still works (grad w.r.t. the whole (Batch,Seq,Feat)
    tensor), but compute_physics_loss's current column-indexing
    (x[:, time_idx:time_idx+1] etc.) assumes a 2D (Batch, Features) tensor --
    it will need updating for 3D inputs (index the LAST timestep,
    x[:, -1, time_idx], to match the timestep this model's output
    corresponds to). Flagging this now since it's a real integration point,
    not something to silently work around independently -- confirm with the
    leader before changing src/losses.py.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_size: int = HIDDEN_SIZE,
        n_lstm_layers: int = N_LSTM_LAYERS,
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.output_dim = output_dim

        # TODO(Kieu): declare 4 separate single-layer LSTMs, e.g.
        #   self.lstm1 = nn.LSTM(input_dim, hidden_size, batch_first=True)
        #   self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        #   self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        #   self.lstm4 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # then the output projection:
        #   self.output_layer = nn.Linear(hidden_size, output_dim)

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the LAST (4th) LSTM layer + output Linear layer
        ONLY, excluding LSTM layers 1-3.

        Required by src.losses.AdaptivePINNLoss.update_weights() for Cho
        2022's Adaptive Normalization gradient-balancing scheme, which is
        defined w.r.t. these "shared last layer" weights specifically --
        same contract as BatteryPINN_Cho2022.shared_parameters()
        (src/models/fcn_cho2022.py), just a different architecture.
        """
        raise NotImplementedError(
            "Kieu: return itertools.chain(self.lstm4.parameters(), self.output_layer.parameters())"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (Batch, Seq_Len, input_dim)
            Feature order per timestep: [Time, Current, Voltage, OCV_Estimated].

        Returns
        -------
        Tensor, shape (Batch, output_dim)
            Predicted temperature at the window's last timestep.
        """
        raise NotImplementedError(
            "Kieu: run x through lstm1->lstm2->lstm3->lstm4 sequentially "
            "(each LSTM's full output sequence feeds the next layer's input), "
            "take the LAST layer's h_T (either lstm4's returned h_n, or "
            "lstm4_output[:, -1, :] -- they're equivalent since batch_first=True "
            "and this is a single-directional LSTM), then self.output_layer(h_T)"
        )
