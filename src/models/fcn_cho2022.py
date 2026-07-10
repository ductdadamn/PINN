"""
Fully-connected PINN baseline, per Cho et al. 2022.

Owner: Kieu.

Purpose (Sprint 2): reproduce the Cho 2022 FCN baseline architecture and
demonstrate that a plain point-wise FCN fails to generalize well to unseen
dynamic-load profiles (trained on DST, evaluated on FUDS) -- motivating the
switch to a sequence model (LSTM) or a properly physics-constrained PINN
later.

Architecture (as specified):
    Input (4 features: Time, Current, Voltage, OCV_Estimated)
        -> Pre-layers:
             - Current -> Sin-activated branch
             - [Time, Voltage, OCV_Estimated] -> Exp-activated branch
        -> Concat(branch outputs)
        -> 4 x FC(145) hidden layers
        -> Output FC(1)  (predicted Temperature)

Implemented by Kieu: pre-layer output width is 16 per branch (32 after
concat) -- not specified by the paper beyond the activation assignment, so
treated as an ordinary architecture hyperparameter. shared_parameters()
(required by src/losses.py's AdaptivePINNLoss) returns fc_stack + output_layer
parameters, excluding the Sin/Exp pre-layer branches.
"""
from itertools import chain
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

INPUT_DIM = 4  # [Time, Current, Voltage, OCV_Estimated]
HIDDEN_DIM = 145
N_HIDDEN_LAYERS = 4
OUTPUT_DIM = 1  # Temperature


class SinActivation(nn.Module):
    """sin(x) activation, used on the Current pre-layer branch."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class ExpActivation(nn.Module):
    """exp(x) activation, used on the [Time, Voltage, OCV_Estimated] pre-layer branch."""

    def forward(self, x: Tensor) -> Tensor:
        # clamp to avoid float32 overflow (exp(x) overflows past x~88.7)
        return torch.exp(torch.clamp(x, max=88.0))


class BatteryPINN_Cho2022(nn.Module):
    """FCN baseline model per Cho 2022.

    Parameters
    ----------
    input_dim : int
        Number of input features (default 4: Time, Current, Voltage, OCV_Estimated).
        Feature order MUST match src.data_loader.FEATURE_COLS.
    hidden_dim : int
        Width of each of the 4 hidden FC layers (default 145, per spec).
    n_hidden_layers : int
        Number of hidden FC layers after the concat (default 4, per spec).
    output_dim : int
        Output width (default 1: Temperature).

    Expected forward() contract
    ----------------------------
    Input:  Tensor, shape (batch, input_dim)   -- point-wise features (sequence_length=1
            samples from BatteryDataset).
    Output: Tensor, shape (batch, output_dim)  -- predicted Temperature, same shape as
            BatteryDataset's target tensor, so it can be compared directly against
            y_true in AdaptivePINNLoss.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_hidden_layers: int = N_HIDDEN_LAYERS,
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.output_dim = output_dim

        self.current_out_dim = 16
        self.other_out_dim = 16

        # Current branch (1 feature)
        self.current_branch = nn.Sequential(nn.Linear(1, self.current_out_dim), SinActivation())
        # others branch [Time, Voltage, OCV_Estimated] (input_dim - 1 = 3 features)
        self.other_branch = nn.Sequential(nn.Linear(self.input_dim - 1, self.other_out_dim), ExpActivation())

        concat_dim = self.current_out_dim + self.other_out_dim

        layers = []
        layers.append(nn.Linear(concat_dim, self.hidden_dim))
        layers.append(nn.Tanh())  # use Tanh as the main activation function since it is smooth on R

        for _ in range(self.n_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())

        self.fc_stack = nn.Sequential(*layers)

        # output layer - predict Temperature
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters of the SHARED last-layer stack only (concat -> 4x145 FC
        -> output), excluding the input-specific Sin/Exp pre-layer branches.

        Required by src.losses.AdaptivePINNLoss.update_weights() for Cho 2022's
        Adaptive Normalization gradient-balancing scheme, which is defined
        w.r.t. these shared weights specifically -- see that module's
        docstring for why.
        """
        return chain(self.fc_stack.parameters(), self.output_layer.parameters())

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)
            Feature order: [Time, Current, Voltage, OCV_Estimated].

        Returns
        -------
        Tensor, shape (batch, output_dim)
            Predicted temperature.
        """
        # Current is index 1; [Time, Voltage, OCV_Estimated] are indices [0, 2, 3]
        current = x[:, [1]]  # Shape: (batch, 1)
        other_features = x[:, [0, 2, 3]]  # Shape: (batch, 3)

        out_current = self.current_branch(current)
        out_other = self.other_branch(other_features)
        out_concat = torch.cat([out_current, out_other], dim=1)
        out_fc = self.fc_stack(out_concat)
        output = self.output_layer(out_fc)
        return output
