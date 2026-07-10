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

SKELETON ONLY -- layer wiring/forward pass intentionally left unimplemented.
Pre-layer output width and activation placement beyond what's specified above
are NOT finalized here; Kieu should confirm exact dims with the leader before
implementing so the architecture matches the paper.
"""
import torch
import torch.nn as nn
from torch import Tensor

INPUT_DIM = 4  # [Time, Current, Voltage, OCV_Estimated]
HIDDEN_DIM = 145
N_HIDDEN_LAYERS = 4
OUTPUT_DIM = 1  # Temperature


class SinActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class ExpActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # use clamp to avoid overflow
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

        # TODO AOI(Kieu): declare pre-layer submodules here, e.g.
        #   self.current_branch = nn.Sequential(nn.Linear(1, ...), SinActivation())
        #   self.other_branch   = nn.Sequential(nn.Linear(input_dim - 1, ...), ExpActivation())
        # then the concat -> 4x145 FC stack -> output layer.
        
        # jjk
        self.current_out_dim = 16
        self.other_out_dim = 16
        
        # Current branch (1 feature)
        self.current_branch = nn.Sequential(nn.Linear(1, self.current_out_dim), SinActivation())
        # others branch [Time, Voltage, OCV_Estimated] (input_dim - 1 = 3 features)
        self.other_branch = nn.Sequential(nn.Linear(self.input_dim - 1, self.other_out_dim), ExpActivation())
        
        concat_dim = self.current_out_dim + self.other_out_dim
        
        layers = []
        layers.append(nn.Linear(concat_dim, self.hidden_dim))
        layers.append(nn.Tanh()) # use Tanh as the main activation function since it is smooth on R
        
        for _ in range(self.n_hidden_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.Tanh())
            
        self.fc_stack = nn.Sequential(*layers)
        
        # output layer - predict Temperature
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

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
        
        # the feature Current (index 1) 
        current = x[:, [1]]  # Shape: (batch, 1)
        # others features
        other_features = x[:, [0, 2, 3]]  # Shape: (batch, 3)
        
        # pass through pre-layer, concatenate, pass through FCN and return output 
        out_current = self.current_branch(current)
        out_other = self.other_branch(other_features)
        out_concat = torch.cat([out_current, out_other], dim=1)
        out_fc = self.fc_stack(out_concat)
        output = self.output_layer(out_fc)
        return output
    