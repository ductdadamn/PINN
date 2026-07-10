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
    """sin(x) activation, used on the Current pre-layer branch."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Kieu: implement sin activation")


class ExpActivation(nn.Module):
    """exp(x) activation, used on the [Time, Voltage, OCV_Estimated] pre-layer branch."""

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Kieu: implement exp activation (watch for overflow)")


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

        # TODO(Kieu): declare pre-layer submodules here, e.g.
        #   self.current_branch = nn.Sequential(nn.Linear(1, ...), SinActivation())
        #   self.other_branch   = nn.Sequential(nn.Linear(input_dim - 1, ...), ExpActivation())
        # then the concat -> 4x145 FC stack -> output layer.

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
        raise NotImplementedError(
            "Kieu: split x into Current vs [Time, Voltage, OCV_Estimated], "
            "run pre-layers, concat, run 4x145 FC stack, output layer"
        )
