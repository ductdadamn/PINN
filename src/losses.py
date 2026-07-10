"""
Adaptive-weighted PINN loss.

Owner: Leader (Kieu/Kem/Cam should treat this module's public interface as
fixed and import it as-is; do not modify without syncing with the leader).

Combines a data-fitting loss (Temperature prediction vs ground truth) with one
or more physics-residual losses, and adaptively re-weights them each step
using gradient-magnitude balancing: the weight of each loss term is scaled
relative to max(|grad_theta Loss_term|), tracked with an exponential moving
average (EMA) so the weighting doesn't jitter step-to-step.

SKELETON ONLY -- method bodies are intentionally left unimplemented
(`raise NotImplementedError`). Do not fill in the weighting formula without
explicit instruction; the exact update rule is core math owned by the
Strategic Planner / leader.
"""
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AdaptivePINNLoss(nn.Module):
    """Adaptive data/physics loss combiner for the PINN battery model.

    Parameters
    ----------
    model : nn.Module
        The network being trained. Needed so gradients of each loss term can
        be computed w.r.t. its parameters for gradient-magnitude balancing.
    alpha_init : float
        Initial weight for the data loss term.
    beta_init : float
        Initial weight for the physics loss term.
    ema_decay : float
        Decay factor for the moving average used to smooth alpha/beta updates
        across steps (closer to 1.0 = slower/smoother adaptation).

    Attributes (expected, to be set in __init__)
    ----------------------------------------------
    alpha : float
        Current adaptive weight applied to the data loss term.
    beta : float
        Current adaptive weight applied to the physics loss term.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        ema_decay: float = 0.9,
    ) -> None:
        super().__init__()
        self.model = model
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.ema_decay = ema_decay
        self.alpha: float = alpha_init
        self.beta: float = beta_init
        # TODO(leader): init any EMA state buffers needed by update_weights()

    def compute_data_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Data-fitting loss.

        Parameters
        ----------
        y_pred : Tensor, shape (batch, 1)
            Model-predicted temperature.
        y_true : Tensor, shape (batch, 1)
            Ground-truth temperature (from BatteryDataset target).

        Returns
        -------
        Tensor, scalar
            Data loss value (e.g. MSE). Must remain differentiable w.r.t.
            `self.model` parameters (do not `.item()` / detach here).
        """
        raise NotImplementedError("Leader: implement data loss")

    def compute_physics_loss(self, *physics_inputs: Tensor) -> Tensor:
        """Physics-residual (PDE) loss enforcing the governing thermal equation.

        Parameters
        ----------
        *physics_inputs : Tensor
            Whatever tensors are needed to evaluate the PDE residual (e.g.
            model inputs with requires_grad=True for autograd derivatives).
            Exact signature to be finalized when the physics term is
            specified by the Strategic Planner.

        Returns
        -------
        Tensor, scalar
            Physics residual loss value. Must remain differentiable w.r.t.
            `self.model` parameters.
        """
        raise NotImplementedError("Leader: implement physics loss (pending PDE spec)")

    def _max_abs_grad(self, loss: Tensor, params: Iterable[Tensor]) -> Tensor:
        """Compute max(|grad_theta loss|) -- the maximum-magnitude gradient of
        `loss` with respect to `params`.

        Parameters
        ----------
        loss : Tensor, scalar
            A differentiable loss term (e.g. output of compute_data_loss).
        params : Iterable[Tensor]
            Model parameters to differentiate w.r.t. (typically
            `self.model.parameters()`).

        Returns
        -------
        Tensor, scalar (0-d)
            The single largest absolute gradient value across all `params`.
        """
        raise NotImplementedError("Leader: implement gradient extraction")

    def update_weights(self, data_loss: Tensor, physics_loss: Tensor) -> Tuple[float, float]:
        """Update self.alpha / self.beta via an EMA of the gradient-balancing
        ratio between compute_data_loss and compute_physics_loss.

        Parameters
        ----------
        data_loss : Tensor, scalar
            Current-step data loss (from compute_data_loss).
        physics_loss : Tensor, scalar
            Current-step physics loss (from compute_physics_loss).

        Returns
        -------
        Tuple[float, float]
            Updated (alpha, beta), also stored on self.
        """
        raise NotImplementedError("Leader: implement EMA-based alpha/beta update")

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        *physics_inputs: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the total adaptively-weighted loss for one training step.

        total_loss = alpha * data_loss + beta * physics_loss

        Parameters
        ----------
        y_pred : Tensor, shape (batch, 1)
            Model output for this batch.
        y_true : Tensor, shape (batch, 1)
            Ground-truth target for this batch.
        *physics_inputs : Tensor
            Forwarded to compute_physics_loss.

        Returns
        -------
        Tuple[Tensor, Dict[str, float]]
            (total_loss, log_dict) where log_dict contains at least
            {"data_loss": float, "physics_loss": float, "alpha": float, "beta": float}
            for logging in scripts/train_fcn.py.
        """
        raise NotImplementedError("Leader: implement forward (combine + update weights)")
