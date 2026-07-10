"""
Adaptive-weighted PINN loss.

Owner: Leader (Kieu/Kem/Cam should treat this module's public interface as
fixed and import it as-is; do not modify without syncing with the leader).

Combines a data-fitting loss (Temperature prediction vs ground truth) with a
physics-residual loss derived from the Lumped Capacitance Model, and
adaptively re-weights them using gradient-magnitude balancing tracked with an
exponential moving average (EMA).

Physics equation (Lumped Capacitance Model), confirmed by the leader:
    f = dT/dt + lambda1 * (V - V_ocv) * I + lambda2 * (T_amb - T) = 0
    Loss_physics = MSE(f)
lambda1, lambda2 are TRAINABLE (nn.Parameter), not fixed physical constants.

IMPORTANT -- scaling: BatteryDataset (src/data_loader.py) Min-Max scales all
features and the target to [0,1] before the model ever sees them. The physics
equation above is only physically meaningful in REAL units (seconds, Volts,
Amps, degrees C). Autograd differentiating the model's scaled output w.r.t.
its scaled input would NOT equal the real dT/dt, and the bilinear
(V - V_ocv) * I term does not factor through independent per-feature Min-Max
scaling by a constant either -- so all quantities in compute_physics_loss are
explicitly unscaled back to real units (via the fitted scalers' data_min_/
data_max_, applied as differentiable tensor ops so the autograd graph stays
intact) before the residual is evaluated. See compute_physics_loss below.

alpha/beta update rule (update_weights) is intentionally left unimplemented
pending the exact formula from the leader -- do not guess at this; it's core
math per project rule.
"""
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor

from src.data_loader import CURRENT_COL, FEATURE_COLS, TIME_COL, VOLTAGE_COL

OCV_COL = "OCV_Estimated"


class AdaptivePINNLoss(nn.Module):
    """Adaptive data/physics loss combiner for the PINN battery model.

    Parameters
    ----------
    model : nn.Module
        The network being trained (e.g. BatteryPINN_Cho2022). Its forward()
        must accept x of shape (batch, len(FEATURE_COLS)) and return
        (batch, 1) predicted temperature in the SAME scaled space as the
        BatteryDataset target.
    feature_scaler : MinMaxScaler
        The scaler fit on the TRAIN (DST) features -- pass
        `train_dataset.feature_scaler` from src.data_loader.build_datasets().
        Column order must match FEATURE_COLS.
    target_scaler : MinMaxScaler
        The scaler fit on the TRAIN (DST) target -- pass
        `train_dataset.target_scaler`.
    alpha_init : float
        Initial weight for the data loss term.
    beta_init : float
        Initial weight for the physics loss term.
    ema_decay : float
        Decay factor for the moving average used to smooth alpha/beta updates
        across steps (closer to 1.0 = slower/smoother adaptation).
    ambient_temp_c : float
        Chamber/ambient temperature T_amb in the Lumped Capacitance Model,
        in degrees C. Default 25.0 -- inferred from the "-25-" in the raw
        CALCE filenames (A1-007-DST-US06-FUDS-25-... / A1-007-OCV-25-...),
        which denotes the test chamber setpoint. Override if that assumption
        is wrong for a given cell/test.

    Attributes
    ----------
    alpha : float
        Current adaptive weight applied to the data loss term.
    beta : float
        Current adaptive weight applied to the physics loss term.
    lambda1, lambda2 : nn.Parameter
        Trainable Lumped Capacitance Model coefficients. Arbitrary small
        positive initial values (0.01) -- not specified by the equation, so
        treated as ordinary learnable weights. Must be included in the
        optimizer's parameter list alongside the model's own parameters,
        e.g.:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(loss_fn.parameters()), lr=...
            )
        (loss_fn.parameters() picks these up automatically since they're
        registered via nn.Parameter on this nn.Module.)
    """

    def __init__(
        self,
        model: nn.Module,
        feature_scaler: MinMaxScaler,
        target_scaler: MinMaxScaler,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        ema_decay: float = 0.9,
        ambient_temp_c: float = 25.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.ema_decay = ema_decay
        self.alpha: float = alpha_init
        self.beta: float = beta_init
        self.ambient_temp_c = ambient_temp_c

        self.register_buffer("feat_min", torch.tensor(feature_scaler.data_min_, dtype=torch.float32))
        self.register_buffer("feat_max", torch.tensor(feature_scaler.data_max_, dtype=torch.float32))
        self.register_buffer("target_min", torch.tensor(target_scaler.data_min_, dtype=torch.float32))
        self.register_buffer("target_max", torch.tensor(target_scaler.data_max_, dtype=torch.float32))

        self.lambda1 = nn.Parameter(torch.tensor(0.01))
        self.lambda2 = nn.Parameter(torch.tensor(0.01))

    def compute_data_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """MSE data loss, in the same (scaled) space as y_pred/y_true.

        Parameters
        ----------
        y_pred : Tensor, shape (batch, 1)
            Model-predicted temperature (scaled space).
        y_true : Tensor, shape (batch, 1)
            Ground-truth temperature (scaled space, from BatteryDataset).

        Returns
        -------
        Tensor, scalar
            Differentiable MSE loss.
        """
        return nn.functional.mse_loss(y_pred, y_true)

    def _unscale(self, x_col: Tensor, col_min: Tensor, col_max: Tensor) -> Tensor:
        """Invert Min-Max scaling for a single column, staying in-graph
        (plain differentiable tensor ops -- NOT sklearn's .inverse_transform,
        which would detach from autograd)."""
        return x_col * (col_max - col_min) + col_min

    def compute_physics_loss(self, x: Tensor, T_pred_scaled: Tensor) -> Tensor:
        """Lumped Capacitance Model residual loss.

            f = dT/dt + lambda1*(V - V_ocv)*I + lambda2*(T_amb - T)
            Loss_physics = mean(f**2)

        All quantities are converted to real physical units before evaluating
        f (see module docstring for why this matters).

        Parameters
        ----------
        x : Tensor, shape (batch, len(FEATURE_COLS)), requires_grad=True
            SCALED model input (same tensor passed to `self.model`), columns
            ordered per FEATURE_COLS = [Time, Current, Voltage, OCV_Estimated].
            Caller (forward()) is responsible for setting requires_grad=True
            before calling self.model(x), so d(T_pred)/d(x) is defined.
        T_pred_scaled : Tensor, shape (batch, 1)
            self.model(x) -- MUST be connected to x's autograd graph (i.e.
            computed from this exact x, not detached / a copy).

        Returns
        -------
        Tensor, scalar
            Differentiable physics residual loss.
        """
        time_idx = FEATURE_COLS.index(TIME_COL)
        current_idx = FEATURE_COLS.index(CURRENT_COL)
        voltage_idx = FEATURE_COLS.index(VOLTAGE_COL)
        ocv_idx = FEATURE_COLS.index(OCV_COL)

        # T back to real degrees C (differentiable affine unscale).
        T_real = self._unscale(T_pred_scaled, self.target_min, self.target_max)

        # dT_real/d(x_time_scaled) via autograd; create_graph=True so this
        # term can itself be backpropagated into the model's parameters.
        dT_dx = torch.autograd.grad(
            outputs=T_real,
            inputs=x,
            grad_outputs=torch.ones_like(T_real),
            create_graph=True,
            retain_graph=True,
        )[0]
        dT_dtscaled = dT_dx[:, time_idx: time_idx + 1]

        # Chain rule: t_scaled = (t_real - t_min)/(t_max - t_min)
        #   => d(t_scaled)/d(t_real) = 1/(t_max - t_min)
        #   => dT_real/dt_real = dT_real/d(t_scaled) * d(t_scaled)/d(t_real)...
        #      i.e. divide by (t_max - t_min).
        time_min = self.feat_min[time_idx]
        time_max = self.feat_max[time_idx]
        dT_dt_real = dT_dtscaled / (time_max - time_min)

        V_real = self._unscale(x[:, voltage_idx: voltage_idx + 1], self.feat_min[voltage_idx], self.feat_max[voltage_idx])
        Vocv_real = self._unscale(x[:, ocv_idx: ocv_idx + 1], self.feat_min[ocv_idx], self.feat_max[ocv_idx])
        I_real = self._unscale(x[:, current_idx: current_idx + 1], self.feat_min[current_idx], self.feat_max[current_idx])

        residual = (
            dT_dt_real
            + self.lambda1 * (V_real - Vocv_real) * I_real
            + self.lambda2 * (self.ambient_temp_c - T_real)
        )
        return torch.mean(residual ** 2)

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
            Returns 0.0 if no parameter received a gradient (e.g. loss does
            not depend on any of them).
        """
        params = [p for p in params if p.requires_grad]
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        abs_maxes = [g.detach().abs().max() for g in grads if g is not None]
        if not abs_maxes:
            return torch.tensor(0.0, device=loss.device)
        return torch.stack(abs_maxes).max()

    def update_weights(self, data_loss: Tensor, physics_loss: Tensor) -> Tuple[float, float]:
        """Update self.alpha / self.beta via an EMA of a gradient-balancing
        ratio between compute_data_loss and compute_physics_loss.

        NOT YET IMPLEMENTED -- the exact update rule (which loss's max vs
        mean, whether alpha is fixed at 1 or also adapts, EMA direction) is
        pending confirmation from the leader/Strategic Planner. Do not guess
        at this formula; it's core math per project rule. `_max_abs_grad` is
        ready to use once the rule is specified.

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
        raise NotImplementedError("Leader: implement EMA-based alpha/beta update once formula is confirmed")

    def forward(self, x: Tensor, y_true: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the total adaptively-weighted loss for one training step.

        total_loss = alpha * data_loss + beta * physics_loss

        Parameters
        ----------
        x : Tensor, shape (batch, len(FEATURE_COLS))
            SCALED model input for this batch (order: Time, Current, Voltage,
            OCV_Estimated). Note: this replaces the earlier skeleton's
            (y_pred, y_true, *physics_inputs) signature -- computing the
            physics residual requires differentiating the model's own output
            w.r.t. its raw input (for dT/dt), so this module now runs
            `self.model(x)` internally rather than accepting a precomputed,
            possibly-detached y_pred.
        y_true : Tensor, shape (batch, 1)
            Ground-truth target for this batch (scaled space).

        Returns
        -------
        Tuple[Tensor, Dict[str, float]]
            (total_loss, log_dict) where log_dict contains
            {"data_loss", "physics_loss", "alpha", "beta", "lambda1", "lambda2"}
            for logging in scripts/train_fcn.py.
        """
        x = x.clone().requires_grad_(True)
        T_pred_scaled = self.model(x)

        data_loss = self.compute_data_loss(T_pred_scaled, y_true)
        physics_loss = self.compute_physics_loss(x, T_pred_scaled)

        alpha, beta = self.update_weights(data_loss, physics_loss)
        total_loss = alpha * data_loss + beta * physics_loss

        log_dict = {
            "data_loss": data_loss.item(),
            "physics_loss": physics_loss.item(),
            "alpha": alpha,
            "beta": beta,
            "lambda1": self.lambda1.item(),
            "lambda2": self.lambda2.item(),
        }
        return total_loss, log_dict
