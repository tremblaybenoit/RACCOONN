import torch
import os
import numpy as np
from typing import Callable
import torch.nn as nn
try:
    from inverse.model.forward import CRTMForward
    from utilities.logic import get_config_path
    # Initialize CRTM forward model
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__name__), 'forward/model/checkpoints/model_v7.ckpt'))
    config_path = os.path.join(get_config_path(), 'model/forward_emulator2.yaml')
    forward = CRTMForward(checkpoint_path=checkpoint_path, config_path=config_path)
except (ImportError, FileNotFoundError, Exception) as e:
    print(f"Error loading CRTM forward model: {e}")
    forward = None


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Mean Squared Error loss function.

    Parameters
    ----------
    pred: torch.Tensor. Predicted tensor.
    target: torch.Tensor. True values.

    Returns
    -------
    torch.Tensor. Mean squared error over the batch.
    """
    return (pred - target) ** 2


class MSE(torch.nn.Module):
    """ Mean Squared Error loss module."""
    def __init__(self):
        """ Initialize the MSE module.

        Returns
        -------
        None.
        """
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the mean squared error between predicted and target tensors.

        Parameters
        ----------
        pred: torch.Tensor. Predicted tensor.
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Mean squared error over the batch.
        """
        return mse(pred, target)


class CholeskyForm(torch.nn.Module):
    """
    Stable Mahalanobis Loss: 0.5 * || L^-1 (pred - target) ||^2
    """

    def __init__(self, matrix: np.ndarray):
        super().__init__()
        # Ensure L is stored as a buffer (device management)
        self.register_buffer('matrix', torch.from_numpy(matrix).float())

    def to(self, device):
        super().to(device)
        self.matrix = self.matrix.to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Flatten to [Batch, N, 1]
        # order: (samples, vars, levels) -> (samples, vars * levels, 1)
        diff = (pred - target).view(pred.shape[0], -1, 1)

        # 2. Solve L @ w = diff for w
        # w = L^-1 @ diff. This is the 'whitened' residual.
        # Since L is lower triangular, this is a very fast/stable back-substitution.
        whitened_diff = torch.linalg.solve_triangular(
            self.matrix, diff, upper=False
        )

        # 3. Return 0.5 * sum(w^2) per batch
        # Result is [Batch]
        return 0.5 * torch.pow(whitened_diff, 2).sum(dim=1).squeeze()


class VerticalSmoothnessLoss(nn.Module):
    """
    Penalizes non-smooth vertical profiles using finite differences.
    Works for 1st order (gradients) or 2nd order (curvature/Laplacian).
    """
    def __init__(self, lambda_smooth: float = 0.1, order: int = 1, edge_weight: float = 1.0):
        """
        Args:
            lambda_smooth: Scaling factor for the penalty.
            order: 1 for first-order (prevents jumps), 2 for second-order (prevents kinks).
            edge_weight: Weight multiplier for the top/bottom of the atmosphere.
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.order = order
        self.edge_weight = edge_weight

    def __call__(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Tensor of shape (Batch, Variables, Levels)
                  e.g., (32, 9, 127)
        Returns:
            Scalar loss value.
        """
        if self.order == 1:
            # First-order difference: x_{i+1} - x_i
            # Result shape: (Batch, Vars, Levels-1)
            diff = pred[:, :, 1:] - pred[:, :, :-1]
        elif self.order == 2:
            # Second-order difference (Laplacian): x_{i+1} - 2x_i + x_{i-1}
            # Result shape: (Batch, Vars, Levels-2)
            diff = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]
        else:
            raise ValueError("Order must be 1 or 2.")

        # Square the differences and average over the batch and levels
        # We use .mean() to keep the loss scale independent of the number of levels
        smoothness_penalty = torch.pow(diff, 2).mean()

        return self.lambda_smooth * smoothness_penalty


def quadratic_form(pred: torch.Tensor, target: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """ Compute the quadratic form of the difference between predicted and target tensors.

    Parameters
    ----------
    pred: torch.Tensor. Predicted tensor.
    target: torch.Tensor. True values.
    matrix: torch.Tensor. Matrix to compute the quadratic form with.

    Returns
    -------
    torch.Tensor. Quadratic form of the difference between predicted and target tensors.
    """

    # Compute difference vector
    dv = pred - target

    return torch.einsum('bi,ij,bj->b', dv, matrix, dv)


class QuadraticForm(torch.nn.Module):
    """ Quadratic form loss module."""
    def __init__(self, matrix: np.ndarray):
        """ Initialize the QuadraticForm module.

        Parameters
        ----------
        matrix: np.ndarray. Matrix to compute the quadratic form with.

        Returns
        -------
        None.
        """
        super().__init__()
        self.matrix = torch.from_numpy(matrix)

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """
        super().to(device)
        self.matrix = self.matrix.to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the quadratic form of the difference between predicted and target tensors.

        Parameters
        ----------
        pred: torch.Tensor. Predicted tensor.
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Quadratic form of the difference between predicted and target tensors.
        """
        return quadratic_form(pred.view(pred.shape[0], -1), target.view(pred.shape[0], -1),
                              self.matrix)


def diagonal_quadratic_form(pred: torch.Tensor, target: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    """ Compute the diagonal quadratic form of the difference between predicted and target tensors.

    Parameters
    ----------
    pred: torch.Tensor. Predicted tensor.
    target: torch.Tensor. True values.
    diag: torch.Tensor. Diagonal elements of the matrix to compute the quadratic form with.

    Returns
    -------
    torch.Tensor. Diagonal quadratic form of the difference between predicted and target tensors.
    """

    # Minor adjustment to avoid division by zero
    eps = 1.e-8
    denom = torch.clamp(diag, min=eps)
    return mse(pred, target)/denom**2


class DiagonalQuadraticForm(torch.nn.Module):
    """ Diagonal quadratic form loss module."""
    def __init__(self):
        """ Initialize the DiagonalQuadraticForm module.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
        """ Compute the diagonal quadratic form of the difference between predicted and target tensors.

        Parameters
        ----------
        pred: torch.Tensor. Predicted tensor.
        target: torch.Tensor. True values.
        diag: torch.Tensor. Diagonal elements of the matrix to compute the quadratic form with.

        Returns
        -------
        torch.Tensor. Diagonal quadratic form of the difference between predicted and target tensors.
        """
        return diagonal_quadratic_form(pred, target, diag)


def kl_divergence_normal(mu_pred, sigma_pred, mu_target, sigma_target):
    """
    Compute the KL divergence between two normal distributions.

    Parameters
    ----------
    mu_pred: torch.Tensor. Mean of the predicted normal distribution.
    sigma_pred: torch.Tensor. Standard deviation of the predicted normal distribution.
    mu_target: torch.Tensor. Mean of the target normal distribution.
    sigma_target: torch.Tensor. Standard deviation of the target normal distribution.

    Returns
    -------
    kl_div: torch.Tensor. KL divergence between the predicted and target normal distributions.
    """

    # Compute variances
    var_pred = sigma_pred ** 2
    var_target = sigma_target ** 2

    # Compute KL divergence between two normal distributions
    return torch.log(sigma_target / sigma_pred) + (var_pred + (mu_pred - mu_target) ** 2) / (2 * var_target) - 0.5


class KLDivNormal(torch.nn.Module):
    """ KL divergence loss module."""
    def __init__(self):
        """ Initialize the KLDivNormal module.

        Returns
        -------
        None.
        """
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the KL divergence between predicted and target normal distributions.

        Parameters
        ----------
        pred: torch.Tensor. Tensor containing predictions: [mean, std].
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Mean KL divergence over the batch.
        """

        # Split input
        mu_pred = pred[:, :10]
        sigma_pred = pred[:, 10:]
        mu_target = target[:, :10]
        sigma_target = target[:, 10:]

        return kl_divergence_normal(mu_pred, sigma_pred, mu_target, sigma_target)


def wasserstein_2_normal(mu_pred, sigma_pred, mu_target, sigma_target):
    """
    Compute the Wasserstein-2 distance between two normal distributions.

    Parameters
    ----------
    mu_pred: torch.Tensor. Mean of the predicted normal distribution.
    sigma_pred: torch.Tensor. Standard deviation of the predicted normal distribution.
    mu_target: torch.Tensor. Mean of the target normal distribution.
    sigma_target: torch.Tensor. Standard deviation of the target normal distribution.

    Returns
    -------
    wasserstein_distance: torch.Tensor. Wasserstein-2 distance between the predicted and target normal distributions.
    """
    return torch.sqrt((mu_pred - mu_target) ** 2 + (sigma_pred - sigma_target) ** 2)


class Wasserstein2Normal(torch.nn.Module):
    """ Wasserstein-2 distance loss module."""
    def __init__(self):
        """ Initialize the Wasserstein2Normal module.

        Returns
        -------
        None.
        """
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute the Wasserstein-2 distance between predicted and target normal distributions.

        Parameters
        ----------
        pred: torch.Tensor. Tensor containing predictions: [mean, std].
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Mean Wasserstein-2 distance over the batch.
        """

        # Split input
        mu_pred = pred[:, :10]
        sigma_pred = pred[:, 10:]
        mu_target = target[:, :10]
        sigma_target = target[:, 10:]

        return wasserstein_2_normal(mu_pred, sigma_pred, mu_target, sigma_target)


class ForwardModel(torch.nn.Module):
    """ Forward loss module."""
    def __init__(self, prof_norm: Callable = None, dtype: str = None):
        """ Initialize the Forward module.

        Parameters
        ----------
        prof_norm: Callable. Function to apply normalization to profiles before the forward model.

        Returns
        -------
        None.
        """
        super().__init__()

        # Forward operator
        self.forward_model = forward
        if dtype:
            self.forward_model.to(None, dtype=getattr(torch, dtype))
        # Apply normalization to profiles
        self.prof_norm = prof_norm

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """
        # Assign forward model to the specified device
        if hasattr(self.forward_model, 'to'):
            self.forward_model = self.forward_model.to(device)
        # Assign profile normalization function to the specified device if it has a 'to' method
        if self.prof_norm is not None and hasattr(self.prof_norm, 'to'):
            self.prof_norm = self.prof_norm.to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: dict) -> torch.Tensor:
        """ Compute loss between a forward-modeled prediction and target.

        Parameters
        ----------
        pred: torch.Tensor. Predicted profile tensor.
        target: dict. Dictionary containing target tensors.

        Returns
        -------
        loss: torch.Tensor. Loss between the forward-modeled prediction and the target.
        """

        # Apply the forward model to the prediction
        input = {'prof': pred if self.prof_norm is None else self.prof_norm(pred),
                 'surf': target['surf'], 'meta': target['meta']}
        forward_pred = self.forward_model(input)

        return forward_pred


class VarLoss(torch.nn.Module):
    """ Universal loss module that combines observation and model losses. """
    def __init__(self, forward_model: Callable, loss_obs: Callable, loss_model: Callable = None, loss_bcs: Callable = None,
                 lambda_obs: float=1.0, lambda_model: float=1.0, lambda_bcs: float=1.0,
                 pressure_filter: np.ndarray=None, clear_sky: bool=False, prof_pred: np.ndarray=None):
        """ Initialize the variational loss module.

        Parameters
        ----------
        forward_model: Callable. Function to apply the forward model.
        loss_obs: Callable or ListConfig. Loss function(s) for observations.
        loss_model: Callable or ListConfig. Loss function(s) for model predictions.
        loss_bcs: Callable or ListConfig. Loss function(s) for boundary conditions.
        lambda_obs: float. Weight for the observation loss.
        lambda_model: float. Weight for the model loss.
        lambda_bcs: float. Weight for the boundary condition loss.
        pressure_filter: Callable. Function to generate a mask for the profile levels to include in the model loss.
        clear_sky: bool. Whether to apply clear-sky filtering.
        prof_pred: np.ndarray. Base profile for clear-sky filtering.

        Returns
        -------
        None.
        """

        super().__init__()
        # Forward model
        self.forward_model = forward_model
        # Loss terms
        self.loss_obs, self.loss_model, self.loss_bcs = loss_obs, loss_model, loss_bcs
        # Weighting factors for the losses
        self.lambda_obs, self.lambda_model, self.lambda_bcs = lambda_obs, lambda_model, lambda_bcs
        # Pressure mask per profile type
        self.pressure_filter = torch.from_numpy(pressure_filter) \
            if pressure_filter is not None and ~pressure_filter.sum() == 0 else None
        # Clear-sky filtering
        self.clear_sky = clear_sky
        if prof_pred is not None:
            # Shape is likely (1, 9, n_levels) based on your context
            self.register_buffer('prof_pred', torch.from_numpy(prof_pred))
        else:
            self.prof_pred = None

    def __call__(self, pred: dict, target: dict) -> tuple[dict, torch.Tensor]:
        """ Compute the combined loss between predicted profiles and target data.

        Parameters
        ----------
        pred: dict. Dictionary containing predicted tensors.
        target: dict. Dictionary containing target tensors.

        Returns
        -------
        loss: dict. Dictionary containing total, observation, and model losses.
        bt_pred: torch.Tensor. Forward-modeled brightness temperature predictions.
        """

        # Mask
        if self.pressure_filter is not None:
            pressure_filter = self.pressure_filter
        else:
            pressure_filter = torch.ones_like(pred['prof'], dtype=torch.bool, device=pred['prof'].device)

        # Compute the forward model output
        if self.clear_sky and self.prof_pred is not None:
            pred_prof = self.prof_pred[:pred['prof'].shape[0]].clone()
            pred_prof[:, 0:1, ...] = pred['prof'][:, 0:1, :]  #  Air temperature
            pred_prof[:, 4:5, ...] = pred['prof'][:, 1:2, :]  #  Ice particle effective radius
            pred_prof[:, 8:9, ...] = pred['prof'][:, 2:3, :]  #  Ozone mixing ratio
            hofx_pred = self.forward_model(pred_prof, target)
        elif self.clear_sky:
            pred_prof = torch.zeros((pred['prof'].shape[0], 9, pred['prof'].shape[2]), device=pred['prof'].device)
            pred_prof[:, 0:1, ...] = pred['prof'][:, 0:1, :]  #  Air temperature
            pred_prof[:, 4:5, ...] = pred['prof'][:, 1:2, :]  #  Ice particle effective radius
            pred_prof[:, 8:9, ...] = pred['prof'][:, 2:3, :]  #  Ozone mixing ratio
            hofx_pred = self.forward_model(pred_prof, target)
        else:
            hofx_pred = self.forward_model(pred['prof'], target)

        # Initialize loss dictionary
        loss = {}

        # Observation loss: Some observation losses may require additional inputs
        if isinstance(self.loss_obs, DiagonalQuadraticForm):
            loss['obs'] = self.loss_obs(hofx_pred[:, :10], target['hofx'][:, :10],
                                        target['hofx'][:, 10:])
        else:
            loss['obs'] = self.loss_obs(hofx_pred[:, :10],
                                        target['hofx'][:, :10])
        # Total
        loss['total'] = self.lambda_obs * loss['obs'].mean()

        # Model losses: Some model losses may require additional inputs
        if self.loss_model is not None:
            if isinstance(self.loss_model, DiagonalQuadraticForm):
                if self.pressure_filter is not None:
                    loss['model'] = self.loss_model(pred['prof'][:, pressure_filter],
                                                    target['prof_background'][:, pressure_filter],
                                                    target['prof_increment'])
                else:
                    loss['model'] = self.loss_model(pred['prof'],
                                                    target['prof_background'],
                                                    target['prof_increment'])
            else:
                if self.pressure_filter is not None:
                    loss['model'] = self.loss_model(pred['prof'][:, pressure_filter],
                                                    target['prof_background'][:, pressure_filter])
                else:
                    loss['model'] = self.loss_model(pred['prof'],
                                                    target['prof_background'])
            # Total
            loss['total'] += self.lambda_model * loss['model'].mean()

        # Boundary condition losses (where the variance is zero)
        if self.loss_bcs is not None and self.pressure_filter is not None:
            loss['bcs'] = self.loss_bcs(pred['prof'][:, ~pressure_filter], target['prof'][:, ~pressure_filter])
            # Total
            loss['total'] += self.lambda_bcs * loss['bcs'].mean()

        return loss, hofx_pred

    def to(self, device):
        """
        Move the module and its components to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module and its components to.

        Returns
        -------
        self: The module itself after moving to the specified device.
        """
        super().to(device)
        # Forward model
        if hasattr(self.forward_model, 'to'):
            self.forward_model = self.forward_model.to(device)
        # Loss functions
        for attr in ['loss_obs', 'loss_model', 'loss_bcs']:
            loss_fn = getattr(self, attr, None)
            if loss_fn is not None and hasattr(loss_fn, 'to'):
                setattr(self, attr, loss_fn.to(device))
        # Pressure filter
        if self.pressure_filter is not None:
            self.pressure_filter = self.pressure_filter.to(device)
        # Profile prediction
        if self.prof_pred is not None and hasattr(self.prof_pred, 'to'):
            self.prof_pred = self.prof_pred.to(device)
        return self


class UncertaintyVarLoss(VarLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize log-variances to 0 (which makes exp(0) = 1.0)
        self.log_var_obs = nn.Parameter(torch.zeros(1))
        self.log_var_model = nn.Parameter(torch.zeros(1))
        self.log_var_bcs = nn.Parameter(torch.zeros(1))

    def forward(self, pred: dict, target: dict) -> tuple[dict, torch.Tensor]:
        # Reuse your existing loss calculation logic
        loss_dict, hofx_pred = super().__call__(pred, target)

        # Apply the uncertainty weighting:
        # Total = (1 / exp(log_var)) * Loss + log_var

        # Observation term
        precision_obs = torch.exp(-self.log_var_obs)
        weighted_loss = precision_obs * loss_dict['obs'].mean() + self.log_var_obs

        # Model (Background) term
        if 'model' in loss_dict:
            precision_model = torch.exp(-self.log_var_model)
            weighted_loss += precision_model * loss_dict['model'].mean() + self.log_var_model

        # BCS term
        if 'bcs' in loss_dict:
            precision_bcs = torch.exp(-self.log_var_bcs)
            weighted_loss += precision_bcs * loss_dict['bcs'].mean() + self.log_var_bcs

        loss_dict['total'] = weighted_loss
        return loss_dict, hofx_pred


    class NormalizedUncertaintyVarLoss(VarLoss):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Learnable parameters
            self.log_var_obs = nn.Parameter(torch.ones(1) * 2.0)
            self.log_var_model = nn.Parameter(torch.zeros(1))

            # Buffers to store the initial loss values (not updated by gradients)
            self.register_buffer('init_obs', torch.tensor(1.0))
            self.register_buffer('init_model', torch.tensor(1.0))
            self.initialized = False

        def forward(self, pred: dict, target: dict) -> tuple[dict, torch.Tensor]:
            # Reuse your existing loss calculation logic
            loss_dict, hofx_pred = super().__call__(pred, target)

            # Capture initial scales on the very first step
            if not self.initialized and self.training:
                self.init_obs.fill_(loss_dict['obs'].detach().mean())
                self.init_model.fill_(loss_dict['model'].detach().mean())
                self.initialized = True

            # Observation term
            precision_obs = torch.exp(-self.log_var_obs)
            weighted_loss = precision_obs * loss_dict['obs'].mean()/ self.init_obs + self.log_var_obs

            # Model (Background) term
            if 'model' in loss_dict:
                precision_model = torch.exp(-self.log_var_model)
                weighted_loss += precision_model * loss_dict['model'].mean()/ self.init_model + self.log_var_model

            # BCS term
            if 'bcs' in loss_dict:
                precision_bcs = torch.exp(-self.log_var_bcs)
                weighted_loss += precision_bcs * loss_dict['bcs'].mean()/ self.init_model + self.log_var_bcs

            loss_dict['total'] = weighted_loss
            return loss_dict, hofx_pred


class FadingAnchorUncertaintyLoss(VarLoss):
    def __init__(self, start_obs_epoch=5, ramp_duration=30, **kwargs):
        super().__init__(**kwargs)
        # Learnable uncertainty parameters
        self.log_var_obs = nn.Parameter(torch.ones(1) * 2.0)
        self.log_var_model = nn.Parameter(torch.zeros(1))

        # Curriculum settings
        self.start_obs_epoch = start_obs_epoch
        self.ramp_duration = ramp_duration

    def forward(self, pred: dict, target: dict, current_epoch: int = 0) -> tuple[dict, torch.Tensor]:
        # 1. Compute Raw Losses
        # Note: We need to normalize scale as discussed (Obs is ~10^5, Model is ~10^-2)
        # We can use a hard-coded scale factor based on your observation
        loss_dict, hofx_pred = super().__call__(pred, target)

        # 2. Compute Progress Factor (0.0 to 1.0)
        # Use a sigmoid or linear ramp to shift focus
        if current_epoch < self.start_obs_epoch:
            progress = 0.0
        else:
            progress = min(1.0, (current_epoch - self.start_obs_epoch) / self.ramp_duration)

        # 3. Apply the "Guiding then Straying" Weights
        # Early on: weight_model is high, weight_obs is low
        # Later: weight_model decreases (straying), weight_obs increases (fitting)

        # Observation Weight: Learns to fit, but restricted by 'progress'
        # We multiply the precision by progress so it starts at 0 weight
        precision_obs = torch.exp(-self.log_var_obs) * progress
        loss_obs_term = precision_obs * loss_dict['obs'].mean() + (self.log_var_obs / (progress + 1e-6))

        # Model Weight: Starts as a strong guide, then becomes a weak regularizer
        # We decay the model precision as we gain confidence in observations
        model_decay = 1.0 - (0.9 * progress)  # Decays from 100% to 10% influence
        precision_model = torch.exp(-self.log_var_model) * model_decay
        loss_model_term = precision_model * loss_dict['model'].mean() + self.log_var_model

        loss_dict['total'] = loss_obs_term + loss_model_term
        return loss_dict, hofx_pred


class FadingAnchorUncertaintyLoss2(VarLoss):
    def __init__(self, start_obs_epoch=2, ramp_duration=30, **kwargs):
        super().__init__(**kwargs)
        self.log_var_obs = nn.Parameter(torch.ones(1) * 2.0)
        self.log_var_model = nn.Parameter(torch.ones(1) * 2.0)

        self.start_obs_epoch = start_obs_epoch
        self.ramp_duration = ramp_duration

        # Buffers to store the scale of the losses at the very first step
        self.register_buffer('scale_obs', torch.tensor(1.0))
        self.register_buffer('scale_model', torch.tensor(1.0))
        self.is_initialized = False

    def __call__(self, pred: dict, target: dict, current_epoch: int = 0) -> tuple[dict, torch.Tensor]:
        # 1. Get RAW losses from Parent VarLoss
        loss_dict, hofx_pred = super().__call__(pred, target)

        l_obs_raw = loss_dict['obs'].mean()
        l_model_raw = loss_dict['model'].mean()

        # 2. Initialize scales on the very first training step
        if not self.is_initialized and self.training:
            # We detach to ensure we don't backprop through the scale initialization
            self.scale_obs.fill_(l_obs_raw.detach())
            self.scale_model.fill_(l_model_raw.detach())
            self.is_initialized = True

        # 3. Normalize losses (bringing them to ~1.0 magnitude)
        # This makes the uncertainty parameters 's' operate on the same scale
        l_obs_norm = l_obs_raw / (self.scale_obs + 1e-8)
        l_model_norm = l_model_raw / (self.scale_model + 1e-8)

        # 4. Progress Factor for Curriculum
        if current_epoch < self.start_obs_epoch:
            progress = 0.0
        else:
            progress = min(1.0, (current_epoch - self.start_obs_epoch) / self.ramp_duration)

        # 5. Apply Weighted Terms
        # Obs: weight grows with progress
        s_obs = torch.nn.functional.softplus(self.log_var_obs)
        precision_obs = torch.exp(-s_obs) * progress
        loss_obs_term = precision_obs * l_obs_norm + (s_obs * progress)

        # Model: weight stays strong then slightly decays to let Obs dominate
        model_decay = 1.0 - (0.7 * progress)  # Decays to 30% of original 'anchor' strength
        s_model = torch.nn.functional.softplus(self.log_var_model)
        precision_model = torch.exp(-s_model) * model_decay
        loss_model_term = precision_model * l_model_norm + s_model

        loss_dict['total'] = loss_obs_term.mean() + loss_model_term.mean()
        return loss_dict, hofx_pred


class FadingAnchorUncertaintyLoss3(VarLoss):
    def __init__(self, start_obs_epoch=2, ramp_duration=30, **kwargs):
        super().__init__(**kwargs)
        # One parameter to rule them all.
        # Initializing at 0.0 means 50/50 importance.
        self.alpha = nn.Parameter(torch.zeros(1))

        self.start_obs_epoch = start_obs_epoch
        self.ramp_duration = ramp_duration

        self.register_buffer('scale_obs', torch.tensor(1.0))
        self.register_buffer('scale_model', torch.tensor(1.0))
        self.is_initialized = False

    def __call__(self, pred: dict, target: dict, current_epoch: int = 0) -> tuple[dict, torch.Tensor]:
        loss_dict, hofx_pred = super().__call__(pred, target)

        l_obs_raw = loss_dict['obs'].mean()
        l_model_raw = loss_dict['model'].mean()

        if not self.is_initialized and self.training:
            self.scale_obs.fill_(l_obs_raw.detach())
            self.scale_model.fill_(l_model_raw.detach())
            self.is_initialized = True

        l_obs_norm = l_obs_raw / (self.scale_obs + 1e-8)
        l_model_norm = l_model_raw / (self.scale_model + 1e-8)

        # Progress Factor
        progress = 0.0 if current_epoch < self.start_obs_epoch else \
            min(1.0, (current_epoch - self.start_obs_epoch) / self.ramp_duration)

        # --- Relative Weighting Logic ---
        # Sigmoid(alpha) gives a value [0, 1].
        # We multiply by 2 so the total weight sum is 2.0.
        if progress > 0:
            prob_obs = torch.sigmoid(self.alpha) * 2.0
            prob_model = 2.0 - prob_obs
        else:
            # By using a constant here, no gradient flows to self.alpha
            prob_obs = torch.tensor(1.0, device=self.alpha.device)
            prob_model = torch.tensor(1.0, device=self.alpha.device)

        # Apply curriculum multipliers
        # Observations start at 0 and grow to their "learned importance"
        w_obs = prob_obs * progress

        # Model starts at full importance and decays slightly to allow
        # observations to take the lead if needed.
        model_decay = 1.0 - (0.7 * progress)
        w_model = prob_model * model_decay

        loss_dict['total'] = (w_obs * l_obs_norm) + (w_model * l_model_norm)

        # Highly recommended: Log these for W&B / Tensorboard
        loss_dict['w_obs_effective'] = w_obs.detach()
        loss_dict['w_model_effective'] = w_model.detach()

        return loss_dict, hofx_pred