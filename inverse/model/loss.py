import torch
import os
import numpy as np
from typing import Callable
import torch.nn as nn
from torch.nn import MSELoss


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


def diagonal_quadratic_huberform(pred: torch.Tensor, target: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
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

    # Compute Huber loss
    delta = 1.0
    diff = (pred - target)/denom
    abs_diff = torch.abs(diff)
    quadratic = torch.minimum(abs_diff, torch.tensor(delta))
    linear = abs_diff - quadratic
    huber_loss = 0.5 * quadratic ** 2 + delta * linear
    return huber_loss


class DiagonalQuadraticHuberForm(torch.nn.Module):
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
        return diagonal_quadratic_huberform(pred, target, diag)


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
    denom = torch.clamp(torch.abs(diag), min=eps)
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
    def __init__(self, checkpoint_path: str = 'forward/model/checkpoints/model_v3.ckpt',
                 config_path: str = 'model/forward_emulator.yaml', prof_norm: Callable = None, dtype: str = None):
        """ Initialize the Forward module.

        Parameters
        ----------
        checkpoint_path: str. Path to the checkpoint file for the forward model.
        config_path: str. Path to the configuration file for the forward model.
        prof_norm: Callable. Function to apply normalization to profiles before the forward model.
        dtype: str. Data type to cast the forward model to (e.g., 'float32', 'float64').

        Returns
        -------
        None.
        """
        super().__init__()

        # Import CRTM forward model
        try:
            from inverse.model.forward import CRTMForward
            from utilities.logic import get_config_path
            # Initialize CRTM forward model
            checkpoint_path = os.path.abspath(
                os.path.join(os.path.dirname(__name__), checkpoint_path))
            config_path = os.path.join(get_config_path(), config_path)
            forward = CRTMForward(checkpoint_path=checkpoint_path, config_path=config_path)
        except (ImportError, FileNotFoundError, Exception) as e:
            print(f"Error loading CRTM forward model: {e}")
            forward = None

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
        input = {'prof': pred if self.prof_norm is None else self.prof_norm(pred.clone()),
                 'surf': target['surf'], 'meta': target['meta']}
        forward_pred = self.forward_model(input)

        return forward_pred


class SobolevRegularization(torch.nn.Module):
    """
    Sobolev Regularization loss module to penalize the magnitude of spatial gradients.
    Enforces smoothness by minimizing ||grad(f(x))||^2.
    """
    def __init__(self, input_keys: list = None):
        """
        Parameters
        ----------
        input_keys: list. The keys in the batch dict to differentiate against.
        """
        super().__init__()
        self.input_keys = ['lat', 'lon'] if input_keys is None else input_keys

    def __call__(self, pred: torch.Tensor, coords: dict) -> torch.Tensor:
        """
        Compute the Sobolev penalty.

        Parameters
        ----------
        pred: torch.Tensor. Predicted output (e.g., prof_white).
                           Shape: (Batch, 270)
        coords: dict. Dictionary containing input tensors with requires_grad=True.

        Returns
        -------
        torch.Tensor. Mean squared gradient across the batch.
        """

        # Ensure we have a flattened representation for differentiation
        # (Batch, N)
        y = pred.view(pred.shape[0], -1)

        # We want to find the gradient of the model output with respect to inputs.
        # Since we want a single scalar loss to minimize, we compute the gradient
        # of the sum of outputs, which is mathematically equivalent to the
        # sum of the gradients for each output feature.
        grad_outputs = torch.ones_like(y)

        # Select input tensors that exist in coords
        inputs = [coords[k] for k in self.input_keys if k in coords]

        # Calculate Jacobian-vector product
        # creates_graph=True allows the optimizer to backpropagate through this gradient
        grads = torch.autograd.grad(
            outputs=y,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )

        total_grad_loss = torch.tensor(0.0, device=pred.device)
        for g in grads:
            if g is not None:
                # g will have the same shape as the input (Batch, 1)
                # We penalize the square of the derivative
                total_grad_loss += torch.mean(g**2)

        return total_grad_loss


class VarLoss(torch.nn.Module):
    """ Universal loss module that combines observation and model losses. """
    def __init__(self, forward_model: Callable, loss_obs: Callable, loss_model: Callable = None, loss_bcs: Callable = None,
                 lambda_obs: float=1.0, lambda_model: float=1.0, lambda_bcs: float=1.0, lambda_sobolev: float=0.0,
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
        lambda_sobolev: float. Weight for the Sobolev regularization loss.
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
        self.lambda_sobolev = lambda_sobolev
        self.sobolev_loss_fn = SobolevRegularization(input_keys=['lat', 'lon', 'scans', 'pressure'])
        # Pressure mask per profile type
        self.pressure_filter = torch.from_numpy(pressure_filter) \
            if pressure_filter is not None else None
        self.pressure_mask = torch.from_numpy(pressure_filter.astype('float32')) if pressure_filter is not None else None
        # Clear-sky filtering
        self.clear_sky = clear_sky
        if prof_pred is not None:
            # Shape is likely (1, 9, n_levels) based on your context
            self.register_buffer('prof_pred', torch.from_numpy(prof_pred))
        else:
            self.prof_pred = None

    def __call__(self, pred: dict, target: dict, input: dict=None) -> tuple[dict, torch.Tensor]:
        """ Compute the combined loss between predicted profiles and target data.

        Parameters
        ----------
        pred: dict. Dictionary containing predicted tensors.
        target: dict. Dictionary containing target tensors.
        input: dict. Dictionary containing input tensors for Sobolev loss.

        Returns
        -------
        loss: dict. Dictionary containing total, observation, and model losses.
        bt_pred: torch.Tensor. Forward-modeled brightness temperature predictions.
        """

        # Mask
        if self.pressure_filter is not None:
            pressure_filter = self.pressure_filter
            # pred['prof'][:, ~pressure_filter] = target['prof_background'][:, ~pressure_filter]
        else:
            pressure_filter = torch.ones_like(pred['prof'], dtype=torch.bool, device=pred['prof'].device)

        # Compute the forward model output
        if self.clear_sky:
            if self.prof_pred is not None:
                pred_prof = self.prof_pred[:pred['prof'].shape[0]].clone()
            else:
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

        # Sobolev regularization loss
        if self.lambda_sobolev > 0.0:
            loss['sobolev'] = self.sobolev_loss_fn(pred['prof'], input)
            loss['total'] += self.lambda_sobolev * loss['sobolev'].mean()

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


class VarLossP(VarLoss):

    def __call__(self, pred: dict, target: dict, input: dict=None) -> tuple[dict, torch.Tensor]:
        """ Compute the combined loss between predicted profiles and target data.

        Parameters
        ----------
        pred: dict. Dictionary containing predicted tensors.
        target: dict. Dictionary containing target tensors.
        input: dict. Dictionary containing input tensors for Sobolev loss.

        Returns
        -------
        loss: dict. Dictionary containing total, observation, and model losses.
        bt_pred: torch.Tensor. Forward-modeled brightness temperature predictions.
        """

        # Mask
        if self.pressure_filter is not None:
            pressure_filter = self.pressure_filter
            pred['prof'][:, ~pressure_filter] = target['prof_background'][:, ~pressure_filter]
            pred['prof_phys'][:, ~pressure_filter] = pred['prof_background_phys'][:, ~pressure_filter]
            pred['prof_min_max'][:, ~pressure_filter] = pred['prof_background_min_max'][:, ~pressure_filter]
            pred['prof_mean_stdev'][:, ~pressure_filter] = pred['prof_background_mean_stdev'][:, ~pressure_filter]
        else:
            pressure_filter = torch.ones_like(pred['prof_phys'], dtype=torch.bool, device=pred['prof_phys'].device)

        # Compute the forward model output
        if self.clear_sky:
            if self.prof_pred is not None:
                pred_prof = self.prof_pred[:pred['prof'].shape[0]].clone()
            else:
                pred_prof = torch.zeros((pred['prof'].shape[0], 9, pred['prof'].shape[2]), device=pred['prof'].device)
            pred_prof[:, 0:1, ...] = pred['prof'][:, 0:1, :]  #  Air temperature
            pred_prof[:, 4:5, ...] = pred['prof'][:, 1:2, :]  #  Ice particle effective radius
            pred_prof[:, 8:9, ...] = pred['prof'][:, 2:3, :]  #  Ozone mixing ratio
            hofx_pred = self.forward_model(pred_prof, target)
        else:
            hofx_pred = self.forward_model(pred['prof_min_max'], target)

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
                    loss['model'] = self.loss_model(pred['prof_mean_stdev'][:, pressure_filter],
                                                    pred['prof_background_mean_stdev'][:, pressure_filter])
                else:
                    loss['model'] = self.loss_model(pred['prof_mean_stdev'],
                                                    pred['prof_background_mean_stdev'])
            # Total
            loss['total'] += self.lambda_model * loss['model'].mean()

        # Sobolev regularization loss
        if self.lambda_sobolev > 0.0:
            loss['sobolev'] = self.sobolev_loss_fn(pred['prof'], input)
            loss['total'] += self.lambda_sobolev * loss['sobolev'].mean()

        # Boundary condition losses (where the variance is zero)
        if self.loss_bcs is not None and self.pressure_filter is not None:
            loss['bcs'] = self.loss_bcs(pred['prof'][:, ~pressure_filter], target['prof'][:, ~pressure_filter])
            # Total
            loss['total'] += self.lambda_bcs * loss['bcs'].mean()

        return loss, hofx_pred


class VarLossPCA(torch.nn.Module):
    """ Universal loss module that combines observation and model losses. """
    def __init__(self, forward_model: Callable, loss_obs: Callable, loss_model: Callable = None,
                 loss_bcs: Callable = None, lambda_obs: float=1.0, lambda_model: float=1.0, lambda_bcs: float=1.0):
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

            Returns
            -------
            None.
        """

        # Class inheritance
        super().__init__()

        # Forward model
        self.forward_model = forward_model
        # Loss terms
        self.loss_obs, self.loss_model, self.loss_bcs = loss_obs, loss_model, loss_bcs
        # Weighting factors for the losses
        self.lambda_obs, self.lambda_model, self.lambda_bcs = lambda_obs, lambda_model, lambda_bcs

    def __call__(self, pred: dict, target: dict, input: dict=None) -> tuple[dict, torch.Tensor]:
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

        # Compute the forward model output
        hofx_pred = self.forward_model(pred['prof_white'], target)

        # Initialize loss dictionary
        loss = {}

        # Observation loss: Some observation losses may require additional inputs
        if isinstance(self.loss_obs, DiagonalQuadraticForm):
            loss['obs'] = self.loss_obs(hofx_pred[:, :10], target['hofx'][:, :10], target['hofx'][:, 10:])
        else:
            loss['obs'] = self.loss_obs(hofx_pred[:, :10], target['hofx'][:, :10])
        # Total
        loss['total'] = self.lambda_obs * loss['obs'].mean()

        # Model losses: Some model losses may require additional inputs
        if self.loss_model is not None:
            if isinstance(self.loss_model, (DiagonalQuadraticForm, DiagonalQuadraticHuberForm)):
                loss['model'] = self.loss_model(pred['prof_white'],
                                                target['prof_white_background'],
                                                target['prof_white_increment'])
            else:
                loss['model'] = self.loss_model(pred['prof_white'],
                                                target['prof_white_background'])
            # Total
            loss['total'] += self.lambda_model * loss['model'].mean()

        # Boundary condition losses (where the variance is zero)
        # if self.loss_bcs is not None:
        #     raise NotImplementedError("Boundary condition loss is not implemented for PCA-based profiles.")

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

        return self


class VarLossHybridPCA(torch.nn.Module):
    """ Universal loss module that combines observation and model losses. """
    def __init__(self, forward_model: Callable, loss_obs: Callable, loss_model: Callable = None, loss_bcs: Callable = None,
                 lambda_obs: float=1.0, lambda_model: float=1.0, lambda_bcs: float=1.0, lambda_sobolev: float=0.0,
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
        lambda_sobolev: float. Weight for the Sobolev regularization loss.
        pressure_filter: Callable. Function to generate a mask for the profile levels to include in the model loss.
        clear_sky: bool. Whether to apply clear-sky filtering.
        prof_pred: np.ndarray. Base profile for clear-sky filtering.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()
        # Forward model
        self.forward_model = forward_model
        # Loss terms
        self.loss_obs, self.loss_model, self.loss_bcs = loss_obs, loss_model, loss_bcs
        # Weighting factors for the losses
        self.lambda_obs, self.lambda_model, self.lambda_bcs = lambda_obs, lambda_model, lambda_bcs
        self.lambda_sobolev = lambda_sobolev
        self.sobolev_loss_fn = SobolevRegularization(input_keys=['lat', 'lon'])
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

    def __call__(self, pred: dict, target: dict, input: dict) -> tuple[dict, torch.Tensor]:
        """ Compute the combined loss between predicted profiles and target data.

        Parameters
        ----------
        pred: dict. Dictionary containing predicted tensors.
        target: dict. Dictionary containing target tensors.
        input: dict. Dictionary containing input tensors for Sobolev loss.

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
        if self.clear_sky:
            if self.prof_pred is not None:
                pred_prof = self.prof_pred[:pred['prof'].shape[0]].clone()
            else:
                pred_prof = torch.zeros((pred['prof'].shape[0], 9, pred['prof'].shape[2]), device=pred['prof'].device)
            pred_prof[:, 0:1, ...] = pred['prof'][:, 0:1, :]  # Air temperature
            pred_prof[:, 4:5, ...] = pred['prof'][:, 1:2, :]  # Ice particle effective radius
            pred_prof[:, 8:9, ...] = pred['prof'][:, 2:3, :]  # Ozone mixing ratio
            hofx_pred = self.forward_model(pred_prof, target)
        else:
            hofx_pred = self.forward_model(pred['prof'], target)

        # Initialize loss dictionary
        loss = {}

        # Observation loss: Some observation losses may require additional inputs
        if isinstance(self.loss_obs, DiagonalQuadraticForm):
            loss['obs'] = self.loss_obs(hofx_pred[:, :10], target['hofx'][:, :10], target['hofx'][:, 10:])
        else:
            loss['obs'] = self.loss_obs(hofx_pred[:, :10], target['hofx'][:, :10])
        # Total
        loss['total'] = self.lambda_obs * loss['obs'].mean()

        # Sobolev regularization loss
        loss['sobolev'] = self.sobolev_loss_fn(pred['prof_white'], input)  # TODO: Switch to prof (physical space) if needed
        loss['total'] += self.lambda_sobolev * loss['sobolev'].mean()

        # Model losses: Some model losses may require additional inputs
        if self.loss_model is not None:
            if isinstance(self.loss_model, (DiagonalQuadraticForm, DiagonalQuadraticHuberForm)):
                loss['model'] = self.loss_model(pred['prof_white'],
                                                target['prof_white_background'],
                                                target['prof_white_increment'])
            else:
                loss['model'] = self.loss_model(pred['prof_white'],
                                                target['prof_white_background'])
            # Total
            loss['total'] += self.lambda_model * loss['model'].mean()

        # Boundary condition losses (where the variance is zero)
        # if self.loss_bcs is not None:
        #     raise NotImplementedError("Boundary condition loss is not implemented for PCA-based profiles.")

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

        return self

class VarLossHybridPCA2(torch.nn.Module):
    """ Universal loss module that combines observation and model losses. """
    def __init__(self, forward_model: Callable, loss_obs: Callable, loss_model: Callable = None, loss_bcs: Callable = None,
                 lambda_obs: float=1.0, lambda_model: float=1.0, lambda_bcs: float=1.0, lambda_sobolev: float=0.0,
                 lambda_model_phys: float=0.0, forward_channels: list=None,
                 pressure_filter: np.ndarray=None, clear_sky: bool=False, prof_pred: np.ndarray=None):
        """ Initialize the variational loss module.

        Parameters
        ----------
        forward_model: Callable. Function to apply the forward model.
        forward_channels: list. List of channel indices to include in the forward model loss.
        loss_obs: Callable or ListConfig. Loss function(s) for observations.
        loss_model: Callable or ListConfig. Loss function(s) for model predictions.
        loss_bcs: Callable or ListConfig. Loss function(s) for boundary conditions.
        lambda_obs: float. Weight for the observation loss.
        lambda_model: float. Weight for the model loss.
        lambda_bcs: float. Weight for the boundary condition loss.
        lambda_model_phys: float. Weight for the model loss.
        lambda_sobolev: float. Weight for the Sobolev regularization loss.
        pressure_filter: Callable. Function to generate a mask for the profile levels to include in the model loss.
        clear_sky: bool. Whether to apply clear-sky filtering.
        prof_pred: np.ndarray. Base profile for clear-sky filtering.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()
        # Forward model
        self.forward_model = forward_model
        if forward_channels is None:
            self.forward_channels = None
        else:
            # register buffer so it moves with model.to(device)
            self.register_buffer('forward_channels', torch.tensor(forward_channels, dtype=torch.long))
        # Loss terms
        self.loss_obs, self.loss_model, self.loss_bcs = loss_obs, loss_model, loss_bcs
        # Weighting factors for the losses
        self.lambda_obs, self.lambda_model, self.lambda_bcs = lambda_obs, lambda_model, lambda_bcs
        self.lambda_model_phys = lambda_model_phys
        self.lambda_sobolev = lambda_sobolev
        self.model_loss_fn = MSE()
        self.sobolev_loss_fn = SobolevRegularization(input_keys=['lat', 'lon'])
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

    def __call__(self, pred: dict, target: dict, input: dict) -> tuple[dict, torch.Tensor]:
        """ Compute the combined loss between predicted profiles and target data.

        Parameters
        ----------
        pred: dict. Dictionary containing predicted tensors.
        target: dict. Dictionary containing target tensors.
        input: dict. Dictionary containing input tensors for Sobolev loss.

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
        if self.clear_sky:
            if self.prof_pred is not None:
                pred_prof = self.prof_pred[:pred['prof'].shape[0]].clone()
            else:
                pred_prof = torch.zeros((pred['prof'].shape[0], 9, pred['prof'].shape[2]), device=pred['prof'].device)
            pred_prof[:, 0:1, ...] = pred['prof'][:, 0:1, :]  # Air temperature
            pred_prof[:, 4:5, ...] = pred['prof'][:, 1:2, :]  # Ice particle effective radius
            pred_prof[:, 8:9, ...] = pred['prof'][:, 2:3, :]  # Ozone mixing ratio
            hofx_pred = self.forward_model(pred_prof, target)
        else:
            hofx_pred = self.forward_model(pred['prof'], target)

        # Initialize loss dictionary
        loss = {}

        # Observation loss: Some observation losses may require additional inputs
        if isinstance(self.loss_obs, DiagonalQuadraticForm):
            if self.forward_channels is not None:
                loss['obs'] = self.loss_obs(hofx_pred[:, self.forward_channels], target['hofx'][:, self.forward_channels],
                                            target['hofx'][:, 10:])
            else:
                loss['obs'] = self.loss_obs(hofx_pred[:, :10], target['hofx'][:, :10], target['hofx'][:, 10:])
        else:
            if self.forward_channels is not None:
                loss['obs'] = self.loss_obs(hofx_pred[:, self.forward_channels], target['hofx'][:, self.forward_channels])
            else:
                loss['obs'] = self.loss_obs(hofx_pred[:, :10], target['hofx'][:, :10])
        # Total
        loss['total'] = self.lambda_obs * loss['obs'].mean()

        # Sobolev regularization loss
        if self.lambda_sobolev > 0.0:
            loss['sobolev'] = self.sobolev_loss_fn(pred['prof_white'], input)  # TODO: Switch to prof (physical space) if needed
            loss['total'] += self.lambda_sobolev * loss['sobolev'].mean()

        # Model losses: Some model losses may require additional inputs
        if self.loss_model is not None:
            if isinstance(self.loss_model, (DiagonalQuadraticForm, DiagonalQuadraticHuberForm)):
                loss['model'] = self.loss_model(pred['prof_white'],
                                                target['prof_white_background'],
                                                target['prof_white_increment'])
            else:
                loss['model'] = self.loss_model(pred['prof_white'],
                                                target['prof_white_background'])
            # Total
            loss['total'] += self.lambda_model * loss['model'].mean()

        if self.model_loss_fn is not None:
            loss['model_phys'] = self.model_loss_fn(pred['prof'],
                                                    target['prof'])
            loss['total'] += self.lambda_model_phys * loss['model_phys'].mean()

        # Boundary condition losses (where the variance is zero)
        # if self.loss_bcs is not None:
        #     raise NotImplementedError("Boundary condition loss is not implemented for PCA-based profiles.")

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

        return self
