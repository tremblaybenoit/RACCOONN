import torch
import os
import numpy as np
from typing import Callable
try:
    from inverse.model.forward import CRTMForward
    from utilities.logic import get_config_path
    # Initialize CRTM forward model
    checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__name__), 'forward/model/checkpoints/model.ckpt'))
    config_path = os.path.join(get_config_path(), 'model/forward_emulator.yaml')
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
        return quadratic_form(pred, target, self.matrix)


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
    eps = torch.finfo(target.dtype).eps
    return mse(pred/(diag + eps), target/(diag + eps))


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
                 pressure_filter: np.ndarray=None):
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
        self.pressure_filter = torch.from_numpy(pressure_filter) if pressure_filter is not None else None

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
        hofx_pred = self.forward_model(pred['prof'], target)

        # Initialize loss dictionary
        loss = {'total': torch.tensor(0.0, device=pred['prof'].device)}

        # Observation loss: Some observation losses may require additional inputs
        if getattr(getattr(self.loss_obs, "func", self.loss_obs), "__name__", None) == 'diagonal_quadratic_form':
            loss['obs'] = self.loss_obs(hofx_pred[:, :10], target['hofx'][:, :10], target['hofx'][:, 10:])
        else:
            loss['obs'] = self.loss_obs(hofx_pred[:, :10], target['hofx'][:, :10])
        # Total
        loss['total'] += self.lambda_obs * torch.nanmean(loss['obs'])

        # Model losses: Some model losses may require additional inputs
        if self.loss_model is not None:
            if getattr(getattr(self.loss_model, "func", self.loss_model), "__name__", None) == 'diagonal_quadratic_form':
                loss['model'] = self.loss_model(pred['prof_norm'][:, pressure_filter], target['prof_norm_background'][:, pressure_filter], target['prof_norm_increment'])
            else:
                loss['model'] = self.loss_model(pred['prof_norm'][:, pressure_filter], target['prof_norm_background'][:, pressure_filter])
            # Total
            loss['total'] += self.lambda_model * torch.nanmean(loss['model'])

        # Boundary condition losses (where the variance is zero)
        if self.loss_bcs is not None and self.pressure_filter is not None:
            loss['bcs'] = self.loss_bcs(pred['prof_norm'][:, ~pressure_filter], target['prof_norm'][:, ~pressure_filter])
            # Total
            loss['total'] += self.lambda_bcs * torch.nanmean(loss['bcs'])

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
        if self.pressure_filter is not None and hasattr(self.pressure_filter, 'to'):
            self.pressure_filter = self.pressure_filter.to(device)
        return self
