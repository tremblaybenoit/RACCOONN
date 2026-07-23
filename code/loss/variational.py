import torch
import numpy as np
from typing import Callable
from code.loss.basic import mse


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
    def __init__(self, matrix: np.ndarray) -> None:
        """ Initialize the QuadraticForm module.

        Parameters
        ----------
        matrix: np.ndarray. Matrix to compute the quadratic form with.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()
        # Store covariance matrix
        self.matrix = torch.from_numpy(matrix)

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """

        # Class inheritance
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


class CholeskyForm(torch.nn.Module):
    """
    Stable Mahalanobis Loss: 0.5 * || L^-1 (pred - target) ||^2
    """

    def __init__(self, matrix: np.ndarray) -> None:
        # Class inheritance
        super().__init__()

        # Ensure L is stored as a buffer (device management)
        self.register_buffer('matrix', torch.from_numpy(matrix).float())

    def to(self, device):
        # Class inheritance
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
        # whitened_diff = torch.linalg.solve_triangular(
        #     self.matrix, diff, upper=False
        # )
        whitened_diff = torch.matmul(self.matrix, diff)

        # 3. Return 0.5 * sum(w^2) per batch
        # Result is [Batch]
        return 0.5 * torch.pow(whitened_diff, 2).sum(dim=1).squeeze()


class VarLoss(torch.nn.Module):
    """ Universal loss module that combines observation and model losses.

    NOTE: hofx predictions must be pre-computed and provided in pred['hofx'].
    This module no longer calls forward_model internally.
    """
    def __init__(self, loss_obs: Callable, loss_model: Callable | None = None, loss_bcs: Callable | None = None,
                 lambda_obs: float=1.0, lambda_model: float=1.0, lambda_bcs: float=1.0,
                 pressure_mask: np.ndarray | None =None):
        """ Initialize the variational loss module.

        Parameters
        ----------
        loss_obs: Callable or ListConfig. Loss function(s) for observations.
        loss_model: Callable or ListConfig. Loss function(s) for model predictions.
        loss_bcs: Callable or ListConfig. Loss function(s) for boundary conditions.
        lambda_obs: float. Weight for the observation loss.
        lambda_model: float. Weight for the model loss.
        lambda_bcs: float. Weight for the boundary condition loss.
        pressure_mask: np.ndarray. Mask for profile levels to include in the model loss.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

        # Loss terms
        self.loss_obs, self.loss_model, self.loss_bcs = loss_obs, loss_model, loss_bcs
        # Weighting factors for the losses
        self.lambda_obs, self.lambda_model, self.lambda_bcs = lambda_obs, lambda_model, lambda_bcs
        # Pressure mask per profile type
        self.pressure_mask = torch.from_numpy(pressure_mask) \
            if pressure_mask is not None else None

    def __call__(self, output: dict, target: dict) -> dict:
        """ Compute the combined loss between predicted profiles and target data.

        Parameters
        ----------
        output: dict. Dictionary containing predicted tensors.
                Must include 'prof' and 'hofx' (pre-computed by the model).
        target: dict. Dictionary containing target tensors.

        Returns
        -------
        loss: dict. Dictionary containing total, observation, and model losses.
        """

        # Mask
        if self.pressure_mask is not None:
            pressure_mask = self.pressure_mask
        else:
            pressure_mask = torch.ones_like(output['prof'], dtype=torch.bool, device=output['prof'].device)

        # Initialize loss dictionary
        loss = {}

        # Observation loss: Some observation losses may require additional inputs
        if isinstance(self.loss_obs, DiagonalQuadraticForm):
            loss['obs'] = self.loss_obs(output['hofx_forward'][:, :10], target['hofx_forward'][:, :10],
                                        target['hofx_forward'][:, 10:])
        else:
            loss['obs'] = self.loss_obs(output['hofx_forward'][:, :10],
                                        target['hofx_forward'][:, :10])
        # Total
        loss['total'] = self.lambda_obs * loss['obs'].mean()

        # Model losses: Some model losses may require additional inputs
        if self.loss_model is not None:
            if self.pressure_mask is not None:
                loss['model'] = self.loss_model(output['prof'][:, pressure_mask],
                                                target['prof_prior'][:, pressure_mask])
            else:
                loss['model'] = self.loss_model(output['prof'],
                                                target['prof_prior'])
            # Total
            loss['total'] += self.lambda_model * loss['model'].mean()

        # Boundary condition losses (where the variance is zero)
        if self.loss_bcs is not None and self.pressure_mask is not None:
            loss['bcs'] = self.loss_bcs(output['prof'][:, ~pressure_mask], target['prof'][:, ~pressure_mask])
            # Total
            loss['total'] += self.lambda_bcs * loss['bcs'].mean()

        return loss

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

        # Class inheritance
        super().to(device)

        # Loss functions
        for attr in ['loss_obs', 'loss_model', 'loss_bcs']:
            loss_fn = getattr(self, attr, None)
            if loss_fn is not None and hasattr(loss_fn, 'to'):
                setattr(self, attr, loss_fn.to(device))
        # Pressure filter
        if self.pressure_mask is not None:
            self.pressure_mask = self.pressure_mask.to(device)
        return self

