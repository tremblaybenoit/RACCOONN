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
        return quadratic_form(pred.reshape(pred.shape[0], -1), target.reshape(pred.shape[0], -1),
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


class DynamicQuadraticForm(torch.nn.Module):
    """ Dynamically compute inverse of R from the inverse of the correlation matrix and the target standard deviation. """

    def __init__(self, matrix: np.ndarray):
        """ Initialize the QuadraticForm module.

        Parameters
        ----------
        matrix: np.ndarray. Correlation matrix inverse to compute the quadratic form with.

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

    def __call__(self, pred: torch.Tensor, diag: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Compute Quadratic form, but first update R dynamically

        Parameters
        ----------
        pred: torch.Tensor. Predicted tensor. Shape: (n_samples, n_channels)
        target: torch.Tensor. True values. Shape: (n_samples, n_channels)
        diag: torch.Tensor. Diagonal elements of the matrix to compute the quadratic form with. Shape: (n_samples, n_channels)

        Returns
        -------
        torch.Tensor. Quadratic form of the difference between predicted and target tensors.
        """

        # Flatten inputs to (n_samples, n_channels)
        p = pred.view(pred.shape[0], -1)
        t = target.view(target.shape[0], -1)
        d = diag.view(diag.shape[0], -1)
        return quadratic_form(p / d, t / d, self.matrix)


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
