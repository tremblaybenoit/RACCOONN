import torch
import numpy as np
from code.loss.basic import mse, l2


class Control(torch.nn.Module):
    """ Control variable loss module."""
    def __init__(self) -> None:
        """ Initialize the L2 module.

        Returns
        -------
        None.
        """

        # Class inheritance
        super().__init__()

    def to(self, device):
        """ Move the module to a specified device.

        Parameters
        ----------
        device: torch.device. Device to move the module to.
        """

        # Class inheritance
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """ Compute the L2 loss between predicted and target tensors.

        Parameters
        ----------
        pred: torch.Tensor. Predicted tensor.
        target: torch.Tensor | None. True values.

        Returns
        -------
        torch.Tensor. L2 loss over the batch.
        """
        return l2(pred)


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
        self.matrix = torch.from_numpy(matrix.astype(np.float32))

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
    """ Cholesky form loss module."""
    def __init__(self, matrix: np.ndarray) -> None:
        """ Initialize the CholeskyForm module.

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
        """ Compute the Cholesky form of the difference between predicted and target tensors.

        Parameters
        ----------
        pred: torch.Tensor. Predicted tensor.
        target: torch.Tensor. True values.

        Returns
        -------
        torch.Tensor. Cholesky form of the difference between predicted and target tensors.
        """
        d = (pred-target).reshape(pred.shape[0], -1)
        z = torch.linalg.solve_triangular(self.matrix, d.unsqueeze(-1), upper=False).squeeze(-1)

        return l2(z)
