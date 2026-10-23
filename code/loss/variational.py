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


class BatchKroneckerCholeskyForm(torch.nn.Module):
    """
    Batch-wide variational loss using a Kronecker-factored covariance:
    B_batch = C_spatial (M x M) x B_vertical (N_feat x N_feat)
    Solved efficiently via triangular backsubstitution without full matrix inversion.
    """

    def __init__(self, vertical_cholesky: np.ndarray, spatial_length_scale: float = 1.0) -> None:
        super().__init__()
        # Store vertical Cholesky factor L_v (shape: [N_feat, N_feat])
        self.register_buffer("L_v", torch.from_numpy(vertical_cholesky.astype(np.float32)))
        self.spatial_length_scale = spatial_length_scale

    def to(self, device):
        super().to(device)
        return self

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred: torch.Tensor, shape (M, n_vars, n_levels) or (M, N_feat)
        target: torch.Tensor, shape same as pred
        lat: torch.Tensor, shape (M,) - latitudes of batch profiles
        lon: torch.Tensor, shape (M,) - longitudes of batch profiles
        """
        M = pred.shape[0]
        d = (pred - target).reshape(M, -1)  # Shape: (M, N_feat)
        N_feat = d.shape[1]

        # 1. Build dynamic spatial correlation matrix C_spatial for the current batch (M x M)
        # Using a simple Gaussian distance kernel based on batch (lat, lon) coordinates
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)

        # Approximate great-circle distance or Euclidean distance on batch coordinates
        lat_diff = lat_rad.unsqueeze(1) - lat_rad.unsqueeze(0)
        lon_diff = lon_rad.unsqueeze(1) - lon_rad.unsqueeze(0)
        dist_sq = lat_diff ** 2 + lon_diff ** 2

        C_spatial = torch.exp(-0.5 * dist_sq / (self.spatial_length_scale ** 2))

        # Add a tiny ridge for positive-definiteness on C_spatial
        C_spatial = C_spatial + 1e-5 * torch.eye(M, device=pred.device, dtype=pred.dtype)

        # 2. Compute Cholesky factor of spatial matrix: L_s (shape: [M, M])
        L_s = torch.linalg.cholesky(C_spatial)

        # 3. Apply Kronecker Cholesky solve efficiently using matrix multiplications:
        # Instead of building the (M*N_feat x M*N_feat) matrix, we solve:
        # L_s^{-1} * d * (L_v^{-1})^T
        # Step A: Solve spatial system (L_s X = d) across batch dimension
        # d shape: (M, N_feat). torch.linalg.solve_triangular handles batched RHS if needed,
        # or we solve system for the (M, M) factor:
        X = torch.linalg.solve_triangular(L_s, d, upper=False)  # Shape: (M, N_feat)

        # Step B: Solve vertical system across feature dimension
        # X = L_v^{-T} X^T -> equivalent to solving L_v Z = X^T for each spatial point
        # L_v is lower triangular, so L_v Z = X^T means X = Z^T L_v^T -> X^T = L_v Z -> Z = L_v^{-1} X^T
        Z_T = torch.linalg.solve_triangular(self.L_v, X.T, upper=False)  # Shape: (N_feat, M)
        Z = Z_T.T  # Shape: (M, N_feat)

        # 4. Total squared Mahalanobis distance
        return torch.sum(Z ** 2) / M
