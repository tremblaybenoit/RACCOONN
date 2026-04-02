import numpy as np
import hydra
from omegaconf import DictConfig
from data.io import load_var
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging
from scipy.linalg import eigh
from typing import Union
import torch

# Initialize logger
logger = logging.getLogger(__name__)


def load_lanczos_buffers(path: str, buffer: str=None) -> dict:
    """ Load Lanczos buffers from a given file.

    Parameters
    ----------
    path: str. Path to the file containing Lanczos buffers.
    buffer: str. Specific buffer to load (not used in this implementation).

    Returns
    -------
    dict. Dictionary containing Lanczos buffers.
    """

    lanczos_buffers = np.load(path, allow_pickle=True)

    if buffer is not None:
        return lanczos_buffers[buffer]
    return {
        'x_b': lanczos_buffers['x_b'],
        'std_devs': lanczos_buffers['std_devs'],
        'L_corr': lanczos_buffers['L_corr'],
        'L_corr_inv': lanczos_buffers['L_corr_inv'],
        'eigenvalues': lanczos_buffers['eigenvalues'],
        'n_comp': lanczos_buffers['n_comp'],
    }


class LanczosProcessor:
    def __init__(self, buffers: dict):
        """
            Implementation of the Control Variable Transform (CVT).

            Parameters
            ----------
            buffers: Lanczos buffers.

            Returns
            -------
            None.
        """

        # Parameters
        self.x_b = np.asarray(buffers['x_b'])  # Mean background profile (V*L, 1)
        # D is the per-feature standard deviation (V*L)
        self.D = np.asarray(buffers['std_devs'])
        # L_corr is the square root of the CORRELATION matrix
        self.L_corr = np.asarray(buffers['L_corr'])
        self.L_corr_inv = np.asarray(buffers['L_corr_inv'])

    def physical_to_control(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
            WHITENING: Takes physical profiles and returns whitened coefficients 'u'.
            If x is (N, V*L), returns (N, n_modes).

            Parameters
            ----------
            x: Physical space data.

            Returns
            -------
            u: Control space data.
        """
        # Ensure x is 2D (samples, features)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Center the data
        anomalies = x - self.x_b

        # Scale to unit variance (Anomalies -> Correlation Space)
        eps = 1e-8
        norm_anomalies = anomalies / (self.D + eps)
        # Whiten via Correlation Basis
        u = np.dot(norm_anomalies, self.L_corr_inv.T)
        return u

    def control_to_physical(self, u: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
            UN-WHITENING: Takes coefficients 'u' and returns physical profiles.
            Used inside the PINN forward pass.

            Parameters
            ----------
            u: Control space data.

            Returns
            -------
            x: Physical space data
        """
        if u.ndim == 1:
            u = u.reshape(1, -1)

        # 1. Un-whiten to Correlation Space
        norm_innovation = np.dot(u, self.L_corr.T)
        # 2. Rescale to Physical Units
        innovation = norm_innovation * self.D
        return self.x_b + innovation


def generate_lanczos_buffers(data: np.ndarray, n_comp: int = 200, alpha: float = 0.) -> dict:
    """
        Generates the B-matrix Eigen-decomposition (Lanczos Space).

        Parameters
        ----------
        data: np.ndarray. Shape (N, V, L) - Synthetic truth profiles.
        n_comp: int. Number of modes to retain.
        alpha: float. Ridge regularization for the B-matrix.

        Returns
        -------
        lanczos_buffers: dict. Dictionary containing Lanczos buffers.
    """

    # Parameters
    n_samples, n_vars, n_levels = data.shape
    d_total = n_vars * n_levels
    flat_data = data.reshape(n_samples, -1)

    # 1. Compute Background (Mean) and Anomalies (Errors)
    x_b = np.mean(flat_data, axis=0)
    err = flat_data - x_b

    # 2. Extract Standard Deviations (D) for each level/variable
    # This is the "Scaling" part of the massage
    std_devs = np.std(err, axis=0)
    std_devs = np.clip(std_devs, a_min=1e-9, a_max=None)  # Prevent div by zero

    # 3. Compute the Correlation Matrix (C)
    # Correlation = Covariance / (std_i * std_j)
    # Alternatively: compute covariance of normalized errors
    norm_err = err / std_devs
    C = np.cov(norm_err, rowvar=False)

    # 4. Regularization (Ridge) on the Correlation Matrix
    # Since C has 1.0 on diagonal, alpha is easy to tune
    C += alpha * np.eye(d_total)

    # 5. Eigen-decomposition of the Correlation Matrix
    eigenvalues, eigenvectors = eigh(C)

    # Sort by variance (descending)
    idx = np.argsort(eigenvalues)[::-1]
    sorted_evals = eigenvalues[idx]
    evals = sorted_evals[:n_comp]
    evecs = eigenvectors[:, idx][:, :n_comp]

    # --- Metrics ---
    total_var = np.sum(sorted_evals)
    explained_var = np.sum(evals)
    evr = (explained_var / total_var) * 100
    theoretical_rmse_corr = np.sqrt(np.sum(sorted_evals[n_comp:]) / d_total)

    logger.info(f"Correlation Basis: {n_comp} modes explain {evr:.4f}% of correlation variance.")
    logger.info(f"Theoretical RMSE (in Correlation Units): {theoretical_rmse_corr:.6f}")

    # 6. Compute Square Root Matrices for Correlation Space
    # L_corr = V * sqrt(Lambda)
    L_corr = evecs @ np.diag(np.sqrt(evals))
    # L_corr_inv = sqrt(Lambda)^-1 * V.T
    L_corr_inv = np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    return {
        'x_b': x_b,  # (V*L,)
        'std_devs': std_devs,  # (V*L,)
        'L_corr': L_corr,  # (V*L, n_comp)
        'L_corr_inv': L_corr_inv,  # (n_comp, V*L)
        'eigenvalues': evals,
        'n_comp': n_comp,
        'metrics': {
            'evr': evr,
            'theoretical_rmse_corr': theoretical_rmse_corr
        }
    }


def project_lanczos(input: DictConfig, output: DictConfig) -> None:
    """
        Project physical profiles into the whitened Lanczos (control) space.

        This is the inverse of the generator: it takes ground truth profiles and
        finds the 'u' coefficients that the MLP should ideally predict.

        Parameters
        ----------
        input: DictConfig. Hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.

        Returns
        -------
        None.
    """
    logger.info("Loading Multivariate Lanczos buffers...")
    # This loads the dict from the new generate_lanczos_buffers
    lanczos_buffs = instantiate(input.buffers.load)

    # Initialize Processor (Update your class to use std_devs/L_corr)
    processor = LanczosProcessor(lanczos_buffs)

    # Load ground truth data (N, V, L)
    logger.info("Loading physical ground truth data...")
    data = load_var(input.data)
    n_samples = data.shape[0]

    # Flatten to (N, V*L)
    flat_data = data.reshape(n_samples, -1)

    # 1. Project: u = L_corr_inv @ ((x - x_b) / std_devs)
    logger.info("Projecting data into whitened correlation space (u-space)...")
    u_space_coefficients = processor.physical_to_control(flat_data)

    # 2. MANDATORY SANITY CHECK
    # Let's ensure our buffers can actually reconstruct the data
    reconstructed_flat = processor.control_to_physical(u_space_coefficients)

    # Calculate RMSE in physical units (e.g., K, g/kg, Pa)
    total_rmse = np.sqrt(np.mean((flat_data - reconstructed_flat) ** 2))

    # Optional: Calculate per-variable RMSE to ensure no one variable is 'lost'
    # Assuming V variables and L levels
    n_vars = data.shape[1]
    n_levels = data.shape[2]
    reconstructed_3d = reconstructed_flat.reshape(n_samples, n_vars, n_levels)

    logger.info("--- Reconstruction Sanity Check ---")
    logger.info(f"Total Combined RMSE: {total_rmse:.6f}")

    for v in range(n_vars):
        v_rmse = np.sqrt(np.mean((data[:, v, :] - reconstructed_3d[:, v, :]) ** 2))
        logger.info(f"Variable {v} Reconstruction RMSE: {v_rmse:.6f}")

    # 3. Save for training the Neural Field / MLP
    if hasattr(output.lanczos, 'save'):
        save_path = instantiate(output.lanczos.save)(u_space_coefficients)
        logger.info(f"Saved {u_space_coefficients.shape[1]} coefficients to {save_path}")

    return


def compute_lanczos(input: DictConfig, output: DictConfig) -> None:
    """ Compute Lanczos decomposition of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.

        Returns
        -------
        None.
    """

    # Load data
    logger.info("Loading data...")
    data = instantiate(input.data.load)

    # Generate PCA buffers
    logger.info("Generating PCA buffers...")
    lanczos_buffs = generate_lanczos_buffers(
        data=data,
        alpha=input.get('alpha', 0.),
        n_comp=input.get('n_comp', 200),
    )

    # Save statistics to file
    logger.info("Saving data to file...")
    if hasattr(output, 'save'):
        save_func = instantiate(output.save)
        save_func(lanczos_buffs)

    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute Lanczos decompositions for datasets as specified in the configuration file.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # Compute model and observation pca decompositions
    if hasattr(config.preparation, "lanczos"):
        for dataset, config_lanczos in config.preparation.lanczos.items():
            logger.info(f"Computing Lanczos decomposition of {dataset} set")
            instantiate(config_lanczos)

    return


if __name__ == '__main__':
    """ Compute Lanczos decomposition of given datasets.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        zarr file containing data statistics.
    """

    main()