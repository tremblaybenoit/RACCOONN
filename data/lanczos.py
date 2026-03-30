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
        'L': lanczos_buffers['L'],
        'L_inv': lanczos_buffers['L_inv'],
        'eigenvalues': lanczos_buffers['eigenvalues'],
        'basis': lanczos_buffers['basis'],
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
        self.L = np.asarray(buffers['L'])  # B^(1/2) matrix (V*L, n_modes)
        self.L_inv = np.asarray(buffers['L_inv'])  # B^(-1/2) matrix (n_modes, V*L)

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

        # Project into whitened space: u = anomalies @ L_inv.T
        u = np.dot(anomalies, self.L_inv.T)
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

        innovation = np.dot(u, self.L.T)
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
    flat_data = data.reshape(n_samples, -1)

    # 1. Compute Background (Mean) and Anomalies (Errors)
    x_b = np.mean(flat_data, axis=0)
    err = flat_data - x_b

    # 2. Compute B-matrix (Covariance of anomalies)
    # B represents the 'links' between all levels and variables
    B = np.cov(err, rowvar=False)

    # 3. Regularization (Ridge) to handle co-linearities
    ridge = alpha * np.trace(B) / B.shape[0]
    B += ridge * np.eye(B.shape[0])

    # 4. Eigen-decomposition (The Lanczos modes)
    # Using eigh because B is symmetric
    eigenvalues, eigenvectors = eigh(B)

    # Sort by variance (descending)
    idx = np.argsort(eigenvalues)[::-1]
    sorted_evals = eigenvalues[idx]
    evals = sorted_evals[:n_comp]
    evecs = eigenvectors[:, idx][:, :n_comp]

    # Explained variance ratio
    total_var = np.sum(sorted_evals)
    explained_var = np.sum(evals)
    evr = (explained_var / total_var) * 100
    ignored_var = np.sum(sorted_evals[n_comp:])
    theoretical_mse = ignored_var / B.shape[0]
    theoretical_rmse = np.sqrt(theoretical_mse)
    logger.info(f"Lanczos Quality: {n_comp} modes explain {evr:.4f}% of variance.")
    logger.info(f"Theoretical Lower Bound for MSE:  {theoretical_mse:.6f}")
    logger.info(f"Theoretical Lower Bound for RMSE: {theoretical_rmse:.6f}")

    # 5. Compute the Square Root Matrix L (B = L L^T)
    # In Lanczos space: L = Eigenvectors * sqrt(Eigenvalues)
    L = evecs @ np.diag(np.sqrt(evals))

    # 6. Compute L_inv for whitening residuals
    L_inv = np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    return {
        'x_b': x_b,
        'L': L,
        'L_inv': L_inv,
        'eigenvalues': evals,
        'basis': evecs,
        'n_comp': n_comp
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
    logger.info("Loading Lanczos buffers...")
    # This would load the dict generated by generate_lanczos_buffers
    lanczos_buffs = instantiate(input.buffers.load)

    # Initialize Processor
    processor = LanczosProcessor(lanczos_buffs)

    # Load ground truth data (N, V, L)
    logger.info("Loading physical ground truth data...")
    data = load_var(input.data)
    n_samples = data.shape[0]

    # Flatten for matrix operations: (N, V*L)
    flat_data = data.reshape(n_samples, -1)

    # Project: u = L_inv @ (x - x_b)
    # These 'u' values are the 'Control Variables' (unit variance, independent)
    logger.info("Projecting data into whitened control space (u-space)...")
    u_space_coefficients = processor.physical_to_control(flat_data)

    # Save for training the Neural Field / MLP
    if hasattr(output.lanczos, 'save'):
        save_func = instantiate(output.lanczos.save)
        save_func(u_space_coefficients)
        logger.info(f"Saved projected coefficients with shape {u_space_coefficients.shape}")

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