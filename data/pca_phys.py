import numpy as np
import hydra
from omegaconf import DictConfig
from data.io import load_var_and_normalize
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging
from sklearn.decomposition import PCA
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from utilities.plot import plot_map, save_plot, flexible_gridspec

# Initialize logger
logger = logging.getLogger(__name__)


def load_pca_buffers(path: str, buffer: str=None) -> dict:
    """ Load PCA buffers from a given file.

    Parameters
    ----------
    path: str. Path to the file containing PCA buffers.
    buffer: str. Specific buffer to load (not used in this implementation).

    Returns
    -------
    dict. Dictionary containing PCA buffers.
    """

    pca_buffers = np.load(path, allow_pickle=True)

    if buffer is not None:
        return pca_buffers[buffer]
    return {
        'mu': pca_buffers['mu'],
        'std': pca_buffers['std'],
        'eigenvalues': pca_buffers['eigenvalues'],
        'scales': pca_buffers['scales'],
        'basis': pca_buffers['basis'],
    }


class PCAProcessor:
    def __init__(self, pca_buffers: dict):
        """Initialize PCA processor with given PCA buffers.

        Parameters
        ----------
        pca_buffers: dict. Dictionary containing PCA buffers with keys:
            - 'mu': Mean of the data.
            - 'std': Standard deviation of the data.
            - 'eigenvalues': Eigenvalues from PCA.
            - 'scales': Scales for whitening.
            - 'basis': PCA basis vectors.

        Returns
        -------
        None.
        """

        self.mu = pca_buffers['mu']
        self.sigma = pca_buffers['std']
        self.eigenvalues = pca_buffers['eigenvalues']
        self.scales = pca_buffers['scales']
        self.basis = pca_buffers['basis']

    def physical_to_standardized(self, x):
        """Converts physical levels to standardized PCA coefficients."""
        z = (x - self.mu) / self.sigma
        w = np.dot(z, self.basis.T)
        return w

    def physical_to_whitened(self, x):
        """Converts physical levels to whitened PCA coefficients."""
        z = (x - self.mu) / self.sigma
        w = np.dot(z, self.basis.T)
        return w / np.sqrt(self.eigenvalues)

    def whitened_to_physical(self, w_hat):
        """The inverse: Whitened coefficients to physical levels."""
        w = w_hat * np.sqrt(self.eigenvalues)
        z = np.dot(w, self.basis)
        return (z * self.sigma) + self.mu

    def standardized_to_physical(self, w):
        """The inverse: Standardized coefficients to physical levels."""
        z = np.dot(w, self.basis)
        return (z * self.sigma) + self.mu


def generate_pca_buffers(data: np.ndarray, mode: str='multivariate', pressure_filter: np.ndarray=None, alpha: float=1e-5) -> dict:
    """
    Generate PCA buffers for a given dataset.

    Parameters
    ----------
    data: np.ndarray. Input data of shape (N, V, L) where N is the number of samples,
          V is the number of variables, and L is the number of levels.
    mode: str. PCA mode, either 'global' or 'local'.
    pressure_filter: np.ndarray. Optional boolean array to filter pressure levels (not used in this implementation).
    alpha: float. Damping factor for eigenvalues to ensure numerical stability.

    Returns
    -------
    Dict containing PCA basis, mean, std, and eigenvalues.
    """

    # Get data shape
    n_samples, n_vars, n_levels = data.shape

    # Standardize
    mu = np.mean(data, axis=0, keepdims=True)  # (V, L)
    std = np.ones_like(mu)  # np.std(data, axis=0, keepdims=True) + 1e-12  # (V, L)
    increment = data - mu
    standardized_data = increment / std  # (N, V, L)

    # Multivariate PCA
    if mode == 'multivariate':

        # Pressure filter
        if pressure_filter is not None:
            mu = mu[:, pressure_filter]
            std = std[:, pressure_filter]
            standardized_data = standardized_data[:, pressure_filter]

        # Decomposition
        flat_z = standardized_data.reshape(n_samples, -1)
        pca = PCA(n_components=flat_z.shape[1])
        pca.fit(flat_z)
        Q = pca.components_.T  # (n_features, n_features), eigenvectors
        lambdas = pca.explained_variance_ # eigenvalues
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        k_var = np.argmax(cumvar >= 0.999) + 1
        lambda_max = lambdas[0]
        eigen_floor = 1e-5
        k_floor = np.sum(lambdas >= eigen_floor * lambda_max)
        k = max(k_var, k_floor)
        Qk = Q[:, :k]  # (n_features, k), leading eigenvectors
        Lk = np.maximum(lambdas[:k], eigen_floor* lambda_max)  # (k,), leading eigenvalues

        pca = PCA(n_components=k)
        pca.fit(flat_z)

        # Compute pseudo-inverse of the covariance matrix
        # If we assume that the background is the mean, then the standardizedf data is the increment, and the covariance matrix is the covariance of the standardized data, which is the identity matrix.
        # The pseudo-inverse of the identity matrix is itself, so we can compute the pseudo-inverse of the covariance matrix in the PCA space as follows:
        max_ev = np.max(pca.explained_variance_)
        damped_ev = pca.explained_variance_ + alpha * max_ev
        B_inv = (Qk * Lk) @ Qk.T
        # B_inv += (1.0 / (alpha * max_ev)) * (np.eye(B_inv.shape[0]) - pca.components_.T @ pca.components_)
        # Make diagonal-only version in case we want to use it for whitening
        B_inv_d = np.diag(np.diag(B_inv))
        B_inv_phys = B_inv * np.outer(1.0/std.flatten(), 1.0/std.flatten())
        B_inv_d_phys = B_inv_d * np.outer(1.0/std.flatten(), 1.0/std.flatten())

        # Store PCA buffers in a dictionary
        pca_buffs = {
            'n_comp': pca.n_components_,
            'basis': pca.components_,
            'mu': mu.flatten(),
            'std': std.flatten(),
            'eigenvalues': pca.explained_variance_,
            'scales': np.sqrt(pca.explained_variance_),
            'scales_inv': 1.0/np.sqrt(pca.explained_variance_.reshape(1, -1)),
            'B_inv': B_inv,
            'B_inv_d': B_inv_d,
            'B_inv_phys': B_inv_phys,
            'B_inv_d_phys': B_inv_d_phys,
        }

    elif mode == 'univariate':
        # Per variable PCA (with variables in the second dimension)
        pca_buffs = {}
        B_inv = []
        B_inv_phys = []
        for v in range(n_vars):

            # Pressure filter
            if pressure_filter is not None:
                mu_v = mu[:, v][:, pressure_filter[v]]
                std_v = std[:, v][:, pressure_filter[v]]
                standardized_data_v = standardized_data[:, v][:, pressure_filter[v]]
                n_levels = pressure_filter[v].sum()
            else:
                mu_v = mu[:, v]
                std_v = std[:, v]
                standardized_data_v = standardized_data[:, v]

            # Decomposition
            flat_z = standardized_data_v.reshape(n_samples, -1)
            pca = PCA(n_components=flat_z.shape[1])
            pca.fit(flat_z)
            Q = pca.components_.T  # (n_features, n_features), eigenvectors
            lambdas = pca.explained_variance_  # eigenvalues
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            k_var = np.argmax(cumvar >= 0.999) + 1
            lambda_max = lambdas[0]
            eigen_floor = 1e-5
            k_floor = np.sum(lambdas >= eigen_floor * lambda_max)
            k = max(k_var, k_floor)
            Qk = Q[:, :k]  # (n_features, k), leading eigenvectors
            Lk = np.maximum(lambdas[:k], eigen_floor* lambda_max)  # (k,), leading eigenvalues
            pca = PCA(n_components=k)
            pca.fit(flat_z)

            # Compute pseudo-inverse of the covariance matrix
            max_ev = np.max(pca.explained_variance_)
            damped_inv_ev = pca.explained_variance_ + alpha * max_ev
            # B_inv_v = pca.components_.T @ np.diag(1.0/damped_inv_ev) @ pca.components_
            # Null-space penalty for this variable
            # B_inv_v += (1.0 / (alpha * max_ev)) * (np.eye(n_levels) - pca.components_.T @ pca.components_)
            B_inv_v = (Qk * Lk) @ Qk.T
            B_inv.append(B_inv_v)
            # Make diagonal-only version in case we want to use it for whitening
            B_inv_d = np.diag(np.diag(B_inv_v))
            B_inv_phys_v = B_inv_v * np.outer(1.0/std_v.flatten(), 1.0/std_v.flatten())
            B_inv_d_phys = B_inv_d * np.outer(1.0/std_v.flatten(), 1.0/std_v.flatten())
            B_inv_phys.append(B_inv_phys_v)

            # Store PCA buffers in a dictionary
            pca_buffs[v] = {
                'basis': pca.components_,
                'mu': mu_v,
                'std': std_v,
                'eigenvalues': pca.explained_variance_,
                'scales': np.sqrt(pca.explained_variance_),
                'scales_inv': 1.0/np.sqrt(pca.explained_variance_.reshape(1, -1)),
                'B_inv': B_inv_v,
                'B_inv_d': B_inv_d,
                'B_inv_phys': B_inv_phys,
                'B_inv_d_phys': B_inv_d_phys,
            }
        # Assemble a pseudo-inverse of the covariance matrix for the full state vector by block-diagonalizing the per-variable pseudo-inverses
        # Assemble Block-Diagonal B_inv
        B_inv = block_diag(*B_inv)
        pca_buffs['B_inv'] = B_inv
        B_inv_phys = block_diag(*B_inv_phys)
        pca_buffs['B_inv_phys'] = B_inv_phys
        # Flattened Mu/Std for easy use in loss function
        pca_buffs['mu'] = mu.flatten()
        pca_buffs['std'] = std.flatten()

    else:
        raise ValueError(f"Invalid PCA mode: {mode}. Must be 'global' or 'local'.")

    # Create a flexible gridspec
    fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                      lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
    ax = get_axes(0, 0)
    # Plot covariance matrix
    plot_map(ax, np.abs(B_inv), title=f"Inverse covariance matrix", plt_origin='upper',
             cb_label=r'Values')
    save_plot(fig, filename='B_inv.png')

    fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                      lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
    ax = get_axes(0, 0)
    # Plot covariance matrix
    plot_map(ax, np.abs(B_inv_phys), title=f"Inverse covariance matrix", plt_origin='upper',
             cb_label=r'Values')
    save_plot(fig, filename='B_inv_phys.png')

    return pca_buffs


def project_pca(input: DictConfig, output: DictConfig, mode='multivariate') -> None:
    """ Project data onto PCA basis.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        mode: str. PCA mode, either 'multivariate' or 'univariate'.

        Returns
        -------
        np.ndarray. Projected data.
    """

    # Load PCA buffers
    logger.info("Loading PCA buffers...")
    pca_buffs = instantiate(input.pca_buffers.load)

    # Global mode
    if mode == 'multivariate':

        # Initialize PCA processor
        pca_processor = PCAProcessor(pca_buffs)

        # Load data
        logger.info("Loading data...")
        data = load_var_and_normalize(input.data)

        # Project data
        logger.info("Projecting data onto PCA basis...")
        n_samples = data.shape[0]
        flat_data = data.reshape(n_samples, -1)
        standardized_pca = pca_processor.physical_to_standardized(flat_data)
        whitened_pca = pca_processor.physical_to_whitened(flat_data)

    # Local mode
    elif mode == 'univariate':

        # Load data
        logger.info("Loading data...")
        data = load_var_and_normalize(input.data)

        # Project data
        logger.info("Projecting data onto PCA basis...")
        n_samples, n_vars, n_levels = data.shape
        standardized_pca = []
        whitened_pca = []
        for v in range(n_vars):
            pca_processor = PCAProcessor(pca_buffs[v])
            standardized_pca.append(pca_processor.physical_to_standardized(data[:, v, :]))
            whitened_pca.append(pca_processor.physical_to_whitened(data[:, v, :]))
        standardized_pca = np.concatenate(standardized_pca, axis=1)
        whitened_pca = np.concatenate(whitened_pca, axis=1)

    else:
        raise ValueError(f"Invalid PCA mode: {mode}. Must be 'global' or 'local'.")

    # Save statistics to file
    logger.info("Saving data to file...")
    if hasattr(output.pca, 'save'):
        save_func = instantiate(output.pca.save)
        save_func(standardized_pca)
    if hasattr(output.pca_white, 'save'):
        save_func = instantiate(output.pca_white.save)
        save_func(whitened_pca)

    return


def compute_pca(input: DictConfig, output: DictConfig, mode: str='multivariate', pressure_filter: np.ndarray = None) -> None:
    """ Compute pca decomposition of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        mode: str. PCA mode, either 'global' or 'local'.
        pressure_filter: np.ndarray. Optional boolean array to filter pressure levels.

        Returns
        -------
        None.
    """

    # Load data
    logger.info("Loading data...")
    data = instantiate(input.data.load)

    # Generate PCA buffers
    logger.info("Generating PCA buffers...")
    pca_buffs = generate_pca_buffers(
        data=data,
        mode=input.get('mode', mode),
        pressure_filter=instantiate(pressure_filter) if pressure_filter is not None else None,
        alpha=input.get('alpha', 1e-5)
    )

    # Save statistics to file
    logger.info("Saving data to file...")
    if hasattr(output, 'save'):
        save_func = instantiate(output.save)
        save_func(pca_buffs)
    breakpoint()
    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute pca decompositions for datasets as specified in the configuration file.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # Compute model and observation pca decompositions
    if hasattr(config.preparation, "pca"):
        for dataset, config_pca in config.preparation.pca.items():
            logger.info(f"Computing PCA decomposition of {dataset} set")
            instantiate(config_pca)

    return


if __name__ == '__main__':
    """ Compute covariance matrices of given datasets.

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