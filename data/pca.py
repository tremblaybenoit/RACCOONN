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


def generate_pca_buffers(data: np.ndarray, mode: str='multivariate', pressure_filter: np.ndarray=None, alpha: float=0) -> dict:
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
    std = np.std(data, axis=0, keepdims=True) + 1e-12  # (V, L)
    vmin = np.min(data, axis=0, keepdims=True)
    vmax = np.max(data, axis=0, keepdims=True)
    increment = data - mu
    increment_std = np.std(increment, axis=0, keepdims=True) + 1e-12
    standardized_data = increment / std  # (N, V, L)
    normalized_data = (data - vmin.min(axis=(0, 2), keepdims=True))/(vmax.max(axis=(0, 2), keepdims=True)-vmin.min(axis=(0, 2), keepdims=True))  # (N, V, L)
    normalized_std = np.std(normalized_data, axis=0, keepdims=True) + 1e-12  # (V, L)
    breakpoint()

    # Multivariate PCA
    if mode == 'multivariate':

        # Pressure filter
        if pressure_filter is not None:
            mu = mu[:, pressure_filter]
            std = std[:, pressure_filter]
            vmin = vmin[:, pressure_filter]
            vmax = vmax[:, pressure_filter]
            standardized_data = standardized_data[:, pressure_filter]

        # Decomposition
        flat_z = standardized_data.reshape(n_samples, -1)
        pca = PCA(n_components=0.9999)
        pca.fit(flat_z)
        # Compute pseudo-inverse of the covariance matrix
        # If we assume that the background is the mean, then the standardizedf data is the increment, and the covariance matrix is the covariance of the standardized data, which is the identity matrix.
        # The pseudo-inverse of the identity matrix is itself, so we can compute the pseudo-inverse of the covariance matrix in the PCA space as follows:
        max_ev = np.max(pca.explained_variance_)
        damped_ev = pca.explained_variance_ + alpha * max_ev
        B_inv = pca.components_.T @ np.diag(1.0/damped_ev) @ pca.components_
        # B_inv += (1.0 / (alpha * max_ev)) * (np.eye(B_inv.shape[0]) - pca.components_.T @ pca.components_)
        # Make diagonal-only version in case we want to use it for whitening
        B_inv_d = np.diag(np.diag(B_inv))

        # Robustly convert inverse-covariance from standardized space to physical units.
        # std has shape (1, V, L) (because keepdims=True). Flatten to match the PCA flattening order.
        std_flat = std.flatten()

        # Clip extremely small standard deviations to avoid huge scaling factors.
        # If your data legitimately contains near-zero-variance levels, consider a larger floor.
        eps_base = 1e-5
        eps = max(eps_base, 0.01 * float(np.median(std_flat)))
        std_floor = np.maximum(std_flat, eps)
        # breakpoint()
        # std_floor = std_flat

        # Scaling factors (1/std) for rows and columns. Use broadcasting to avoid an explicit outer allocation.
        factors = 1.0 / std_floor
        # Apply row and column scaling: (D^{-1} B_inv D^{-1}) = diag(factors) @ B_inv @ diag(factors)
        B_inv_phys = (factors[:, None] * B_inv) * factors[None, :]
        B_inv_d_phys = (factors[:, None] * B_inv_d) * factors[None, :]

        # Debug / diagnostic info (useful when B_inv_phys has unexpectedly large values)
        try:
            max_abs = float(np.max(np.abs(B_inv_phys)))
            mean_abs = float(np.mean(np.abs(B_inv_phys)))
        except Exception:
            max_abs = None
            mean_abs = None
        print(f"std_flat: min={std_flat.min():.3e}, median={np.median(std_flat):.3e}, max={std_flat.max():.3e}, eps={eps}")
        print(f"B_inv_phys: max_abs={max_abs}, mean_abs={mean_abs}")

        # Store PCA buffers in a dictionary
        pca_buffs = {
            'n_comp': pca.n_components_,
            'basis': pca.components_,
            'mu': mu,
            'std': std,
            'vmin': vmin,
            'vmax': vmax,
            'eigenvalues': pca.explained_variance_,
            'scales': np.sqrt(pca.explained_variance_),
            'scales_inv': 1.0/np.sqrt(pca.explained_variance_.reshape(1, -1)),
            'B_inv': B_inv,
            'B_inv_d': B_inv_d,
            'B_inv_phys': B_inv_phys,
            'B_inv_d_phys': np.diag(1.0/increment_std.flatten()**2),
        }

    elif mode == 'univariate':
        # Per variable PCA (with variables in the second dimension)
        pca_buffs = {'n_comp': 0}
        B_inv = []
        B_inv_phys = []
        for v in range(n_vars):

            # Pressure filter
            if pressure_filter is not None:
                mu_v = mu[:, v][:, pressure_filter[v]]
                std_v = std[:, v][:, pressure_filter[v]]
                vmin_v = vmin[:, v][:, pressure_filter[v]]
                vmax_v = vmax[:, v][:, pressure_filter[v]]
                standardized_data_v = standardized_data[:, v][:, pressure_filter[v]]
                n_levels = pressure_filter[v].sum()
            else:
                mu_v = mu[:, v]
                std_v = std[:, v]
                vmin_v = vmin[:, v]
                vmax_v = vmax[:, v]
                standardized_data_v = standardized_data[:, v]

            # Decomposition
            flat_z = standardized_data_v.reshape(n_samples, -1)
            pca = PCA(n_components=0.9999)
            pca.fit(flat_z)

            # Compute pseudo-inverse of the covariance matrix
            max_ev = np.max(pca.explained_variance_)
            damped_inv_ev = pca.explained_variance_ + alpha * max_ev
            B_inv_v = pca.components_.T @ np.diag(1.0/damped_inv_ev) @ pca.components_
            # Null-space penalty for this variable
            # B_inv_v += (1.0 / (alpha * max_ev)) * (np.eye(n_levels) - pca.components_.T @ pca.components_)
            B_inv.append(B_inv_v)
            # Make diagonal-only version in case we want to use it for whitening
            B_inv_d = np.diag(np.diag(B_inv_v))

            # Robust scaling for per-variable inverse covariance (avoid huge values when std is tiny)
            std_v_flat = std_v.flatten()
            eps_v = max(1e-5, 0.01 * float(np.median(std_v_flat)))
            std_v_floor = np.maximum(std_v_flat, eps_v)
            factors_v = 1.0 / std_v_floor
            # B_inv_phys_v = (factors_v[:, None] * pca.components_.T @ np.diag(1.0/pca.explained_variance_) @ pca.components_) * factors_v[None, :]
            B_inv_phys_v = (factors_v[:, None] * B_inv_v) * factors_v[None, :]
            B_inv_d_phys = (factors_v[:, None] * B_inv_d) * factors_v[None, :]
            B_inv_phys.append(B_inv_phys_v)
            # Debug logging for this variable
            try:
                print(f"var={v} std_v: min={std_v_flat.min():.3e}, median={np.median(std_v_flat):.3e}, max={std_v_flat.max():.3e}, eps_v={eps_v}")
                print(f"var={v} B_inv_phys_v: max_abs={float(np.max(np.abs(B_inv_phys_v))):.3e}")
            except Exception:
                pass

            # Store PCA buffers in a dictionary
            pca_buffs[v] = {
                'n_comp': pca.n_components_,
                'basis': pca.components_,
                'mu': mu_v,
                'std': std_v,
                'vmin': vmin_v,
                'vmax': vmax_v,
                'eigenvalues': pca.explained_variance_,
                'scales': np.sqrt(pca.explained_variance_),
                'scales_inv': 1.0/np.sqrt(pca.explained_variance_.reshape(1, -1)),
                'B_inv': B_inv_v,
                'B_inv_d': B_inv_d,
                'B_inv_phys': B_inv_phys_v,
                'B_inv_d_phys': B_inv_d_phys,
            }

        # Combine the number of components, basis, mu, std, and eigenvalues across variables for easy use in the loss function
        pca_buffs['n_comp'] = sum(pca_buffs[v]['n_comp'] for v in range(n_vars))
        pca_buffs['basis'] = block_diag(*[pca_buffs[v]['basis'] for v in range(n_vars)])
        pca_buffs['eigenvalues'] = np.concatenate([pca_buffs[v]['eigenvalues'] for v in range(n_vars)])
        pca_buffs['mu'] = mu  # np.stack([pca_buffs[v]['mu'] for v in range(n_vars)])
        pca_buffs['std'] = std  # np.stack([pca_buffs[v]['std'] for v in range(n_vars)])
        pca_buffs['vmin'] = vmin  # np.concatenate([pca_buffs[v]['vmin'] for v in range(n_vars)])
        pca_buffs['vmax'] = vmax  # np.concatenate([pca_buffs[v]['vmax'] for v in range(n_vars)])
        pca_buffs['scales'] = np.sqrt(pca_buffs['eigenvalues'])
        pca_buffs['scales_inv'] = 1.0/np.sqrt(pca_buffs['eigenvalues'].reshape(1, -1))
        # Assemble a pseudo-inverse of the covariance matrix for the full state vector by block-diagonalizing the per-variable pseudo-inverses
        # Assemble Block-Diagonal B_inv
        B_inv = block_diag(*B_inv)
        pca_buffs['B_inv'] = B_inv
        pca_buffs['B_inv_d'] = block_diag(*[pca_buffs[v]['B_inv_d'] for v in range(n_vars)])
        B_inv_phys = block_diag(*B_inv_phys)
        pca_buffs['B_inv_phys'] = B_inv_phys
        pca_buffs['B_inv_d_phys'] = block_diag(*[pca_buffs[v]['B_inv_d_phys'] for v in range(n_vars)])

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
        alpha=input.get('alpha', 1e-4)
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