import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig
from data.io import load_var_and_normalize
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging
from sklearn.decomposition import PCA
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
        'scales_z_var': pca_buffers['scales_z_var'],
        'scales_z_std': pca_buffers['scales_z_std'],
        'basis': pca_buffers['basis'],
        # 'sym_log_scales': pca_buffers['sym_log_scales'],
        # 'sym_log_z_var': pca_buffers['sym_log_z_var'],
        # 'sym_log_z_std': pca_buffers['sym_log_z_std'],
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


def generate_pca_buffers(data: np.ndarray, mode: str='global', n_comp: int=270):
    """
    Generate PCA buffers for a given dataset.

    Parameters
    ----------
    data: np.ndarray. Input data of shape (N, V, L) where N is the number of samples,
          V is the number of variables, and L is the number of levels.
    mode: str. PCA mode, either 'global' or 'local'.
    n_comp: int. Number of principal components to retain.

    Returns
    -------
    Dict containing PCA basis, mean, std, and eigenvalues.
    """

    # Get data shape
    n_samples, n_vars, n_levels = data.shape

    # Standardize
    mu = np.mean(data, axis=0, keepdims=True)  # (V, L)
    std = np.std(data, axis=0, keepdims=True) + 1e-12  # (V, L)
    increment = data - mu
    standardized_data = increment / std
    # pressure_filter = np.load('../scene_clouds_float32/pressure_filter.npy')
    # standardized_data = standardized_data[..., pressure_filter]

    if mode == 'global':
        # Flatten: (N, V*L)
        pca = PCA().fit(standardized_data.reshape(n_samples, -1))
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Find K for specific thresholds
        k_99 = np.argmax(cumulative_variance >= 0.999) + 1
        k_9999 = np.argmax(cumulative_variance >= 0.9999) + 1

        print(f"Components for 99.9% variance: {k_99}")
        print(f"Components for 99.99% variance: {k_9999}")
        breakpoint()

        plt.figure(figsize=(10, 4))
        plt.plot(cumulative_variance)
        plt.axhline(y=0.999, color='r', linestyle='--')
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Components")
        plt.savefig("cumulative_explained_variance.png")
        plt.close()

        # Decomposition
        flat_z = standardized_data.reshape(n_samples, -1)
        pca = PCA(n_components=n_comp)
        pca.fit(flat_z)
        # Compute pseudo-inverse of the PCA transformation matrix for whitening
        B_pca_inv = pca.components_.T @ np.diag(1.0/pca.explained_variance_) @ pca.components_
        # Store PCA buffers in a dictionary
        pca_buffs = {
            'n_comp': n_comp,
            'basis': pca.components_,
            'mu': mu.flatten(),
            'std': std.flatten(),
            'eigenvalues': pca.explained_variance_,
            'scales': np.sqrt(pca.explained_variance_),
            'scales_inv': 1.0/np.sqrt(pca.explained_variance_.reshape(1, -1)),
            'B_inv': B_pca_inv,
        }

        # Initialize PCA processor
        pca_processor = PCAProcessor(pca_buffs)
        # Data
        whitened_data = pca_processor.physical_to_whitened(data.reshape(n_samples, -1))
        # sym_log_data = sym_log(whitened_data, inverse_transform=False)
        # Mean
        whitened_mu0 = whitened_data.mean(axis=0, keepdims=True)
        whitened_mu1 = pca_processor.physical_to_whitened(mu.reshape(1, -1))
        # breakpoint()
        # sym_log_mu0 = sym_log_data.mean(axis=0, keepdims=True)
        # sym_log_mu1 = sym_log(whitened_mu1, inverse_transform=False)
        # Increments
        whitened_increment0 = whitened_data - whitened_mu0
        whitened_increment1 = whitened_data - whitened_mu1
        # sym_log_increment0 = sym_log_data - sym_log_mu0
        # sym_log_increment1 = sym_log(whitened_increment1, inverse_transform=False)
        # sym_log_increment2 = sym_log_data - sym_log_mu1
        # sym_log_increment3 = sym_log(whitened_increment0, inverse_transform=False)
        # Variances and stds
        # sym_log_cov0 = np.cov(sym_log_increment0, rowvar=False)
        # sym_log_var0 = np.diag(sym_log_cov0).reshape(1, -1)
        # sym_log_std0 = np.sqrt(sym_log_var0)
        # sym_log_cov_inv0 = np.linalg.inv(sym_log_cov0).astype(data.dtype)
        # sym_log_cov1 = np.cov(sym_log_increment1, rowvar=False)
        # sym_log_var1 = np.diag(sym_log_cov1).reshape(1, -1)
        # sym_log_std1 = np.sqrt(sym_log_var1)
        # sym_log_cov_inv1 = np.linalg.inv(sym_log_cov1).astype(data.dtype)
        # sym_log_cov2 = np.cov(sym_log_increment2, rowvar=False)
        # sym_log_var2 = np.diag(sym_log_cov2).reshape(1, -1)
        # sym_log_std2 = np.sqrt(sym_log_var2)
        # sym_log_cov_inv2 = np.linalg.inv(sym_log_cov2).astype(data.dtype)
        #
        pca_buffs['whitened_mu'] = whitened_mu1
        pca_buffs['scales_z_var'] = whitened_data.var(axis=0, keepdims=True)
        pca_buffs['scales_z_std'] = np.sqrt(pca_buffs['scales_z_var'])
        # pca_buffs['sym_log_z_var'] = sym_log(whitened_data, inverse_transform=False).var(axis=0, keepdims=True)
        # pca_buffs['sym_log_z_std'] = np.sqrt(pca_buffs['sym_log_z_var'])
        # pca_buffs['sym_log_var0'] = sym_log_var0
        # pca_buffs['sym_log_std0'] = sym_log_std0
        # pca_buffs['sym_log_var'] = sym_log_var1
        # pca_buffs['sym_log_std'] = sym_log_std1
        # pca_buffs['sym_log_diag_cov'] = 1.0/sym_log_cov_inv0
        # pca_buffs['sym_log_cov'] = sym_log_cov1
        # pca_buffs['sym_log_diag_cov'] = 1.0/np.diag(sym_log_cov_inv1).reshape(1, -1)
        # pca_buffs['sym_log_diag_cov_inv0'] = sym_log_cov_inv0
        # pca_buffs['sym_log_cov_inv'] = sym_log_cov_inv1
        # pca_buffs['sym_log_diag_cov_inv'] = np.diag(sym_log_cov_inv1).reshape(1, -1)


        # fig, get_axes = flexible_gridspec(cell_widths=[4.0, 4.0], cell_heights=[4.0, 4.0], lefts=[1.00, 1.00],
        #                                   rights=[1.00, 1.00], bottoms=[1.00, 1.00], tops=[1.00, 1.00])
        # ax0 = get_axes(0, 0)
        # plot_map(ax0, sym_log_cov_inv0, title=f"Inverse covariance matrix", plt_origin='upper',
        #          cb_label=r'Values (divided by 10$^4$)')
        # ax1 = get_axes(0, 1)
        # plot_map(ax1, sym_log_cov_inv1, title=f"Inverse covariance matrix (alt method)", plt_origin='upper',
        #          cb_label=r'Values (divided by 10$^4$)')
        # save_plot(fig, "pca_inverse_covariance_matrices.png")
        # plt.close(fig)

        return pca_buffs
    elif mode == 'local':
        # Per variable PCA (with variables in the second dimension)
        pca_buffs = {}
        for v in range(n_vars):
            pca = PCA()
            pca.fit(standardized_data[:, v, :])
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            # Find K for specific thresholds
            k_99 = np.argmax(cumulative_variance >= 0.999) + 1
            k_9999 = np.argmax(cumulative_variance >= 0.9999) + 1
            print(f"Components for 99.9% variance: {k_99}")
            print(f"Components for 99.99% variance: {k_9999}")

            # Decomposition
            flat_z = standardized_data[:, v, :].reshape(n_samples, -1)
            pca = PCA(n_components=n_comp)
            pca.fit(flat_z)
            pca_buffs[v] = {
                'basis': pca.components_,
                'mu': mu[0:1, v],
                'std': std[0:1, v],
                'eigenvalues': pca.explained_variance_,
                'scales': np.sqrt(pca.explained_variance_),
                'scales_inv': 1.0/np.sqrt(pca.explained_variance_.reshape(1, -1)),
            }
        return pca_buffs
    else:
        raise ValueError(f"Invalid PCA mode: {mode}. Must be 'global' or 'local'.")


def project_pca(input: DictConfig, output: DictConfig, mode='global', n_comp: int=270) -> None:
    """ Project data onto PCA basis.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        mode: str. PCA mode, either 'global' or 'local'.
        n_comp: int. Number of principal components to retain.

        Returns
        -------
        np.ndarray. Projected data.
    """

    # Load PCA buffers
    logger.info("Loading PCA buffers...")
    pca_buffs = instantiate(input.pca_buffers.load)

    # Global mode
    if mode == 'global':

        # Initialize PCA processor
        pca_processor = PCAProcessor(pca_buffs)

        # Load data
        logger.info("Loading data...")
        data = load_var_and_normalize(input.data)

        # Project data
        logger.info("Projecting data onto PCA basis...")
        n_samples = data.shape[0]
        flat_data = data.reshape(n_samples, -1)
        standardized_data = pca_processor.physical_to_standardized(flat_data)
        whitened_data = pca_processor.physical_to_whitened(flat_data)

    # Local mode
    elif mode == 'local':

        # Load data
        logger.info("Loading data...")
        data = load_var_and_normalize(input.data)

        # Project data
        logger.info("Projecting data onto PCA basis...")
        n_samples, n_vars, n_levels = data.shape
        standardized_data = np.zeros((n_samples, n_vars, n_comp))
        whitened_data = np.zeros((n_samples, n_vars, n_comp))
        for v in range(n_vars):
            pca_processor = PCAProcessor(pca_buffs[v])
            standardized_data[:, v, :] = pca_processor.physical_to_standardized(data[:, v, :])
            whitened_data[:, v, :] = pca_processor.physical_to_whitened(data[:, v, :])

    else:
        raise ValueError(f"Invalid PCA mode: {mode}. Must be 'global' or 'local'.")

    # Save statistics to file
    logger.info("Saving data to file...")
    if hasattr(output.pca, 'save'):
        save_func = instantiate(output.pca.save)
        save_func(standardized_data)
    if hasattr(output.pca_white, 'save'):
        save_func = instantiate(output.pca_white.save)
        save_func(whitened_data)

    return


def compute_pca(input: DictConfig, output: DictConfig, mode: str='global', n_comp: int=270) -> None:
    """ Compute pca decomposition of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        mode: str. PCA mode, either 'global' or 'local'.
        n_comp: int. Number of principal components to retain.

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
        n_comp=input.get('n_comp', n_comp)
    )

    # Save statistics to file
    logger.info("Saving data to file...")
    if hasattr(output, 'save'):
        save_func = instantiate(output.save)
        save_func(pca_buffs)

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