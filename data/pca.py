import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig
from data.io import load_var_and_normalize
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging
from sklearn.decomposition import PCA
from data.transformations import sym_log


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
        'basis': pca_buffers['basis']
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
    mu = np.mean(data, axis=0)  # (V, L)
    std = np.std(data, axis=0)  # (V, L)
    standardized_data = (data - mu) / std

    """
    pca = PCA().fit(standardized_data.reshape(n_samples, -1))
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find K for specific thresholds
    k_99 = np.argmax(cumulative_variance >= 0.999) + 1
    k_9999 = np.argmax(cumulative_variance >= 0.9999) + 1

    print(f"Components for 99.9% variance: {k_99}")
    print(f"Components for 99.99% variance: {k_9999}")

    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_variance)
    plt.axhline(y=0.999, color='r', linestyle='--')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.savefig("cumulative_explained_variance.png")
    plt.close()
    """


    if mode == 'global':
        # Flatten: (N, V*L)
        flat_z = standardized_data.reshape(n_samples, -1)
        pca = PCA(n_components=n_comp)
        pca.fit(flat_z)
        pca_buffs = {
            'basis': pca.components_,
            'mu': mu.flatten(),
            'std': std.flatten(),
            'eigenvalues': pca.explained_variance_,
            'scales': np.sqrt(pca.explained_variance_),
            'sym_log_scales': sym_log(np.sqrt(pca.explained_variance_), inverse_transform=False),
        }

        # Initialize PCA processor
        pca_processor = PCAProcessor(pca_buffs)
        projected_z = pca_processor.physical_to_whitened(flat_z)
        pca_buffs['sym_log_z_var'] = sym_log(projected_z, inverse_transform=False).var(axis=0, keepdims=True)

        return pca_buffs
    else:
        raise NotImplementedError(f"PCA mode '{mode}' is not implemented.")


def project_pca(input: DictConfig, output: DictConfig) -> None:
    """ Project data onto PCA basis.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.

        Returns
        -------
        np.ndarray. Projected data.
    """

    # Load PCA buffers
    logger.info("Loading PCA buffers...")
    pca_buffs = instantiate(input.pca_buffers.load)

    # Initialize PCA processor
    pca_processor = PCAProcessor(pca_buffs)

    # Load data
    logger.info("Loading data...")
    data = load_var_and_normalize(input.data)

    # Project data
    logger.info("Projecting data onto PCA basis...")
    n_samples = data.shape[0]
    flat_data = data.reshape(n_samples, -1)
    pca_data = pca_processor.physical_to_standardized(flat_data)
    projected_data = pca_processor.physical_to_whitened(flat_data)
    sym_log_data = sym_log(projected_data, inverse_transform=False)
    """    # Assuming 'whitened_coeffs' is your (N, 270) array
    abs_coeffs = np.abs(projected_data)

    print(f"Total coefficients: {abs_coeffs.size}")
    print(f"Values > 3 sigma:  {np.sum(abs_coeffs > 3)}  ({np.sum(abs_coeffs > 3) / abs_coeffs.size:.4%})")
    print(f"Values > 10 sigma: {np.sum(abs_coeffs > 10)} ({np.sum(abs_coeffs > 10) / abs_coeffs.size:.4%})")
    print(f"Values > 30 sigma: {np.sum(abs_coeffs > 30)} ({np.sum(abs_coeffs > 30) / abs_coeffs.size:.4%})")

    # Check if outliers are concentrated in specific components (the 'tail' of the PCA)
    outliers_per_component = np.sum(abs_coeffs > 10, axis=0)

    # 1. Identify the 'Extreme' sample
    idx = np.argmax(np.abs(projected_data).max(axis=1))
    extreme_w = projected_data[idx]
    # 2. Reconstruct it normally
    phys_full = pca_processor.whitened_to_physical(extreme_w)

    # 3. 'Clip' the outliers to a reasonable range (e.g., +/- 3)
    clipped_w = np.clip(extreme_w, -3, 3)
    phys_clipped = pca_processor.whitened_to_physical(clipped_w)

    # 1. Grab a sample index that has a >30 sigma value
    extreme_idx = np.where(np.abs(projected_data) > 30)[0][0]

    # 2. Get the two physical reconstructions (assuming you have these arrays from Pdb)
    # phys_full and phys_clipped for that extreme_idx

    plt.figure(figsize=(10, 6))

    # Plotting indices as a proxy for height/pressure levels
    # Note: Usually indices 0-127 are Temp, 128-255 are Q, etc.
    indices = np.arange(len(phys_full))

    plt.plot(indices, phys_full, label='Full Reconstruction (with Outliers)', alpha=0.8, color='crimson')
    plt.plot(indices, phys_clipped, label='Clipped Reconstruction (Standardized)', alpha=0.8, color='black',
             linestyle='--')

    # Highlight the delta
    plt.fill_between(indices, phys_full, phys_clipped, color='gray', alpha=0.3, label='Information Lost by Clipping')

    plt.title(f"Impact of Whitened Outliers on Physical Profile (Sample {extreme_idx})")
    plt.xlabel("Feature Index (Vertical Levels/Variables)")
    plt.ylabel("Physical Value (Normalized)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig("project_explained_variance.png")
    """

    # Save statistics to file
    logger.info("Saving data to file...")
    if hasattr(output.pca, 'save'):
        save_func = instantiate(output.pca.save)
        save_func(pca_data)
    if hasattr(output.pca_white, 'save'):
        save_func = instantiate(output.pca_white.save)
        save_func(projected_data)
    if hasattr(output.pca_sym, 'save'):
        save_func = instantiate(output.pca_sym.save)
        save_func(sym_log_data)

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