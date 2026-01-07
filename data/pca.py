import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig
from data.io import load_var_and_normalize
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging
from sklearn.decomposition import PCA


# Initialize logger
logger = logging.getLogger(__name__)

def generate_pca_buffers(data: np.ndarray, mode: str='global', n_comp: int=50):
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

    pca = PCA().fit(standardized_data)
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
    breakpoint()

    if mode == 'global':
        # Flatten: (N, V*L)
        flat_z = standardized_data.reshape(n_samples, -1)
        pca = PCA(n_components=n_comp)
        pca.fit(flat_z)

        return {
            'basis': pca.components_,
            'mu': mu.flatten(),
            'std': std.flatten(),
            'eigenvalues': pca.explained_variance_
        }
    else:
        raise NotImplementedError(f"PCA mode '{mode}' is not implemented.")


def compute_pca(input: DictConfig, output: DictConfig) -> None:
    """ Compute pca decomposition of a given dataset.

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
    data = load_var_and_normalize(input.data)

    # Generate PCA buffers
    logger.info("Generating PCA buffers...")
    pca_buffs = generate_pca_buffers(
        data=data,
        mode=input.get('mode', 'global'),
        n_comp=input.get('n_comp', 50)
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