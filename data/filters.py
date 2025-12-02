import numpy as np
import torch
from typing import Union
import hydra
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def clearsky_filter(prof: Union[np.ndarray, torch.Tensor], split: Union[np.ndarray, torch.tensor] = None) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Filter out profiles with clear skies.

        Parameters
        ----------
        prof: np.ndarray or torch.Tensor. Input profiles of shape (n_samples, n_profiles, n_levels).
        split: np.ndarray or torch.tensor. Indices to split the profiles.

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating cloudy/clear-sky profiles.
    """

    if isinstance(prof, torch.Tensor):
        clrsky = (prof[:, 5, :].sum(dim=1) == 0) & (prof[:, 6, :].sum(dim=1) == 0)
        clrsky = clrsky & (prof[:, 7, :].sum(dim=1) == 0)
    else:
        clrsky = (prof[:, 5, :].sum(axis=1) == 0) & (prof[:, 6, :].sum(axis=1) == 0)
        clrsky = clrsky & (prof[:, 7, :].sum(axis=1) == 0)

    # Apply split
    if split is not None:
        clrsky = clrsky[split]
    return clrsky


def cloud_filter(prof: Union[np.ndarray, torch.Tensor], split: Union[np.ndarray, torch.tensor] = None) \
        -> Union[np.ndarray, torch.Tensor]:
    """ Filter out profiles with clear skies.

        Parameters
        ----------
        prof: np.ndarray or torch.Tensor. Input profiles of shape (n_samples, n_profiles, n_levels).
        split: np.ndarray or torch.tensor. Indices to split the profiles.

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating cloudy/clear-sky profiles.
    """

    return ~clearsky_filter(prof, split=split)


def pressure_filter(prof: Union[np.ndarray, torch.Tensor], threshold: float = 1.e-6) -> Union[np.ndarray, torch.Tensor]:
    """ Filter out profiles based on a pressure threshold.

        Parameters
        ----------
        prof: np.ndarray or torch.Tensor. Input profiles of shape (n_samples, n_profiles, n_levels).
        threshold: float. Variance threshold to filter profiles.

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating profiles above the pressure threshold.
    """

    if isinstance(prof, torch.Tensor):
        # Compute variance of the profiles
        prof_var = torch.var(prof, dim=0, keepdim=True).squeeze()
    else:
        prof_var = prof.var(axis=0, keepdims=True).squeeze()

    return prof_var > threshold

def daytime_filter(meta: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """ Filter out profiles based on daytime/nighttime condition.

        Parameters
        ----------
        meta: np.ndarray or torch.Tensor. Metadata array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating daytime profiles.
    """

    return meta[:, 6] >= 90


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute statistics of a given dataset.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # Compute model and observation covariance matrices
    if hasattr(config.preparation, "covariance"):
        for dataset, config_covariance in config.preparation.covariance.items():
            logger.info(f"Computing error covariance matrix of {dataset} set")
            instantiate(config_covariance)

    return


if __name__ == '__main__':
    """ Compute various filters.

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