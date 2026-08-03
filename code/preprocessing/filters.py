import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def composite_mask(masks: list[np.ndarray]) -> np.ndarray:
    """ Combine multiple masks into a single composite mask.

        Parameters
        ----------
        masks: list of np.ndarray. List of boolean masks to combine.

        Returns
        -------
        np.ndarray. Composite boolean mask where True indicates data points that pass all individual masks.
    """

    # Initialize composite mask with the first mask in the list
    composite = masks[0].copy()

    # Combine all masks using logical AND
    for mask in masks[1:]:
        composite &= mask

    return composite


def spatial_mask(lat: np.ndarray | torch.Tensor, lon: np.ndarray | torch.Tensor,
                 lat_min: float  = -90, lat_max: float = 90, lon_min: float  = -180,
                 lon_max: float = 180) -> np.ndarray | torch.Tensor:
    """ Filter out data based on spatial mask.

        Parameters
        ----------
        lat: np.ndarray or torch.Tensor. Latitude of input data.
        lon: np.ndarray or torch.Tensor. Longitude of input data.
        lat_min: float. Minimum latitude.
        lat_max: float. Maximum latitude.
        lon_min: float. Minimum longitude.
        lon_max: float. Maximum longitude.

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating data within the specified spatial bounds.
    """

    # Initialize mask
    mask = np.ones_like(lat, dtype=bool) if isinstance(lat, np.ndarray) else torch.ones_like(lat, dtype=torch.bool)

    # Apply latitude and longitude bounds
    mask &= lat >= lat_min
    mask &= lat <= lat_max
    mask &= lon >= lon_min
    mask &= lon <= lon_max

    return mask


def temporal_mask(scans: np.ndarray | torch.Tensor, scans_min: float = 0, scans_max: float = 1) -> np.ndarray | torch.Tensor:
    """ Filter out data based on temporal mask.

        Parameters
        ----------
        scans: np.ndarray or torch.Tensor. Input scans.
        scans_min: float. Minimum scan value.
        scans_max: float. Maximum scan value.

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating data within the specified time range.
    """

    # Initialize mask
    mask = np.ones_like(scans, dtype=bool) if isinstance(scans, np.ndarray) else torch.ones_like(scans, dtype=torch.bool)

    # Apply temporal bounds
    mask &= scans >= scans_min
    mask &= scans <= scans_max

    return mask


def clear_mask(prof: np.ndarray | torch.Tensor, split: np.ndarray | torch.Tensor | None = None) \
        -> np.ndarray | torch.Tensor:
    """ Filter out profiles with clear skies.

        Parameters
        ----------
        prof: np.ndarray or torch.Tensor. Input profiles of shape (n_samples, n_profiles, n_levels).
        split: np.ndarray or torch.tensor. Indices to split the profiles.

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating cloudy/clear-sky profiles.
    """

    # Check if data is not already filtered for clear sky
    if prof.shape[1] > 3:
        if isinstance(prof, torch.Tensor):
            c = (prof[:, 5, :].sum(dim=1) == 0) & (prof[:, 6, :].sum(dim=1) == 0)
            c = c & (prof[:, 7, :].sum(dim=1) == 0)
        else:
            c = (prof[:, 5, :].sum(axis=1) == 0) & (prof[:, 6, :].sum(axis=1) == 0)
            c = c & (prof[:, 7, :].sum(axis=1) == 0)
    else:
        if isinstance(prof, torch.Tensor):
            c = torch.ones((prof.shape[0],), dtype=torch.bool)
        else:
            c = np.ones((prof.shape[0],), dtype=bool)

    # Apply split
    if split is not None:
        c = c[split]
    return c


def cloud_mask(prof: np.ndarray | torch.Tensor, split: np.ndarray | torch.Tensor | None = None) \
        -> np.ndarray | torch.Tensor:
    """ Filter out profiles with clear skies.

        Parameters
        ----------
        prof: np.ndarray or torch.Tensor. Input profiles of shape (n_samples, n_profiles, n_levels).
        split: np.ndarray or torch.tensor. Indices to split the profiles.

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating cloudy/clear-sky profiles.
    """

    return ~clear_mask(prof, split=split)


def pressure_mask(stats: dict[str, np.ndarray],
                  threshold: float = 0.005) -> np.ndarray:
    """ Determine which pressure levels are non-constant based on coefficient of variation.

        Identifies pressure levels that have meaningful variability across the dataset.
        A level is considered "constant" if its coefficient of variation (stdev/mean)
        is below the threshold.

        Parameters
        ----------
        stats: dict[str, np.ndarray]. Statistics dictionary with keys:
               - 'mean': np.ndarray of shape (n_levels,), mean pressure per level
               - 'stdev': np.ndarray of shape (n_levels,), standard deviation per level
               Typically obtained from compute_dataset_statistics() or read from file.
        threshold: float. Coefficient of variation threshold. Default 0.005 (0.5%).
                   Levels with CV < threshold are considered constant.

        Returns
        -------
        np.ndarray. Boolean mask of shape (n_levels,) where:
               True = level is non-constant (has meaningful variation)
               False = level is essentially constant (CV below threshold)
    """

    # Extract mean and stdev
    mean = np.asarray(stats.get('mean'))
    stdev = np.asarray(stats.get('stdev'))

    # Compute coefficient of variation
    # Add small epsilon to avoid division by zero
    cv = stdev / (np.abs(mean) + 1.e-10)

    # Boolean mask: True where CV >= threshold (non-constant)
    mask = np.asarray(cv >= threshold, dtype=bool)

    return mask


def daytime_mask(meta: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """ Filter out profiles based on daytime/nighttime condition.

        Parameters
        ----------
        meta: np.ndarray or torch.Tensor. Metadata array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating daytime profiles.
    """

    return meta[:, 6] >= 90


def nighttime_mask(meta: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """ Filter out profiles based on daytime/nighttime condition.

        Parameters
        ----------
        meta: np.ndarray or torch.Tensor. Metadata array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray or torch.Tensor. Boolean mask indicating nighttime profiles.
    """

    return ~daytime_mask(meta)


def compute_mask(input: DictConfig, output: DictConfig) -> None:
    """ Compute masks for a given dataset.

        Parameters
        ----------
        input: DictConfig. Input configuration.
        output: DictConfig. Output configuration.

        Returns
        -------
        None.
    """
    # TODO: Make more uniform with other preprocessing functions

    # Instantiate mask
    mask = instantiate(input)

    # Save function
    if hasattr(output, 'save'):
        # Create directory if needed
        if hasattr(output.save, 'path'):
            logger.info(f"Saving statistics to file {output.save.path}.")
            os.makedirs(os.path.dirname(output.save.path), exist_ok=True)
        save_fn = instantiate(output.save)
        save_fn(mask)

    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
        Compute various data filters/masks.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Compute filters
    if hasattr(config.preprocessing, "filters"):
        for key, config in config.preprocessing.filters.items():
            logger.info(f"Computing mask: {key}")
            instantiate(config)

    return


if __name__ == '__main__':
    """ Compute various data filters/masks.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        Masks and filters.
    """

    main()