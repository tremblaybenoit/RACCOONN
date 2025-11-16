import numpy as np
import torch
from typing import Union


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

    if isinstance(prof, torch.Tensor):
        clrsky = (prof[:, 5, :].sum(dim=1) == 0) & (prof[:, 6, :].sum(dim=1) == 0)
        clrsky = clrsky & (prof[:, 7, :].sum(dim=1) == 0)
    else:
        clrsky = np.logical_and(prof[:, 5, :].sum(axis=1) == 0, prof[:, 6, :].sum(axis=1) == 0,
                                prof[:, 7, :].sum(axis=1) == 0)

    # Apply split
    if split is not None:
        clrsky = clrsky[split]
    return ~clrsky


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
