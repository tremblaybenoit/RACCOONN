import numpy as np


def pa_to_hpa(pressure: float | np.ndarray) -> float | np.ndarray:
    """
    Convert pressure from Pascals (Pa) to hectoPascals (hPa).

    Parameters
    ----------
    pressure: float or np.ndarray. Pressure in Pascals.

    Returns
    -------
    float or np.ndarray. Pressure in hectoPascals.
    """
    return pressure / 100.0


def hpa_to_pa(pressure: float | np.ndarray) -> float | np.ndarray:
    """
    Convert pressure from hectoPascals (hPa) to Pascals (Pa).

    Parameters
    ----------
    pressure: float or np.ndarray. Pressure in hectoPascals.

    Returns
    -------
    float or np.ndarray. Pressure in Pascals.
    """
    return pressure * 100.0
