import numpy as np


# RMSE computations
def rmse(errors, axis=0):
    """ Computes the root mean squared error (RMSE) between predicted and true values.

    Parameters
    ----------
    errors : np.ndarray. Array of errors (predicted - true values).
    axis : int. Axis along which to compute the RMSE (default is 0).

    Returns
    -------
    np.ndarray. RMSE values.
    """

    return np.sqrt(np.mean(errors ** 2, axis=axis))
