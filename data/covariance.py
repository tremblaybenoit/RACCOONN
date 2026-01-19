import numpy as np
import hydra
from omegaconf import DictConfig
from data.io import load_var_and_normalize
from data.statistics import reduction_shape, batch_statistics
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
from utilities.plot import plot_map, save_plot, flexible_gridspec
import os
import logging
import torch


# Initialize logger
logger = logging.getLogger(__name__)


def verify_torch_numpy_flattening(n_samples, n_vars, n_levels):
    """
    Verifies that NumPy and PyTorch flattening (reshape vs view)
    produce identical vector indices for the same multi-dimensional coordinate.
    """
    # 1. Create a coordinate-coded array
    # Logic: (Var * 1000) + Level. e.g., Var 2, Level 45 -> 2045.0
    arr_np = np.zeros((n_samples, n_vars, n_levels), dtype=np.float32)
    for v in range(n_vars):
        for l in range(n_levels):
            arr_np[:, v, l] = (v * 1000) + l

    # 2. NumPy Flattening (as done in your covariance script)
    flat_np = arr_np.reshape(n_samples, -1)

    # 3. PyTorch Flattening (as done in your VarLoss / QuadraticForm)
    arr_pt = torch.from_numpy(arr_np)
    flat_pt = arr_pt.view(n_samples, -1)

    # 4. Comparative Checks
    # Check 1: Do the libraries agree with each other?
    mismatch_count = np.sum(flat_np != flat_pt.numpy())

    # Check 2: Verify specific semantic indices
    # Let's check Var 1, Level 0. In row-major, this should be at index [n_levels]
    test_v, test_l = 1, 0
    expected_val = (test_v * 1000) + test_l
    actual_idx = test_v * n_levels + test_l

    val_at_idx_np = flat_np[0, actual_idx]
    val_at_idx_pt = flat_pt[0, actual_idx].item()

    print(f"--- Flattening Consistency Check ---")
    print(f"Total Mismatches (NP vs PT): {mismatch_count}")
    print(f"Value at index {actual_idx}:")
    print(f"  Expected: {expected_val}")
    print(f"  NumPy:    {val_at_idx_np}")
    print(f"  PyTorch:  {val_at_idx_pt}")

    if mismatch_count == 0 and val_at_idx_np == expected_val:
        print("SUCCESS: Flattening is consistent and Row-Major (C-style).")
    else:
        print("FAILURE: Mismatch in flattening logic!")


def prepare_stable_cholesky(err_data: np.ndarray, ridge_factor: float=1e-6, max_cond: float=1e6) -> np.ndarray:
    """ Prepare a stable Cholesky factor from error data.

        Parameters
        ----------
        err_data: np.ndarray. Error data of shape (n_samples, n_variables...).
        ridge_factor: float. Factor to scale the ridge added to the diagonal.
        max_cond: float. Maximum allowed condition number (max eigenvalue / min eigenvalue).

        Returns
        -------
        np.ndarray. Lower Cholesky factor L such that B = L @ L.T.
    """

    # 1. Force everything to float64 immediately for math stability
    target_dtype = err_data.dtype
    flat_err = err_data.reshape(err_data.shape[0], -1).astype(np.float64)

    # B will be float64
    B = np.cov(flat_err, rowvar=False)

    # 2. Add Ridge (Explicitly float64)
    ridge_val = np.mean(np.diag(B)) * ridge_factor
    B += np.eye(B.shape[0], dtype=np.float64) * ridge_val

    # 3. Eigen-Value Capping
    # This is the most sensitive part; float64 is mandatory here
    vals, vecs = np.linalg.eigh(B)
    min_val = np.max(vals) / max_cond
    vals_stable = np.maximum(vals, min_val)

    # Reconstruct B_stable
    bm_stable = vecs @ np.diag(vals_stable) @ vecs.T

    # 4. Compute Cholesky Factor L
    lm = np.linalg.cholesky(bm_stable)

    # 5. Return to the original dtype (e.g., float32) for the NN
    return lm.astype(target_dtype)


def verify_cholesky_reconstruction(B_stable: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Verifies that the Cholesky factor accurately reconstructs the covariance matrix.

    Parameters
    ----------
    B_stable: np.ndarray. The stable covariance matrix.
    L: np.ndarray. The lower Cholesky factor such that B = L @ L.T.

    Returns
    -------
    np.ndarray. Absolute differences between B_stable and reconstructed B.
    """
    # 1. Reconstruct: B_reconstructed = L @ L.T
    B_rec = L @ L.T

    # 2. Compute Residuals
    abs_diff = np.abs(B_stable - B_rec)
    max_err = np.max(abs_diff)
    mean_err = np.mean(abs_diff)

    # 3. Check Condition Numbers (The 'Health' of the matrix)
    # Higher condition numbers = more numerical instability
    cond_B = np.linalg.cond(B_stable)
    cond_L = np.linalg.cond(L)

    print("--- Cholesky Sanity Check ---")
    print(f"Max Absolute Reconstruction Error: {max_err:.2e}")
    print(f"Mean Absolute Reconstruction Error: {mean_err:.2e}")
    print(f"Condition Number of B: {cond_B:.2e}")
    print(f"Condition Number of L: {cond_L:.2e} (Should be ~sqrt of B)")

    # Threshold for float32 safety
    if max_err > 1e-4:
        print("⚠️ WARNING: Significant reconstruction error. Check your dtypes.")
    if cond_B > 1e10:
        print("⚠️ WARNING: B is extremely ill-conditioned. NN training may be unstable.")
    else:
        print("✅ SUCCESS: Matrix is stable and reconstruction is accurate.")

    return abs_diff


def innovation_error(data: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """ Return uncertainy estimation of the radiance data.

        Parameters
        ----------
        data: np.ndarray. Radiances of shape (n_samples, n_channels).

        Returns
        -------
        np.ndarray. Spatiotemporal standard deviation of the dataset.
    """
    return data[:, :10]-ref[:, :10]



def innovation_uncertainty(data: np.ndarray) -> np.ndarray:
    """ Return uncertainy estimation of the radiance data.

        Parameters
        ----------
        data: np.ndarray. Radiances of shape (n_samples, n_channels).

        Returns
        -------
        np.ndarray. Spatiotemporal standard deviation of the dataset.
    """

    # Return the standard deviation (last 10 values)
    return data[:, 10:]


def background_climatology(data: np.ndarray, axis: int=0, keepdims: bool=True) \
        -> np.ndarray:
    """ Compute spatiotemporal mean of a given dataset.

        Parameters
        ----------
        data: np.ndarray. Input profiles of shape (n_samples, n_profiles, n_levels).
        axis: int or tuple of int. Axis or axes along which the means are computed.
        keepdims: bool. If True, the reduced axes are left in the result as dimensions with size one.

        Returns
        -------
        np.ndarray. Spatiotemporal mean of the dataset.
    """

    # Compute mean
    data_mean = batch_statistics(data, which=['mean'], axis=axis)['mean']
    # Apply keepdims if necessary
    if keepdims and data_mean.ndim < data.ndim:
        data_mean = np.expand_dims(data_mean, axis=axis)
    return data_mean


def background_increment(config_true: DictConfig, config_background: DictConfig) -> np.ndarray:
    """ Compute error between ground truth and background.

        Parameters
        ----------
        config_true: DictConfig. Configuration for the ground truth dataset.
        config_background: DictConfig. Configuration for the background dataset.

        Returns
        -------
        np.ndarray. Error between ground truth and background.
    """

    # Truth - Background
    x_true = load_var_and_normalize(config_true)
    x_background = load_var_and_normalize(config_background)

    # Check dimensions and add new axis if necessary
    if x_true.ndim != x_background.ndim:
        if x_background.ndim == x_true.ndim - 1:
            x_background = x_background[np.newaxis, :]
        else:
            raise ValueError("The shapes of the true and background data do not match.")

    # Return increment
    return np.subtract(x_true, x_background, out=x_true)


def save_cholesky_factor(err_data, inflation=1.0, ridge=1e-6):
    """
    Computes and saves the Lower Cholesky factor L.
    B = L @ L.T
    """
    # 1. Compute Standard Covariance
    flat_err = err_data.reshape(err_data.shape[0], -1)
    cov = np.cov(flat_err, rowvar=False) * inflation

    # 2. Add Ridge (Tikhonov Regularization)
    # This ensures the matrix is strictly Positive Definite
    cov = cov + np.eye(cov.shape[1]) * ridge

    # 3. Compute Lower Cholesky
    try:
        L = np.linalg.cholesky(cov)
        logger.info("Successfully computed Cholesky factor L.")
    except np.linalg.LinAlgError:
        # If it fails, the ridge was too small or data is constant
        eigvals = np.linalg.eigvalsh(cov)
        logger.error(f"Matrix not PD. Min eigenvalue: {np.min(eigvals)}")
        raise

    return L.astype(np.float32)


def covariance_matrix(input: DictConfig, output: DictConfig, plot_flag: bool=True, recenter: bool=True,
                      inflation: float=1.0, var_threshold: float=1.e-8, cholesky: bool = False) -> None:
    """ Compute statistics of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        plot_flag: bool. If True, plot the covariance matrix.
        recenter: bool. If True, recenter the error by removing the mean.
        inflation: float. Inflation factor to apply to the covariance matrix.
        var_threshold: float. Variance threshold to filter variables.
        cholesky: bool. If True, compute and verify Cholesky factor for stability.

        Returns
        -------
        None.
    """

    # Error
    logger.info("Loading data...")
    err = load_var_and_normalize(input.err)

    # Pressure filter (background only)
    if hasattr(input, 'pressure_filter'):
        logger.info("Applying pressure filter...")
        pressure_filter = instantiate(input.pressure_filter.load)
        err = np.take(err.reshape(err.shape[0], -1), np.flatnonzero(pressure_filter), axis=1)

    # Compute Cholesky factor for stability verification
    if cholesky:
        logger.info("Preparing stable Cholesky factor...")
        L = prepare_stable_cholesky(err, ridge_factor=1e-6, max_cond=1e6)
        # Verify Cholesky reconstruction
        logger.info("Verifying Cholesky reconstruction...")
        verify_cholesky_reconstruction(np.cov(err.reshape(err.shape[0], -1), rowvar=False), L)
        breakpoint()

    # Compute covariance matrix
    if recenter:
        logger.info("Computing covariance matrix with recentered data...")
        err -= np.mean(err, axis=0, keepdims=True)

        # Compute Cholesky factor for stability verification
        logger.info("Preparing stable Cholesky factor...")
        if cholesky:
            L = prepare_stable_cholesky(err, ridge_factor=1e-6, max_cond=1e6)
            # Verify Cholesky reconstruction
            logger.info("Verifying Cholesky reconstruction...")
            verify_cholesky_reconstruction(np.cov(err.reshape(err.shape[0], -1), rowvar=False), L)
            breakpoint()

    # Compute covariance matrix
    logger.info("Computing covariance matrix...")
    cov = np.cov(err.reshape(err.shape[0], -1), rowvar=False)*inflation

    # Add diagonal
    if hasattr(input, 'sigma'):
        logger.info("Applying sigma scaling...")
        sigma_values = instantiate(input.sigma)
        sigma_values = sigma_values.reshape(sigma_values.shape[0], -1)
        # Add to diagonal
        for i in range(cov.shape[0]):
            cov[i, i] += sigma_values[i]**2

    # Apply variance thresholding
    logger.info("Applying variance thresholding...")
    var = np.diag(cov)
    low_var_indices = np.where(var < var_threshold)[0]
    if low_var_indices.size > 0:
        # breakpoint()
        logger.info(f"Variables below variance threshold ({var_threshold}): {low_var_indices.size}")
        # Set low-variance values to threshold
        for idx in low_var_indices:
            cov[idx, idx] = cov[idx, idx]  # var_threshold

    # Compute inverse covariance matrix
    logger.info("Computing inverse covariance matrix...")
    cov_inv = np.linalg.inv(cov).astype(err.dtype)
    var_inv = np.diag(cov_inv)
    vif = var*var_inv

    # Check if covariance matrix is positive definite
    if np.any(np.linalg.eigvals(cov_inv) <= 0):
        raise ValueError("Inverse covariance matrix is not positive definite.")

    # Save statistics to file
    logger.info("Saving inverse covariance matrix to file...")
    if hasattr(output, 'save'):
        save_func = instantiate(output.save)
        save_func(cov_inv)

    # Plot covariance matrix if required
    if plot_flag:
        logger.info("Plotting inverse covariance matrix...")
        # Create a flexible gridspec
        fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                          lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
        ax = get_axes(0, 0)
        # Plot covariance matrix
        plot_map(ax, cov_inv, title=f"Inverse covariance matrix", plt_origin='upper',
                 cb_label=r'Values (divided by 10$^4$)')
        save_plot(fig, filename=os.path.splitext(output.path)[0] + '.png')

    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute covariance matrices of given datasets.

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