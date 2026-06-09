import numpy as np
from scipy.linalg import block_diag
import hydra
from omegaconf import DictConfig
from data.io import load_var, load_var_and_normalize
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
from utilities.plot import plot_map, save_plot, flexible_gridspec
import os
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def obs_error(data: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """ Return innovation of the radiance data.

        Parameters
        ----------
        data: np.ndarray. Radiances of shape (n_samples, n_channels).
        obs:

        Returns
        -------
        np.ndarray. Solution.
    """
    return data[:, :10]-obs[:, :10]


def obs_uncertainty(data: np.ndarray) -> np.ndarray:
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


def increment(config_true: DictConfig, config_background: DictConfig) -> np.ndarray:
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


def background_from_perturbations(input: DictConfig, output: DictConfig) -> np.ndarray | None:
    """ Compute the background from the model error covariance matrix and perturbations.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.

        Returns
        -------
        None.
    """

    # Load Cholesky matrix
    cov_cholesky = load_var(input.cholesky)
    # Load samples
    x_t = load_var_and_normalize(input.prof)
    x_dims = x_t.shape
    x_t = x_t.reshape(x_dims[0], -1)

    # Compute random perturbations from a normal distribution
    logger.info(f"Generating independent random perturbations for {x_dims[0]} samples...")
    p = np.random.normal(0, 1, size=(cov_cholesky.shape[1], x_dims[0]))
    dx = (cov_cholesky @ p).T

    # Pressure filter
    if hasattr(input, 'pressure_filter') and input.pressure_filter is not None:
        logger.info("Applying pressure filter...")
        # Load filter
        pressure_filter = instantiate(input.pressure_filter.load)
        if x_dims[1] != pressure_filter.shape[0]:
            pressure_filter = np.take(pressure_filter, [0, 4, 8], axis=0)
        # Compute background by applying perturbations to samples
        logger.info("Apply perturbations...")
        x_t[:, np.flatnonzero(pressure_filter)] += dx
    else:
        # Compute background by applying perturbations to samples
        logger.info("Apply perturbations...")
        x_t += dx

    # Reshape
    x_t = x_t.reshape(x_dims)

    # Unnormalize the data prior to saving
    if hasattr(input.prof, 'normalization'):
        norm_func = instantiate(input.prof.normalization, inverse_transform=True)
        x_t = norm_func(x_t)

    # Save to file
    if output is not None and hasattr(output, 'save'):
        logger.info(f"Saving background to {output.path}...")
        save_func = instantiate(output.save)
        save_func(x_t)
        return None
    else:
        return x_t


def climatological_matrix(input: DictConfig, output: DictConfig, scaling_factor: float = 1.0, regularization_factor: float = 1.0,
                          plot_flag: bool=True, recenter: bool=False, univariate: bool = False) -> None:
    """ Compute climatological covariance matrix of a given dataset.

        Parameters
        ----------
        input: DictConfig. Main hydra configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration.
        plot_flag: bool. If True, plot the covariance matrix.
        recenter: bool. If True, recenter by removing the mean.

        Returns
        -------
        None.
    """

    # Begin by loading the data and normalizing it
    logger.info("Loading data...")
    data = load_var_and_normalize(input.data)
    n_samples, n_vars = data.shape[0], data.shape[1]

    # Apply recentering
    if recenter:
        logger.info("Recentering data around the mean...")
        # Compute spatiotemporal mean
        mu = np.mean(data, axis=0)
        # Compute anomalies (thruth - climatological mean)
        data -= mu

    # Pressure filter (background only)
    if hasattr(input, 'pressure_filter') and input.pressure_filter is not None:
        # Load filter
        pressure_filter = instantiate(input.pressure_filter.load)
        if n_vars != pressure_filter.shape[0]:
            pressure_filter = np.take(pressure_filter, [0, 4, 8], axis=0)
    else:
        pressure_filter = None

    # Univariate matrix computation steps
    if univariate:
        # Check dimensions
        if data.ndim <=2:
            raise ValueError("Univariate covariance matrix computation requires data with more than 2 dimensions.")
        # Initialize empty matrix
        m = []
        # Loop over varialbes
        for i in range(n_vars):
            # Compute sub-matrix
            data_i = data[:, i]
            # Apply pressure filter
            if pressure_filter is not None:
                logger.info(f"Applying pressure filter for variable {i}...")
                # Remove constant pressure levels from the data
                data_i = np.take(data_i, np.flatnonzero(pressure_filter[i]), axis=1)
            # Store diagonal block
            logger.info(f"Computing univariate covariance matrix for variable {i}...")
            m.append(scaling_factor*(data_i.T @ data_i)/(n_samples-1))
        # Assemble
        del data_i
        cov = {'matrix': block_diag(*m)}
    # Multivariate matrix computation steps
    else:
        # Flatten data
        data = data.reshape(n_samples, -1)
        # Apply pressure filter
        if pressure_filter is not None:
            logger.info("Applying pressure filter...")
            # Remove constant pressure levels from the data
            data = np.take(data, np.flatnonzero(pressure_filter), axis=1)
        # Compute matrix
        logger.info("Computing covariance matrix...")
        cov = {'matrix': scaling_factor*(data.T @ data)/(n_samples-1)}
    # Release memory
    del data

    # Apply regularization
    if regularization_factor > 0:
        logger.info("Applying regularization factor...")
        cov['matrix'] += regularization_factor * np.mean(np.diag(cov['matrix'])) * np.eye(cov['matrix'].shape[0])
    # Compute Cholesky decomposition
    if hasattr(output, 'cholesky'):
        logger.info("Computing Cholesky decomposition...")
        cov['cholesky'] = np.linalg.cholesky(cov['matrix'])
    # Compute matrix inverse
    if hasattr(output, 'inverse'):
        logger.info("Computing the inverse of the covariance matrix...")
        cov['inverse'] = np.linalg.inv(cov['matrix'])
    # Compute matrix pseudo-inverse
    if hasattr(output, 'pseudo_inverse'):
        logger.info("Computing the pseudo-inverse of the covariance matrix...")
        rcond = output.pseudo_inverse.get('params.rcond', 1.e-3)
        cov['pseudo_inverse'] = np.linalg.pinv(cov['matrix'], rcond=rcond)

    # Loop over keys
    for key in list(cov.keys()):
        # Save to file
        if hasattr(output, key):
            if hasattr(output[key], 'save'):
               logger.info(f"Saving {key} matrix...")
               save_func = instantiate(output[key].save)
               save_func(cov[key])
            # Plot covariance matrix if requested
            if plot_flag and key in ['matrix', 'inverse', 'pseudo_inverse']:
                logger.info(f"Plotting {key} matrix...")
                fig, get_axes = flexible_gridspec(cell_widths=[4.0], cell_heights=[4.0],
                                                  lefts=[1.00], rights=[1.00], bottoms=[1.00], tops=[1.00])
                ax = get_axes(0, 0)
                plot_map(ax, cov[key], title=f"Covariance matrix: {key}", plt_origin='upper', cb_label=r'Values')
                save_plot(fig, filename=os.path.splitext(output[key].path)[0] + '.png')

    return


def covariance_matrix(input: DictConfig, output: DictConfig, plot_flag: bool=True, recenter: bool=False,
                      inflation: float=1.0, var_threshold: float=1.e-7, cholesky: bool = True) -> None:
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
        # for idx in low_var_indices:
        #     cov[idx, idx] = cov[idx, idx] + 1.
        good_var_indices = np.where(var >= var_threshold)[0]
        cov = np.cov(err.reshape(err.shape[0], -1)[:, good_var_indices], rowvar=False)*inflation
        var = np.diag(cov)

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
    breakpoint()

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