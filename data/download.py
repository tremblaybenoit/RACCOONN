import os
import logging
import hydra
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from utilities.logic import get_config_path


# Initialize logger
logger = logging.getLogger(__name__)


def download_data(dir: str, url: str) -> None:
    """ Download data for CRTM forward model.

        Parameters
        ----------
        dir: str. Directory to download data to.
        url: str. URL to download data from.

        Returns
        -------
        None.
    """

    # Download zipped data from url, unzip, and have directories from zipped data populate args.data_dir
    zip_path = os.path.join(dir, "tmp.zip")
    logger.info(f"Downloading data from {url} to {dir}...")
    os.system(f"wget -O {zip_path} {url}")
    logger.info(f"Unzipping data to {dir}...")
    os.system(f"unzip {zip_path} -d {dir}")
    logger.info("Cleaning up temporary files...")
    os.remove(zip_path)
    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Download data for CRTM forward model.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # Compute model and observation covariance matrices
    if hasattr(config.data, 'download'):
        logger.info("Downloading data...")
        instantiate(config.data.download)

    return


if __name__ == '__main__':
    """ Compute statistics of a given dataset.

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
