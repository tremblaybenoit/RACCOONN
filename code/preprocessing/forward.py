import hydra
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging

# Initialize logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """
    Compute forward-modeled radiances.

    Parameters
    ----------
    config: DictConfig. Main hydra configuration file containing all model hyperparameters.

    Returns
    -------
    None.
    """

    # Compute forward-modeled radiances
    if hasattr(config.preprocessing, "forward"):
        # If single operation, execute
        if hasattr(config.preprocessing.forward, "_target_"):
            logger.info(f"Computing forward-modeled radiances...")
            instantiate(config.preprocessing.forward)
        # Execute individual operations
        else:
            for dataset, config_forward in config.preprocessing.forward.items():
                if hasattr(config_forward, '_target_'):
                    logger.info(f"Computing forward-modeled radiances for {dataset}")
                    instantiate(config_forward)

    return


if __name__ == '__main__':
    """ Compute forward-modeled radiances of given datasets.

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