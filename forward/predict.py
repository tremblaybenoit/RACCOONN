import logging
import hydra
from omegaconf import DictConfig
from forward.utilities.instantiators import instantiate
from forward.train import CRTMEmulator
from forward.utilities.logic import get_config_path


# Initialize logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Train neural network based on set of configurations.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Initialize trainer object
    logger.info("Initializing CRTM emulator...")
    crtm = CRTMEmulator(config)

    # Evaluate on test set
    logger.info("Testing CRTM emulator...")
    pred = crtm.predict(config.loader)

    # Save predictions to file
    logger.info("Saving predictions to file...")
    save_function = instantiate(config.data.stage.predict.hofx.save)
    save_function(pred)


if __name__ == '__main__':
    """ Predict using the CRTM emulator.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        checkpoint: Training weights & biases.
    """

    main()
