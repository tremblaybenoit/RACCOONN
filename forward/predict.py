import logging
import hydra
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from forward.train import Operator
from utilities.logic import get_config_path
import torch
torch.set_float32_matmul_precision('high')

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
    logger.info("Initializing forward model...")
    forward_model = Operator(config)

    # Evaluate on prediction set
    logger.info("Predicting using the forward model...")
    pred = forward_model.predict(config.loader)

    # Save predictions to file
    logger.info("Saving predictions to file...")
    save_function = instantiate(config.loader.stage.predict.results.hofx.save)
    save_function(pred)


if __name__ == '__main__':
    """ Predict using the forward model.

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
