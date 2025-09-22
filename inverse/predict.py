import logging
import hydra
from omegaconf import DictConfig
from utilities.instantiators import instantiate
from inverse.train import Operator
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
    logger.info("Initializing inverse model...")
    inverse_model = Operator(config)

    # Evaluate on prediction set
    logger.info("Predicting using the inverse model...")
    pred = inverse_model.predict(config.loader)

    # Save predictions to file
    logger.info("Saving predictions to file...")
    save_function = instantiate(config.loader.stage.predict.results.prof.save)
    save_function(pred)


if __name__ == '__main__':
    """ Predict using the inverse operator.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        Prediction.
    """

    main()
