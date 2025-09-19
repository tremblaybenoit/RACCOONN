import logging
import hydra
from omegaconf import DictConfig
import torch
from forward.train import Operator
from utilities.logic import get_config_path
# torch.set_float32_matmul_precision('high')
torch.set_default_dtype(torch.float64)

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
    logger.info("Initializing the inverse model...")
    inverse_model = Operator(config)

    # Train the model
    logger.info("Training the inverse model...")
    inverse_model.train()


if __name__ == '__main__':
    """ Train radiative transfer inverse operator.

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
