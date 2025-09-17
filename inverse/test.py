import logging
import hydra
from omegaconf import DictConfig
from inverse.train import InverseOperator
from forward.utilities.logic import get_config_path
import torch
# torch.set_float32_matmul_precision('high')
torch.set_default_dtype(torch.float64)

# Initialize logger
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Test neural network based on set of configurations.

        Parameters
        ----------
        config: str. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Initialize trainer object
    inv = InverseOperator(config)

    # Evaluate on test set
    inv.test()


if __name__ == '__main__':
    """ Test inverse operator.

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
