import numpy as np
import os
import hydra
from omegaconf import DictConfig
from code.data.io import load_variable
from code.model.forward import ForwardModel
from utilities.instantiators import instantiate
from utilities.logic import get_config_path
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def load_forward_model(forward_model: DictConfig | ForwardModel) -> ForwardModel:
    """ Load a forward model from a configuration or an existing instance.

    Parameters
    ----------
    forward_model: DictConfig or ForwardModel. The configuration for the forward model or an existing instance.

    Returns
    -------
    ForwardModel. The loaded and initialized forward model.
    """

    if isinstance(forward_model, DictConfig):
        forward_model = instantiate(forward_model)
    elif isinstance(forward_model, ForwardModel):
        forward_model = forward_model
    else:
        raise ValueError("forward_model must be a DictConfig or a ForwardModel instance.")

    # Load checkpoint if ckpt_path is provided
    if hasattr(forward_model, 'ckpt_path') and forward_model.ckpt_path:
        forward_model.load_ckpt(freeze=True)  # type: ignore

    # Explicitly freeze all forward model parameters (no gradients)
    for param in forward_model.parameters():
        param.requires_grad = False

    # Set to eval mode for batch norm / dropout
    forward_model.eval()

    return forward_model

def compute_bt(input: DictConfig, output: DictConfig) -> None:
    """ Forward-model brightness temperatures from available data based on set of configurations.

        Parameters
        ----------
        input: DictConfig. Input configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Load forward model
    logger.info(f"Loading forward model...")
    config_forward = input.forward_model
    forward_model = load_forward_model(config_forward)

    # Loop over datasets
    for stage, config_stage in input.stage.items():
        # Load data
        logger.info(f"Loading input data for stage {stage}...")
        config_variables = config_stage.variables
        if 'prof' not in config_variables or 'meta' not in config_variables or 'surf' not in config_variables:
            raise ValueError("Input data configuration must contain 'prof', 'meta', and 'surf' keys.")
        input_variables = {
            'prof': load_variable(config_variables.prof),
            'meta': load_variable(config_variables.meta),
            'surf': load_variable(config_variables.surf),
        }

        # Compute brightness temperatures
        logger.info(f"Forward-modeling brightness temperatures...")
        forward_model_output = forward_model.forward(input_variables)

        # Save output data
        logger.info(f"Saving output data for stage {stage}...")
        # Save results
        for key, value in forward_model_output.items():
            config_variables = output[stage].variables
            if hasattr(config_variables, key):
                if hasattr(getattr(config_variables, key), 'save'):
                    if hasattr(getattr(config_variables, key), 'path'):
                        os.makedirs(os.path.dirname(getattr(config_variables, key).path), exist_ok=True)
                    save_func = instantiate(getattr(config_variables, key).save)
                    save_func(value)

    return


def compute_bt_v0(input: DictConfig, output: DictConfig) -> None:
    """ Forward-model brightness temperatures from available data based on set of configurations.

        Parameters
        ----------
        input: DictConfig. Input configuration file containing all model hyperparameters.
        output: DictConfig. Output configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Load forward model
    logger.info(f"Loading forward model...")
    config_forward = input.forward_model
    forward_model = load_forward_model(config_forward)

    # Load data
    logger.info(f"Loading input data...")
    config_data = input.data
    if 'prof' not in config_data or 'meta' not in config_data or 'surf' not in config_data:
        raise ValueError("Input data configuration must contain 'prof', 'meta', and 'surf' keys.")
    input_data = {
        'prof': load_variable(config_data.prof),
        'meta': load_variable(config_data.meta),
        'surf': load_variable(config_data.surf),
    }

    # Compute brightness temperatures
    logger.info(f"Forward-modeling brightness temperatures...")
    forward_model_output = forward_model.forward(input_data)

    # Save output data
    logger.info(f"Saving output data...")
    # Save results
    for key, value in forward_model_output.items():
        if hasattr(output, key):
            if hasattr(getattr(output, key), 'save'):
                if hasattr(getattr(output, key), 'path'):
                    os.makedirs(os.path.dirname(getattr(output, key).path), exist_ok=True)
                save_func = instantiate(getattr(output, key).save)
                save_func(value)

    return


@hydra.main(version_base=None, config_path=get_config_path(), config_name="default")
def main(config: DictConfig) -> None:
    """ Forward-model brightness temperatures from available data based on set of configurations.

        Parameters
        ----------
        config: DictConfig. Main hydra configuration file containing all model hyperparameters.

        Returns
        -------
        None.
    """

    # Compute latitude and longitude offsets
    if hasattr(config.preprocessing, "forward"):
        # If single operation, execute
        if hasattr(config.preprocessing.forward, "_target_"):
            logger.info(f"Forward-modeling brightness temperatures...")
            instantiate(config.preprocessing.forward)
        # Execute individual operations
        else:
            for dataset, config_forward in config.preprocessing.forward.items():
                logger.info(f"Forward-modeling brightness temperatures of {dataset} set...")
                instantiate(config_forward)

    return


if __name__ == '__main__':
    """ Forward-model from available data.

        Parameters
        ----------
        --config_path: str. Directory containing configuration file.
        --config_name: str. Configuration filename.
        +experiment: str. Experiment configuration filename to override default configuration.

        Returns
        -------
        Brightness temperatures as computed by the forward model.
    """

    main()