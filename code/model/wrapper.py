from omegaconf import DictConfig, ListConfig, OmegaConf
import logging


# Initialize logger
logger = logging.getLogger(__name__)

def load_model_from_config(path: str) -> DictConfig | ListConfig:
    """
    Load model configuration from a saved, resolved YAML file.

    This function loads the model configuration (typically model.yaml
    saved during model training). The returned DictConfig is then passed
    to the appropriate model class, which handles instantiation and checkpoint loading.

    Parameters
    ----------
    path : str
        Path to the model configuration YAML file (e.g., /path/to/checkpoints/model.yaml)

    Returns
    -------
    DictConfig | ListConfig
        Configuration that will be instantiated by ForwardModel.__init__
    """

    # Load model configuration
    logger.info(f"Loading model config from: {path}")
    config = OmegaConf.load(path)
    OmegaConf.update(config, "optimizer", None)
    OmegaConf.update(config, "scheduler", None)
    OmegaConf.update(config, "loss", None)
    return config
